"""Core r2d1 tracker.

r2d1 is a courier between local GPU/notebook runs and persistent Cloudflare R2
storage. D1 is optional and used only as a lightweight SQL ledger.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

import boto3
import requests
from botocore.config import Config

from .credentials import missing_d1, missing_error, missing_r2, r2d1_config


def _now() -> str:
    # Sortable UTC timestamp.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "-", s.strip()).strip("-._")
    return s or "job"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, default=str)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Job:
    """Handle for one r2d1 training/run job."""

    def __init__(self, tracker: "Tracker", job_id: str, name: str, *, r2_prefix: str):
        self._t = tracker
        self.job_id = str(job_id)
        self.name = name
        self.r2_prefix = r2_prefix.rstrip("/") + "/"
        self._pending_upload: Optional[threading.Thread] = None
        self._pending_error: Optional[BaseException] = None

    # ------------------------ metrics / D1 ------------------------
    def d1(self, *, epoch: int | None = None, metrics: Optional[Mapping[str, Any]] = None, duration_sec: float | None = None, **kwargs: Any) -> None:
        """Log metrics. Uses D1 if configured; otherwise records nothing but warns once."""
        data: dict[str, Any] = {}
        if metrics:
            data.update(metrics)
        data.update(kwargs)
        self._t._log_metrics(self, epoch=epoch, metrics=data, duration_sec=duration_sec)

    log = d1

    # ------------------------ R2 checkpoints ------------------------
    def r2(
        self,
        *,
        epoch: int,
        files: Mapping[str, Any],
        metrics: Optional[Mapping[str, Any]] = None,
        keep_last: int = 2,
        checkpoint_index: int | None = None,
    ) -> str:
        """Synchronously upload one checkpoint/artifact bundle to R2."""
        self._t._ensure_r2()
        return self._t._upload_checkpoint(self, epoch=epoch, files=files, metrics=metrics, keep_last=keep_last, checkpoint_index=checkpoint_index)

    checkpoint = r2

    def r2_async(
        self,
        *,
        epoch: int,
        files: Mapping[str, Any],
        metrics: Optional[Mapping[str, Any]] = None,
        keep_last: int = 2,
        checkpoint_index: int | None = None,
    ) -> None:
        """Upload in a background thread. Only one upload is active at a time."""
        self.wait()
        captured = dict(files)
        captured_metrics = dict(metrics or {})
        captured_index = checkpoint_index

        def _run() -> None:
            try:
                self.r2(epoch=epoch, files=captured, metrics=captured_metrics, keep_last=keep_last, checkpoint_index=captured_index)
            except BaseException as e:  # preserve for wait()
                self._pending_error = e

        self._pending_upload = threading.Thread(target=_run, daemon=True)
        self._pending_upload.start()

    def wait(self) -> None:
        """Wait for any async upload and re-raise upload failures."""
        if self._pending_upload is not None:
            self._pending_upload.join()
            self._pending_upload = None
        if self._pending_error is not None:
            e = self._pending_error
            self._pending_error = None
            raise e

    def complete(self) -> None:
        """Mark complete if D1 is available, and write R2 job metadata."""
        self.wait()
        self._t._mark_job(self, status="completed")
        print(f"[r2d1] ✓ job {self.job_id} completed")

    def interrupt(self) -> None:
        """Optional graceful status marker. Hard preemption is inferred by timestamps."""
        self._t._mark_job(self, status="interrupted")
        print(f"[r2d1] ⚡ job {self.job_id} interrupted")

    def load_latest(self) -> dict[str, bytes]:
        """Download files from the latest committed checkpoint."""
        self._t._ensure_r2()
        latest = self._t._read_latest(self)
        prefix = latest.get("checkpoint_prefix")
        if not prefix:
            raise ValueError(f"No checkpoint found for job {self.job_id}")
        manifest = self._t._get_json(prefix + "manifest.json")
        files: dict[str, bytes] = {}
        for name in manifest.get("files", {}):
            obj = self._t._s3().get_object(Bucket=self._t.r2_bucket, Key=prefix + name)
            files[name] = obj["Body"].read()
        print(f"[r2d1] ✓ loaded checkpoint job={self.job_id} epoch={manifest.get('epoch')} files={list(files)}")
        return files

    def status(self) -> dict[str, Any]:
        if self._t.has_d1:
            job = self._t._d1("SELECT * FROM jobs WHERE id=?", [self.job_id])
            epochs = self._t._d1("SELECT * FROM epochs WHERE job_id=? ORDER BY epoch", [self.job_id])
            return {
                "job": job["result"][0]["results"][0] if job.get("result") else {},
                "epochs": epochs["result"][0]["results"] if epochs.get("result") else [],
            }
        return self._t._read_job_json(self)


class Tracker:
    """r2d1 tracker.

    Tracker.from_env() is lazy: it discovers available secrets/tokens but does not
    require R2 or D1 immediately. Starting a job or uploading to R2 validates R2.
    D1 remains optional.
    """

    def __init__(
        self,
        *,
        account_id: str | None = None,
        api_token: str | None = None,
        d1_database_id: str | None = None,
        r2_bucket: str | None = None,
        r2_access_key: str | None = None,
        r2_secret_key: str | None = None,
        r2_endpoint_url: str | None = None,
    ):
        self.account_id = account_id
        self.api_token = api_token
        self.d1_database_id = d1_database_id
        self.r2_bucket = r2_bucket
        self.r2_access_key = r2_access_key
        self.r2_secret_key = r2_secret_key
        self.r2_endpoint_url = r2_endpoint_url
        self._s3_client = None
        self._d1_ready = False
        self._warned: set[str] = set()

    @classmethod
    def from_env(cls, *, discover_common_tokens: bool = True) -> "Tracker":
        cfg = r2d1_config(discover_common_tokens=discover_common_tokens)
        return cls(**cfg)

    @property
    def has_r2(self) -> bool:
        return not missing_r2(self.__dict__)

    @property
    def has_d1(self) -> bool:
        # D1 needs account_id too, plus api token and database id.
        return bool(self.account_id and self.api_token and self.d1_database_id)

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            print(msg)
            self._warned.add(key)

    def _ensure_r2(self) -> None:
        missing = missing_r2(self.__dict__)
        if missing:
            raise missing_error("R2", missing)

    def _s3(self):
        self._ensure_r2()
        if self._s3_client is None:
            endpoint = self.r2_endpoint_url or f"https://{self.account_id}.r2.cloudflarestorage.com"
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=self.r2_access_key,
                aws_secret_access_key=self.r2_secret_key,
                config=Config(signature_version="s3v4"),
                region_name="auto",
            )
        return self._s3_client

    @property
    def _d1_url(self) -> str:
        return (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self.account_id}/d1/database/{self.d1_database_id}/query"
        )

    def _d1(self, sql: str, params: Optional[list[Any]] = None) -> dict[str, Any]:
        if not self.has_d1:
            raise RuntimeError("D1 is not configured")
        r = requests.post(
            self._d1_url,
            headers={"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"},
            json={"sql": sql, "params": params or []},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("success", False):
            raise RuntimeError(f"D1 query failed: {data}")
        return data

    def _init_tables(self) -> None:
        if not self.has_d1 or self._d1_ready:
            return
        self._d1(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                dataset_key TEXT,
                config_json TEXT,
                tags_json TEXT,
                status TEXT DEFAULT 'running',
                r2_prefix TEXT,
                last_checkpoint_prefix TEXT,
                last_checkpoint_epoch INTEGER,
                submitted_at TEXT,
                updated_at TEXT
            )
            """
        )
        self._d1(
            """
            CREATE TABLE IF NOT EXISTS epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT,
                epoch INTEGER,
                loss REAL,
                accuracy REAL,
                duration_sec REAL,
                metrics_json TEXT,
                logged_at TEXT
            )
            """
        )
        self._d1_ready = True

    def start_job(
        self,
        name: str,
        *,
        dataset_key: str | None = None,
        config: Mapping[str, Any] | None = None,
        tags: list[str] | tuple[str, ...] | None = None,
        dataset_size_mb: float | None = None,
    ) -> Job:
        """Start a job. R2 is required here; D1 is optional."""
        del dataset_size_mb  # kept for backward-ish compatibility; config can store this if needed.
        self._ensure_r2()
        if not self.has_d1:
            md = missing_d1(self.__dict__)
            self._warn_once(
                "d1_missing",
                "[r2d1] ⚠ D1 credentials not configured; running in R2-only mode. "
                f"Missing optional: {', '.join(md) if md else 'D1 config'}. "
                "Checkpoints still upload to R2; epoch.d1(...) metrics are not written to D1.",
            )

        now = _now()
        if self.has_d1:
            self._init_tables()
            res = self._d1(
                """
                INSERT INTO jobs (name, dataset_key, config_json, tags_json, status, submitted_at, updated_at)
                VALUES (?, ?, ?, ?, 'running', ?, ?)
                """,
                [name, dataset_key, _json_dumps(config or {}), _json_dumps(list(tags or [])), now, now],
            )
            job_id = str(res["result"][0]["meta"].get("last_row_id"))
            prefix = f"jobs/job_{job_id}/"
            self._d1("UPDATE jobs SET r2_prefix=? WHERE id=?", [prefix, job_id])
        else:
            job_id = f"{_slug(name)}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:8]}"
            prefix = f"jobs/{job_id}/"

        job = Job(self, job_id=job_id, name=name, r2_prefix=prefix)
        self._put_json(
            prefix + "job.json",
            {
                "job_id": job_id,
                "name": name,
                "dataset_key": dataset_key,
                "config": config or {},
                "tags": list(tags or []),
                "status": "running",
                "created_at": now,
                "updated_at": now,
                "mode": "r2+d1" if self.has_d1 else "r2-only",
            },
        )
        print(f"[r2d1] ✓ started job {job_id}: {name}")
        return job

    def resume_job(self, job_id: str | int) -> Job:
        self._ensure_r2()
        jid = str(job_id)
        if self.has_d1:
            self._init_tables()
            self._d1("UPDATE jobs SET status='running', updated_at=? WHERE id=?", [_now(), jid])
            res = self._d1("SELECT name, r2_prefix FROM jobs WHERE id=?", [jid])
            rows = res["result"][0]["results"]
            if not rows:
                raise ValueError(f"No D1 job found for id={jid}")
            name = rows[0].get("name") or f"job_{jid}"
            prefix = rows[0].get("r2_prefix") or f"jobs/job_{jid}/"
        else:
            name = jid
            prefix = f"jobs/{jid}/" if not jid.startswith("jobs/") else jid.rstrip("/") + "/"
        print(f"[r2d1] ✓ resuming job {jid}")
        return Job(self, jid, name, r2_prefix=prefix)

    def get_job(self, job_id: str | int) -> Job:
        jid = str(job_id)
        if self.has_d1:
            res = self._d1("SELECT name, r2_prefix FROM jobs WHERE id=?", [jid])
            rows = res["result"][0]["results"]
            if rows:
                return Job(self, jid, rows[0].get("name") or f"job_{jid}", r2_prefix=rows[0].get("r2_prefix") or f"jobs/job_{jid}/")
        return Job(self, jid, jid, r2_prefix=f"jobs/{jid}/")

    def list_jobs(self) -> list[dict[str, Any]]:
        if self.has_d1:
            self._init_tables()
            return self._d1("SELECT * FROM jobs ORDER BY id DESC")["result"][0]["results"]
        self._warn_once("list_jobs_r2", "[r2d1] D1 not configured; list_jobs() is limited in R2-only mode.")
        return []

    def job(self, name: str, **start_kwargs: Any):
        """Decorator that starts/completes a job around a function."""
        def _decorator(fn):
            def _wrapped(*args, **kwargs):
                job = self.start_job(name, **start_kwargs)
                result = fn(job, *args, **kwargs)
                job.complete()
                return result
            return _wrapped
        return _decorator

    # ------------------------ internal R2 helpers ------------------------
    def _put_json(self, key: str, obj: Any) -> None:
        body = _json_dumps(obj).encode("utf-8")
        self._s3().put_object(
            Bucket=self.r2_bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )

    def _get_json(self, key: str) -> dict[str, Any]:
        obj = self._s3().get_object(Bucket=self.r2_bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def _read_latest(self, job: Job) -> dict[str, Any]:
        # Prefer R2 latest.json. This works in both R2-only and D1 modes.
        try:
            return self._get_json(job.r2_prefix + "latest.json")
        except Exception:
            if self.has_d1:
                res = self._d1("SELECT last_checkpoint_prefix, last_checkpoint_epoch FROM jobs WHERE id=?", [job.job_id])
                rows = res["result"][0]["results"]
                if rows and rows[0].get("last_checkpoint_prefix"):
                    return {
                        "checkpoint_prefix": rows[0]["last_checkpoint_prefix"],
                        "epoch": rows[0].get("last_checkpoint_epoch"),
                    }
            return {}

    def _read_job_json(self, job: Job) -> dict[str, Any]:
        try:
            return self._get_json(job.r2_prefix + "job.json")
        except Exception:
            return {"job_id": job.job_id, "name": job.name, "r2_prefix": job.r2_prefix}

    def _prepare_object(self, value: Any) -> tuple[str, int, str, Optional[Path], Optional[bytes]]:
        """Return content_type, size, sha256, path, bytes for upload."""
        if isinstance(value, os.PathLike) or (isinstance(value, str) and Path(value).exists()):
            path = Path(value)
            size = path.stat().st_size
            digest = _sha256_file(path)
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return ctype, size, digest, path, None
        if isinstance(value, bytes):
            data = value
            return "application/octet-stream", len(data), _sha256_bytes(data), None, data
        if isinstance(value, str):
            data = value.encode("utf-8")
            return "text/plain; charset=utf-8", len(data), _sha256_bytes(data), None, data
        data = _json_dumps(value).encode("utf-8")
        return "application/json", len(data), _sha256_bytes(data), None, data

    def _upload_value(self, key: str, value: Any) -> dict[str, Any]:
        ctype, size, digest, path, data = self._prepare_object(value)
        if path is not None:
            extra = {"ContentType": ctype}
            self._s3().upload_file(str(path), self.r2_bucket, key, ExtraArgs=extra)
        else:
            self._s3().put_object(Bucket=self.r2_bucket, Key=key, Body=data, ContentType=ctype)
        return {"size": size, "sha256": digest, "content_type": ctype}

    def _upload_checkpoint(self, job: Job, *, epoch: int, files: Mapping[str, Any], metrics: Optional[Mapping[str, Any]], keep_last: int, checkpoint_index: int | None = None) -> str:
        keep_last = max(1, int(keep_last))
        # Slot is based on checkpoint count, not raw epoch, so checkpoint_every=10
        # with epochs 0,10,20 alternates slot_0, slot_1, slot_0.
        slot_counter = int(epoch if checkpoint_index is None else checkpoint_index)
        slot = slot_counter % keep_last
        prefix = f"{job.r2_prefix}checkpoints/slot_{slot}/"
        manifest_files: dict[str, Any] = {}
        for name, value in files.items():
            clean = str(name).lstrip("/")
            if ".." in Path(clean).parts:
                raise ValueError(f"Unsafe artifact name: {name!r}")
            manifest_files[clean] = self._upload_value(prefix + clean, value)

        manifest = {
            "format": "r2d1-checkpoint-manifest-v1",
            "job_id": job.job_id,
            "job_name": job.name,
            "epoch": epoch,
            "slot": slot,
            "checkpoint_prefix": prefix,
            "created_at": _now(),
            "metrics": dict(metrics or {}),
            "files": manifest_files,
        }
        self._put_json(prefix + "manifest.json", manifest)
        latest = {
            "format": "r2d1-latest-v1",
            "job_id": job.job_id,
            "job_name": job.name,
            "epoch": epoch,
            "slot": slot,
            "checkpoint_prefix": prefix,
            "updated_at": _now(),
        }
        self._put_json(job.r2_prefix + "latest.json", latest)

        if self.has_d1:
            self._init_tables()
            self._d1(
                "UPDATE jobs SET last_checkpoint_prefix=?, last_checkpoint_epoch=?, updated_at=? WHERE id=?",
                [prefix, epoch, _now(), job.job_id],
            )
        self._mark_job(job, status="running", quiet=True)
        print(f"[r2d1] ✓ checkpoint job={job.job_id} epoch={epoch} slot={slot} files={list(files)}")
        return prefix

    # ------------------------ internal D1/metadata helpers ------------------------
    def _log_metrics(self, job: Job, *, epoch: int | None, metrics: Mapping[str, Any], duration_sec: float | None) -> None:
        if not self.has_d1:
            self._warn_once("d1_metrics", "[r2d1] ⚠ D1 not configured; epoch.d1(...) metrics are not written to D1.")
            return
        self._init_tables()
        loss = metrics.get("loss")
        accuracy = metrics.get("accuracy")
        self._d1(
            """
            INSERT INTO epochs (job_id, epoch, loss, accuracy, duration_sec, metrics_json, logged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [job.job_id, epoch, loss, accuracy, duration_sec, _json_dumps(dict(metrics)), _now()],
        )
        self._d1("UPDATE jobs SET updated_at=? WHERE id=?", [_now(), job.job_id])

    def _mark_job(self, job: Job, *, status: str, quiet: bool = False) -> None:
        if self.has_d1:
            self._init_tables()
            self._d1("UPDATE jobs SET status=?, updated_at=? WHERE id=?", [status, _now(), job.job_id])
        # update R2 job.json opportunistically
        try:
            current = self._read_job_json(job)
            current.update({"status": status, "updated_at": _now()})
            self._put_json(job.r2_prefix + "job.json", current)
        except Exception:
            if not quiet:
                raise


# Top-level convenience API. Users do not need Tracker.from_env() for the common path.
def start_job(name: str, **kwargs: Any) -> Job:
    tracker = Tracker.from_env()
    return tracker.start_job(name, **kwargs)


def resume_job(job_id: str | int, **kwargs: Any) -> Job:
    del kwargs
    tracker = Tracker.from_env()
    return tracker.resume_job(job_id)


def get_job(job_id: str | int, **kwargs: Any) -> Job:
    del kwargs
    tracker = Tracker.from_env()
    return tracker.get_job(job_id)
