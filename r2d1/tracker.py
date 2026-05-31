"""
r2d1 core tracker.

- D1 stores job rows, epoch metrics, and checkpoint pointers.
- R2 stores artifacts/checkpoint files plus a manifest.json per checkpoint.
- D1 is only updated to point at a checkpoint after R2 upload + manifest succeed.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import posixpath
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError("r2d1 requires requests. Install with `pip install r2d1`." ) from exc

try:
    import boto3
    from botocore.config import Config
except ImportError as exc:  # pragma: no cover
    raise ImportError("r2d1 requires boto3. Install with `pip install r2d1`.") from exc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(obj: Any) -> Any:
    """JSON helper for common ML scalar/array values."""
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _json_dumps(data: Optional[Mapping[str, Any]]) -> str:
    return json.dumps(data or {}, default=_json_default, sort_keys=True, separators=(",", ":"))


def _safe_name(name: str) -> str:
    """Reject absolute paths and traversal while allowing nested artifact names."""
    name = str(name).replace("\\", "/")
    name = posixpath.normpath(name)
    if name in {".", ""} or name.startswith("/") or name.startswith("../") or "/../" in name:
        raise ValueError(f"unsafe artifact name: {name!r}")
    return name


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class _UploadItem:
    filename: str
    source: Any
    kind: str
    size: int
    sha256: str
    content_type: Optional[str] = None
    temp_path: Optional[Path] = None

    def cleanup(self) -> None:
        if self.temp_path is not None:
            try:
                self.temp_path.unlink(missing_ok=True)
            except Exception:
                pass


class Job:
    """Handle for a single training run."""

    def __init__(self, job_id: int, tracker: "Tracker"):
        self.job_id = int(job_id)
        self._t = tracker
        self._pending_upload: Optional[threading.Thread] = None
        self._pending_error: Optional[BaseException] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Metrics -> D1
    # ------------------------------------------------------------------

    def d1(
        self,
        *,
        epoch: int,
        duration_sec: Optional[float] = None,
        metrics: Optional[Mapping[str, Any]] = None,
        step: Optional[int] = None,
        checkpoint_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Write small JSON-serializable metrics/metadata to D1."""
        data: Dict[str, Any] = {}
        if metrics:
            data.update(dict(metrics))
        data.update(kwargs)

        loss = data.get("loss")
        accuracy = data.get("accuracy", data.get("acc"))
        logged_at = _now()
        metrics_json = _json_dumps(data)

        self._t._d1(
            """
            INSERT INTO epochs
                (job_id, epoch, step, loss, accuracy, duration_sec, metrics_json, checkpoint_prefix, logged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, epoch) DO UPDATE SET
                step=excluded.step,
                loss=excluded.loss,
                accuracy=excluded.accuracy,
                duration_sec=excluded.duration_sec,
                metrics_json=excluded.metrics_json,
                checkpoint_prefix=COALESCE(excluded.checkpoint_prefix, epochs.checkpoint_prefix),
                logged_at=excluded.logged_at
            """,
            [self.job_id, int(epoch), step, loss, accuracy, duration_sec, metrics_json, checkpoint_prefix, logged_at],
        )

    log = d1

    # ------------------------------------------------------------------
    # Artifacts/checkpoints -> R2
    # ------------------------------------------------------------------

    def r2(
        self,
        *,
        epoch: int,
        files: Mapping[str, Any],
        metrics: Optional[Mapping[str, Any]] = None,
        keep_last: int = 2,
        checkpoint_every: int = 1,
    ) -> str:
        """
        Upload files/artifacts to R2 and commit the checkpoint pointer in D1.

        D1 points only at complete checkpoints: files first, manifest second,
        D1 update last.
        """
        if not files:
            raise ValueError("files must not be empty")
        if keep_last <= 0:
            raise ValueError("keep_last must be >= 1")
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be >= 1")

        slot = (int(epoch) // int(checkpoint_every)) % int(keep_last)
        prefix = f"jobs/job_{self.job_id}/checkpoints/slot_{slot}/"
        manifest_key = prefix + "manifest.json"

        items = []
        try:
            for filename, source in files.items():
                items.append(self._prepare_item(filename, source))

            manifest_files: Dict[str, Dict[str, Any]] = {}
            for item in items:
                key = prefix + item.filename
                self._upload_item(key, item)
                manifest_files[item.filename] = {
                    "key": key,
                    "size": item.size,
                    "sha256": item.sha256,
                    "kind": item.kind,
                }
                if item.content_type:
                    manifest_files[item.filename]["content_type"] = item.content_type

            manifest = {
                "r2d1_version": "0.1.0",
                "job_id": self.job_id,
                "epoch": int(epoch),
                "slot": slot,
                "prefix": prefix,
                "created_at": _now(),
                "metrics": dict(metrics or {}),
                "files": manifest_files,
                "complete": True,
            }
            manifest_bytes = json.dumps(manifest, default=_json_default, indent=2, sort_keys=True).encode("utf-8")
            self._t._s3.put_object(
                Bucket=self._t.bucket,
                Key=manifest_key,
                Body=manifest_bytes,
                ContentType="application/json",
            )

            self._commit_checkpoint(
                epoch=int(epoch),
                slot=slot,
                prefix=prefix,
                manifest_key=manifest_key,
                metrics=metrics,
            )
            print(
                f"[r2d1] ✓ r2 checkpoint job={self.job_id} epoch={epoch} slot={slot} files={list(files.keys())}",
                flush=True,
            )
            return prefix
        finally:
            for item in items:
                item.cleanup()

    checkpoint = r2
    save = r2

    def r2_async(
        self,
        *,
        epoch: int,
        files: Mapping[str, Any],
        metrics: Optional[Mapping[str, Any]] = None,
        keep_last: int = 2,
        checkpoint_every: int = 1,
    ) -> None:
        """Start a background R2 upload. Only one upload runs at a time."""
        self.wait()

        # Capture the mapping immediately. Path sources remain paths and are
        # streamed by the background thread; bytes/dicts remain immutable enough
        # for normal use. Caller should not delete path files until wait/complete.
        captured = dict(files)
        captured_metrics = dict(metrics or {})

        def _run() -> None:
            try:
                self.r2(
                    epoch=epoch,
                    files=captured,
                    metrics=captured_metrics,
                    keep_last=keep_last,
                    checkpoint_every=checkpoint_every,
                )
            except BaseException as exc:  # save and re-raise on wait()
                with self._lock:
                    self._pending_error = exc

        self._pending_upload = threading.Thread(target=_run, name=f"r2d1-upload-job-{self.job_id}", daemon=True)
        self._pending_upload.start()

    save_async = r2_async

    def wait(self) -> None:
        """Block until any in-flight upload is complete, then surface upload errors."""
        thread = self._pending_upload
        if thread is not None:
            thread.join()
            self._pending_upload = None
        with self._lock:
            err = self._pending_error
            self._pending_error = None
        if err is not None:
            raise RuntimeError(f"r2d1 async upload failed for job {self.job_id}") from err

    def load_latest(self, *, include_manifest: bool = False) -> Any:
        """
        Download files from the latest committed checkpoint.

        Returns dict filename -> bytes, or (files, manifest) if include_manifest=True.
        """
        row = self._single_row(
            "SELECT last_checkpoint_prefix, last_checkpoint_manifest_key FROM jobs WHERE id=?",
            [self.job_id],
        )
        prefix = row.get("last_checkpoint_prefix") if row else None
        manifest_key = row.get("last_checkpoint_manifest_key") if row else None
        if not prefix or not manifest_key:
            raise ValueError(f"No checkpoint found for job {self.job_id}")

        manifest_obj = self._t._s3.get_object(Bucket=self._t.bucket, Key=manifest_key)
        manifest = json.loads(manifest_obj["Body"].read().decode("utf-8"))
        if not manifest.get("complete"):
            raise ValueError(f"Latest checkpoint manifest is not complete: {manifest_key}")

        files: Dict[str, bytes] = {}
        for filename, meta in manifest.get("files", {}).items():
            key = meta["key"]
            body = self._t._s3.get_object(Bucket=self._t.bucket, Key=key)["Body"].read()
            expected = meta.get("sha256")
            if expected and _sha256_bytes(body) != expected:
                raise ValueError(f"Checksum mismatch for {filename!r} in checkpoint {manifest_key}")
            files[filename] = body

        print(f"[r2d1] ✓ loaded checkpoint job={self.job_id} epoch={manifest.get('epoch')} files={list(files.keys())}")
        return (files, manifest) if include_manifest else files

    # ------------------------------------------------------------------
    # Status/lifecycle
    # ------------------------------------------------------------------

    def complete(self) -> None:
        self.wait()
        self._t._d1("UPDATE jobs SET status='completed', updated_at=? WHERE id=?", [_now(), self.job_id])
        print(f"[r2d1] ✓ job {self.job_id} completed", flush=True)

    def interrupt(self) -> None:
        # Do not wait forever on Ctrl-C. Mark interrupted; caller can resume from
        # the latest already-committed checkpoint.
        self._t._d1("UPDATE jobs SET status='interrupted', updated_at=? WHERE id=?", [_now(), self.job_id])
        print(f"[r2d1] ⚡ job {self.job_id} interrupted", flush=True)

    def status(self) -> Dict[str, Any]:
        job = self._single_row("SELECT * FROM jobs WHERE id=?", [self.job_id])
        epochs = self._rows("SELECT * FROM epochs WHERE job_id=? ORDER BY epoch", [self.job_id])
        checkpoints = self._rows("SELECT * FROM checkpoints WHERE job_id=? ORDER BY epoch", [self.job_id])
        return {"job": job, "epochs": epochs, "checkpoints": checkpoints}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _commit_checkpoint(
        self,
        *,
        epoch: int,
        slot: int,
        prefix: str,
        manifest_key: str,
        metrics: Optional[Mapping[str, Any]],
    ) -> None:
        now = _now()
        metrics_json = _json_dumps(metrics)
        self._t._d1(
            """
            INSERT INTO checkpoints
                (job_id, epoch, slot, prefix, manifest_key, metrics_json, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'complete', ?)
            ON CONFLICT(job_id, epoch) DO UPDATE SET
                slot=excluded.slot,
                prefix=excluded.prefix,
                manifest_key=excluded.manifest_key,
                metrics_json=excluded.metrics_json,
                status='complete',
                created_at=excluded.created_at
            """,
            [self.job_id, epoch, slot, prefix, manifest_key, metrics_json, now],
        )
        self._t._d1(
            """
            UPDATE jobs
            SET last_checkpoint_epoch=?,
                last_checkpoint_prefix=?,
                last_checkpoint_slot=?,
                last_checkpoint_manifest_key=?,
                updated_at=?
            WHERE id=?
            """,
            [epoch, prefix, slot, manifest_key, now, self.job_id],
        )
        self.d1(epoch=epoch, metrics=metrics or {}, checkpoint_prefix=prefix)

    def _single_row(self, sql: str, params: Optional[Iterable[Any]] = None) -> Optional[Dict[str, Any]]:
        rows = self._rows(sql, params)
        return rows[0] if rows else None

    def _rows(self, sql: str, params: Optional[Iterable[Any]] = None) -> list:
        result = self._t._d1(sql, list(params or []))
        return result.get("result", [{}])[0].get("results", [])

    def _prepare_item(self, filename: str, source: Any) -> _UploadItem:
        filename = _safe_name(filename)

        if isinstance(source, Path):
            path = source
            if not path.is_file():
                raise FileNotFoundError(path)
            return _UploadItem(filename, path, "path", path.stat().st_size, _sha256_path(path))

        if isinstance(source, str):
            maybe_path = Path(source)
            if maybe_path.exists() and maybe_path.is_file():
                return _UploadItem(filename, maybe_path, "path", maybe_path.stat().st_size, _sha256_path(maybe_path))
            data = source.encode("utf-8")
            return _UploadItem(filename, data, "text", len(data), _sha256_bytes(data), "text/plain; charset=utf-8")

        if isinstance(source, (bytes, bytearray, memoryview)):
            data = bytes(source)
            return _UploadItem(filename, data, "bytes", len(data), _sha256_bytes(data))

        if hasattr(source, "read"):
            # File-like sources may not be rewindable. Spool to a temp file so we
            # can compute checksum and upload safely.
            fd, temp_name = tempfile.mkstemp(prefix="r2d1-fileobj-")
            os.close(fd)
            temp_path = Path(temp_name)
            h = hashlib.sha256()
            size = 0
            with temp_path.open("wb") as out:
                while True:
                    chunk = source.read(1024 * 1024)
                    if chunk in (b"", "", None):
                        break
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8")
                    out.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
            return _UploadItem(filename, temp_path, "fileobj", size, h.hexdigest(), temp_path=temp_path)

        # JSON-ish objects. This includes dict/list configs.
        data = json.dumps(source, default=_json_default, indent=2, sort_keys=True).encode("utf-8")
        return _UploadItem(filename, data, "json", len(data), _sha256_bytes(data), "application/json")

    def _upload_item(self, key: str, item: _UploadItem) -> None:
        extra_args = {}
        if item.content_type:
            extra_args["ContentType"] = item.content_type

        if isinstance(item.source, Path):
            with item.source.open("rb") as f:
                self._t._s3.upload_fileobj(f, self._t.bucket, key, ExtraArgs=extra_args or None)
        elif isinstance(item.source, (bytes, bytearray)):
            self._t._s3.put_object(Bucket=self._t.bucket, Key=key, Body=bytes(item.source), **extra_args)
        else:
            raise TypeError(f"unsupported upload source for {item.filename!r}: {type(item.source)!r}")


class Tracker:
    """Cloudflare R2 + D1 tracker."""

    def __init__(
        self,
        *,
        account_id: str,
        api_token: str,
        d1_database_id: str,
        r2_bucket: str,
        r2_access_key: str,
        r2_secret_key: str,
        r2_endpoint_url: Optional[str] = None,
    ):
        self.account_id = account_id
        self.bucket = r2_bucket
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        self._d1_url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/d1/database/{d1_database_id}/query"
        )
        self._s3 = boto3.client(
            "s3",
            endpoint_url=r2_endpoint_url or f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=r2_access_key,
            aws_secret_access_key=r2_secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        self._init_tables()

    @classmethod
    def from_env(cls) -> "Tracker":
        """Construct from R2D1_* environment variables."""
        required = {
            "account_id": "R2D1_ACCOUNT_ID",
            "api_token": "R2D1_API_TOKEN",
            "d1_database_id": "R2D1_D1_DATABASE_ID",
            "r2_bucket": "R2D1_R2_BUCKET",
            "r2_access_key": "R2D1_R2_ACCESS_KEY",
            "r2_secret_key": "R2D1_R2_SECRET_KEY",
        }
        values = {}
        missing = []
        for arg, env in required.items():
            val = os.environ.get(env)
            if not val:
                missing.append(env)
            else:
                values[arg] = val
        if missing:
            raise EnvironmentError("Missing environment variables: " + ", ".join(missing))
        endpoint = os.environ.get("R2D1_R2_ENDPOINT_URL")
        if endpoint:
            values["r2_endpoint_url"] = endpoint
        return cls(**values)  # type: ignore[arg-type]

    def _d1(self, sql: str, params: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
        response = requests.post(
            self._d1_url,
            headers=self._headers,
            json={"sql": sql, "params": list(params or [])},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success", True):
            raise RuntimeError(f"D1 query failed: {payload}")
        return payload

    def _init_tables(self) -> None:
        self._d1(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                dataset_key TEXT,
                dataset_size_mb REAL,
                status TEXT NOT NULL DEFAULT 'pending',
                config_json TEXT,
                tags_json TEXT,
                last_checkpoint_epoch INTEGER,
                last_checkpoint_prefix TEXT,
                last_checkpoint_slot INTEGER,
                last_checkpoint_manifest_key TEXT,
                submitted_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._d1(
            """
            CREATE TABLE IF NOT EXISTS epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                step INTEGER,
                loss REAL,
                accuracy REAL,
                duration_sec REAL,
                metrics_json TEXT,
                checkpoint_prefix TEXT,
                logged_at TEXT NOT NULL,
                UNIQUE(job_id, epoch),
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )
        self._d1(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                slot INTEGER NOT NULL,
                prefix TEXT NOT NULL,
                manifest_key TEXT NOT NULL,
                metrics_json TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(job_id, epoch),
                FOREIGN KEY(job_id) REFERENCES jobs(id)
            )
            """
        )

    def start_job(
        self,
        name: str,
        *,
        dataset_key: Optional[str] = None,
        dataset_size_mb: Optional[float] = None,
        config: Optional[Mapping[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> Job:
        now = _now()
        result = self._d1(
            """
            INSERT INTO jobs
                (name, dataset_key, dataset_size_mb, status, config_json, tags_json, submitted_at, updated_at)
            VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
            """,
            [name, dataset_key, dataset_size_mb, _json_dumps(config), _json_dumps({"tags": list(tags or [])}), now, now],
        )
        job_id = result.get("result", [{}])[0].get("meta", {}).get("last_row_id")
        if job_id is None:
            # Fallback for unusual D1 responses.
            row = self._d1("SELECT id FROM jobs WHERE name=? AND submitted_at=? ORDER BY id DESC LIMIT 1", [name, now])
            job_id = row["result"][0]["results"][0]["id"]
        print(f"[r2d1] ✓ started job {job_id}: {name}", flush=True)
        return Job(int(job_id), self)

    def resume_job(self, job_id: int) -> Job:
        self._d1("UPDATE jobs SET status='running', updated_at=? WHERE id=?", [_now(), int(job_id)])
        print(f"[r2d1] ✓ resuming job {job_id}", flush=True)
        return Job(int(job_id), self)

    def get_job(self, job_id: int) -> Job:
        return Job(int(job_id), self)

    def list_jobs(self) -> list:
        return self._d1(
            """
            SELECT id, name, dataset_key, status, last_checkpoint_epoch,
                   last_checkpoint_prefix, submitted_at, updated_at
            FROM jobs
            ORDER BY id DESC
            """
        ).get("result", [{}])[0].get("results", [])

    def job(
        self,
        *,
        name: str,
        dataset_key: Optional[str] = None,
        dataset_size_mb: Optional[float] = None,
        config: Optional[Mapping[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        resume_job_id: Optional[int] = None,
    ):
        """Decorator that injects a Job and handles lifecycle."""
        from .loop import job_decorator

        return job_decorator(
            self,
            name=name,
            dataset_key=dataset_key,
            dataset_size_mb=dataset_size_mb,
            config=config,
            tags=tags,
            resume_job_id=resume_job_id,
        )
