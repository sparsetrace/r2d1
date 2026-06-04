"""
r2d1.fetcher — Pull checkpoints and datasets from Cloudflare R2 to local disk.

Only handles R2 sources. HuggingFace is out of scope for now.

URI schemes:
    r2://jobs/<job_id>/latest    latest checkpoint for a job (D1 → R2 fallback)
    r2://jobs/<job_id>/<name>    specific named checkpoint folder
    r2://<any/prefix>            arbitrary R2 prefix → local folder

Fetcher is model-agnostic. It puts files on disk and returns a FetchInfo.
It has no idea what DDIT or any training code will do with those files.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .d1 import D1Client
from .secrets import r2d1_config, missing_r2, config_error, r2_client_from_cfg


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class FetchInfo:
    local_dir: Path             # where files landed on disk
    source:    str              # original URI passed to pull()
    found:     bool  = True     # False → no checkpoint exists, fresh start
    cached:    bool  = False    # True → already on disk, download skipped
    epoch:     Optional[int] = None   # set for r2://jobs/.../latest pulls
    size_mb:   float = 0.0


# ── Fetcher ───────────────────────────────────────────────────────────────────

class Fetcher:
    """
    Pull checkpoints and data from R2 to local disk.

    Typical bob.py usage:

        # Build from the secrets block in bob_job.json
        fetcher = Fetcher.from_config(cfg["secrets"])

        # Pull latest checkpoint (returns found=False on fresh start — no exception)
        info = fetcher.pull(f"r2://jobs/{job_id}/latest", dest="/root/checkpoints")
        if info.found:
            print(f"resuming from epoch {info.epoch} at {info.local_dir}")

        # Pull a dataset stored in R2
        fetcher.pull("r2://datasets/mnist", dest="/root/data")
    """

    def __init__(self, r2_client, bucket: str, d1_client: Optional[D1Client] = None):
        self._r2     = r2_client
        self._bucket = bucket
        self._d1     = d1_client

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, secrets: dict) -> "Fetcher":
        """
        Build from the 'secrets' block of bob_job.json.

        Exports the secrets to os.environ so downstream libraries
        (boto3, requests) pick them up automatically, then constructs
        the R2 client directly.

        Args:
            secrets: the dict at cfg["secrets"] from bob_job.json
        """
        import os
        for key, val in secrets.items():
            if val and key != "_comment":
                os.environ.setdefault(key, str(val))

        cfg     = r2d1_config()
        missing = missing_r2(cfg)
        if missing:
            raise config_error("R2", missing)

        r2 = r2_client_from_cfg(cfg)
        d1 = D1Client.from_env_optional()
        return cls(r2_client=r2, bucket=cfg["r2_bucket"], d1_client=d1)

    @classmethod
    def from_env(cls) -> "Fetcher":
        """Build from os.environ (for CLI use or when secrets already exported)."""
        cfg     = r2d1_config()
        missing = missing_r2(cfg)
        if missing:
            raise config_error("R2", missing)
        r2 = r2_client_from_cfg(cfg)
        d1 = D1Client.from_env_optional()
        return cls(r2_client=r2, bucket=cfg["r2_bucket"], d1_client=d1)

    # ── Public API ────────────────────────────────────────────────────────────

    def pull(
        self,
        source: str,
        dest:   str | Path,
        force:  bool = False,
    ) -> FetchInfo:
        """
        Pull an R2 source to local disk.

        Args:
            source:  r2:// URI
            dest:    local directory to download into
            force:   re-download even if files already exist on disk

        Returns:
            FetchInfo — always returns, never raises on "not found".
            Check info.found to distinguish fresh-start from resume.
        """
        dest = Path(dest)
        if not source.startswith("r2://"):
            raise ValueError(
                f"[r2d1] Fetcher only handles r2:// URIs, got: {source!r}\n"
                "For local paths pass the path directly to your training code."
            )
        return self._pull_r2(source, dest, force)

    # ── R2 routing ────────────────────────────────────────────────────────────

    def _pull_r2(self, source: str, dest: Path, force: bool) -> FetchInfo:
        path  = source[len("r2://"):]
        parts = path.split("/")

        # r2://jobs/<id>/latest  →  special: find latest epoch
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "latest":
            return self._pull_latest(job_id=parts[1], dest=dest, force=force)

        # everything else → download the prefix as-is
        return self._download_prefix(prefix=path, dest=dest, force=force,
                                     source=source)

    # ── Latest-checkpoint logic ───────────────────────────────────────────────

    def _pull_latest(self, job_id: str, dest: Path, force: bool) -> FetchInfo:
        source = f"r2://jobs/{job_id}/latest"
        prefix, epoch = self._find_latest_prefix(job_id)

        if prefix is None:
            print(f"[r2d1] no checkpoint found for job {job_id!r} — fresh start")
            dest.mkdir(parents=True, exist_ok=True)
            return FetchInfo(local_dir=dest, source=source, found=False)

        name      = prefix.rstrip("/").split("/")[-1]   # chk_0042
        local_dir = dest / name

        if local_dir.exists() and not force:
            print(f"[r2d1] {name} already on disk — skipping download")
            return FetchInfo(local_dir=local_dir, source=source,
                             found=True, cached=True, epoch=epoch)

        info       = self._download_prefix(prefix, local_dir, force, source)
        info.epoch = epoch
        info.found = True
        return info

    def _find_latest_prefix(self, job_id: str) -> tuple[Optional[str], Optional[int]]:
        """
        Return (r2_prefix, epoch) for the highest-epoch checkpoint of job_id.
        Queries D1 first (one fast SQL row). Falls back to scanning R2 objects.
        Returns (None, None) if no checkpoint exists.
        """
        # ── D1 fast path ──────────────────────────────────────────────────
        if self._d1:
            try:
                row = self._d1.query_one(
                    "SELECT r2_prefix, epoch FROM checkpoints "
                    "WHERE job_id = ? ORDER BY epoch DESC LIMIT 1",
                    [job_id],
                )
                if row:
                    print(f"[r2d1] D1: latest checkpoint is epoch {row['epoch']}")
                    return row["r2_prefix"], int(row["epoch"])
            except Exception as exc:
                print(f"[r2d1] D1 query failed, falling back to R2 scan: {exc}")

        # ── R2 fallback — scan for chk_N/ prefixes ────────────────────────
        list_prefix = f"jobs/{job_id}/"
        paginator   = self._r2.get_paginator("list_objects_v2")
        best_epoch  = -1
        best_prefix = None

        for page in paginator.paginate(
            Bucket    = self._bucket,
            Prefix    = list_prefix,
            Delimiter = "/",
        ):
            for cp in page.get("CommonPrefixes", []):
                folder = cp["Prefix"].rstrip("/")      # jobs/<id>/chk_0042
                name   = folder.split("/")[-1]         # chk_0042

                # Try reading the sidecar JSON for the epoch number
                sidecar_key = f"{folder}/{name}.json"
                ep = None
                try:
                    obj  = self._r2.get_object(Bucket=self._bucket, Key=sidecar_key)
                    meta = json.loads(obj["Body"].read())
                    ep   = int(meta.get("epoch", -1))
                except Exception:
                    pass

                # Fall back to parsing epoch from folder name (chk_0042 → 42)
                if ep is None:
                    try:
                        ep = int(name.split("_")[-1])
                    except ValueError:
                        continue

                if ep > best_epoch:
                    best_epoch  = ep
                    best_prefix = folder

        if best_prefix:
            print(f"[r2d1] R2 scan: latest checkpoint is epoch {best_epoch}")
            return best_prefix, best_epoch
        return None, None

    # ── Core download ─────────────────────────────────────────────────────────

    def _download_prefix(
        self,
        prefix: str,
        dest:   Path,
        force:  bool,
        source: str,
    ) -> FetchInfo:
        """
        Download every object under R2 prefix/ into local dest/.
        Preserves sub-directory structure. Skips files already on disk
        unless force=True.
        """
        dest.mkdir(parents=True, exist_ok=True)
        paginator  = self._r2.get_paginator("list_objects_v2")
        total_mb   = 0.0
        n_files    = 0
        n_skipped  = 0

        for page in paginator.paginate(
            Bucket = self._bucket,
            Prefix = prefix.rstrip("/") + "/",
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel = key[len(prefix):].lstrip("/")
                if not rel:
                    continue

                local_f = dest / rel
                local_f.parent.mkdir(parents=True, exist_ok=True)

                if local_f.exists() and not force:
                    n_skipped += 1
                    continue

                size_mb = obj.get("Size", 0) / 1_048_576
                print(f"[r2d1] ↓ {rel}  ({size_mb:.1f} MB)")
                self._r2.download_file(self._bucket, key, str(local_f))
                total_mb += size_mb
                n_files  += 1

        if n_skipped:
            print(f"[r2d1] {n_skipped} file(s) already on disk — skipped")
        print(f"[r2d1] downloaded {n_files} file(s) ({total_mb:.1f} MB) → {dest}")
        return FetchInfo(local_dir=dest, source=source, size_mb=total_mb)
