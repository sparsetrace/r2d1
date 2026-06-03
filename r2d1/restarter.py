"""
r2d1.restarter — Pull the latest checkpoint from R2 to local disk.

Discovery order:
  1. Query D1 for the row with the highest epoch  (fast, one SQL call)
  2. If D1 unavailable, scan R2 objects and read sidecar JSON  (fallback)

Called by bob.py before launching the model.  The model sees only a local
directory and a RestartInfo dataclass — no cloud knowledge required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import boto3

from .d1 import D1Client
from .secrets import r2d1_config, missing_r2, missing_error


@dataclass
class RestartInfo:
    """Returned by Restarter.pull(). Passed directly to the model."""
    found:     bool
    epoch:     int        = 0
    name:      str        = ""
    local_dir: Path       = Path(".")
    meta:      dict       = field(default_factory=dict)


class Restarter:
    """
    Downloads the latest checkpoint for a job from R2 to a local directory.

    Usage (in bob.py, before launching the model):
        info = Restarter.from_env().pull(job_id="abc123", dest="/workspace/checkpoints")
        if info.found:
            # pass info.local_dir to the model
            ...
    """

    def __init__(self, r2_client, bucket: str, d1_client: Optional[D1Client] = None):
        self._r2     = r2_client
        self._bucket = bucket
        self._d1     = d1_client

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "Restarter":
        cfg = r2d1_config()
        missing = missing_r2(cfg)
        if missing:
            raise missing_error("R2", missing)

        endpoint = cfg["r2_endpoint_url"] or (
            f"https://{cfg['account_id']}.r2.cloudflarestorage.com"
        )
        r2 = boto3.client(
            "s3",
            endpoint_url          = endpoint,
            aws_access_key_id     = cfg["r2_access_key"],
            aws_secret_access_key = cfg["r2_secret_key"],
            region_name           = "auto",
        )
        d1 = D1Client.from_env_optional()
        return cls(r2_client=r2, bucket=cfg["r2_bucket"], d1_client=d1)

    # ── public API ────────────────────────────────────────────────────────────

    def pull(
        self,
        job_id: str,
        dest:   str | Path = "/workspace/checkpoints",
    ) -> RestartInfo:
        """
        Find and download the latest checkpoint for job_id.

        Returns RestartInfo(found=False) if no checkpoint exists.
        Returns RestartInfo(found=True, epoch=N, local_dir=Path(...)) on success.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

        meta = self._find_latest(job_id)
        if meta is None:
            print(f"[r2d1] no checkpoint found for job {job_id} — fresh start")
            return RestartInfo(found=False)

        name      = meta["name"]
        r2_prefix = meta["r2_prefix"]
        local_dir = dest / name
        local_dir.mkdir(parents=True, exist_ok=True)

        self._download_prefix(r2_prefix, local_dir)

        print(f"[r2d1] checkpoint {name} (epoch {meta['epoch']}) ready at {local_dir}")
        return RestartInfo(
            found     = True,
            epoch     = int(meta.get("epoch", 0)),
            name      = name,
            local_dir = local_dir,
            meta      = meta,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _find_latest(self, job_id: str) -> Optional[dict]:
        # Fast path: D1 query
        if self._d1:
            try:
                row = self._d1.query_one(
                    "SELECT * FROM checkpoints "
                    "WHERE job_id = ? "
                    "ORDER BY epoch DESC LIMIT 1",
                    [job_id],
                )
                if row:
                    # metadata is stored as JSON string in D1
                    row["metadata"] = json.loads(row.get("metadata") or "{}")
                    return row
            except Exception as e:
                print(f"[r2d1] D1 query failed ({e}), falling back to R2 scan")

        # Fallback: scan R2 for sidecar objects
        return self._scan_r2(job_id)

    def _scan_r2(self, job_id: str) -> Optional[dict]:
        """List all chk_*.json objects for job_id and return the highest-epoch one."""
        prefix    = f"jobs/{job_id}/"
        paginator = self._r2.get_paginator("list_objects_v2")
        sidecars  = []

        try:
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    # only top-level sidecars: jobs/<id>/chk_N.json
                    rel = key[len(prefix):]
                    if rel.startswith("chk_") and rel.endswith(".json") and "/" not in rel:
                        sidecars.append(key)
        except Exception as e:
            print(f"[r2d1] R2 scan failed: {e}")
            return None

        if not sidecars:
            return None

        # sidecars sort lexicographically correctly: chk_0042 > chk_0010
        latest_key = sorted(sidecars)[-1]

        try:
            obj  = self._r2.get_object(Bucket=self._bucket, Key=latest_key)
            meta = json.loads(obj["Body"].read())
            # derive r2_prefix from the sidecar key
            meta.setdefault("r2_prefix", latest_key.replace(".json", ""))
            meta.setdefault("name", Path(latest_key).stem)
            return meta
        except Exception as e:
            print(f"[r2d1] could not read sidecar {latest_key}: {e}")
            return None

    def _download_prefix(self, r2_prefix: str, local_dir: Path) -> None:
        """Download all objects under r2_prefix to local_dir."""
        paginator = self._r2.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=f"{r2_prefix}/"):
            for obj in page.get("Contents", []):
                key   = obj["Key"]
                fname = key.split("/")[-1]
                dest  = local_dir / fname
                print(f"[r2d1] downloading {fname}")
                self._r2.download_file(self._bucket, key, str(dest))
