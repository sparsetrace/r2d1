"""
r2d1.courier — Directory watcher that ships checkpoints to R2 and D1.

The sidecar convention:
    <watch_dir>/chk_0042/          # folder  → R2
    <watch_dir>/chk_0042.json      # sidecar → D1 + triggers ship

Courier polls for new .json sidecars. When one is found:
  1. Upload every file in the matching folder to R2.
  2. Upsert a metadata row to D1.
  3. Mark the sidecar as shipped (in-memory) to avoid re-shipping.

Launch modes:
  In-process  : courier.watch(dir, job_id)           # background thread
  Subprocess  : python -m r2d1 watch <dir> --job-id  # fully decoupled
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import boto3

from .d1 import D1Client
from .secrets import r2d1_config, missing_r2, missing_error, secret


class CourierError(RuntimeError):
    pass


class _AsyncUploader:
    """
    Background thread that drains an upload queue.
    Training / the caller never blocks on network I/O.
    """

    def __init__(self, r2_client, bucket: str):
        self._r2      = r2_client
        self._bucket  = bucket
        self._queue: queue.Queue = queue.Queue()
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, local_path: Path, r2_key: str) -> None:
        """Non-blocking enqueue."""
        self._queue.put((local_path, r2_key))

    def flush(self, timeout: int = 300) -> None:
        """Block until the queue is empty (all uploads done)."""
        self._queue.join()

    def _worker(self) -> None:
        while True:
            local_path, r2_key = self._queue.get()
            try:
                self._upload(local_path, r2_key)
            except Exception as e:
                print(f"[r2d1] upload error {r2_key}: {e}")
            finally:
                self._queue.task_done()

    def _upload(self, local_path: Path, r2_key: str) -> None:
        self._r2.upload_file(str(local_path), self._bucket, r2_key)
        print(f"[r2d1] uploaded {local_path.name} → {r2_key}")


class Courier:
    """
    Watches a local directory for checkpoint sidecars and ships them to R2 + D1.

    Usage (in-process background thread):
        courier = Courier.from_env()
        courier.watch("./checkpoints", job_id="abc123")
        # ... training happens here ...
        courier.flush(timeout=300)

    Usage (subprocess / bob.py):
        python -m r2d1 watch ./checkpoints --job-id abc123 --poll-every 30
    """

    def __init__(
        self,
        r2_client,
        bucket:     str,
        d1_client:  Optional[D1Client] = None,
    ):
        self._uploader  = _AsyncUploader(r2_client, bucket)
        self._d1        = d1_client
        self._shipped:  set[str] = set()   # sidecar names already processed
        self._d1_warned = False

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "Courier":
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

    def watch(
        self,
        watch_dir:  str | Path,
        job_id:     str,
        poll_every: int  = 30,
        blocking:   bool = False,
    ) -> None:
        """
        Start watching watch_dir for new sidecars.

        blocking=False (default): returns immediately, runs in background thread.
        blocking=True: blocks forever (use for subprocess / CLI mode).
        """
        t = threading.Thread(
            target  = self._loop,
            args    = (Path(watch_dir), job_id, poll_every),
            daemon  = True,
        )
        t.start()
        if blocking:
            t.join()

    def flush(self, timeout: int = 300) -> None:
        """Wait until all queued uploads have completed."""
        self._uploader.flush(timeout=timeout)

    # ── internal ──────────────────────────────────────────────────────────────

    def _loop(self, watch_dir: Path, job_id: str, poll_every: int) -> None:
        print(f"[r2d1] courier watching {watch_dir} every {poll_every}s")
        while True:
            self._scan(watch_dir, job_id)
            time.sleep(poll_every)

    def _scan(self, watch_dir: Path, job_id: str) -> None:
        if not watch_dir.exists():
            return
        for sidecar in sorted(watch_dir.glob("chk_*.json")):
            if sidecar.name in self._shipped:
                continue
            self._process(sidecar, job_id)

    def _process(self, sidecar: Path, job_id: str) -> None:
        # parse sidecar
        try:
            meta = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[r2d1] could not read {sidecar.name}: {e}")
            return

        name    = meta.get("name", sidecar.stem)
        chk_dir = sidecar.with_suffix("")   # chk_0042.json → chk_0042/

        if not chk_dir.is_dir():
            print(f"[r2d1] folder {chk_dir.name} missing, skipping")
            return

        r2_prefix = f"jobs/{job_id}/{name}"

        # enqueue all files in the folder
        for f in sorted(chk_dir.iterdir()):
            if f.is_file():
                self._uploader.submit(f, f"{r2_prefix}/{f.name}")

        # also ship the sidecar itself to R2
        self._uploader.submit(sidecar, f"{r2_prefix}/{sidecar.name}")

        # D1 upsert (heartbeat + metadata)
        if self._d1:
            try:
                self._d1.upsert("checkpoints", {
                    "job_id":    job_id,
                    "name":      name,
                    "epoch":     meta.get("epoch", -1),
                    "timestamp": meta.get("timestamp", time.time()),
                    "r2_prefix": r2_prefix,
                    "metadata":  json.dumps(meta.get("metadata", {})),
                })
                print(f"[r2d1] D1 upserted {name}")
            except Exception as e:
                print(f"[r2d1] D1 upsert failed for {name}: {e}")
        else:
            if not self._d1_warned:
                print("[r2d1] D1 not configured — running in R2-only mode")
                self._d1_warned = True

        self._shipped.add(sidecar.name)
        print(f"[r2d1] {name} queued for shipping")
