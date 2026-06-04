"""
r2d1.courier — Watch a local directory and ship new checkpoints to R2 + D1.

The sidecar convention:
    <watch_dir>/slot_0/        checkpoint folder  → uploaded to R2
    <watch_dir>/slot_0.json    JSON sidecar       → triggers ship, upserted to D1

Courier polls for new or updated .json sidecars. When one appears:
  1. Enqueue every file in the matching folder for upload to R2 (async).
  2. Upsert a metadata row to D1 (heartbeat).
  3. Record (name, epoch) as shipped to avoid re-shipping the same version.

Keying on (name, epoch) means slot_0.json can be re-shipped each time DDIT
overwrites it with a new epoch — R2 always has the latest version of each slot.

Works with any checkpoint format — torch .pt, safetensors, HF save_pretrained(),
or anything else. r2d1 never inspects file contents.

Launch modes:
    In-process:  courier.watch(dir, job_id)           # background thread
    Subprocess:  python -m r2d1 watch <dir> --job-id  # fully isolated
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Optional

from .d1 import D1Client
from .secrets import r2d1_config, missing_r2, config_error, r2_client_from_cfg


class _AsyncUploader:
    """
    Drains an upload queue in a background daemon thread.
    The caller (training code) never blocks on network I/O.
    """

    def __init__(self, r2_client, bucket: str):
        self._r2     = r2_client
        self._bucket = bucket
        self._q: queue.Queue = queue.Queue()
        self._t = threading.Thread(target=self._worker, daemon=True)
        self._t.start()

    def submit(self, local_path: Path, r2_key: str) -> None:
        """Non-blocking enqueue."""
        self._q.put((local_path, r2_key))

    def flush(self, timeout: int = 300) -> None:
        """Block until all queued uploads are done."""
        self._q.join()

    def _worker(self) -> None:
        while True:
            local_path, r2_key = self._q.get()
            try:
                self._r2.upload_file(str(local_path), self._bucket, r2_key)
                print(f"[r2d1] ↑ {local_path.name} → {r2_key}")
            except Exception as exc:
                print(f"[r2d1] upload error {r2_key}: {exc}")
            finally:
                self._q.task_done()


class Courier:
    """
    Ship checkpoints to R2 + D1 as they appear on disk.

    Supports slot-based rotation (slot_0, slot_1, ...) — each slot is
    re-shipped every time DDIT overwrites it with a new epoch. R2 always
    holds the latest version of each slot.

    Typical bob.py usage:

        courier = Courier.from_config(cfg["secrets"])
        courier.watch("/root/checkpoints", job_id=job_id)
        # ... DDIT trains, writes slot_N/ + slot_N.json ...
        courier.flush(timeout=300)
    """

    def __init__(
        self,
        r2_client,
        bucket:    str,
        d1_client: Optional[D1Client] = None,
    ):
        self._uploader = _AsyncUploader(r2_client, bucket)
        self._d1       = d1_client
        # keyed on "slot_0.json:42" — (filename:epoch) so same slot at a
        # new epoch is treated as a new event and re-shipped
        self._shipped:  set[str] = set()
        self._d1_warned = False

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, secrets: dict) -> "Courier":
        """Build from the 'secrets' block of bob_job.json."""
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
    def from_env(cls) -> "Courier":
        """Build from os.environ (CLI or when secrets already exported)."""
        cfg     = r2d1_config()
        missing = missing_r2(cfg)
        if missing:
            raise config_error("R2", missing)
        r2 = r2_client_from_cfg(cfg)
        d1 = D1Client.from_env_optional()
        return cls(r2_client=r2, bucket=cfg["r2_bucket"], d1_client=d1)

    # ── Public API ────────────────────────────────────────────────────────────

    def watch(
        self,
        watch_dir:  str | Path,
        job_id:     str,
        poll_every: int  = 30,
        blocking:   bool = False,
    ) -> None:
        """
        Start watching watch_dir for new or updated sidecars.

        blocking=False (default): starts a daemon thread and returns immediately.
        blocking=True:  blocks forever — use this in subprocess/CLI mode.
        """
        t = threading.Thread(
            target = self._loop,
            args   = (Path(watch_dir), job_id, poll_every),
            daemon = True,
        )
        t.start()
        if blocking:
            t.join()

    def flush(self, timeout: int = 300) -> None:
        """Block until all queued R2 uploads are complete."""
        self._uploader.flush(timeout=timeout)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self, watch_dir: Path, job_id: str, poll_every: int) -> None:
        print(f"[r2d1] courier watching {watch_dir} (poll every {poll_every}s)")
        while True:
            self._scan(watch_dir, job_id)
            time.sleep(poll_every)

    def _scan(self, watch_dir: Path, job_id: str) -> None:
        if not watch_dir.exists():
            return
        # match both chk_*.json and slot_*.json
        for sidecar in sorted(watch_dir.glob("*.json")):
            if not (sidecar.stem.startswith("chk_") or
                    sidecar.stem.startswith("slot_")):
                continue
            self._process(sidecar, job_id)

    def _process(self, sidecar: Path, job_id: str) -> None:
        try:
            meta = json.loads(sidecar.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[r2d1] cannot read {sidecar.name}: {exc}")
            return

        name  = meta.get("name", sidecar.stem)
        epoch = meta.get("epoch", 0)

        # Key on (filename, epoch) so the same slot at a new epoch re-ships
        ship_key = f"{sidecar.name}:{epoch}"
        if ship_key in self._shipped:
            return

        chk_dir = sidecar.with_suffix("")   # slot_0.json → slot_0/

        if not chk_dir.is_dir():
            print(f"[r2d1] folder {chk_dir.name}/ not found — skipping")
            return

        r2_prefix = f"jobs/{job_id}/checkpoints/{name}"

        # Enqueue all checkpoint files
        for f in sorted(chk_dir.iterdir()):
            if f.is_file():
                self._uploader.submit(f, f"{r2_prefix}/{f.name}")

        # Also ship the sidecar itself
        self._uploader.submit(sidecar, f"{r2_prefix}/{sidecar.name}")

        # D1 heartbeat row
        if self._d1:
            try:
                self._d1.upsert("checkpoints", {
                    "job_id":    job_id,
                    "name":      name,
                    "epoch":     epoch,
                    "timestamp": meta.get("timestamp", time.time()),
                    "r2_prefix": r2_prefix,
                    "metadata":  json.dumps(meta.get("metadata", {})),
                })
                print(f"[r2d1] D1 heartbeat → {name} epoch {epoch}")
            except Exception as exc:
                print(f"[r2d1] D1 upsert failed for {name}: {exc}")
        else:
            if not self._d1_warned:
                print("[r2d1] D1 not configured — R2-only mode")
                self._d1_warned = True

        self._shipped.add(ship_key)
        print(f"[r2d1] {name} epoch {epoch} queued for upload → {r2_prefix}")
