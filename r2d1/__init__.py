"""
r2d1 — tiny ML experiment tracking on Cloudflare R2 + D1.

Public API:
    from r2d1 import Tracker, r2d1

    job = tracker.start_job("run")
    for epoch in r2d1(range(100), job=job):
        epoch.d1(loss=0.1)              # metrics / metadata -> D1
        epoch.r2({"ckpt.bin": path})   # artifacts / checkpoints -> R2
"""
from .tracker import Tracker, Job
from .loop import r2d1, Epoch

__version__ = "0.1.0"
__all__ = ["Tracker", "Job", "r2d1", "Epoch"]
