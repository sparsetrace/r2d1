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
from .credentials import (
    secret,
    require_secret,
    lookup_secret,
    export_secrets,
    r2d1_config,
    load_dotenv,
    MissingSecretError,
)

__version__ = "0.1.5"
__all__ = [
    "Tracker",
    "Job",
    "r2d1",
    "Epoch",
    "secret",
    "require_secret",
    "lookup_secret",
    "export_secrets",
    "r2d1_config",
    "load_dotenv",
    "MissingSecretError",
]
