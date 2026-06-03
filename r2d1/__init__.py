"""
r2d1 — Lightweight ML checkpoint courier.

Cloudflare R2 for durable artifacts, D1 for metadata and heartbeat.
Model-agnostic: works with any training code that follows the sidecar convention.

Quick start:
    from r2d1 import Courier, Restarter

    # Ship checkpoints as they appear (background thread)
    courier = Courier.from_env()
    courier.watch("./checkpoints", job_id="abc123")
    # ... training writes chk_N/ + chk_N.json ...
    courier.flush()

    # Pull latest checkpoint before training resumes (blocking)
    info = Restarter.from_env().pull(job_id="abc123", dest="./checkpoints")
    if info.found:
        # load from info.local_dir
        ...

Sidecar convention:
    <watch_dir>/
    |-- chk_0042/          # checkpoint folder  -> R2
    `-- chk_0042.json      # JSON sidecar       -> D1 + triggers ship

See r2d1_notes.pdf for full design documentation.
"""

from .courier   import Courier
from .restarter import Restarter, RestartInfo
from .d1        import D1Client, D1Error
from .secrets   import (
    secret,
    require_secret,
    export_secrets,
    discover_common_secrets,
    r2d1_config,
    MissingSecretError,
)

__all__ = [
    "Courier",
    "Restarter",
    "RestartInfo",
    "D1Client",
    "D1Error",
    "secret",
    "require_secret",
    "export_secrets",
    "discover_common_secrets",
    "r2d1_config",
    "MissingSecretError",
]

__version__ = "0.1.0"
