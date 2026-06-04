"""
r2d1 — Lightweight ML checkpoint courier for Cloudflare R2 + D1.

    from r2d1 import Courier, Fetcher, FetchInfo

Courier  — watch a local directory and ship new checkpoints to R2 + D1
Fetcher  — pull checkpoints or datasets from R2 to local disk

Both classes accept .from_config(secrets_dict) to read credentials
directly from the bob_job.json secrets block, or .from_env() when
credentials are already in os.environ.
"""

from .courier import Courier
from .fetcher import Fetcher, FetchInfo

__all__ = ["Courier", "Fetcher", "FetchInfo"]
__version__ = "0.2.0"
