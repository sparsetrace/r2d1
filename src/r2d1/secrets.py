"""
r2d1.secrets — Credential resolution.

In normal deployment bob.py reads secrets from bob_job.json and exports
them to os.environ before importing r2d1. So r2d1 just reads os.environ.

Search order (never overrides an already-set var):
  1. os.environ  (set by bob.py from bob_job.json)
  2. .env file in cwd or parents  (local dev only, requires python-dotenv)

R2D1_* names are canonical. A small set of common aliases is accepted so
the library works outside of the bob.py context too (e.g. local dev with
AWS_ or CF_ env vars already set).
"""
from __future__ import annotations

import os
from typing import Optional

# canonical key → accepted env var names (first match wins, R2D1_* first)
_ALIASES: dict[str, list[str]] = {
    "account_id":     ["R2D1_ACCOUNT_ID",    "CLOUDFLARE_ACCOUNT_ID"],
    "r2_bucket":      ["R2D1_R2_BUCKET",      "R2_BUCKET"],
    "r2_access_key":  ["R2D1_R2_ACCESS_KEY",  "AWS_ACCESS_KEY_ID"],
    "r2_secret_key":  ["R2D1_R2_SECRET_KEY",  "AWS_SECRET_ACCESS_KEY"],
    "r2_endpoint_url":["R2D1_R2_ENDPOINT_URL"],
    "api_token":      ["R2D1_API_TOKEN",       "CLOUDFLARE_API_TOKEN"],
    "d1_database_id": ["R2D1_D1_DATABASE_ID",  "D1_DATABASE_ID"],
    "session_token":  ["R2D1_SESSION_TOKEN"],
}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv, find_dotenv
        path = find_dotenv(usecwd=True)
        if path:
            load_dotenv(path, override=False)
    except ImportError:
        pass


def _resolve(cfg_key: str) -> Optional[str]:
    _load_dotenv()
    for name in _ALIASES.get(cfg_key, []):
        val = os.environ.get(name)
        if val:
            return val
    return None


def r2d1_config() -> dict:
    """Return resolved config dict. Missing optional keys are None."""
    return {k: _resolve(k) for k in _ALIASES}


def missing_r2(cfg: dict) -> list[str]:
    return [k for k in ("account_id", "r2_bucket", "r2_access_key", "r2_secret_key")
            if not cfg.get(k)]


def missing_d1(cfg: dict) -> list[str]:
    return [k for k in ("api_token", "d1_database_id") if not cfg.get(k)]


def config_error(component: str, missing: list[str]) -> RuntimeError:
    names = [_ALIASES[k][0] for k in missing]
    return RuntimeError(
        f"[r2d1] missing {component} credentials: {', '.join(names)}\n"
        "These should be present in os.environ — "
        "check that bob.py exported secrets from bob_job.json before importing r2d1."
    )


def r2_client_from_cfg(cfg: dict):
    import boto3
    endpoint = cfg.get("r2_endpoint_url") or (
        f"https://{cfg['account_id']}.r2.cloudflarestorage.com"
    )
    kwargs = dict(                              # ← was missing
        endpoint_url          = endpoint,
        aws_access_key_id     = cfg["r2_access_key"],
        aws_secret_access_key = cfg["r2_secret_key"],
        region_name           = "auto",
    )
    if cfg.get("session_token"):
        kwargs["aws_session_token"] = cfg["session_token"]
    return boto3.client("s3", **kwargs)
