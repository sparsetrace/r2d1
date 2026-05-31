"""
r2d1 credential and secret discovery.

The design goal is notebook/cloud portability without hardcoding secrets in code.

Search sources:
  1. .env in current directory or parents, without overriding existing env vars
  2. os.environ
  3. Google Colab userdata
  4. Kaggle UserSecretsClient

Modal, Vast.ai, RunPod, Docker, CI, SageMaker, Vertex, Lightning AI, etc. are
covered by os.environ once those platforms inject secrets into the process.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional


class MissingSecretError(RuntimeError):
    """Raised when a required secret cannot be found after all lookups."""


@dataclass(frozen=True)
class SecretLookup:
    name: str
    value: Optional[str]
    found_as: Optional[str]
    source: Optional[str]
    tried_names: tuple[str, ...]


# Canonical names and aliases. The canonical name is what gets populated into
# os.environ when set_env=True. R2D1_* names take precedence over generic names.
DEFAULT_SECRET_ALIASES: dict[str, tuple[str, ...]] = {
    # Hugging Face
    "HF_TOKEN": (
        "HF_TOKEN",
        "HF_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
    ),
    # GitHub / Git providers
    "GITHUB_TOKEN": ("GITHUB_TOKEN", "GH_TOKEN"),
    "GITLAB_TOKEN": ("GITLAB_TOKEN", "GL_TOKEN"),
    # Experiment/model services
    "WANDB_API_KEY": ("WANDB_API_KEY", "WANDB_KEY"),
    "COMET_API_KEY": ("COMET_API_KEY",),
    "NEPTUNE_API_TOKEN": ("NEPTUNE_API_TOKEN",),
    "MLFLOW_TRACKING_URI": ("MLFLOW_TRACKING_URI",),
    "MLFLOW_TRACKING_TOKEN": ("MLFLOW_TRACKING_TOKEN",),
    # LLM/data APIs users often need in notebooks
    "OPENAI_API_KEY": ("OPENAI_API_KEY",),
    "ANTHROPIC_API_KEY": ("ANTHROPIC_API_KEY",),
    "GOOGLE_API_KEY": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    # Kaggle
    "KAGGLE_USERNAME": ("KAGGLE_USERNAME",),
    "KAGGLE_KEY": ("KAGGLE_KEY", "KAGGLE_API_KEY"),
    # Provider APIs; these are optional tokens for user workflows, not used by r2d1 itself.
    "MODAL_TOKEN_ID": ("MODAL_TOKEN_ID",),
    "MODAL_TOKEN_SECRET": ("MODAL_TOKEN_SECRET",),
    "VAST_API_KEY": ("VAST_API_KEY", "VASTAI_API_KEY"),
    # Cloudflare / R2D1 canonical bundle
    "R2D1_ACCOUNT_ID": (
        "R2D1_ACCOUNT_ID",
        "CLOUDFLARE_ACCOUNT_ID",
        "CF_ACCOUNT_ID",
    ),
    "R2D1_API_TOKEN": (
        "R2D1_API_TOKEN",
        "CLOUDFLARE_API_TOKEN",
        "CF_API_TOKEN",
    ),
    "R2D1_D1_DATABASE_ID": (
        "R2D1_D1_DATABASE_ID",
        "D1_DATABASE_ID",
        "CLOUDFLARE_D1_DATABASE_ID",
        "CF_D1_DATABASE_ID",
    ),
    "R2D1_R2_BUCKET": (
        "R2D1_R2_BUCKET",
        "R2_BUCKET",
        "CLOUDFLARE_R2_BUCKET",
        "CF_R2_BUCKET",
    ),
    "R2D1_R2_ACCESS_KEY": (
        "R2D1_R2_ACCESS_KEY",
        "R2_ACCESS_KEY",
        "R2_ACCESS_KEY_ID",
        "CLOUDFLARE_R2_ACCESS_KEY",
        # Generic S3 envs last by design to avoid accidentally preferring AWS over R2-specific names.
        "AWS_ACCESS_KEY_ID",
    ),
    "R2D1_R2_SECRET_KEY": (
        "R2D1_R2_SECRET_KEY",
        "R2_SECRET_KEY",
        "R2_SECRET_ACCESS_KEY",
        "CLOUDFLARE_R2_SECRET_KEY",
        "AWS_SECRET_ACCESS_KEY",
    ),
    "R2D1_R2_ENDPOINT_URL": (
        "R2D1_R2_ENDPOINT_URL",
        "R2_ENDPOINT_URL",
        "CLOUDFLARE_R2_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
    ),
}

COMMON_OPTIONAL_SECRETS: tuple[str, ...] = (
    "HF_TOKEN",
    "GITHUB_TOKEN",
    "GITLAB_TOKEN",
    "WANDB_API_KEY",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "COMET_API_KEY",
    "NEPTUNE_API_TOKEN",
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_TOKEN",
    "MODAL_TOKEN_ID",
    "MODAL_TOKEN_SECRET",
    "VAST_API_KEY",
)

R2_REQUIRED_SECRETS: tuple[str, ...] = (
    "R2D1_ACCOUNT_ID",
    "R2D1_R2_BUCKET",
    "R2D1_R2_ACCESS_KEY",
    "R2D1_R2_SECRET_KEY",
)

D1_OPTIONAL_SECRETS: tuple[str, ...] = (
    "R2D1_API_TOKEN",
    "R2D1_D1_DATABASE_ID",
)

R2D1_ALL_SECRETS: tuple[str, ...] = (
    "R2D1_ACCOUNT_ID",
    "R2D1_API_TOKEN",
    "R2D1_D1_DATABASE_ID",
    "R2D1_R2_BUCKET",
    "R2D1_R2_ACCESS_KEY",
    "R2D1_R2_SECRET_KEY",
    "R2D1_R2_ENDPOINT_URL",
)

_DOTENV_LOADED: set[Path] = set()


def _candidate_dotenv_paths(start: Path | None = None) -> list[Path]:
    start = (start or Path.cwd()).resolve()
    if start.is_file():
        start = start.parent
    paths: list[Path] = []
    cur = start
    while True:
        p = cur / ".env"
        if p.exists() and p.is_file():
            paths.append(p)
        if cur.parent == cur:
            break
        cur = cur.parent
    return paths


def _fallback_load_dotenv(path: Path) -> None:
    """Small .env loader supporting KEY=VALUE and export KEY=VALUE."""
    line_re = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if "#" in value and not value.startswith(("'", '"')):
            value = value.split("#", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_dotenv(path: str | os.PathLike[str] | None = None) -> None:
    """Best-effort .env loading. Existing os.environ values are not overwritten."""
    paths = [Path(path).expanduser().resolve()] if path else _candidate_dotenv_paths()
    for p in paths:
        if p in _DOTENV_LOADED:
            continue
        try:
            from dotenv import load_dotenv as _load_dotenv  # type: ignore
            _load_dotenv(p, override=False)
        except Exception:
            _fallback_load_dotenv(p)
        _DOTENV_LOADED.add(p)


def names_for(name: str, aliases: Optional[Iterable[str]] = None) -> tuple[str, ...]:
    names = list(DEFAULT_SECRET_ALIASES.get(name, (name,)))
    for a in aliases or ():
        if a not in names:
            names.append(a)
    return tuple(names)


def _lookup_env(names: Iterable[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v, n, "environment/.env"
    return None, None, None


def _lookup_colab(names: Iterable[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        from google.colab import userdata  # type: ignore
    except Exception:
        return None, None, None
    for n in names:
        try:
            v = userdata.get(n)
        except Exception:
            v = None
        if v:
            return v, n, "Google Colab userdata"
    return None, None, None


def _lookup_kaggle(names: Iterable[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        return None, None, None
    try:
        client = UserSecretsClient()
    except Exception:
        return None, None, None
    for n in names:
        try:
            v = client.get_secret(n)
        except Exception:
            v = None
        if v:
            return v, n, "Kaggle UserSecretsClient"
    return None, None, None


def lookup_secret(
    name: str,
    *,
    aliases: Optional[Iterable[str]] = None,
    set_env: bool = True,
    env_name: Optional[str] = None,
    dotenv_path: str | os.PathLike[str] | None = None,
    load_dotenv_first: bool = True,
) -> SecretLookup:
    """Return structured lookup info without raising."""
    tried = names_for(name, aliases)
    if load_dotenv_first:
        load_dotenv(dotenv_path)

    value, found_as, source = _lookup_env(tried)
    if not value:
        value, found_as, source = _lookup_colab(tried)
    if not value:
        value, found_as, source = _lookup_kaggle(tried)

    if value and set_env:
        canonical = env_name or name
        os.environ[canonical] = value
        if found_as and found_as != canonical:
            os.environ.setdefault(found_as, value)

    return SecretLookup(
        name=name,
        value=value,
        found_as=found_as,
        source=source,
        tried_names=tried,
    )


def _locations_text() -> str:
    return (
        "local .env files, os.environ, Google Colab userdata, and Kaggle UserSecretsClient.\n"
        "Platform-injected secrets from Modal, Vast.ai, RunPod, Docker, CI, SageMaker, Vertex, "
        "Lightning AI, Paperspace, and similar providers are covered by os.environ."
    )


def secret(
    name: str,
    *,
    aliases: Optional[Iterable[str]] = None,
    required: bool = True,
    set_env: bool = True,
    env_name: Optional[str] = None,
    dotenv_path: str | os.PathLike[str] | None = None,
    load_dotenv_first: bool = True,
) -> Optional[str]:
    """
    Resolve a token/secret across common local, cloud, and notebook environments.

    If found and set_env=True, the value is copied into os.environ[env_name or name].
    If required=True and missing, MissingSecretError is raised only after all sources
    have been tried.
    """
    info = lookup_secret(
        name,
        aliases=aliases,
        set_env=set_env,
        env_name=env_name,
        dotenv_path=dotenv_path,
        load_dotenv_first=load_dotenv_first,
    )
    if info.value:
        return info.value
    if required:
        raise MissingSecretError(
            f"Missing required secret {name!r}.\n"
            f"Tried names: {', '.join(info.tried_names)}\n"
            f"Searched: {_locations_text()}"
        )
    return None


def require_secret(name: str, **kwargs) -> str:
    value = secret(name, required=True, **kwargs)
    assert value is not None
    return value


def export_secrets(
    names: Iterable[str] | Mapping[str, Iterable[str]],
    *,
    required: bool = False,
) -> dict[str, Optional[str]]:
    """Resolve several secrets and populate os.environ for any that are found."""
    out: dict[str, Optional[str]] = {}
    if isinstance(names, Mapping):
        for name, aliases in names.items():
            out[name] = secret(name, aliases=aliases, required=required, set_env=True)
    else:
        for name in names:
            out[name] = secret(name, required=required, set_env=True)
    return out


def discover_common_secrets(*, required: bool = False) -> dict[str, Optional[str]]:
    """Opportunistically load common ML/dev tokens such as HF_TOKEN and GITHUB_TOKEN."""
    return export_secrets(COMMON_OPTIONAL_SECRETS, required=required)


def r2d1_config(*, discover_common_tokens: bool = True) -> dict[str, Optional[str]]:
    """
    Resolve the R2D1 credential bundle without raising.

    R2 is validated lazily when a job/checkpoint is started. D1 is optional.
    """
    if discover_common_tokens:
        discover_common_secrets(required=False)
    return {
        "account_id": secret("R2D1_ACCOUNT_ID", required=False),
        "api_token": secret("R2D1_API_TOKEN", required=False),
        "d1_database_id": secret("R2D1_D1_DATABASE_ID", required=False),
        "r2_bucket": secret("R2D1_R2_BUCKET", required=False),
        "r2_access_key": secret("R2D1_R2_ACCESS_KEY", required=False),
        "r2_secret_key": secret("R2D1_R2_SECRET_KEY", required=False),
        "r2_endpoint_url": secret("R2D1_R2_ENDPOINT_URL", required=False),
    }


def missing_r2(cfg: Mapping[str, Optional[str]]) -> list[str]:
    checks = {
        "R2D1_ACCOUNT_ID": cfg.get("account_id"),
        "R2D1_R2_BUCKET": cfg.get("r2_bucket"),
        "R2D1_R2_ACCESS_KEY": cfg.get("r2_access_key"),
        "R2D1_R2_SECRET_KEY": cfg.get("r2_secret_key"),
    }
    return [k for k, v in checks.items() if not v]


def missing_d1(cfg: Mapping[str, Optional[str]]) -> list[str]:
    checks = {
        "R2D1_API_TOKEN": cfg.get("api_token"),
        "R2D1_D1_DATABASE_ID": cfg.get("d1_database_id"),
    }
    return [k for k, v in checks.items() if not v]


def missing_error(kind: str, missing: Iterable[str]) -> MissingSecretError:
    lines = [f"Missing required {kind} credentials:"]
    for name in missing:
        lines.append(f"  - {name}  (aliases: {', '.join(names_for(name))})")
    lines.append(f"Searched: {_locations_text()}")
    return MissingSecretError("\n".join(lines))
