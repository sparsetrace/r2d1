"""Secret discovery helpers for r2d1.

The public entry point is :func:`secret`:

    from r2d1 import secret
    hf_token = secret("HF_TOKEN", aliases=["HF_HUB_TOKEN"], required=False)

It searches common local/cloud/notebook locations, returns the secret as a
string, and by default copies the discovered value into ``os.environ`` under the
canonical name.  R2D1 itself uses the same machinery in ``Tracker.from_env()``.

Search model
------------
1. Local .env file, loaded with override=False.
2. os.environ.
3. Google Colab notebook secrets via google.colab.userdata, if available.
4. Kaggle notebook secrets via kaggle_secrets.UserSecretsClient, if available.

Most GPU/cloud providers -- Modal, Vast.ai, RunPod, Docker, GitHub Actions,
SageMaker, Vertex AI Workbench, Lightning AI, Paperspace, JupyterHub, Hugging
Face Spaces, etc. -- inject secrets as environment variables inside the job.
Those are intentionally covered by the os.environ step instead of using
provider-specific SDKs.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional


class MissingSecretError(RuntimeError):
    """Raised when a required secret cannot be found after all lookups."""


# Canonical names and common aliases.  Users can still pass aliases=[...] for
# project-specific names.  Values intentionally include the canonical name first
# for clear error messages and stable env export behavior.
DEFAULT_SECRET_ALIASES: Mapping[str, tuple[str, ...]] = {
    # Hugging Face
    "HF_TOKEN": (
        "HF_TOKEN",
        "HF_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ),
    # GitHub / git private repos
    "GITHUB_TOKEN": (
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "GIT_TOKEN",
    ),
    # Common experiment/model APIs
    "WANDB_API_KEY": ("WANDB_API_KEY", "WANDB_KEY"),
    "OPENAI_API_KEY": ("OPENAI_API_KEY",),
    "ANTHROPIC_API_KEY": ("ANTHROPIC_API_KEY",),
    "GOOGLE_API_KEY": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "KAGGLE_USERNAME": ("KAGGLE_USERNAME",),
    "KAGGLE_KEY": ("KAGGLE_KEY", "KAGGLE_API_KEY"),
    # Modal CLI tokens, if a user wants to resolve them in notebooks.
    "MODAL_TOKEN_ID": ("MODAL_TOKEN_ID",),
    "MODAL_TOKEN_SECRET": ("MODAL_TOKEN_SECRET",),
    # Vast.ai API key names seen in scripts/templates.
    "VAST_API_KEY": ("VAST_API_KEY", "VASTAI_API_KEY", "VAST_AI_API_KEY"),
    # Cloudflare / R2D1 bundle
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
        "R2_ACCESS_KEY_SECRET_ID",
        "AWS_ACCESS_KEY_ID",
    ),
    "R2D1_R2_SECRET_KEY": (
        "R2D1_R2_SECRET_KEY",
        "R2_SECRET_KEY",
        "R2_SECRET_ACCESS_KEY",
        "R2_SECRET_ACCESS_KEY_SECRET",
        "AWS_SECRET_ACCESS_KEY",
    ),
    "R2D1_R2_ENDPOINT_URL": (
        "R2D1_R2_ENDPOINT_URL",
        "R2_ENDPOINT_URL",
        "CLOUDFLARE_R2_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
    ),
}

R2D1_REQUIRED_SECRET_NAMES: tuple[str, ...] = (
    "R2D1_ACCOUNT_ID",
    "R2D1_API_TOKEN",
    "R2D1_D1_DATABASE_ID",
    "R2D1_R2_BUCKET",
    "R2D1_R2_ACCESS_KEY",
    "R2D1_R2_SECRET_KEY",
)

R2D1_OPTIONAL_SECRET_NAMES: tuple[str, ...] = (
    "R2D1_R2_ENDPOINT_URL",
)

PLATFORM_ENV_NOTE = (
    "Modal, Vast.ai, RunPod, Docker, GitHub Actions, SageMaker, Vertex AI "
    "Workbench, Lightning AI, Paperspace, JupyterHub, Hugging Face Spaces, "
    "and similar providers are covered when they inject secrets into os.environ."
)


@dataclass(frozen=True)
class SecretLookup:
    """Debug record for where a secret was found.

    The secret value itself is included only because callers explicitly request a
    lookup; do not print this object in logs.  For normal use prefer
    ``secret(...)``, which returns just the string.
    """

    name: str
    value: Optional[str]
    found_as: Optional[str]
    source: Optional[str]
    candidates: tuple[str, ...]
    checked: tuple[str, ...]

    @property
    def found(self) -> bool:
        return bool(self.value)


_DOTENV_LOADED: set[str] = set()


def _unique_names(name: str, aliases: Optional[Iterable[str]] = None) -> list[str]:
    out: list[str] = []
    for item in [*DEFAULT_SECRET_ALIASES.get(name, (name,)), *(aliases or ())]:
        s = str(item).strip()
        if s and s not in out:
            out.append(s)
    return out


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_dotenv_file(path: Path, *, override: bool = False) -> bool:
    """Small fallback parser for KEY=value and export KEY=value lines."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False

    line_re = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$")
    loaded_any = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        # Strip a simple trailing comment for unquoted values.
        if value and not value.startswith(("'", '"')) and " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        value = _strip_quotes(value)
        if override or key not in os.environ:
            os.environ[key] = value
            loaded_any = True
    return loaded_any


def _dotenv_candidates(dotenv_path: Optional[str | os.PathLike[str]]) -> list[Path]:
    if dotenv_path is not None:
        return [Path(dotenv_path).expanduser()]

    # Notebook-friendly: .env in cwd or any parent.  python-dotenv's find_dotenv
    # can be brittle inside notebooks depending on stack inspection, so we keep
    # our own explicit fallback path list.
    cwd = Path.cwd().resolve()
    return [p / ".env" for p in [cwd, *cwd.parents]]


def load_dotenv(
    *,
    dotenv_path: Optional[str | os.PathLike[str]] = None,
    override: bool = False,
) -> bool:
    """Load a local .env file without making python-dotenv a hard runtime detail.

    Existing environment variables are preserved unless ``override=True``.  The
    function returns True if a candidate .env was found/loaded.
    """
    candidates = _dotenv_candidates(dotenv_path)

    # Prefer python-dotenv when installed because it supports more syntax.
    try:
        from dotenv import load_dotenv as _load_dotenv  # type: ignore

        for path in candidates:
            if path.is_file():
                key = str(path.resolve())
                # Avoid re-loading the same file repeatedly during r2d1_config().
                if override or key not in _DOTENV_LOADED:
                    _load_dotenv(dotenv_path=path, override=override)
                    _DOTENV_LOADED.add(key)
                return True
    except Exception:
        pass

    for path in candidates:
        if path.is_file():
            key = str(path.resolve())
            if override or key not in _DOTENV_LOADED:
                _parse_dotenv_file(path, override=override)
                _DOTENV_LOADED.add(key)
            return True
    return False


def _from_env(candidates: Iterable[str]) -> tuple[Optional[str], Optional[str]]:
    for name in candidates:
        value = os.environ.get(name)
        if value:
            return str(value), name
    return None, None


def _from_colab(candidates: Iterable[str]) -> tuple[Optional[str], Optional[str], str]:
    try:
        from google.colab import userdata  # type: ignore
    except Exception as exc:
        return None, None, f"Google Colab userdata unavailable ({type(exc).__name__})"

    for name in candidates:
        try:
            value = userdata.get(name)
        except Exception:
            value = None
        if value:
            return str(value), name, "Google Colab userdata"
    return None, None, "Google Colab userdata checked"


def _from_kaggle(candidates: Iterable[str]) -> tuple[Optional[str], Optional[str], str]:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception as exc:
        return None, None, f"Kaggle UserSecretsClient unavailable ({type(exc).__name__})"

    try:
        client = UserSecretsClient()
    except Exception as exc:
        return None, None, f"Kaggle UserSecretsClient unavailable ({type(exc).__name__})"

    for name in candidates:
        try:
            value = client.get_secret(name)
        except Exception:
            value = None
        if value:
            return str(value), name, "Kaggle UserSecretsClient"
    return None, None, "Kaggle UserSecretsClient checked"


def lookup_secret(
    name: str,
    *,
    aliases: Optional[Iterable[str]] = None,
    set_env: bool = True,
    env_name: Optional[str] = None,
    load_dotenv_first: bool = True,
    dotenv_path: Optional[str | os.PathLike[str]] = None,
) -> SecretLookup:
    """Return detailed secret lookup info without raising.

    Normal users should call :func:`secret`.  This function is useful for tests
    and diagnostics because it records candidate names and checked locations.
    """
    candidates = tuple(_unique_names(name, aliases))
    checked: list[str] = []

    if load_dotenv_first:
        found_dotenv = load_dotenv(dotenv_path=dotenv_path, override=False)
        checked.append("local .env" + (" loaded" if found_dotenv else " not found"))

    value, found_as = _from_env(candidates)
    checked.append("os.environ")
    if value:
        if set_env:
            canonical = env_name or name
            os.environ[canonical] = value
            if found_as and found_as != canonical:
                os.environ.setdefault(found_as, value)
        return SecretLookup(name, value, found_as, "os.environ", candidates, tuple(checked))

    value, found_as, note = _from_colab(candidates)
    checked.append(note)
    if value:
        if set_env:
            canonical = env_name or name
            os.environ[canonical] = value
            if found_as and found_as != canonical:
                os.environ.setdefault(found_as, value)
        return SecretLookup(name, value, found_as, "Google Colab userdata", candidates, tuple(checked))

    value, found_as, note = _from_kaggle(candidates)
    checked.append(note)
    if value:
        if set_env:
            canonical = env_name or name
            os.environ[canonical] = value
            if found_as and found_as != canonical:
                os.environ.setdefault(found_as, value)
        return SecretLookup(name, value, found_as, "Kaggle UserSecretsClient", candidates, tuple(checked))

    checked.append(PLATFORM_ENV_NOTE)
    return SecretLookup(name, None, None, None, candidates, tuple(checked))


def _missing_secret_message(name: str, result: SecretLookup) -> str:
    return (
        f"Missing required secret {name!r}.\n"
        f"Tried names: {', '.join(result.candidates)}\n"
        "Search locations checked:\n  - "
        + "\n  - ".join(result.checked)
    )


def secret(
    name: str,
    *,
    aliases: Optional[Iterable[str]] = None,
    required: bool = True,
    set_env: bool = True,
    env_name: Optional[str] = None,
    load_dotenv_first: bool = True,
    dotenv_path: Optional[str | os.PathLike[str]] = None,
) -> Optional[str]:
    """Find a secret/token across common notebook and cloud environments.

    Parameters
    ----------
    name:
        Canonical environment variable name, e.g. ``HF_TOKEN`` or
        ``R2D1_ACCOUNT_ID``.
    aliases:
        Extra variable/secret names to try after built-in aliases.
    required:
        Raise :class:`MissingSecretError` if not found.  If False, return None.
    set_env:
        If found, write the value into ``os.environ[env_name or name]``.  This
        lets downstream libraries discover the token normally.
    env_name:
        Optional canonical environment variable name to populate.

    Returns
    -------
    str | None
        The discovered secret string, or None when ``required=False`` and the
        secret is absent.
    """
    result = lookup_secret(
        name,
        aliases=aliases,
        set_env=set_env,
        env_name=env_name,
        load_dotenv_first=load_dotenv_first,
        dotenv_path=dotenv_path,
    )
    if result.value:
        return result.value
    if required:
        raise MissingSecretError(_missing_secret_message(name, result))
    return None


def require_secret(name: str, **kwargs) -> str:
    """Strict variant of secret(...)."""
    value = secret(name, required=True, **kwargs)
    assert value is not None
    return value


def r2d1_config(
    *,
    required: bool = True,
    dotenv_path: Optional[str | os.PathLike[str]] = None,
    set_env: bool = True,
) -> dict[str, Optional[str]]:
    """Load the Cloudflare R2/D1 credential bundle for Tracker.from_env().

    Required names by default:
        R2D1_ACCOUNT_ID, R2D1_API_TOKEN, R2D1_D1_DATABASE_ID,
        R2D1_R2_BUCKET, R2D1_R2_ACCESS_KEY, R2D1_R2_SECRET_KEY

    Optional:
        R2D1_R2_ENDPOINT_URL
    """
    cfg: dict[str, Optional[str]] = {}
    results: dict[str, SecretLookup] = {}

    for name in [*R2D1_REQUIRED_SECRET_NAMES, *R2D1_OPTIONAL_SECRET_NAMES]:
        result = lookup_secret(
            name,
            set_env=set_env,
            dotenv_path=dotenv_path,
        )
        results[name] = result

    mapping = {
        "account_id": "R2D1_ACCOUNT_ID",
        "api_token": "R2D1_API_TOKEN",
        "d1_database_id": "R2D1_D1_DATABASE_ID",
        "r2_bucket": "R2D1_R2_BUCKET",
        "r2_access_key": "R2D1_R2_ACCESS_KEY",
        "r2_secret_key": "R2D1_R2_SECRET_KEY",
        "r2_endpoint_url": "R2D1_R2_ENDPOINT_URL",
    }
    for field, name in mapping.items():
        cfg[field] = results[name].value

    if not cfg.get("r2_endpoint_url") and cfg.get("account_id"):
        cfg["r2_endpoint_url"] = f"https://{cfg['account_id']}.r2.cloudflarestorage.com"

    missing = [name for name in R2D1_REQUIRED_SECRET_NAMES if not results[name].value]
    if missing and required:
        lines = ["Missing required R2D1 secrets:"]
        for name in missing:
            result = results[name]
            lines.append(f"\n  {name}\n    tried names: {', '.join(result.candidates)}")
        # The checked locations are identical enough to show once.
        exemplar = results[missing[0]]
        lines.append("\nSearch locations checked:\n  - " + "\n  - ".join(exemplar.checked))
        raise MissingSecretError("\n".join(lines))

    return cfg


def export_secrets(
    names: Iterable[str] | Mapping[str, Iterable[str]],
    *,
    required: bool = False,
    dotenv_path: Optional[str | os.PathLike[str]] = None,
) -> dict[str, Optional[str]]:
    """Resolve several secrets and populate os.environ.

    Examples
    --------
    >>> export_secrets(["HF_TOKEN", "GITHUB_TOKEN"], required=False)
    >>> export_secrets({"HF_TOKEN": ["HF_HUB_TOKEN"]}, required=False)
    """
    out: dict[str, Optional[str]] = {}
    if isinstance(names, Mapping):
        for name, aliases in names.items():
            out[name] = secret(name, aliases=aliases, required=required, set_env=True, dotenv_path=dotenv_path)
    else:
        for name in names:
            out[name] = secret(name, required=required, set_env=True, dotenv_path=dotenv_path)
    return out
