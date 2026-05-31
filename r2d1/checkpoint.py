"""Optional checkpoint helpers.

These are convenience utilities only. The core r2d1 path is framework-agnostic:
epoch.r2({"checkpoint.pt": Path(...)}) ships whatever files your code produces.
"""
from __future__ import annotations

import io
import pickle
from typing import Any

import numpy as np


def _is_torch_module(obj: Any) -> bool:
    try:
        import torch.nn as nn
        return isinstance(obj, nn.Module)
    except Exception:
        return False


def _torch_state_to_np(state: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in state.items():
        try:
            out[k] = v.detach().cpu().numpy()
        except Exception:
            out[k] = v
    return out


def _jax_to_np(pytree: Any) -> Any:
    try:
        import jax
        return jax.tree_util.tree_map(np.array, pytree)
    except Exception:
        return pytree


def serialize(epoch: int, model_or_params: Any, optimizer_state: Any = None, loss: float | None = None) -> bytes:
    """Serialize a simple torch module or JAX pytree to bytes.

    Prefer native checkpoint formats for serious training. This helper is best for
    small demos/tests.
    """
    if _is_torch_module(model_or_params):
        framework = "torch"
        params = _torch_state_to_np(model_or_params.state_dict())
        opt = _torch_state_to_np(optimizer_state.state_dict()) if optimizer_state else None
    else:
        framework = "jax-or-numpy"
        params = _jax_to_np(model_or_params)
        opt = _jax_to_np(optimizer_state) if optimizer_state is not None else None
    buf = io.BytesIO()
    pickle.dump({"epoch": epoch, "loss": loss, "framework": framework, "params": params, "optimizer": opt}, buf)
    return buf.getvalue()


def deserialize(data: bytes) -> dict[str, Any]:
    """Deserialize helper payload. Returns a raw dict."""
    return pickle.load(io.BytesIO(data))
