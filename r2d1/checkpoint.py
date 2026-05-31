"""
Optional checkpoint serialization helpers.

Core r2d1 is framework-agnostic: it ships whatever files you pass to epoch.r2().
These helpers are convenience functions for small/simple PyTorch or JAX cases.
For serious training runs, prefer framework-native checkpointing such as safetensors,
Orbax, Flax serialization, torch.save, or your own format, then pass Paths to r2d1.
"""
from __future__ import annotations

import io
import pickle
from typing import Any, Optional, Tuple

import numpy as np


class UnsafePickleWarning(RuntimeWarning):
    """Pickle checkpoints should only be loaded from trusted sources."""


def _is_torch_module(obj: Any) -> bool:
    try:
        import torch.nn as nn  # type: ignore

        return isinstance(obj, nn.Module)
    except ImportError:
        return False


def _torch_tree_to_numpy(obj: Any) -> Any:
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None

    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, dict):
        return {k: _torch_tree_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_torch_tree_to_numpy(v) for v in obj)
    return obj


def _numpy_tree_to_torch(obj: Any) -> Any:
    import torch  # type: ignore

    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {k: _numpy_tree_to_torch(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_numpy_tree_to_torch(v) for v in obj)
    return obj


def _jax_to_numpy(pytree: Any) -> Any:
    import jax  # type: ignore

    return jax.tree_util.tree_map(lambda x: np.asarray(x) if hasattr(x, "shape") else x, pytree)


def _numpy_to_jax(pytree: Any) -> Any:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore

    return jax.tree_util.tree_map(lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x, pytree)


def serialize(epoch: int, model_or_params: Any, optimizer_state: Optional[Any] = None, loss: Optional[float] = None) -> bytes:
    """
    Serialize a simple PyTorch module or JAX pytree checkpoint to bytes.

    Note: uses pickle. Only deserialize data you trust.
    """
    if _is_torch_module(model_or_params):
        framework = "torch"
        params = _torch_tree_to_numpy(model_or_params.state_dict())
        optimizer = _torch_tree_to_numpy(optimizer_state.state_dict()) if optimizer_state is not None else None
    else:
        framework = "jax"
        params = _jax_to_numpy(model_or_params)
        optimizer = _jax_to_numpy(optimizer_state) if optimizer_state is not None else None

    payload = {
        "epoch": int(epoch),
        "loss": loss,
        "framework": framework,
        "params": params,
        "optimizer": optimizer,
    }
    buf = io.BytesIO()
    pickle.dump(payload, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()


def deserialize(data: bytes, model_or_params: Optional[Any] = None, optimizer_state: Optional[Any] = None) -> Tuple[int, Optional[float], Any, Any]:
    """
    Deserialize checkpoint bytes.

    For torch, pass model_or_params and optionally optimizer_state to load in-place.
    For JAX, returns new pytrees.
    """
    payload = pickle.load(io.BytesIO(data))
    epoch = payload["epoch"]
    loss = payload.get("loss")
    framework = payload.get("framework")

    if framework == "torch" and model_or_params is not None:
        model_or_params.load_state_dict(_numpy_tree_to_torch(payload["params"]))
        if optimizer_state is not None and payload.get("optimizer") is not None:
            optimizer_state.load_state_dict(_numpy_tree_to_torch(payload["optimizer"]))
        return epoch, loss, model_or_params, optimizer_state

    if framework == "jax":
        params = _numpy_to_jax(payload["params"])
        optimizer = _numpy_to_jax(payload["optimizer"]) if payload.get("optimizer") is not None else None
        return epoch, loss, params, optimizer

    return epoch, loss, payload.get("params"), payload.get("optimizer")
