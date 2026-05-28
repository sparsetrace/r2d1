"""
Checkpoint serialization.
Auto-detects PyTorch (nn.Module) vs JAX (param pytree).
Always serializes to numpy so checkpoints are framework-agnostic.
"""
import io
import pickle
import numpy as np


def _is_torch(obj):
    try:
        import torch.nn as nn
        return isinstance(obj, nn.Module)
    except ImportError:
        return False


def _torch_to_np(state_dict):
    return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}


def _np_to_torch(d):
    import torch
    return {k: torch.tensor(v) for k, v in d.items()}


def _jax_to_np(pytree):
    import jax
    return jax.tree_util.tree_map(np.array, pytree)


def _np_to_jax(pytree):
    import jax.numpy as jnp
    import jax
    return jax.tree_util.tree_map(jnp.array, pytree)


def serialize(epoch, model_or_params, optimizer_state=None, loss=None):
    """Serialize checkpoint to bytes. Works for torch or JAX."""
    if _is_torch(model_or_params):
        framework = 'torch'
        params_np = _torch_to_np(model_or_params.state_dict())
        opt_np    = _torch_to_np(optimizer_state.state_dict()) if optimizer_state else None
    else:
        framework = 'jax'
        params_np = _jax_to_np(model_or_params)
        opt_np    = _jax_to_np(optimizer_state) if optimizer_state else None

    buf = io.BytesIO()
    pickle.dump({
        'epoch': epoch, 'loss': loss,
        'framework': framework,
        'params': params_np, 'optimizer': opt_np,
    }, buf)
    buf.seek(0)
    return buf.getvalue()


def deserialize(data, model_or_params=None, optimizer_state=None):
    """
    Deserialize checkpoint bytes.
    Returns (epoch, loss, params, optimizer_state).
    For torch: loads weights in-place and returns the same objects.
    For JAX:   returns new pytrees.
    """
    payload = pickle.load(io.BytesIO(data))
    epoch, loss = payload['epoch'], payload['loss']
    framework   = payload['framework']

    if framework == 'torch' and model_or_params is not None:
        model_or_params.load_state_dict(_np_to_torch(payload['params']))
        if optimizer_state and payload['optimizer']:
            optimizer_state.load_state_dict(_np_to_torch(payload['optimizer']))
        return epoch, loss, model_or_params, optimizer_state

    if framework == 'jax':
        params = _np_to_jax(payload['params'])
        opt    = _np_to_jax(payload['optimizer']) if payload['optimizer'] else None
        return epoch, loss, params, opt

    # fallback: return raw numpy
    return epoch, loss, payload['params'], payload['optimizer']
