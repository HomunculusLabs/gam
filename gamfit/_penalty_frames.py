"""Per-frame penalty adapters.

The dataclass penalty wrappers in :mod:`gamfit._penalties` expose
``value_grad(t)`` and ``hvp(t, v)``. The Rust kernel returns NumPy
arrays; for the torch and jax frames we wrap the same kernel so the
output is a native tensor with an autograd graph connecting ``value``
back to ``t``.

The math lives in one place: the Rust ``analytic_penalty_value_grad`` /
``analytic_penalty_hvp`` PyFFI functions. Each frame adapter is a thin
detach-cast-wrap layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._penalties import _penalty_value_grad_via_rust, _penalty_hvp_via_rust
from ._penalty_bridge import torch_value_grad_from_rust


# ---------------------------------------------------------------------------
# Torch frame
# ---------------------------------------------------------------------------


def torch_penalty_value_grad(wrapper: Any, t: Any) -> tuple[Any, Any]:
    """Torch-frame ``(value, grad)``, autograd-connected to ``t`` to second order.

    ``torch.autograd.grad(value, t)`` runs the Rust gradient, and a second
    gradient through it runs the Rust Hessian-vector product (see
    :func:`gamfit._penalty_bridge.torch_value_grad_from_rust`).
    """
    return torch_value_grad_from_rust(
        t,
        lambda x: _penalty_value_grad_via_rust(wrapper, x),
        lambda x, v: _penalty_hvp_via_rust(wrapper, x, v),
    )


def torch_penalty_hvp(wrapper: Any, t: Any, v: Any) -> Any:
    """Torch-frame Hessian-vector product. Pure forward (no autograd hook)."""
    from ._frame import import_torch
    from ._frame_torch import from_numpy_like, to_numpy_f64

    _ = import_torch()
    hv_np = _penalty_hvp_via_rust(wrapper, to_numpy_f64(t), to_numpy_f64(v))
    return from_numpy_like(np.asarray(hv_np, dtype=np.float64), t)


# ---------------------------------------------------------------------------
# JAX frame
# ---------------------------------------------------------------------------


def jax_penalty_value_grad(wrapper: Any, t: Any) -> tuple[Any, Any]:
    """JAX-frame ``(value, grad)`` differentiable via ``jax.grad``.

    Thin adapter over :func:`jax_value_grad_from_rust`: the only
    wrapper-specific piece is the host callback that runs the Rust
    ``analytic_penalty_value_grad`` kernel for this dataclass wrapper.
    """
    from ._penalty_bridge import jax_value_grad_from_rust

    shape = tuple(int(s) for s in t.shape)

    def _callback(x_np: np.ndarray) -> tuple[float, np.ndarray]:
        return _penalty_value_grad_via_rust(wrapper, x_np)

    return jax_value_grad_from_rust(
        type(wrapper).__name__, shape, _callback, ref=t
    )


def jax_penalty_hvp(wrapper: Any, t: Any, v: Any) -> Any:
    """JAX-frame Hessian-vector product through :func:`jax.pure_callback`."""
    from ._frame import import_jax
    from ._frame_jax import from_numpy_like, to_numpy_f64

    _ = import_jax()
    hv_np = _penalty_hvp_via_rust(wrapper, to_numpy_f64(t), to_numpy_f64(v))
    return from_numpy_like(np.asarray(hv_np, dtype=np.float64), t)


__all__ = [
    "torch_penalty_value_grad",
    "torch_penalty_hvp",
    "jax_penalty_value_grad",
    "jax_penalty_hvp",
]
