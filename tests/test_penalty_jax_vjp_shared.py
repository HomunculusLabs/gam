"""Contract test for the shared JAX value/grad custom-VJP wrapper.

Issue #417: ``gamfit._penalty_jax_vjp.jax_value_grad_from_rust`` is the
single marshalling contract behind the penalty-wrapper JAX path
(``gamfit._penalty_frames.jax_penalty_value_grad``).

Two things are asserted here:

1. The shared wrapper's ``custom_vjp`` is mathematically correct — its
   backward pass recovers the kernel's analytic gradient, verified by
   :func:`jax.test_util.check_grads` (first- and second-order, both ``fwd``
   and ``rev``) against finite differences.
2. A real penalty routed through that wrapper returns, in the JAX frame, the
   NumPy-frame value and gradient of the same Rust kernel.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gamfit._rust")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from jax.test_util import check_grads  # noqa: E402

jax.config.update("jax_enable_x64", True)

from gamfit._penalty_jax_vjp import jax_value_grad_from_rust  # noqa: E402


def test_shared_wrapper_check_grads_anisotropic_quadratic() -> None:
    """``check_grads`` on the wrapper with a non-trivial analytic kernel.

    The callback computes ``½ · Σ wᵢ tᵢ²`` and its exact gradient ``w ⊙ t``.
    The wrapper must route the forward through ``pure_callback`` and pull the
    analytic gradient through its backward — so autodiff of the scalar value
    matches finite differences to ``check_grads`` tolerance.
    """
    rng = np.random.default_rng(417)
    shape = (4, 3)
    weights = rng.uniform(0.3, 2.5, size=shape)

    def _callback(x_np: np.ndarray) -> tuple[float, np.ndarray]:
        x = np.asarray(x_np, dtype=np.float64).reshape(shape)
        value = 0.5 * float(np.sum(weights * x * x))
        grad = weights * x
        return value, grad

    def _scalar(x: object) -> object:
        value, _grad = jax_value_grad_from_rust(
            "aniso_quadratic", shape, _callback, ref=x
        )
        return value

    x0 = jnp.asarray(rng.standard_normal(shape))

    # The second return slot must equal the analytic gradient at x0.
    value_j, grad_j = jax_value_grad_from_rust(
        "aniso_quadratic", shape, _callback, ref=x0
    )
    np.testing.assert_allclose(
        np.asarray(value_j), 0.5 * np.sum(weights * np.asarray(x0) ** 2), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(grad_j), weights * np.asarray(x0), rtol=1e-12
    )

    # The custom_vjp backward must agree with finite differences (and, at
    # second order, stay self-consistent) in both fwd and rev modes.
    check_grads(_scalar, (x0,), order=2, modes=("fwd", "rev"), atol=1e-5, rtol=1e-5)


def test_penalty_wrapper_jax_frame_matches_its_numpy_frame() -> None:
    """A real penalty obeys the shared wrapper contract in the JAX frame.

    * the JAX-frame ``(value, grad)`` equals the NumPy-frame ground truth from
      the very same penalty (no dtype/shape drift across frames),
    * ``jax.grad`` of the scalar value recovers that analytic gradient
      (the ``custom_vjp`` backward is wired through the shared engine).
    """
    from gamfit._penalties import MechanismSparsityPenalty

    rng = np.random.default_rng(2417)
    d_latent, p_features = 3, 4
    penalty = MechanismSparsityPenalty(
        feature_groups=[[0, 1], [2, 3]],
        weight=0.7,
        n_eff=50.0,
        smoothing_eps=1.0e-6,
    )
    w_np = rng.standard_normal((d_latent, p_features))
    w_j = jnp.asarray(w_np)

    # NumPy-frame ground truth (same Rust kernel, no JAX).
    base_value, base_grad = penalty.value_grad(w_np)

    # JAX-frame value/grad must match the NumPy-frame baseline exactly.
    value_j, grad_j = penalty.value_grad(w_j)
    np.testing.assert_allclose(float(value_j), float(base_value), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(grad_j), np.asarray(base_grad), rtol=1e-12, atol=1e-12
    )
    # jax.grad through the shared custom_vjp recovers that analytic grad.
    grad_autodiff = jax.grad(lambda x: penalty.value_grad(x)[0])(w_j)
    np.testing.assert_allclose(
        np.asarray(grad_autodiff), np.asarray(base_grad), rtol=1e-6, atol=1e-9
    )
