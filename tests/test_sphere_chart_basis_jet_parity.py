"""Cross-language parity / regression lock for the sphere basis + jet (#404).

Issue #404 was a *drift* bug: the sphere basis and its first jet were
implemented twice — once in the core Rust SAE path and once behind the PyFFI
``basis_with_jet`` helper — and the two copies disagreed. The Rust-side parity
guard pins the core surface; this file pins the **Python** surface, i.e. the
boundary where the drift actually lived, so an independent PyFFI/Torch copy can
never silently reappear.

The subject changed in #2602. ``b4150e873`` ("delete the (lat, lon) sphere
chart, no legacy path") removed ``SphereChartEvaluator``,
``sphere_chart_basis_jet``, ``SPHERE_CHART_PENALTY_DIAGONAL`` and both
``SphereChart`` plan variants, because S² admits no global 2-D chart and the
pole was a wall rather than a point. ``basis_with_jet("sphere", ...)`` now
routes to ``AmbientSphereHarmonicEvaluator`` on the **ambient** unit 3-vector
(``crates/gam-pyffi/src/model/model_ffi.rs:2717``). This file was left asserting
the deleted chart — 7 columns on ``(lat, lon)``, a tabulated
``[1e-8, 1, 1, 1, 4, 4, 4]`` penalty diagonal, and the ``chain_lat`` pole gate —
so all three of its Rust-backed tests died in ``check_coords`` with
"expected ambient dim == 3 (x, y, z), got 2" and the parity boundary went
unguarded.

Every assertion below is DERIVED, not tabulated:

  * width ``(degree + 1)²`` and jet width 3, from the ``(l, m)`` enumeration in
    ``AmbientSphereHarmonicEvaluator::new``;
  * the penalty diagonal is ``l2_gram_weight · [l(l+1)]²`` per column, read off
    ``spectral_modes()`` and the assembly at ``model_ffi.rs:3583``;
  * the normalization is checked by the spherical-harmonic **addition
    theorem**, ``Σ_m Y_lm(u)² = (2l + 1)/(4π)`` for every unit ``u`` — a
    convention-free, coordinate-free identity that no tabulated column list
    could substitute for;
  * the jet is checked against central finite differences of Φ in all three
    ambient axes. Because the ambient form has no clamp and no kink, this is a
    clean oracle at EVERY point, including the two former poles that used to
    need special-casing;
  * the deleted chart cannot come back: a 2-column ``(lat, lon)`` input must be
    rejected, not silently reinterpreted.
"""
from __future__ import annotations

import math
from importlib import import_module
from typing import Any

import numpy as np

pytest: Any = import_module("pytest")
gamfit = pytest.importorskip("gamfit")

# The PyFFI dispatch hard-codes degree 2 for `kind="sphere"`
# (`model_ffi.rs:2717`: `ambient_sphere_basis_with_jet(py, t, 2)`).
DEGREE = 2
WIDTH = (DEGREE + 1) ** 2


def _rust_sphere(coords: np.ndarray):
    """Evaluate the sphere basis through the PyFFI ``basis_with_jet`` dispatch.

    This is the exact entry point the Torch manifold-SAE path uses
    (``kind="sphere"``), so it exercises the PyFFI helper the issue named.
    """
    rust = gamfit._rust  # type: ignore[attr-defined]
    phi, jet, penalty = rust.basis_with_jet(
        "sphere", np.ascontiguousarray(coords, dtype=np.float64), {}
    )
    return np.asarray(phi), np.asarray(jet), np.asarray(penalty)


def _unit_rows() -> np.ndarray:
    """Fixed, reproducible unit vectors including both former chart poles.

    The poles are ordinary points for the ambient basis; including them is the
    positive statement that the pole wall is gone, replacing the old
    ``chain_lat`` gating tests.
    """
    rng = np.random.default_rng(404)
    raw = rng.standard_normal((12, 3))
    unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float64)
    return np.ascontiguousarray(np.vstack([unit, poles]))


def test_sphere_basis_width_and_penalty_are_the_ambient_harmonic_spectrum():
    coords = _unit_rows()
    phi, jet, penalty = _rust_sphere(coords)

    assert phi.shape == (coords.shape[0], WIDTH)
    assert jet.shape == (coords.shape[0], WIDTH, 3)
    assert penalty.shape == (WIDTH, WIDTH)

    # `penalty[c, c] = l2_gram_weight * laplace_eigenvalue^2` with
    # `laplace_eigenvalue = l(l+1)` and `l2_gram_weight = 1`, over the columns
    # in `(l, m)` order: one l=0, three l=1, five l=2.
    expected_diagonal = [
        float(degree * (degree + 1)) ** 2
        for degree in range(DEGREE + 1)
        for _ in range(2 * degree + 1)
    ]
    np.testing.assert_array_almost_equal(np.diag(penalty), expected_diagonal, decimal=12)
    # The penalty is diagonal in this basis; nothing off the diagonal.
    np.testing.assert_array_almost_equal(
        penalty - np.diag(np.diag(penalty)), np.zeros((WIDTH, WIDTH)), decimal=15
    )


def test_degree_blocks_satisfy_the_spherical_harmonic_addition_theorem():
    """`Σ_m Y_lm(u)² = (2l + 1)/(4π)` for every unit `u`.

    This pins the normalization AND the completeness of each degree block
    without tabulating a single column, and it is rotation-invariant, so it
    holds identically at the two points the deleted chart could not represent.
    """
    coords = _unit_rows()
    phi, _jet, _penalty = _rust_sphere(coords)

    start = 0
    for degree in range(DEGREE + 1):
        width = 2 * degree + 1
        block = phi[:, start : start + width]
        expected = (2 * degree + 1) / (4.0 * math.pi)
        np.testing.assert_allclose(
            np.sum(block**2, axis=1),
            np.full(coords.shape[0], expected),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"degree {degree} block violates the addition theorem",
        )
        start += width
    assert start == WIDTH


def test_jet_matches_finite_differences_in_every_ambient_axis():
    """The ambient form is a polynomial in (x, y, z): no clamp, no kink, so a
    central stencil is a valid oracle at every point — including the poles,
    which the chart form had to gate."""
    h = 1e-6
    coords = _unit_rows()
    _phi, jet, _penalty = _rust_sphere(coords)
    for axis in range(3):
        plus = coords.copy()
        minus = coords.copy()
        plus[:, axis] += h
        minus[:, axis] -= h
        phi_plus, _, _ = _rust_sphere(plus)
        phi_minus, _, _ = _rust_sphere(minus)
        fd = (phi_plus - phi_minus) / (2.0 * h)
        np.testing.assert_allclose(
            jet[:, :, axis],
            fd,
            rtol=0.0,
            atol=1e-6,
            err_msg=f"jet ambient axis {axis} disagrees with finite differences",
        )
    # A jet that is identically zero would pass an FD check vacuously.
    assert np.max(np.abs(jet)) > 1e-3


def test_chart_coordinates_are_rejected_not_reinterpreted():
    """The `(lat, lon)` chart is gone; a 2-column input must fail loudly rather
    than be read as a truncated ambient vector."""
    chart = np.array([[0.3, 0.9], [-1.2, 2.1]], dtype=np.float64)
    with pytest.raises(ValueError, match=r"expected ambient dim == 3"):
        _rust_sphere(chart)


def test_torch_autograd_backward_contracts_the_ambient_jet():
    """The production consumer: ``_BasisWithJetFn.backward`` contracts the saved
    jet, so the gradient of ``phi.sum()`` must match central FD of the same
    scalar in every ambient axis."""
    torch = pytest.importorskip("torch")
    mod = import_module("gamfit.torch.manifold_sae")
    basis_fn = mod._BasisWithJetFn

    coords = _unit_rows()[:4]
    t = torch.tensor(coords, dtype=torch.float64, requires_grad=True)
    phi = basis_fn.apply(t, "sphere", "{}")
    assert tuple(phi.shape) == (coords.shape[0], WIDTH)
    phi.sum().backward()
    grad = t.grad
    assert grad is not None and torch.isfinite(grad).all()

    def _phi_sum(coords_tensor: "torch.Tensor") -> float:
        with torch.no_grad():
            out = basis_fn.apply(coords_tensor, "sphere", "{}")
        return float(out.sum().item())

    h = 1e-6
    base = torch.tensor(coords, dtype=torch.float64)
    for row in range(coords.shape[0]):
        for axis in range(3):
            plus = base.clone()
            minus = base.clone()
            plus[row, axis] += h
            minus[row, axis] -= h
            fd = (_phi_sum(plus) - _phi_sum(minus)) / (2.0 * h)
            assert abs(grad[row, axis].item() - fd) <= 1e-6, (
                f"row {row} ambient axis {axis}: autograd "
                f"{grad[row, axis].item()} vs finite difference {fd}"
            )
