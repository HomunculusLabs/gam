"""Backward checks for ``gamfit.torch.gaussian_reml_fit_blocks``."""

from __future__ import annotations

import pytest

gt = pytest.importorskip("gamfit.torch")
torch = pytest.importorskip("torch")


def _smooth_block(
    n: int,
    k: int,
    phase: float,
    frequency: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.remainder(
        frequency * torch.arange(n, dtype=torch.float64) / n + phase,
        1.0,
    )
    return gt.periodic_spline_curve_basis(t, k)


def _block_setup(
    n: int = 40,
    k: int = 6,
    blocks: int = 3,
    seed: int = 0,
    *,
    identified: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    from gamfit.torch.fit import _weighted_sum_to_zero_chart

    g = torch.Generator().manual_seed(seed)
    designs: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    y = torch.zeros(n, 1, dtype=torch.float64)
    for block in range(blocks):
        # Phase shifts alone span the same cyclic-spline function space and
        # therefore make the per-block smoothing parameters non-identifiable.
        # Distinct frequencies exercise a genuinely joint, identified REML
        # optimum while preserving the same periodic basis construction.
        design, penalty = _smooth_block(
            n,
            k,
            phase=0.017 * block,
            frequency=block + 1,
        )
        if identified:
            design, penalty, _ = _weighted_sum_to_zero_chart(design, penalty, None)
        coef = torch.randn(design.shape[1], 1, generator=g, dtype=torch.float64)
        designs.append(design)
        penalties.append(penalty)
        y = y + design @ coef
    noise = 0.05 * torch.randn(n, 1, generator=g, dtype=torch.float64)
    return designs, penalties, (y + noise).squeeze(1)


def _scalar_loss(result) -> torch.Tensor:
    return (
        result.fitted.sum()
        + result.lambdas.sum()
        + result.log_lambdas.sum()
        + result.reml_score
        + result.edf.sum()
        + sum(coef.sum() for coef in result.coefficients)
    )


def test_f1_matches_single_smooth_forward_and_backward() -> None:
    designs, penalties, y = _block_setup(
        n=60, k=8, blocks=1, seed=1, identified=False,
    )
    design, penalty = designs[0], penalties[0]

    x_single = design.clone().requires_grad_(True)
    p_single = penalty.clone().requires_grad_(True)
    y_single = y.clone().requires_grad_(True)
    single = gt.gaussian_reml_fit(x_single, y_single.unsqueeze(1), p_single)
    grad_single = torch.autograd.grad(
        single.coefficients.square().sum(),
        [x_single, p_single, y_single],
    )

    x_blocks = design.clone().requires_grad_(True)
    p_blocks = penalty.clone().requires_grad_(True)
    y_blocks = y.clone().requires_grad_(True)
    blocks = gt.gaussian_reml_fit_blocks([x_blocks], [p_blocks], y_blocks)
    grad_blocks = torch.autograd.grad(
        blocks.coefficients[0].square().sum(),
        [x_blocks, p_blocks, y_blocks],
    )

    torch.testing.assert_close(blocks.coefficients[0], single.coefficients, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(blocks.fitted, single.fitted, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(blocks.lambdas[0], single.lam.reshape(-1)[0], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        blocks.reml_score.reshape(-1)[0],
        single.reml_score.reshape(-1)[0],
        rtol=1e-10,
        atol=1e-10,
    )
    for actual, expected in zip(grad_blocks, grad_single):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_raw_shared_periodic_null_direction_is_rejected_on_both_paths() -> None:
    import numpy as np

    from gamfit import _api

    designs, penalties, y = _block_setup(
        n=30, k=6, blocks=2, seed=2, identified=False,
    )
    match = "joint coefficient map is not identified"
    with pytest.raises(ValueError, match=match):
        gt.gaussian_reml_fit_blocks(designs, penalties, y)

    with pytest.raises(ValueError, match=match):
        _api.gaussian_reml_fit_blocks_backward(
            [design.detach().numpy() for design in designs],
            [penalty.detach().numpy() for penalty in penalties],
            y.detach().numpy().reshape(-1, 1),
            np.zeros(2, dtype=np.float64),
            grad_reml_score=1.0,
        )


def test_public_blocks_gradcheck() -> None:
    designs, penalties, y = _block_setup(n=24, k=5, blocks=3, seed=3)
    inputs = tuple(
        tensor.clone().requires_grad_(True)
        for tensor in [*designs, *penalties, y.unsqueeze(1)]
    )

    def fit_loss(*args: torch.Tensor) -> torch.Tensor:
        split = (len(args) - 1) // 2
        return _scalar_loss(
            gt.gaussian_reml_fit_blocks(
                list(args[:split]),
                list(args[split:-1]),
                args[-1],
            )
        )

    assert torch.autograd.gradcheck(
        fit_loss,
        inputs,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        nondet_tol=1e-6,
    )


def test_blocks_function_gradcheck() -> None:
    from gamfit.torch._reml import _GaussianRemlFitBlocksFn

    designs, penalties, y = _block_setup(n=18, k=4, blocks=2, seed=11)
    inputs = tuple(
        tensor.clone().requires_grad_(True)
        for tensor in [y.unsqueeze(1), *designs, *penalties]
    )

    def fit_loss(y_arg: torch.Tensor, *blocks: torch.Tensor) -> torch.Tensor:
        n_blocks = len(blocks) // 2
        out = _GaussianRemlFitBlocksFn.apply(
            y_arg,
            None,
            None,
            n_blocks,
            *blocks,
        )
        coefficients, fitted, lambdas, log_lambdas, reml_score, edf = out
        return (
            coefficients.sum()
            + fitted.sum()
            + lambdas.sum()
            + log_lambdas.sum()
            + reml_score
            + edf.sum()
        )

    assert torch.autograd.gradcheck(
        fit_loss,
        inputs,
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
        nondet_tol=1e-6,
    )


def test_block_order_permutation_equivariance() -> None:
    designs, penalties, y = _block_setup(n=50, k=7, blocks=2, seed=101)
    forward = gt.gaussian_reml_fit_blocks(designs, penalties, y)
    swapped = gt.gaussian_reml_fit_blocks(
        [designs[1], designs[0]],
        [penalties[1], penalties[0]],
        y,
    )

    # The old lambda-ratio smoke was not a gamfit invariant: a joint two-block
    # fit receives one response and cannot bind named target components to
    # named blocks. Block-order equivariance is the stable invariant here.
    torch.testing.assert_close(swapped.fitted, forward.fitted, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(swapped.reml_score, forward.reml_score, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(swapped.lambdas, forward.lambdas.flip(0), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(swapped.edf, forward.edf.flip(0), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(swapped.coefficients[0], forward.coefficients[1], rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(swapped.coefficients[1], forward.coefficients[0], rtol=1e-10, atol=1e-10)
