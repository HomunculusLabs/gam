#!/usr/bin/env python3
"""App C causal half — synthetic patched-reconstruction demo for the spike-level
edit op (defined below), validating the *mechanism* end to end without a model.

The claim under test: removing ONE point mass from a superposed circle code edits
exactly that instance and leaves the other instance(s) untouched — a per-instance
causal handle that a linear SAE, which has no separate coordinate for the two
firings of the same feature, cannot provide.

Construction
------------
1. Plant a two-instance measure ``{(a1, t1), (a2, t2)}`` on an ``H``-harmonic
   circle and synthesize its ``2H`` block code ``z``.
2. Lift it into a ``p``-space with a random orthonormal-ish decoder ``D`` (2H×p):
   ``x = z · D`` — the "activation" a token carrying both instances would show.
3. Build ``Δx`` for ``remove(spike 0)`` via :func:`spike_edit_delta_x` and patch
   ``x' = x + Δx``.
4. Re-encode ``x'`` to its harmonic code and check: it equals the single-instance
   code ``a2·u(t2)`` (the surviving instance) to machine precision, while the
   removed instance's content is gone.
5. Two matched-filter "downstream readouts" — one tuned to ``t1``, one to ``t2`` —
   confirm the ``t1`` readout collapses and the ``t2`` readout is unchanged: the
   edit moved the readout for that instance ONLY.

Also exercises ``move`` and ``scale`` edits. Exit code is nonzero if any
tolerance fails, so this doubles as a self-test.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


# The spike-level edit op. It is analysis math for this demo, so it lives here
# rather than in the production gamfit package.


@dataclass(frozen=True)
class MeasureSpikeEdit:
    """A single-spike edit of a recovered circle measure.

    Attributes
    ----------
    kind
        ``"remove"`` (delete the spike), ``"move"`` (retune its position to
        ``new_t``), or ``"scale"`` (multiply its amplitude by ``factor``).
    spike_index
        Index into the row's spike list (positions sorted ascending, matching
        the Rust readout's ordering).
    new_t
        Target circle position for ``"move"`` (in ``[0, 1)``); ignored otherwise.
    factor
        Amplitude multiplier for ``"scale"`` (``0`` reproduces ``"remove"``);
        ignored otherwise.
    """

    kind: str
    spike_index: int
    new_t: float | None = None
    factor: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("remove", "move", "scale"):
            raise ValueError(f"kind must be remove|move|scale; got {self.kind!r}")
        if self.spike_index < 0:
            raise ValueError("spike_index must be non-negative")
        if self.kind == "move" and self.new_t is None:
            raise ValueError("move edit requires new_t")
        if self.kind == "scale" and self.factor is None:
            raise ValueError("scale edit requires factor")


def harmonic_code_features(t: Any, n_harmonics: int) -> np.ndarray:
    """Harmonic frame row(s) ``u(t) = [cos 2πt, sin 2πt, ..., cos 2πHt,
    sin 2πHt]`` for harmonics ``1..H``. Scalar ``t`` returns ``(2H,)``; a ``(k,)``
    array returns ``(k, 2H)``. This is the exact convention
    ``gam_sae::sparse_dict::coordinate`` uses for the ``(c_h, s_h)`` code."""
    t_arr = np.asarray(t, dtype=np.float64)
    scalar = t_arr.ndim == 0
    t_col = np.atleast_1d(t_arr)
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.cos(2.0 * np.pi * h * t_col))
        cols.append(np.sin(2.0 * np.pi * h * t_col))
    feats = np.stack(cols, axis=1)  # (k, 2H)
    return feats[0] if scalar else feats


def synthesize_measure_code(spikes: Any, n_harmonics: int) -> np.ndarray:
    """Re-synthesize the ``2H`` within-block code ``z = Σ_j a_j u(t_j)`` from a
    list of ``(amplitude, t)`` point masses. The exact forward map super-
    resolution inverts; an empty measure yields the zero code."""
    z = np.zeros(2 * n_harmonics, dtype=np.float64)
    for amplitude, t in spikes:
        z += float(amplitude) * harmonic_code_features(float(t), n_harmonics)
    return z


def apply_spike_edit(spikes: Any, edit: MeasureSpikeEdit) -> list[tuple[float, float]]:
    """Return a new ``(amplitude, t)`` measure with ``edit`` applied to one spike.
    The input is left unmodified; ``"remove"`` drops the spike, ``"move"`` retunes
    its position, ``"scale"`` multiplies its amplitude."""
    out = [(float(a), float(t)) for a, t in spikes]
    if edit.spike_index >= len(out):
        raise IndexError(
            f"spike_index {edit.spike_index} out of range for {len(out)} spikes"
        )
    a, t = out[edit.spike_index]
    if edit.kind == "remove":
        del out[edit.spike_index]
    elif edit.kind == "move":
        out[edit.spike_index] = (a, float(edit.new_t) % 1.0)
    else:  # scale
        out[edit.spike_index] = (a * float(edit.factor), t)
    return out


def spike_edit_code_delta(spikes: Any, edit: MeasureSpikeEdit, n_harmonics: int) -> np.ndarray:
    """The ``2H`` change ``Δz = synth(edited) − synth(original)`` for a single-
    spike edit. For ``"remove"`` this is exactly ``−a_j u(t_j)`` — the isolated
    contribution of the removed instance, nothing else."""
    before = synthesize_measure_code(spikes, n_harmonics)
    after = synthesize_measure_code(apply_spike_edit(spikes, edit), n_harmonics)
    return after - before


def spike_edit_delta_x(decoder: Any, spikes: Any, edit: MeasureSpikeEdit) -> np.ndarray:
    """Lift a single-spike code edit into the atom's ambient p-space:
    ``Δx = Δz · D`` where ``decoder`` is the ``2H × p`` block decoder (code →
    activation). The returned ``(p,)`` vector is the p-space move that removes /
    moves / rescales exactly one instance of the circle feature."""
    D = np.asarray(decoder, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError(f"decoder must be 2-D (2H x p); got shape {D.shape}")
    n_harmonics = D.shape[0] // 2
    if 2 * n_harmonics != D.shape[0]:
        raise ValueError(f"decoder rows {D.shape[0]} must be even (2H)")
    delta_z = spike_edit_code_delta(spikes, edit, n_harmonics)
    return delta_z @ D


def random_decoder(n_harmonics: int, p: int, seed: int) -> np.ndarray:
    """A (2H x p) decoder with orthonormal rows (so encode is D's transpose)."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((2 * n_harmonics, p))
    # Orthonormalize rows via QR on the transpose.
    q, _ = np.linalg.qr(m.T)
    return q[:, : 2 * n_harmonics].T  # (2H, p), rows orthonormal


def encode(x: np.ndarray, decoder: np.ndarray) -> np.ndarray:
    """Least-squares harmonic code of x on the frame: z = (D Dᵀ)^{-1} D x."""
    DDt = decoder @ decoder.T
    return np.linalg.solve(DDt, decoder @ x)


def instance_amplitudes(x, decoder, positions, n_harmonics):
    """The per-instance amplitudes ``[a_1, ..., a_k]`` — the least-squares solve of
    ``encode(x) = Σ_j a_j u(t_j)`` on the known support ``positions``. This is the
    instance-resolved readout super-resolution provides (it removes the harmonic
    cross-talk a raw matched filter suffers): the amplitude of *each* instance,
    which is exactly what a per-instance downstream head reads."""
    z = encode(x, decoder)
    vander = np.stack([harmonic_code_features(t, n_harmonics) for t in positions], axis=1)  # (2H, k)
    amps, *_ = np.linalg.lstsq(vander, z, rcond=None)
    return amps


def main() -> int:
    n_harmonics = 4
    p = 32
    t1, a1 = 0.15, 1.0
    t2, a2 = 0.60, 0.8
    measure = [(a1, t1), (a2, t2)]

    decoder = random_decoder(n_harmonics, p, seed=7)
    z = synthesize_measure_code(measure, n_harmonics)
    x = z @ decoder  # (p,)

    results = {}
    tol = 1e-9
    ok = True

    # --- remove spike 0 (the t1 instance) ---
    edit = MeasureSpikeEdit(kind="remove", spike_index=0)
    dx = spike_edit_delta_x(decoder, measure, edit)
    x_patched = x + dx
    z_patched = encode(x_patched, decoder)
    z_survivor = synthesize_measure_code([(a2, t2)], n_harmonics)
    z_removed = synthesize_measure_code([(a1, t1)], n_harmonics)
    err_survivor = float(np.linalg.norm(z_patched - z_survivor))
    dist_removed = float(np.linalg.norm(z_patched - z_removed))
    remove_ok = err_survivor < tol
    ok = ok and remove_ok
    results["remove"] = {
        "code_matches_surviving_instance_err": err_survivor,
        "distance_to_removed_instance": dist_removed,
        "ok": remove_ok,
    }

    # instance-resolved readouts before/after removing spike 0 (the t1 instance).
    # The amplitude of instance a should go to ~0; instance b's amplitude must be
    # IDENTICAL (per-instance specificity — the claim a linear SAE cannot make).
    amps_before = instance_amplitudes(x, decoder, [t1, t2], n_harmonics)
    amps_after = instance_amplitudes(x_patched, decoder, [t1, t2], n_harmonics)
    a1_before, a2_before = float(amps_before[0]), float(amps_before[1])
    a1_after, a2_after = float(amps_after[0]), float(amps_after[1])
    a1_zeroed = abs(a1_after) < 1e-9
    a2_unchanged = abs(a2_after - a2_before) < 1e-9
    readout_ok = a1_zeroed and a2_unchanged
    ok = ok and readout_ok
    results["instance_resolved_readouts"] = {
        "instance_a_amplitude_before": a1_before,
        "instance_a_amplitude_after": a1_after,
        "instance_b_amplitude_before": a2_before,
        "instance_b_amplitude_after": a2_after,
        "instance_a_zeroed": a1_zeroed,
        "instance_b_unchanged": a2_unchanged,
        "ok": readout_ok,
    }

    # --- move spike 1 (the t2 instance) to a new position ---
    new_t = 0.85
    edit_move = MeasureSpikeEdit(kind="move", spike_index=1, new_t=new_t)
    dx_move = spike_edit_delta_x(decoder, measure, edit_move)
    z_moved = encode(x + dx_move, decoder)
    z_expected_move = synthesize_measure_code([(a1, t1), (a2, new_t)], n_harmonics)
    move_err = float(np.linalg.norm(z_moved - z_expected_move))
    move_ok = move_err < tol
    ok = ok and move_ok
    results["move"] = {"err": move_err, "ok": move_ok, "new_t": new_t}

    # --- scale spike 0 by 0.5 ---
    edit_scale = MeasureSpikeEdit(kind="scale", spike_index=0, factor=0.5)
    dx_scale = spike_edit_delta_x(decoder, measure, edit_scale)
    z_scaled = encode(x + dx_scale, decoder)
    z_expected_scale = synthesize_measure_code([(0.5 * a1, t1), (a2, t2)], n_harmonics)
    scale_err = float(np.linalg.norm(z_scaled - z_expected_scale))
    scale_ok = scale_err < tol
    ok = ok and scale_ok
    results["scale"] = {"err": scale_err, "ok": scale_ok}

    results["all_ok"] = ok
    print(json.dumps(results, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
