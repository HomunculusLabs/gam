"""SO(2)/SO(3) representations and the gauge companion — thin FFI shims.

The numerical work (SO(2)/SO(3) representations, their JVPs, and the
gauge-companion HSV/RGB/LCh loss) lives in `gam-pyffi`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ._binding import rust_module


GroupName = Literal["SO2", "SO3", "R1", "Trivial"]
AuxName = Literal["HSV", "RGB", "LCh"]


# ---------------------------------------------------------------------------
# Group representations and JVPs (thin FFI shims).
# ---------------------------------------------------------------------------

def rho_so2(theta: np.ndarray) -> np.ndarray:
    """SO(2) rep. theta: (...,) -> (..., 2, 2)."""
    arr = np.asarray(theta, dtype=np.float64)
    flat = np.ascontiguousarray(arr.reshape(-1))
    out = rust_module().equivariant_rho_so2(flat)
    return out.reshape(arr.shape + (2, 2))


def rho_so2_jvp(theta: np.ndarray) -> np.ndarray:
    """d/dθ ρ_SO2(θ). Returns (..., 2, 2)."""
    arr = np.asarray(theta, dtype=np.float64)
    flat = np.ascontiguousarray(arr.reshape(-1))
    out = rust_module().equivariant_rho_so2_jvp(flat)
    return out.reshape(arr.shape + (2, 2))


def rho_so3(omega: np.ndarray) -> np.ndarray:
    """SO(3) rep via Rodrigues. omega: (..., 3) -> (..., 3, 3)."""
    arr = np.asarray(omega, dtype=np.float64)
    # allow-list (a): FFI input validation.
    if arr.shape[-1] != 3:
        raise ValueError("rho_so3 requires last axis of length 3")
    flat = np.ascontiguousarray(arr.reshape(-1, 3))
    out = rust_module().equivariant_rho_so3(flat)
    return out.reshape(arr.shape[:-1] + (3, 3))


def rho_so3_jvp(omega: np.ndarray, domega: np.ndarray) -> np.ndarray:
    """Directional derivative of ρ_SO3 at ω in direction dω."""
    arr = np.asarray(omega, dtype=np.float64)
    darr = np.asarray(domega, dtype=np.float64)
    # allow-list (a): FFI input validation.
    if arr.shape[-1] != 3 or darr.shape[-1] != 3:
        raise ValueError("rho_so3_jvp requires last axis of length 3 on both inputs")
    bd = np.broadcast_to(darr, arr.shape)
    flat_o = np.ascontiguousarray(arr.reshape(-1, 3))
    flat_d = np.ascontiguousarray(bd.reshape(-1, 3))
    out = rust_module().equivariant_rho_so3_jvp(flat_o, flat_d)
    return out.reshape(arr.shape[:-1] + (3, 3))


def rho(group: GroupName, g: np.ndarray) -> np.ndarray:
    arr = np.asarray(g, dtype=np.float64)
    return rust_module().equivariant_rho(group, np.ascontiguousarray(arr))


# ---------------------------------------------------------------------------
# gauge_companion — auxiliary-supervised gauge-fix recipe as a one-shot helper
# ---------------------------------------------------------------------------

@dataclass
class GaugeCompanion:
    """Auxiliary-supervised gauge-fix recipe wrapped into one object."""
    aux: AuxName
    d_aux: int = 3
    target: str = "lie"
    weight: float = 1.0
    aux_values: np.ndarray | None = None

    def __post_init__(self) -> None:
        # allow-list (e): dataclass typed config normalization.
        if self.aux_values is not None:
            self.aux_values = np.ascontiguousarray(np.asarray(self.aux_values, dtype=np.float64))

    def loss(self, theta: np.ndarray) -> float:
        """Hue circular MSE + sat/val cos-alignment."""
        return float(
            rust_module().equivariant_gauge_companion_loss(
                self.aux_values,
                np.ascontiguousarray(np.asarray(theta, dtype=np.float64)),
                int(self.d_aux),
                float(self.weight),
            )
        )


def gauge_companion(aux: AuxName = "HSV", d_aux: int = 3, weight: float = 1.0,
                    target: str = "lie", aux_values: np.ndarray | None = None) -> GaugeCompanion:
    """One-call gauge-fix companion. See `GaugeCompanion` docstring."""
    return GaugeCompanion(aux=aux, d_aux=d_aux, weight=weight, target=target,
                          aux_values=aux_values)


__all__ = [
    "GroupName",
    "rho", "rho_so2", "rho_so2_jvp", "rho_so3", "rho_so3_jvp",
    "GaugeCompanion", "gauge_companion",
]
