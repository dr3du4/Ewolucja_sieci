"""
Simplified UAV Energy Model.

Captures propulsion + transmission power per UAV per time step,
sums to an energy budget over a mission, and flags infeasibility
against a per-UAV battery budget.

Model:
    P_propulsion(v) = P_hover + k_drag * v²        [W]
    P_tx           = 10^(tx_power_dbm / 10) / 1000 [W]
    E_uav          = sum_t (P_propulsion(v_t) + P_tx) * dt  [J]

References (for the simplified form):
    Zeng & Zhang, "Energy-Efficient UAV Communication With
    Trajectory Optimization," IEEE TWC, 2017 (eq. 8 reduced).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class EnergyParams:
    """UAV propulsion + battery parameters."""
    p_hover_w: float = 100.0      # Hover power [W]
    k_drag: float = 0.5           # Quadratic drag coefficient [W·s²/m²]
    battery_j: float = 15_000.0   # Per-UAV battery budget [J]


def propulsion_power_w(speed_m_s: np.ndarray, params: EnergyParams) -> np.ndarray:
    """P(v) = P_hover + k·v². Vectorized over arbitrary shape."""
    return params.p_hover_w + params.k_drag * speed_m_s ** 2


def tx_power_w(tx_power_dbm: float) -> float:
    """Convert transmit power from dBm to linear Watts."""
    return 10.0 ** (tx_power_dbm / 10.0) / 1000.0
