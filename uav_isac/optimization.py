"""
Joint trajectory optimization for UAV-ISAC.

Optimizes UAV positions Q ∈ R^((T+1) × N × 2) for a fixed sensing power
ratio α. Uses scipy SLSQP with greedy warm-start; falls back to
trust-constr if SLSQP fails to converge.

Objective (scalarized, α-weighted):
    J(Q) = (1-α) · (-mean_rate / R0)  +  α · (mean_crb / C0)

where R0 = 100 Mbps, C0 = 1 m are normalizers keeping both terms O(1).

Constraints:
    - speed:     ‖Q[t+1] - Q[t]‖² ≤ (v_max · dt)²
    - area:      0 ≤ Q[t,n,d] ≤ area_size   (box bounds)
    - collision: ‖Q[t,n] - Q[t,m]‖² ≥ d_min²   for all pairs (if N ≥ 2)
    - energy:    Σ_t (P_hover + k_drag·v_t² + P_tx)·dt ≤ battery_j  per UAV

The first waypoint Q[0] is fixed to the scenario's initial UAV position
(not part of decision variables).
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from .scenario import Scenario, ScenarioParams
from .channel_model import ChannelModel, ChannelParams
from .sensing import SensingModel, SensingParams
from .energy import EnergyParams, tx_power_w


# Normalization constants so rate and CRB terms are O(1) in the objective
RATE_NORM_BPS = 100e6   # 100 Mbps
CRB_NORM_M = 1.0        # 1 m


def joint_optimize_trajectory(
    scenario_params: ScenarioParams,
    channel_params: ChannelParams,
    sensing_params: SensingParams,
    energy_params: EnergyParams,
    alpha: float,
    d_min: float = 30.0,
    warm_start_traj: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-7,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Optimize UAV trajectory for given α.

    Returns
    -------
    Q_opt : (n_steps+1, n_uavs, 2) — optimized 2-D positions per step
    info  : dict with keys: converged, method, iterations, obj_value, fallback_used
    """
    # ------------------------------------------------------------------
    # Setup — geometry, scenarios, channel/sensing models
    # ------------------------------------------------------------------
    n_uavs = scenario_params.n_uavs
    dt = scenario_params.dt
    n_steps = int(scenario_params.mission_duration / dt)
    area = scenario_params.area_size
    v_max = scenario_params.uav_max_speed
    altitude = scenario_params.uav_altitude
    v_max_step_sq = (v_max * dt) ** 2
    d_min_sq = d_min ** 2

    # Pre-compute static user/target positions (joint opt for #17 scope: static only)
    sc = Scenario(scenario_params)
    user_pos = sc.user_positions.copy()
    target_pos = sc.target_positions.copy()
    Q_init = sc.uav_positions[:, :2].copy()  # (n_uavs, 2) — fixed starting point

    channel = ChannelModel(channel_params)
    sensing = SensingModel(sensing_params, channel_params)

    # Power split (fixed by α throughout the mission)
    tx_dbm = channel_params.tx_power_dbm
    comm_dbm = tx_dbm + 10.0 * np.log10(max(1.0 - alpha, 1e-6))
    sense_dbm = tx_dbm + 10.0 * np.log10(max(alpha, 1e-6))
    comm_scale_lin = 10.0 ** ((comm_dbm - tx_dbm) / 10.0)

    p_tx_w = tx_power_w(tx_dbm)

    # ------------------------------------------------------------------
    # Warm-start — greedy trajectory if none provided
    # ------------------------------------------------------------------
    if warm_start_traj is None:
        from .simulation import ISACSimulation
        sp = SensingParams(**sensing_params.__dict__)
        sp.sensing_power_ratio = float(alpha)
        sim = ISACSimulation(scenario_params, channel_params, sp, energy_params)
        warm = sim.run(trajectory_policy="greedy")
        warm_start_traj = warm.uav_trajectories[:, :, :2].copy()

    # Decision variables: positions at t = 1..n_steps (t=0 is fixed)
    n_free_steps = n_steps
    x0_flat = warm_start_traj[1:].reshape(-1)
    n_vars = x0_flat.size

    def unpack(x_flat: np.ndarray) -> np.ndarray:
        """Rebuild full trajectory (n_steps+1, n_uavs, 2) from free vars."""
        Q_free = x_flat.reshape(n_free_steps, n_uavs, 2)
        Q_full = np.empty((n_steps + 1, n_uavs, 2))
        Q_full[0] = Q_init
        Q_full[1:] = Q_free
        return Q_full

    # ------------------------------------------------------------------
    # Objective: weighted (rate, CRB) scalarization
    # ------------------------------------------------------------------
    def objective(x_flat: np.ndarray) -> float:
        Q_full = unpack(x_flat)
        # Metrics are evaluated at the position the UAV occupies during
        # the step — matches simulation.run() semantics (compute then move).
        Q_active = Q_full[:n_steps]  # (n_steps, n_uavs, 2)
        uav_3d = np.concatenate(
            [Q_active, np.full((n_steps, n_uavs, 1), altitude)], axis=-1
        )

        rates = np.empty(n_steps)
        crbs = np.empty(n_steps)
        for t in range(n_steps):
            cap = channel.capacity_bps(uav_3d[t], user_pos) * comm_scale_lin
            best_rate = np.max(cap, axis=0)  # (K,)
            rates[t] = np.mean(best_rate)

            crb = sensing.crb_range_rmse(uav_3d[t], target_pos, sense_dbm)
            best_crb = np.min(crb, axis=0)  # (Q,)
            crbs[t] = np.mean(best_crb)

        mean_rate = np.mean(rates)
        mean_crb = np.mean(crbs)
        return (
            (1.0 - alpha) * (-mean_rate / RATE_NORM_BPS)
            + alpha * (mean_crb / CRB_NORM_M)
        )

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    def speed_ineq(x_flat: np.ndarray) -> np.ndarray:
        """(v_max·dt)² - ‖Q[t+1]-Q[t]‖²  ≥ 0, flattened over (t, n)."""
        Q_full = unpack(x_flat)
        diffs = np.diff(Q_full, axis=0)  # (n_steps, n_uavs, 2)
        step_sq = np.sum(diffs ** 2, axis=-1).reshape(-1)
        return v_max_step_sq - step_sq

    def collision_ineq(x_flat: np.ndarray) -> np.ndarray:
        """‖Q[t,n]-Q[t,m]‖² - d_min² ≥ 0 for all pairs (n,m), all t."""
        Q_full = unpack(x_flat)
        parts = []
        for n in range(n_uavs):
            for m in range(n + 1, n_uavs):
                d = Q_full[:, n, :] - Q_full[:, m, :]
                parts.append(np.sum(d ** 2, axis=-1) - d_min_sq)
        return np.concatenate(parts) if parts else np.array([0.0])

    def energy_ineq(x_flat: np.ndarray) -> np.ndarray:
        """battery_j - E_uav ≥ 0 per UAV."""
        Q_full = unpack(x_flat)
        diffs = np.diff(Q_full, axis=0)
        speeds_sq = np.sum(diffs ** 2, axis=-1) / (dt ** 2)  # (n_steps, n_uavs)
        p_prop = energy_params.p_hover_w + energy_params.k_drag * speeds_sq
        e_per_uav = (p_prop + p_tx_w).sum(axis=0) * dt
        return energy_params.battery_j - e_per_uav

    constraints = [
        {"type": "ineq", "fun": speed_ineq},
        {"type": "ineq", "fun": energy_ineq},
    ]
    if n_uavs >= 2:
        constraints.append({"type": "ineq", "fun": collision_ineq})

    bounds = [(0.0, area)] * n_vars

    # ------------------------------------------------------------------
    # Solve — SLSQP first, fall back to trust-constr
    # ------------------------------------------------------------------
    # Build constraints. trust-constr accepts NonlinearConstraint objects;
    # bounds are passed via `bounds=`. SLSQP is *not* used here because in
    # this problem class the warm-started gradient is small enough that
    # SLSQP's line search declares convergence after one step. trust-constr
    # uses a trust-region SQP with proper second-order info and reliably
    # finds non-trivial improvements.
    nlc = [
        NonlinearConstraint(speed_ineq, 0, np.inf),
        NonlinearConstraint(energy_ineq, 0, np.inf),
    ]
    if n_uavs >= 2:
        nlc.append(NonlinearConstraint(collision_ineq, 0, np.inf))

    info = {"fallback_used": False}
    try:
        res = minimize(
            objective,
            x0_flat,
            method="trust-constr",
            bounds=bounds,
            constraints=nlc,
            options={
                "maxiter": max_iter,
                "xtol": tol,
                "gtol": 1e-8,
                "verbose": 2 if verbose else 0,
            },
        )
        info["method"] = "trust-constr"
    except Exception as e:
        if verbose:
            print(f"  trust-constr raised: {e}; returning warm-start")
        info["method"] = "warm-start"
        info["converged"] = False
        info["iterations"] = 0
        info["obj_value"] = float(objective(x0_flat))
        return warm_start_traj.copy(), info

    info["converged"] = bool(res.success)
    info["iterations"] = int(getattr(res, "nit", 0))
    info["obj_value"] = float(res.fun)
    return unpack(res.x), info
