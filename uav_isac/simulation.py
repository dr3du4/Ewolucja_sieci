"""
ISAC Simulation Engine.

Runs the full simulation loop: UAV trajectory, communication throughput,
sensing performance, and resource allocation trade-off.

Supports comparison of 1-UAV vs multi-UAV scenarios.
"""

import numpy as np
from dataclasses import dataclass, field
from .channel_model import ChannelModel, ChannelParams
from .scenario import Scenario, ScenarioParams
from .sensing import SensingModel, SensingParams
from .energy import EnergyParams, propulsion_power_w, tx_power_w


@dataclass
class SimulationResults:
    """Container for simulation output data."""
    n_uavs: int = 0
    n_steps: int = 0
    time: np.ndarray = field(default_factory=lambda: np.array([]))

    # Per-step metrics (averaged over users/targets)
    avg_user_rate_bps: np.ndarray = field(default_factory=lambda: np.array([]))
    min_user_rate_bps: np.ndarray = field(default_factory=lambda: np.array([]))
    sum_rate_bps: np.ndarray = field(default_factory=lambda: np.array([]))
    avg_sensing_snr_db: np.ndarray = field(default_factory=lambda: np.array([]))
    avg_crb_rmse_m: np.ndarray = field(default_factory=lambda: np.array([]))

    # Aggregated
    total_avg_rate: float = 0.0
    total_min_rate: float = 0.0
    total_sum_rate: float = 0.0
    total_avg_crb: float = 0.0
    total_avg_sensing_snr: float = 0.0

    # Trajectory
    uav_trajectories: np.ndarray = field(default_factory=lambda: np.array([]))
    user_history: np.ndarray = field(default_factory=lambda: np.array([]))
    target_history: np.ndarray = field(default_factory=lambda: np.array([]))

    # Energy (per UAV [J], plus aggregates and feasibility flag)
    energy_consumed_j_per_uav: np.ndarray = field(default_factory=lambda: np.array([]))
    energy_consumed_j_avg: float = 0.0
    mission_feasible: bool = True
    infeasible_uavs: list = field(default_factory=list)


class ISACSimulation:
    """
    Main simulation engine for UAV-ISAC.

    Supports different resource allocation strategies and
    trajectory policies.
    """

    def __init__(
        self,
        scenario_params: ScenarioParams | None = None,
        channel_params: ChannelParams | None = None,
        sensing_params: SensingParams | None = None,
        energy_params: EnergyParams | None = None,
        dt_lookahead: float = 2.0,
    ):
        self.scenario_params = scenario_params or ScenarioParams()
        self.channel_params = channel_params or ChannelParams()
        self.sensing_params = sensing_params or SensingParams()
        self.energy_params = energy_params or EnergyParams()
        self.dt_lookahead = dt_lookahead

        self.channel = ChannelModel(self.channel_params)
        self.sensing = SensingModel(self.sensing_params, self.channel_params)

    # ------------------------------------------------------------------
    # Trajectory policies
    # ------------------------------------------------------------------

    def _circular_trajectory(
        self,
        scenario: Scenario,
        step: int,
    ) -> np.ndarray:
        """
        Simple circular trajectory around center of area.

        Each UAV flies a circle with offset phase.
        Returns (n_uavs, 2) velocity vectors.
        """
        n = scenario.params.n_uavs
        s = scenario.params.area_size
        center = np.array([s / 2, s / 2])
        radius = s * 0.3
        omega = 2 * np.pi / scenario.n_steps  # one full circle per mission

        velocities = np.zeros((n, 2))
        for i in range(n):
            phase = omega * step + (2 * np.pi * i / n)
            # Desired position on circle
            target = center + radius * np.array(
                [np.cos(phase), np.sin(phase)]
            )
            # Velocity toward target
            diff = target - scenario.uav_positions[i, :2]
            dist = np.linalg.norm(diff)
            if dist > 1.0:
                velocities[i] = (
                    diff / dist * min(dist / scenario.params.dt,
                                       scenario.params.uav_max_speed)
                )
        return velocities

    def _greedy_trajectory(
        self,
        scenario: Scenario,
    ) -> np.ndarray:
        """
        Greedy trajectory: each UAV moves toward its nearest user
        while considering sensing targets.

        Balances communication (move to users) and sensing (stay in
        range of targets) using the sensing power ratio as weight.
        """
        n = scenario.params.n_uavs
        alpha = self.sensing_params.sensing_power_ratio  # sensing weight
        velocities = np.zeros((n, 2))

        # Predict-ahead: aim at where users/targets will be in dt_lookahead seconds.
        # For static entities velocities are zero, so predicted == current.
        predicted_users = (
            scenario.user_positions[:, :2]
            + scenario.user_velocities * self.dt_lookahead
        )
        predicted_users = np.clip(
            predicted_users, 0, scenario.params.area_size
        )
        predicted_targets = (
            scenario.target_positions[:, :2]
            + scenario.target_velocities * self.dt_lookahead
        )
        predicted_targets = np.clip(
            predicted_targets, 0, scenario.params.area_size
        )

        for i in range(n):
            uav_xy = scenario.uav_positions[i, :2]

            # --- Communication pull: toward centroid of nearest (predicted) users ---
            user_dists = np.linalg.norm(predicted_users - uav_xy, axis=1)
            # Assign users to nearest UAV (simple partitioning)
            n_per_uav = max(1, scenario.params.n_users // n)
            nearest_users = np.argsort(user_dists)[:n_per_uav]
            comm_target = predicted_users[nearest_users].mean(axis=0)

            # --- Sensing pull: toward nearest (predicted) sensing target ---
            if scenario.params.n_targets > 0:
                tgt_dists = np.linalg.norm(predicted_targets - uav_xy, axis=1)
                nearest_tgt = np.argmin(tgt_dists)
                sense_target = predicted_targets[nearest_tgt]
            else:
                sense_target = comm_target

            # Weighted combination
            goal = (1 - alpha) * comm_target + alpha * sense_target
            diff = goal - uav_xy
            dist = np.linalg.norm(diff)

            if dist > 1.0:
                speed = min(
                    dist / scenario.params.dt,
                    scenario.params.uav_max_speed,
                )
                velocities[i] = diff / dist * speed

        return velocities

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------

    def run(
        self,
        trajectory_policy: str = "greedy",
        sensing_power_ratios: np.ndarray | None = None,
    ) -> SimulationResults:
        """
        Run the full ISAC simulation.

        Parameters
        ----------
        trajectory_policy : "greedy" | "circular" | "hover"
        sensing_power_ratios : (n_steps,) array of alpha values
            for time-varying resource allocation. If None, uses
            the fixed ratio from SensingParams.

        Returns
        -------
        SimulationResults
        """
        scenario = Scenario(self.scenario_params)
        n_steps = scenario.n_steps

        # Storage
        time = np.zeros(n_steps)
        avg_rate = np.zeros(n_steps)
        min_rate = np.zeros(n_steps)
        sum_rate = np.zeros(n_steps)
        avg_snr_s = np.zeros(n_steps)
        avg_crb = np.zeros(n_steps)

        for t in range(n_steps):
            # --- Compute communication metrics ---
            comm_power_ratio = 1.0 - self.sensing_params.sensing_power_ratio
            if sensing_power_ratios is not None:
                comm_power_ratio = 1.0 - sensing_power_ratios[t]

            comm_power_dbm = (
                self.channel_params.tx_power_dbm
                + 10.0 * np.log10(max(comm_power_ratio, 1e-6))
            )

            # Capacity for each (UAV, user) pair
            cap = self.channel.capacity_bps(
                scenario.uav_positions, scenario.user_positions
            )  # (N, K)
            # Scale by actual comm power vs default
            power_offset_db = comm_power_dbm - self.channel_params.tx_power_dbm
            cap *= 10.0 ** (power_offset_db / 10.0)

            # User association: each user connects to best UAV
            best_rate = np.max(cap, axis=0)  # (K,)
            avg_rate[t] = np.mean(best_rate)
            min_rate[t] = np.min(best_rate)
            sum_rate[t] = np.sum(best_rate)

            # --- Compute sensing metrics ---
            s_ratio = self.sensing_params.sensing_power_ratio
            if sensing_power_ratios is not None:
                s_ratio = sensing_power_ratios[t]

            sensing_power_dbm = (
                self.channel_params.tx_power_dbm
                + 10.0 * np.log10(max(s_ratio, 1e-6))
            )

            snr_s = self.sensing.sensing_snr_db(
                scenario.uav_positions,
                scenario.target_positions,
                sensing_power_dbm,
            )  # (N, Q)
            # Best sensing SNR per target (from any UAV)
            best_snr = np.max(snr_s, axis=0)  # (Q,)
            avg_snr_s[t] = np.mean(best_snr)

            crb = self.sensing.crb_range_rmse(
                scenario.uav_positions,
                scenario.target_positions,
                sensing_power_dbm,
            )
            best_crb = np.min(crb, axis=0)  # best CRB per target
            avg_crb[t] = np.mean(best_crb)

            time[t] = scenario.time

            # --- Move ---
            if trajectory_policy == "greedy":
                vel = self._greedy_trajectory(scenario)
            elif trajectory_policy == "circular":
                vel = self._circular_trajectory(scenario, t)
            else:  # hover
                vel = None

            scenario.step(vel)

        # --- Energy ---
        # Effective per-step speed reconstructed from the trajectory (m/s).
        # Shape: (n_steps, n_uavs)
        diffs = np.diff(scenario.uav_trajectory[:, :, :2], axis=0)
        speeds = np.linalg.norm(diffs, axis=2) / self.scenario_params.dt
        p_prop = propulsion_power_w(speeds, self.energy_params)
        p_tx = tx_power_w(self.channel_params.tx_power_dbm)  # scalar [W]
        energy_per_uav = (p_prop + p_tx).sum(axis=0) * self.scenario_params.dt
        infeasible = np.where(
            energy_per_uav > self.energy_params.battery_j
        )[0].tolist()

        # --- Aggregate results ---
        results = SimulationResults(
            n_uavs=self.scenario_params.n_uavs,
            n_steps=n_steps,
            time=time,
            avg_user_rate_bps=avg_rate,
            min_user_rate_bps=min_rate,
            sum_rate_bps=sum_rate,
            avg_sensing_snr_db=avg_snr_s,
            avg_crb_rmse_m=avg_crb,
            total_avg_rate=float(np.mean(avg_rate)),
            total_min_rate=float(np.mean(min_rate)),
            total_sum_rate=float(np.mean(sum_rate)),
            total_avg_crb=float(np.mean(avg_crb)),
            total_avg_sensing_snr=float(np.mean(avg_snr_s)),
            uav_trajectories=scenario.uav_trajectory.copy(),
            user_history=scenario.user_history.copy(),
            target_history=scenario.target_history.copy(),
            energy_consumed_j_per_uav=energy_per_uav,
            energy_consumed_j_avg=float(np.mean(energy_per_uav)),
            mission_feasible=len(infeasible) == 0,
            infeasible_uavs=infeasible,
        )
        return results

    # ------------------------------------------------------------------
    # Pareto trade-off sweep
    # ------------------------------------------------------------------

    def sweep_power_allocation(
        self,
        alphas: np.ndarray | None = None,
        trajectory_policy: str = "greedy",
    ) -> list[SimulationResults]:
        """
        Run simulations for different sensing/communication power splits.

        Parameters
        ----------
        alphas : array of sensing power ratios (0 to 1).
            0 = all power to communication, 1 = all to sensing.

        Returns
        -------
        List of SimulationResults, one per alpha value.
        """
        if alphas is None:
            alphas = np.linspace(0.1, 0.9, 9)

        results = []
        for alpha in alphas:
            self.sensing_params.sensing_power_ratio = float(alpha)
            res = self.run(trajectory_policy=trajectory_policy)
            results.append(res)

        return results
