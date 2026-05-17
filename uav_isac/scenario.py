"""
Scenario Generator for UAV-ISAC Simulation.

Creates and manages simulation scenarios with UAVs, ground users,
and sensing targets on a configurable area.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ScenarioParams:
    """Parameters defining a simulation scenario."""
    # Area
    area_size: float = 500.0          # Square area side length [m]

    # UAVs
    n_uavs: int = 2                   # Number of UAVs
    uav_altitude: float = 100.0       # Fixed altitude [m]
    uav_max_speed: float = 20.0       # Max speed [m/s]

    # Ground users
    n_users: int = 4                  # Number of ground users
    user_mobility: Literal["static", "random_walk", "linear"] = "static"
    user_max_speed: float = 1.4       # Max user speed [m/s] (if mobile)

    # Sensing targets
    n_targets: int = 2                # Number of sensing targets
    target_mobility: Literal["static", "random_walk", "linear"] = "static"
    target_max_speed: float = 5.0     # Max target speed [m/s]

    # Mission
    mission_duration: float = 60.0    # Total mission time [s]
    dt: float = 1.0                   # Time step [s]

    # Random seed
    seed: int | None = None


class Scenario:
    """
    Generates and manages a UAV-ISAC simulation scenario.

    Attributes
    ----------
    uav_positions : (n_uavs, 3)       — current UAV positions [x, y, z]
    user_positions : (n_users, 3)      — current ground user positions
    target_positions : (n_targets, 3)  — current sensing target positions
    time : float                       — current simulation time [s]
    """

    def __init__(self, params: ScenarioParams | None = None):
        self.params = params or ScenarioParams()
        self.rng = np.random.default_rng(self.params.seed)
        self.time = 0.0
        self.n_steps = int(self.params.mission_duration / self.params.dt)

        # Initialize positions
        self.uav_positions = self._init_uav_positions()
        self.user_positions = self._init_ground_positions(self.params.n_users)
        self.target_positions = self._init_ground_positions(self.params.n_targets)

        # Initialize per-entity velocities (used by predict-ahead policies)
        self.user_velocities = self._init_velocities(
            self.params.n_users,
            self.params.user_max_speed,
            self.params.user_mobility,
        )
        self.target_velocities = self._init_velocities(
            self.params.n_targets,
            self.params.target_max_speed,
            self.params.target_mobility,
        )

        # Trajectory storage: (n_steps+1, n_uavs, 3)
        self.uav_trajectory = np.zeros(
            (self.n_steps + 1, self.params.n_uavs, 3)
        )
        self.uav_trajectory[0] = self.uav_positions.copy()

        # User/target position history
        self.user_history = np.zeros(
            (self.n_steps + 1, self.params.n_users, 3)
        )
        self.user_history[0] = self.user_positions.copy()

        self.target_history = np.zeros(
            (self.n_steps + 1, self.params.n_targets, 3)
        )
        self.target_history[0] = self.target_positions.copy()

        self._step_idx = 0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_uav_positions(self) -> np.ndarray:
        """Place UAVs evenly spaced in the area at fixed altitude."""
        n = self.params.n_uavs
        s = self.params.area_size
        h = self.params.uav_altitude

        if n == 1:
            return np.array([[s / 2, s / 2, h]])

        # Spread UAVs on a grid-like pattern
        positions = np.zeros((n, 3))
        margin = s * 0.2
        xs = np.linspace(margin, s - margin, max(int(np.ceil(np.sqrt(n))), 2))
        ys = np.linspace(margin, s - margin, max(int(np.ceil(n / len(xs))), 2))
        grid = np.array(np.meshgrid(xs, ys)).T.reshape(-1, 2)
        positions[:, :2] = grid[:n]
        positions[:, 2] = h
        return positions

    def _init_ground_positions(self, count: int) -> np.ndarray:
        """Random ground positions within the area (z=0)."""
        s = self.params.area_size
        pos = np.zeros((count, 3))
        pos[:, 0] = self.rng.uniform(0, s, count)
        pos[:, 1] = self.rng.uniform(0, s, count)
        return pos

    def _init_velocities(
        self, count: int, max_speed: float, mobility: str
    ) -> np.ndarray:
        """Initial (vx, vy) per entity. Persistent for `linear`, zeros otherwise."""
        if mobility == "linear" and count > 0:
            angle = self.rng.uniform(0, 2 * np.pi, count)
            speed = self.rng.uniform(0.3 * max_speed, max_speed, count)
            return np.column_stack([speed * np.cos(angle), speed * np.sin(angle)])
        return np.zeros((count, 2))

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self, uav_velocities: np.ndarray | None = None) -> None:
        """
        Advance simulation by one time step.

        Parameters
        ----------
        uav_velocities : (n_uavs, 2) — desired [vx, vy] per UAV.
            Speed is clipped to uav_max_speed. If None, UAVs hover.
        """
        dt = self.params.dt
        self._step_idx += 1
        self.time += dt

        # --- UAV movement ---
        if uav_velocities is not None:
            vel = np.atleast_2d(uav_velocities)[:, :2]
            speed = np.linalg.norm(vel, axis=1, keepdims=True)
            max_s = self.params.uav_max_speed
            safe_speed = np.maximum(speed, 1e-9)  # avoid 0/0 in np.where
            scale = np.where(speed > max_s, max_s / safe_speed, 1.0)
            vel = vel * scale
            self.uav_positions[:, :2] += vel * dt
        # Clip to area
        self.uav_positions[:, :2] = np.clip(
            self.uav_positions[:, :2], 0, self.params.area_size
        )
        self.uav_trajectory[self._step_idx] = self.uav_positions.copy()

        # --- User movement ---
        self.user_velocities = self._move_ground(
            self.user_positions,
            self.user_velocities,
            self.params.user_mobility,
            self.params.user_max_speed,
            dt,
        )
        self.user_history[self._step_idx] = self.user_positions.copy()

        # --- Target movement ---
        self.target_velocities = self._move_ground(
            self.target_positions,
            self.target_velocities,
            self.params.target_mobility,
            self.params.target_max_speed,
            dt,
        )
        self.target_history[self._step_idx] = self.target_positions.copy()

    def _move_ground(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        mobility: str,
        max_speed: float,
        dt: float,
    ) -> np.ndarray:
        """Advance ground entities in-place, return updated velocity array."""
        if mobility == "static" or positions.shape[0] == 0:
            return velocities

        if mobility == "random_walk":
            n = positions.shape[0]
            angle = self.rng.uniform(0, 2 * np.pi, n)
            speed = self.rng.uniform(0, max_speed, n)
            velocities = np.column_stack(
                [speed * np.cos(angle), speed * np.sin(angle)]
            )

        new_xy = positions[:, :2] + velocities * dt
        # Reflect velocity on walls (matters for `linear`; harmless for `random_walk`)
        out_x = (new_xy[:, 0] < 0) | (new_xy[:, 0] > self.params.area_size)
        out_y = (new_xy[:, 1] < 0) | (new_xy[:, 1] > self.params.area_size)
        velocities[out_x, 0] *= -1
        velocities[out_y, 1] *= -1
        positions[:, :2] = np.clip(new_xy, 0, self.params.area_size)
        return velocities

    def reset(self) -> None:
        """Reset scenario to initial state."""
        self.time = 0.0
        self._step_idx = 0
        self.uav_positions = self._init_uav_positions()
        self.user_positions = self._init_ground_positions(self.params.n_users)
        self.target_positions = self._init_ground_positions(self.params.n_targets)
        self.user_velocities = self._init_velocities(
            self.params.n_users,
            self.params.user_max_speed,
            self.params.user_mobility,
        )
        self.target_velocities = self._init_velocities(
            self.params.n_targets,
            self.params.target_max_speed,
            self.params.target_mobility,
        )
        self.uav_trajectory[0] = self.uav_positions.copy()
        self.user_history[0] = self.user_positions.copy()
        self.target_history[0] = self.target_positions.copy()

    @property
    def is_done(self) -> bool:
        return self._step_idx >= self.n_steps
