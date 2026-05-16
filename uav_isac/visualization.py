"""
Visualization module for UAV-ISAC Simulation.

Generates plots: Pareto curves, trajectory maps, time-series,
and comparison charts for 1-UAV vs multi-UAV scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from .simulation import SimulationResults


def set_plot_style():
    """Set consistent plot styling."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 2,
        "figure.dpi": 150,
    })


# ------------------------------------------------------------------
# Pareto / Trade-off curves
# ------------------------------------------------------------------


def plot_pareto_curve(
    results_list: list[SimulationResults],
    alphas: np.ndarray,
    title: str = "Communication-Sensing Trade-off",
    save_path: str | None = None,
):
    """
    Pareto curve: avg user rate vs CRB RMSE for different power splits.
    """
    set_plot_style()
    fig, ax = plt.subplots()

    rates = [r.total_avg_rate / 1e6 for r in results_list]  # Mbps
    crbs = [r.total_avg_crb for r in results_list]           # meters

    sc = ax.scatter(crbs, rates, c=alphas, cmap="coolwarm", s=100, zorder=5)
    ax.plot(crbs, rates, "k--", alpha=0.5)
    plt.colorbar(sc, ax=ax, label="Sensing power ratio (α)")

    ax.set_xlabel("Avg CRB RMSE [m] (lower = better sensing)")
    ax.set_ylabel("Avg User Rate [Mbps] (higher = better comms)")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_pareto_multi(
    results_dict: dict[str, list[SimulationResults]],
    alphas: np.ndarray,
    title: str = "Pareto Trade-off Comparison",
    save_path: str | None = None,
):
    """
    Pareto curves for multiple scenario variants on a single axis.

    Parameters
    ----------
    results_dict : {label: list[SimulationResults]} — one Pareto sweep per label
    """
    set_plot_style()
    fig, ax = plt.subplots()

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    palette = [
        "#e74c3c", "#2ecc71", "#3498db", "#f39c12",
        "#9b59b6", "#1abc9c", "#34495e", "#e67e22",
    ]
    for i, (label, results) in enumerate(results_dict.items()):
        rates = [r.total_avg_rate / 1e6 for r in results]
        crbs = [r.total_avg_crb for r in results]
        ax.plot(
            crbs, rates,
            marker=markers[i % len(markers)],
            linestyle="-",
            label=label,
            color=palette[i % len(palette)],
        )

    ax.set_xlabel("Avg CRB RMSE [m] (lower = better sensing)")
    ax.set_ylabel("Avg User Rate [Mbps] (higher = better comms)")
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_pareto_comparison(
    results_1uav: list[SimulationResults],
    results_2uav: list[SimulationResults],
    alphas: np.ndarray,
    save_path: str | None = None,
):
    """
    Compare Pareto frontiers of 1-UAV vs 2-UAV systems.
    """
    set_plot_style()
    fig, ax = plt.subplots()

    rates_1 = [r.total_avg_rate / 1e6 for r in results_1uav]
    crbs_1 = [r.total_avg_crb for r in results_1uav]
    rates_2 = [r.total_avg_rate / 1e6 for r in results_2uav]
    crbs_2 = [r.total_avg_crb for r in results_2uav]

    ax.plot(crbs_1, rates_1, "o-", label="1 UAV", color="#e74c3c")
    ax.plot(crbs_2, rates_2, "s-", label="2 UAVs", color="#2ecc71")

    ax.set_xlabel("Avg CRB RMSE [m]")
    ax.set_ylabel("Avg User Rate [Mbps]")
    ax.set_title("1 UAV vs 2 UAVs — Pareto Trade-off")
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------
# Time-series
# ------------------------------------------------------------------


def plot_time_series(
    results: SimulationResults,
    save_path: str | None = None,
):
    """Plot communication and sensing metrics over time."""
    set_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    t = results.time

    # Communication
    ax = axes[0]
    ax.plot(t, results.avg_user_rate_bps / 1e6, label="Avg rate")
    ax.plot(t, results.min_user_rate_bps / 1e6, label="Min rate", ls="--")
    ax.set_ylabel("User Rate [Mbps]")
    ax.set_title(f"ISAC Performance — {results.n_uavs} UAV(s)")
    ax.legend()

    # Sensing
    ax = axes[1]
    ax.plot(t, results.avg_crb_rmse_m, label="Avg CRB RMSE", color="orange")
    ax.set_ylabel("CRB RMSE [m]")
    ax.set_xlabel("Time [s]")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------
# Trajectory map
# ------------------------------------------------------------------


def _has_motion(history: np.ndarray | None) -> bool:
    """True if a (steps, n, 3) history shows non-trivial movement."""
    if history is None or history.size == 0 or history.shape[0] < 2:
        return False
    return bool(np.any(history.std(axis=0) > 1e-3))


def plot_trajectory_map(
    results: SimulationResults,
    scenario_params=None,
    user_positions: np.ndarray | None = None,
    target_positions: np.ndarray | None = None,
    user_history: np.ndarray | None = None,
    target_history: np.ndarray | None = None,
    save_path: str | None = None,
):
    """
    2D map showing UAV trajectories, users, and targets.

    If `user_history` / `target_history` are supplied and contain motion,
    the entity's path is rendered as a thin trail with a marker at the
    final position. Falls back to a single point from `user_positions`
    / `target_positions` for static entities.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    area = 500.0
    if scenario_params:
        area = scenario_params.area_size

    # Fall back to results attributes if explicit history not passed
    if user_history is None:
        user_history = getattr(results, "user_history", None)
    if target_history is None:
        target_history = getattr(results, "target_history", None)

    # UAV trajectories
    traj = results.uav_trajectories  # (steps, n_uavs, 3)
    colors = plt.cm.Set1(np.linspace(0, 1, results.n_uavs))
    for i in range(results.n_uavs):
        ax.plot(
            traj[:, i, 0], traj[:, i, 1],
            color=colors[i], label=f"UAV {i+1}",
        )
        ax.scatter(
            traj[0, i, 0], traj[0, i, 1],
            color=colors[i], marker="^", s=150, zorder=5, edgecolors="k",
        )
        ax.scatter(
            traj[-1, i, 0], traj[-1, i, 1],
            color=colors[i], marker="v", s=150, zorder=5, edgecolors="k",
        )

    # Users
    if _has_motion(user_history):
        for k in range(user_history.shape[1]):
            ax.plot(
                user_history[:, k, 0], user_history[:, k, 1],
                color="blue", alpha=0.35, linewidth=0.8, zorder=3,
            )
        ax.scatter(
            user_history[-1, :, 0], user_history[-1, :, 1],
            marker="o", s=80, c="blue", label="Users (final)", zorder=4,
        )
    elif user_positions is not None:
        ax.scatter(
            user_positions[:, 0], user_positions[:, 1],
            marker="o", s=80, c="blue", label="Users", zorder=4,
        )

    # Targets
    if _has_motion(target_history):
        for q in range(target_history.shape[1]):
            ax.plot(
                target_history[:, q, 0], target_history[:, q, 1],
                color="red", alpha=0.35, linewidth=0.8, zorder=3,
            )
        ax.scatter(
            target_history[-1, :, 0], target_history[-1, :, 1],
            marker="x", s=100, c="red", linewidths=2,
            label="Sensing Targets (final)", zorder=4,
        )
    elif target_positions is not None:
        ax.scatter(
            target_positions[:, 0], target_positions[:, 1],
            marker="x", s=100, c="red", linewidths=2,
            label="Sensing Targets", zorder=4,
        )

    ax.set_xlim(0, area)
    ax.set_ylim(0, area)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"UAV Trajectories — {results.n_uavs} UAV(s)")
    ax.legend(loc="upper right")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ------------------------------------------------------------------
# Bar chart comparison
# ------------------------------------------------------------------


def plot_comparison_bars(
    results_dict: dict[str, SimulationResults],
    save_path: str | None = None,
):
    """
    Bar chart comparing key metrics across scenarios.

    Parameters
    ----------
    results_dict : {"1 UAV": results1, "2 UAVs": results2, ...}
    """
    set_plot_style()
    labels = list(results_dict.keys())
    n = len(labels)

    metrics = {
        "Avg Rate [Mbps]": [r.total_avg_rate / 1e6 for r in results_dict.values()],
        "Min Rate [Mbps]": [r.total_min_rate / 1e6 for r in results_dict.values()],
        "CRB RMSE [m]": [r.total_avg_crb for r in results_dict.values()],
        "Sensing SNR [dB]": [r.total_avg_sensing_snr for r in results_dict.values()],
    }

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    for ax, (name, values), color in zip(axes, metrics.items(), colors):
        bars = ax.bar(labels, values, color=color, alpha=0.8)
        ax.set_title(name)
        ax.bar_label(bars, fmt="%.2f", fontsize=9)

    plt.suptitle("Scenario Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
