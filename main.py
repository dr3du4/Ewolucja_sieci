"""
UAV-ISAC Simulation — Main Entry Point.

Runs 1-UAV vs 2-UAV comparison with Pareto trade-off analysis.

Usage:
    python main.py
"""

import numpy as np
from pathlib import Path

from uav_isac import (
    ChannelParams,
    ScenarioParams,
    SensingParams,
    ISACSimulation,
)
from uav_isac.visualization import (
    plot_pareto_curve,
    plot_pareto_comparison,
    plot_time_series,
    plot_trajectory_map,
    plot_comparison_bars,
)


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # ---- Common parameters ----
    channel_params = ChannelParams(
        freq_ghz=2.0,
        tx_power_dbm=30.0,
        bandwidth_mhz=10.0,
    )
    sensing_params = SensingParams(
        n_antennas=8,
        n_pulses=64,
        radar_cross_section=1.0,
    )

    seed = 42
    alphas = np.linspace(0.1, 0.9, 9)

    # ==================================================================
    # 1-UAV scenario
    # ==================================================================
    print("=" * 60)
    print("Running 1-UAV scenario...")
    print("=" * 60)

    scenario_1uav = ScenarioParams(
        area_size=500.0,
        n_uavs=1,
        uav_altitude=100.0,
        n_users=4,
        n_targets=2,
        mission_duration=60.0,
        dt=1.0,
        seed=seed,
    )

    sim_1 = ISACSimulation(scenario_1uav, channel_params, SensingParams(**sensing_params.__dict__))

    # Single run (default alpha=0.5)
    results_1 = sim_1.run(trajectory_policy="greedy")
    print(f"  Avg rate: {results_1.total_avg_rate/1e6:.2f} Mbps")
    print(f"  CRB RMSE: {results_1.total_avg_crb:.2f} m")

    # Pareto sweep
    results_1_pareto = []
    for a in alphas:
        sp = SensingParams(**sensing_params.__dict__)
        sp.sensing_power_ratio = float(a)
        sim = ISACSimulation(scenario_1uav, channel_params, sp)
        results_1_pareto.append(sim.run(trajectory_policy="greedy"))
        print(f"  α={a:.1f} → rate={results_1_pareto[-1].total_avg_rate/1e6:.2f} Mbps, CRB={results_1_pareto[-1].total_avg_crb:.2f} m")

    # ==================================================================
    # 2-UAV scenario
    # ==================================================================
    print("\n" + "=" * 60)
    print("Running 2-UAV scenario...")
    print("=" * 60)

    scenario_2uav = ScenarioParams(
        area_size=500.0,
        n_uavs=2,
        uav_altitude=100.0,
        n_users=4,
        n_targets=2,
        mission_duration=60.0,
        dt=1.0,
        seed=seed,
    )

    sim_2 = ISACSimulation(scenario_2uav, channel_params, SensingParams(**sensing_params.__dict__))
    results_2 = sim_2.run(trajectory_policy="greedy")
    print(f"  Avg rate: {results_2.total_avg_rate/1e6:.2f} Mbps")
    print(f"  CRB RMSE: {results_2.total_avg_crb:.2f} m")

    # Pareto sweep
    results_2_pareto = []
    for a in alphas:
        sp = SensingParams(**sensing_params.__dict__)
        sp.sensing_power_ratio = float(a)
        sim = ISACSimulation(scenario_2uav, channel_params, sp)
        results_2_pareto.append(sim.run(trajectory_policy="greedy"))
        print(f"  α={a:.1f} → rate={results_2_pareto[-1].total_avg_rate/1e6:.2f} Mbps, CRB={results_2_pareto[-1].total_avg_crb:.2f} m")

    # ==================================================================
    # Generate plots
    # ==================================================================
    print("\nGenerating plots...")

    # Pareto curves
    plot_pareto_curve(
        results_1_pareto, alphas,
        title="1 UAV — Communication-Sensing Trade-off",
        save_path=str(output_dir / "pareto_1uav.png"),
    )
    plot_pareto_curve(
        results_2_pareto, alphas,
        title="2 UAVs — Communication-Sensing Trade-off",
        save_path=str(output_dir / "pareto_2uav.png"),
    )

    # Pareto comparison
    plot_pareto_comparison(
        results_1_pareto, results_2_pareto, alphas,
        save_path=str(output_dir / "pareto_comparison.png"),
    )

    # Time series
    plot_time_series(results_1, save_path=str(output_dir / "timeseries_1uav.png"))
    plot_time_series(results_2, save_path=str(output_dir / "timeseries_2uav.png"))

    # Trajectory maps (use scenario to get user/target positions)
    from uav_isac.scenario import Scenario
    sc1 = Scenario(scenario_1uav)
    plot_trajectory_map(
        results_1, scenario_1uav,
        user_positions=sc1.user_positions,
        target_positions=sc1.target_positions,
        save_path=str(output_dir / "trajectory_1uav.png"),
    )
    sc2 = Scenario(scenario_2uav)
    plot_trajectory_map(
        results_2, scenario_2uav,
        user_positions=sc2.user_positions,
        target_positions=sc2.target_positions,
        save_path=str(output_dir / "trajectory_2uav.png"),
    )

    # Bar comparison
    plot_comparison_bars(
        {"1 UAV": results_1, "2 UAVs": results_2},
        save_path=str(output_dir / "comparison_bars.png"),
    )

    # ==================================================================
    # Export CSV
    # ==================================================================
    print("Exporting CSV data...")
    header = "alpha,avg_rate_mbps_1uav,crb_rmse_1uav,avg_rate_mbps_2uav,crb_rmse_2uav"
    rows = []
    for i, a in enumerate(alphas):
        rows.append(
            f"{a:.1f},"
            f"{results_1_pareto[i].total_avg_rate/1e6:.4f},"
            f"{results_1_pareto[i].total_avg_crb:.4f},"
            f"{results_2_pareto[i].total_avg_rate/1e6:.4f},"
            f"{results_2_pareto[i].total_avg_crb:.4f}"
        )
    csv_path = output_dir / "pareto_data.csv"
    csv_path.write_text(header + "\n" + "\n".join(rows))

    print(f"\nAll results saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
