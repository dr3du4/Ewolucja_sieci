"""
UAV-ISAC Simulation — Main Entry Point.

Runs 1-UAV vs 2-UAV comparison with Pareto trade-off analysis
across four mobility configurations:
    static, mobile_users, mobile_targets, mobile_both.

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
    plot_pareto_multi,
    plot_time_series,
    plot_trajectory_map,
    plot_comparison_bars,
)


MOBILITY_CONFIGS = [
    # (name, user_mobility, target_mobility)
    ("static",         "static", "static"),
    ("mobile_users",   "linear", "static"),
    ("mobile_targets", "static", "linear"),
    ("mobile_both",    "linear", "linear"),
]


def build_scenario(n_uavs: int, user_mob: str, target_mob: str, seed: int) -> ScenarioParams:
    return ScenarioParams(
        area_size=500.0,
        n_uavs=n_uavs,
        uav_altitude=100.0,
        n_users=4,
        n_targets=2,
        mission_duration=60.0,
        dt=1.0,
        user_mobility=user_mob,
        user_max_speed=1.4,
        target_mobility=target_mob,
        target_max_speed=5.0,
        seed=seed,
    )


def run_pareto(scenario_params, channel_params, sensing_params_template, alphas, label):
    """Run single greedy + Pareto sweep over alpha. Returns (single, [pareto])."""
    sim = ISACSimulation(
        scenario_params,
        channel_params,
        SensingParams(**sensing_params_template.__dict__),
    )
    single = sim.run(trajectory_policy="greedy")
    print(
        f"[{label}] avg rate: {single.total_avg_rate/1e6:.2f} Mbps,"
        f" CRB RMSE: {single.total_avg_crb:.2f} m"
    )

    pareto = []
    for a in alphas:
        sp = SensingParams(**sensing_params_template.__dict__)
        sp.sensing_power_ratio = float(a)
        s = ISACSimulation(scenario_params, channel_params, sp)
        res = s.run(trajectory_policy="greedy")
        pareto.append(res)
        print(
            f"  α={a:.1f} → rate={res.total_avg_rate/1e6:.2f} Mbps,"
            f" CRB={res.total_avg_crb:.2f} m"
        )
    return single, pareto


def main():
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    channel_params = ChannelParams(
        freq_ghz=2.0,
        tx_power_dbm=30.0,
        bandwidth_mhz=10.0,
    )
    sensing_params_template = SensingParams(
        n_antennas=8,
        n_pulses=64,
        radar_cross_section=1.0,
    )

    seed = 42
    alphas = np.linspace(0.1, 0.9, 9)

    # ==================================================================
    # Run all (n_uavs × mobility) configurations
    # ==================================================================
    # runs[(n_uavs, mob_name)] = (scenario_params, single_result, pareto_list)
    runs: dict[tuple[int, str], tuple] = {}

    for n_uavs in (1, 2):
        for mob_name, user_mob, target_mob in MOBILITY_CONFIGS:
            label = f"{n_uavs}uav_{mob_name}"
            print("\n" + "=" * 60)
            print(f"Running {label}")
            print("=" * 60)

            sp = build_scenario(n_uavs, user_mob, target_mob, seed)
            single, pareto = run_pareto(
                sp, channel_params, sensing_params_template, alphas, label
            )
            runs[(n_uavs, mob_name)] = (sp, single, pareto)

    # ==================================================================
    # Pareto plots
    # ==================================================================
    print("\nGenerating plots...")

    # Per-scenario Pareto curves (static, existing names preserved)
    plot_pareto_curve(
        runs[(1, "static")][2], alphas,
        title="1 UAV — Communication-Sensing Trade-off",
        save_path=str(output_dir / "pareto_1uav.png"),
    )
    plot_pareto_curve(
        runs[(2, "static")][2], alphas,
        title="2 UAVs — Communication-Sensing Trade-off",
        save_path=str(output_dir / "pareto_2uav.png"),
    )

    # 1 vs 2 UAV (static)
    plot_pareto_comparison(
        runs[(1, "static")][2], runs[(2, "static")][2], alphas,
        save_path=str(output_dir / "pareto_comparison.png"),
    )

    # Mobility sweep within each n_uavs
    for n_uavs in (1, 2):
        bundle = {
            mob_name: runs[(n_uavs, mob_name)][2]
            for mob_name, _, _ in MOBILITY_CONFIGS
        }
        plot_pareto_multi(
            bundle, alphas,
            title=f"{n_uavs} UAV(s) — Pareto across mobility modes",
            save_path=str(output_dir / f"pareto_mobility_{n_uavs}uav.png"),
        )

    # 1 vs 2 UAV under each mobility mode
    for mob_name, _, _ in MOBILITY_CONFIGS:
        bundle = {
            f"1 UAV": runs[(1, mob_name)][2],
            f"2 UAVs": runs[(2, mob_name)][2],
        }
        plot_pareto_multi(
            bundle, alphas,
            title=f"{mob_name} — 1 UAV vs 2 UAVs",
            save_path=str(output_dir / f"pareto_uavs_{mob_name}.png"),
        )

    # ==================================================================
    # Time series (static baseline only — others omitted to keep results/ tidy)
    # ==================================================================
    plot_time_series(
        runs[(1, "static")][1],
        save_path=str(output_dir / "timeseries_1uav.png"),
    )
    plot_time_series(
        runs[(2, "static")][1],
        save_path=str(output_dir / "timeseries_2uav.png"),
    )

    # ==================================================================
    # Trajectory maps for every (n_uavs, mobility)
    # ==================================================================
    for (n_uavs, mob_name), (sp, single, _) in runs.items():
        plot_trajectory_map(
            single, sp,
            user_positions=single.user_history[0] if single.user_history.size else None,
            target_positions=single.target_history[0] if single.target_history.size else None,
            user_history=single.user_history,
            target_history=single.target_history,
            save_path=str(output_dir / f"trajectory_{n_uavs}uav_{mob_name}.png"),
        )

    # ==================================================================
    # Comparison bars
    # ==================================================================
    plot_comparison_bars(
        {"1 UAV": runs[(1, "static")][1], "2 UAVs": runs[(2, "static")][1]},
        save_path=str(output_dir / "comparison_bars.png"),
    )
    plot_comparison_bars(
        {"1 UAV": runs[(1, "mobile_both")][1], "2 UAVs": runs[(2, "mobile_both")][1]},
        save_path=str(output_dir / "comparison_bars_mobile_both.png"),
    )

    # ==================================================================
    # Export CSV (long format: mobility, n_uavs, alpha, rate, crb)
    # ==================================================================
    print("Exporting CSV data...")
    rows = ["mobility,n_uavs,alpha,avg_rate_mbps,crb_rmse_m"]
    for (n_uavs, mob_name), (_, _, pareto) in runs.items():
        for a, r in zip(alphas, pareto):
            rows.append(
                f"{mob_name},{n_uavs},{a:.1f},"
                f"{r.total_avg_rate/1e6:.4f},{r.total_avg_crb:.4f}"
            )
    (output_dir / "pareto_data.csv").write_text("\n".join(rows) + "\n")

    print(f"\nAll results saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
