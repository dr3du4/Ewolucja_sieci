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
    EnergyParams,
    ISACSimulation,
    joint_optimize_trajectory,
)
from uav_isac.visualization import (
    plot_pareto_curve,
    plot_pareto_comparison,
    plot_pareto_multi,
    plot_time_series,
    plot_trajectory_map,
    plot_comparison_bars,
    plot_energy_per_alpha,
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


def _feas_tag(res) -> str:
    if res.mission_feasible:
        return ""
    return f" [INFEASIBLE: UAVs {res.infeasible_uavs}]"


def run_pareto(
    scenario_params, channel_params, sensing_params_template,
    energy_params, alphas, label,
):
    """Run single greedy + Pareto sweep over alpha. Returns (single, [pareto])."""
    sim = ISACSimulation(
        scenario_params,
        channel_params,
        SensingParams(**sensing_params_template.__dict__),
        energy_params,
    )
    single = sim.run(trajectory_policy="greedy")
    print(
        f"[{label}] avg rate: {single.total_avg_rate/1e6:.2f} Mbps,"
        f" CRB RMSE: {single.total_avg_crb:.2f} m,"
        f" E: {single.energy_consumed_j_avg/1000:.2f} kJ/UAV"
        f"{_feas_tag(single)}"
    )

    pareto = []
    for a in alphas:
        sp = SensingParams(**sensing_params_template.__dict__)
        sp.sensing_power_ratio = float(a)
        s = ISACSimulation(scenario_params, channel_params, sp, energy_params)
        res = s.run(trajectory_policy="greedy")
        pareto.append(res)
        print(
            f"  α={a:.1f} → rate={res.total_avg_rate/1e6:.2f} Mbps,"
            f" CRB={res.total_avg_crb:.2f} m,"
            f" E={res.energy_consumed_j_avg/1000:.2f} kJ"
            f"{_feas_tag(res)}"
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
    energy_params = EnergyParams(
        p_hover_w=100.0,
        k_drag=0.5,
        battery_j=15_000.0,
    )

    seed = 42
    alphas = np.linspace(0.1, 0.9, 9)

    # ==================================================================
    # Run all (n_uavs × mobility) configurations
    # ==================================================================
    # runs[(n_uavs, mob_name)] = (scenario_params, single_result, pareto_list)
    runs: dict[tuple[int, str], tuple] = {}

    for n_uavs in (1, 2, 3, 4):
        for mob_name, user_mob, target_mob in MOBILITY_CONFIGS:
            label = f"{n_uavs}uav_{mob_name}"
            print("\n" + "=" * 60)
            print(f"Running {label}")
            print("=" * 60)

            sp = build_scenario(n_uavs, user_mob, target_mob, seed)
            single, pareto = run_pareto(
                sp, channel_params, sensing_params_template,
                energy_params, alphas, label,
            )
            runs[(n_uavs, mob_name)] = (sp, single, pareto)

    # ==================================================================
    # Joint optimization (static only, both 1-UAV and 2-UAV)
    # SLSQP with greedy warm-start; α swept identically to greedy so
    # Pareto curves overlay directly for comparison.
    # ==================================================================
    # joint_runs[n_uavs] = list[SimulationResults] indexed by alpha
    joint_runs: dict[int, list] = {}

    for n_uavs in (1, 2):
        label = f"{n_uavs}uav_joint"
        print("\n" + "=" * 60)
        print(f"Running {label} (SLSQP, static, warm-start = greedy)")
        print("=" * 60)
        sp = build_scenario(n_uavs, "static", "static", seed)

        sweep = []
        for a in alphas:
            sens = SensingParams(**sensing_params_template.__dict__)
            sens.sensing_power_ratio = float(a)
            Q_opt, info = joint_optimize_trajectory(
                sp, channel_params, sens, energy_params, alpha=float(a),
                d_min=30.0, max_iter=30, tol=1e-6, verbose=False,
            )
            # Replay the optimized trajectory through the simulator to get
            # consistent SimulationResults (same metrics path as greedy/separate)
            sim = ISACSimulation(sp, channel_params, sens, energy_params)
            res = sim.run(
                trajectory_policy="preset",
                preset_trajectory=Q_opt,
            )
            sweep.append(res)
            tag = "✓" if info["converged"] else "✗"
            fb = " [fallback]" if info["fallback_used"] else ""
            print(
                f"  α={a:.1f} [{info['method']}{fb} {tag} iter={info['iterations']}]"
                f" → rate={res.total_avg_rate/1e6:.2f} Mbps,"
                f" CRB={res.total_avg_crb:.2f} m,"
                f" E={res.energy_consumed_j_avg/1000:.2f} kJ"
            )
        joint_runs[n_uavs] = sweep

    # ==================================================================
    # Pareto plots
    # ==================================================================
    print("\nGenerating plots...")

    # Per-scenario Pareto curves (static, one per n_uavs)
    for n_uavs in (1, 2, 3, 4):
        plot_pareto_curve(
            runs[(n_uavs, "static")][2], alphas,
            title=f"{n_uavs} UAV(s) — Communication-Sensing Trade-off",
            save_path=str(output_dir / f"pareto_{n_uavs}uav.png"),
        )

    # 1 vs 2 UAV (static) — kept for backward compat
    plot_pareto_comparison(
        runs[(1, "static")][2], runs[(2, "static")][2], alphas,
        save_path=str(output_dir / "pareto_comparison.png"),
    )

    # Cooperation gain: all four n_uavs on one chart (static mobility)
    plot_pareto_multi(
        {f"{n} UAV{'s' if n > 1 else ''}": runs[(n, "static")][2]
         for n in (1, 2, 3, 4)},
        alphas,
        title="Cooperation gain — Pareto vs number of UAVs (static)",
        save_path=str(output_dir / "pareto_n_uavs_static.png"),
    )

    # Mobility sweep within each n_uavs
    for n_uavs in (1, 2, 3, 4):
        bundle = {
            mob_name: runs[(n_uavs, mob_name)][2]
            for mob_name, _, _ in MOBILITY_CONFIGS
        }
        plot_pareto_multi(
            bundle, alphas,
            title=f"{n_uavs} UAV(s) — Pareto across mobility modes",
            save_path=str(output_dir / f"pareto_mobility_{n_uavs}uav.png"),
        )

    # All n_uavs under each mobility mode
    for mob_name, _, _ in MOBILITY_CONFIGS:
        bundle = {
            f"{n} UAV{'s' if n > 1 else ''}": runs[(n, mob_name)][2]
            for n in (1, 2, 3, 4)
        }
        plot_pareto_multi(
            bundle, alphas,
            title=f"{mob_name} — number of UAVs comparison",
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
    # Energy vs α — one chart per n_uavs, all mobility configs overlaid
    # ==================================================================
    for n_uavs in (1, 2, 3, 4):
        bundle = {
            mob_name: runs[(n_uavs, mob_name)][2]
            for mob_name, _, _ in MOBILITY_CONFIGS
        }
        plot_energy_per_alpha(
            bundle, alphas,
            title=f"{n_uavs} UAV(s) — Energy budget vs α",
            battery_j=energy_params.battery_j,
            save_path=str(output_dir / f"energy_per_alpha_{n_uavs}uav.png"),
        )

    # Cross-cut: energy per α with all n_uavs overlaid (static)
    plot_energy_per_alpha(
        {f"{n} UAV{'s' if n > 1 else ''}": runs[(n, "static")][2]
         for n in (1, 2, 3, 4)},
        alphas,
        title="Energy vs α — cooperation gain (static)",
        battery_j=energy_params.battery_j,
        save_path=str(output_dir / "energy_n_uavs_static.png"),
    )

    # ==================================================================
    # Joint vs greedy: Pareto, energy, and trajectory maps (static)
    # ==================================================================
    for n_uavs in (1, 2):
        bundle = {
            "greedy": runs[(n_uavs, "static")][2],
            "joint (SLSQP)": joint_runs[n_uavs],
        }
        plot_pareto_multi(
            bundle, alphas,
            title=f"{n_uavs} UAV(s) — greedy vs joint optimization (static)",
            save_path=str(output_dir / f"pareto_joint_{n_uavs}uav.png"),
        )
        plot_energy_per_alpha(
            bundle, alphas,
            title=f"{n_uavs} UAV(s) — Energy: greedy vs joint (static)",
            battery_j=energy_params.battery_j,
            save_path=str(output_dir / f"energy_joint_{n_uavs}uav.png"),
        )

    # Trajectory map for joint @ α=0.5 (middle of Pareto)
    mid_idx = len(alphas) // 2
    for n_uavs in (1, 2):
        sp = build_scenario(n_uavs, "static", "static", seed)
        res = joint_runs[n_uavs][mid_idx]
        plot_trajectory_map(
            res, sp,
            user_positions=res.user_history[0] if res.user_history.size else None,
            target_positions=res.target_history[0] if res.target_history.size else None,
            user_history=res.user_history,
            target_history=res.target_history,
            save_path=str(output_dir / f"trajectory_joint_{n_uavs}uav_alpha0.5.png"),
        )

    # ==================================================================
    # Comparison bars
    # ==================================================================
    plot_comparison_bars(
        {f"{n} UAV{'s' if n > 1 else ''}": runs[(n, "static")][1]
         for n in (1, 2, 3, 4)},
        save_path=str(output_dir / "comparison_bars.png"),
    )
    plot_comparison_bars(
        {f"{n} UAV{'s' if n > 1 else ''}": runs[(n, "mobile_both")][1]
         for n in (1, 2, 3, 4)},
        save_path=str(output_dir / "comparison_bars_mobile_both.png"),
    )
    plot_comparison_bars(
        {
            "greedy 2 UAV": runs[(2, "static")][1],
            "joint 2 UAV": joint_runs[2][mid_idx],
        },
        save_path=str(output_dir / "comparison_bars_greedy_vs_joint.png"),
    )

    # ==================================================================
    # Export CSV (long format: mobility, n_uavs, alpha, rate, crb)
    # ==================================================================
    print("Exporting CSV data...")
    rows = [
        "policy,mobility,n_uavs,alpha,avg_rate_mbps,crb_rmse_m,"
        "energy_kj_per_uav,feasible"
    ]
    for (n_uavs, mob_name), (_, _, pareto) in runs.items():
        for a, r in zip(alphas, pareto):
            rows.append(
                f"greedy,{mob_name},{n_uavs},{a:.1f},"
                f"{r.total_avg_rate/1e6:.4f},{r.total_avg_crb:.4f},"
                f"{r.energy_consumed_j_avg/1000:.4f},"
                f"{int(r.mission_feasible)}"
            )
    for n_uavs, sweep in joint_runs.items():
        for a, r in zip(alphas, sweep):
            rows.append(
                f"joint,static,{n_uavs},{a:.1f},"
                f"{r.total_avg_rate/1e6:.4f},{r.total_avg_crb:.4f},"
                f"{r.energy_consumed_j_avg/1000:.4f},"
                f"{int(r.mission_feasible)}"
            )
    (output_dir / "pareto_data.csv").write_text("\n".join(rows) + "\n")

    print(f"\nAll results saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
