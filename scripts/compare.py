#!/usr/bin/env python3
"""
Fair multi-run scenario comparison (sim-plant overrides) for decimated RL control.

This is a decimated variant of:
  scripts/compare_sim_plant.py

It keeps the same comparison_runs semantics (including per-run sim-plant overrides),
but evaluates controllers with:
  - physics at --physics-hz (default 1000 Hz), and
  - outer controller step at --rl-hz (default 100 Hz),
by wrapping the Flightmare VecEnv with VecPhysicsDecimationWrapper.

Important:
  - For plotting and metrics, this script uses dt = 1/rl_hz (controller-step time).
  - For simulator integration, env.quadrotor_env.sim_dt is forced to 1/physics_hz.
  - Each ``<plot_dir>/<run>/<scenario>/rollout_data.pkl`` includes ``extras`` with
    physics_hz, rl_hz, and reward_aggregate when applicable.
  - Per-scenario plots (when --save_plots) include ``metrics_<Controller>_barrier_values.png``:
    uses scenario ``position_barriers`` when present; otherwise the same barriers as
    ``--cbf_config`` (with ``q - r_uav`` as in ``CBFFilter``).
"""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import scripts.compare_core as cs
import scripts.compare_fair as fr
import scripts.compare_sim_plant as sim_plant
from scripts.train import VecPhysicsDecimationWrapper

ALL_CONTROLLERS = fr.ALL_CONTROLLERS


def _validate_decimation(physics_hz: float, rl_hz: float) -> Tuple[int, float, float]:
    ph = float(physics_hz)
    rlh = float(rl_hz)
    if ph <= 0.0 or rlh <= 0.0:
        raise ValueError("physics_hz and rl_hz must be positive")
    ratio = ph / rlh
    n_substeps = int(round(ratio))
    if abs(ratio - n_substeps) > 1e-6:
        raise ValueError(
            f"physics_hz / rl_hz must be an integer; got {ph}/{rlh} = {ratio}"
        )
    return n_substeps, 1.0 / ph, 1.0 / rlh


@contextmanager
def _patch_compare_env_build_for_decimation(
    physics_dt: float,
    rl_dt: float,
    n_substeps: int,
    reward_aggregate: str,
):
    """Temporarily patch compare_core env build to use decimated timing."""
    orig_build_env_cfg = cs._build_env_cfg
    orig_make_env = cs._make_env

    def _build_env_cfg_decimated(scfg: Dict[str, Any]) -> Dict[str, Any]:
        cfg = orig_build_env_cfg(scfg)
        max_steps = int(scfg.get("max_episode_steps", 1000))
        cfg.setdefault("env", {})
        cfg["env"].setdefault("quadrotor_env", {})
        cfg["env"]["quadrotor_env"]["sim_dt"] = float(physics_dt)
        # Keep wall-clock horizon consistent with outer RL timing.
        cfg["env"]["quadrotor_env"]["max_t"] = float(max_steps) * float(rl_dt)
        return cfg

    def _make_env_decimated(cfg: Dict[str, Any]):
        base_env = orig_make_env(cfg)
        return VecPhysicsDecimationWrapper(
            base_env,
            n_substeps=int(n_substeps),
            aggregate_reward=str(reward_aggregate),
        )

    cs._build_env_cfg = _build_env_cfg_decimated
    cs._make_env = _make_env_decimated
    try:
        yield
    finally:
        cs._build_env_cfg = orig_build_env_cfg
        cs._make_env = orig_make_env


def _run_one_comparison_run_decimated(
    run_idx: int,
    run_cfg: Dict[str, Any],
    scfg: Dict[str, Any],
    args: argparse.Namespace,
    temp_files: List[str],
):
    run_scfg = sim_plant._scfg_with_quadrotor_overrides(scfg, run_cfg)
    qo = run_cfg.get("quadrotor_overrides") or {}
    if qo:
        merged_q = run_scfg.get("quadrotor") or {}
        sim_mass_override = run_scfg.get("_sim_mass_override")
        effective_sim_mass = (
            float(sim_mass_override)
            if sim_mass_override is not None
            else merged_q.get("mass")
        )
        nominal_action_mass = run_scfg.get(
            "_nominal_action_mass",
            merged_q.get("mass"),
        )
        print(
            "  Simulation plant: "
            f"mass={effective_sim_mass}, motor_tau={merged_q.get('motor_tau')} "
            f"(overrides applied: {qo})",
        )
        if sim_mass_override is not None:
            print(
                "  Action scaling (frozen) uses nominal mass "
                f"{nominal_action_mass} kg (sim-plant mass override "
                f"{sim_mass_override} kg applied via setEnvMasses after env build).",
            )

    n_substeps, physics_dt, rl_dt = _validate_decimation(args.physics_hz, args.rl_hz)
    cfg_rl_dt = run_scfg.get("sim_dt", None)
    if cfg_rl_dt is not None and abs(float(cfg_rl_dt) - rl_dt) > 1e-9:
        print(
            f"  Note: scenarios sim_dt={cfg_rl_dt} overridden to controller dt={rl_dt:.6f}s "
            f"for plotting/metrics.",
        )
    run_scfg["sim_dt"] = rl_dt

    print(
        f"  Decimation: physics_hz={args.physics_hz:g}, rl_hz={args.rl_hz:g}, "
        f"n_substeps={n_substeps}, reward_aggregate={args.reward_aggregate}"
    )

    with _patch_compare_env_build_for_decimation(
        physics_dt=physics_dt,
        rl_dt=rl_dt,
        n_substeps=n_substeps,
        reward_aggregate=args.reward_aggregate,
    ):
        return fr._run_one_comparison_run(run_idx, run_cfg, run_scfg, args, temp_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fair scenario comparison with per-run sim-plant overrides + decimated "
            "controller timing (physics_hz vs rl_hz)."
        ),
    )
    parser.add_argument(
        "--scenarios_config",
        type=str,
        default=os.path.join(_REPO_ROOT, "configs", "scenarious.yaml"),
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--cbf_config", type=str, default=None)
    parser.add_argument("--mpc_config", type=str, default=None)
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=None,
        choices=ALL_CONTROLLERS,
        help="Controllers to run (default: all)",
    )
    parser.add_argument(
        "--skip_controllers",
        nargs="+",
        default=None,
        choices=ALL_CONTROLLERS,
        help="Exclude these controllers (applied after --controllers).",
    )
    parser.add_argument("--n_seeds", type=int, default=1, help="Stochastic rollouts per scenario (mean±std).")
    parser.add_argument("--save_plots", action="store_true", default=False)
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help=(
            "Root directory for this evaluation (summary CSV, logs, config copies, nested plots). "
            "Default: comparison_plots/<folder_name_next_to_checkpoint> under the repo."
        ),
    )
    parser.add_argument("--comparison_base_seed", type=int, default=None)
    parser.add_argument(
        "--no_sync_stochastic_seeds",
        action="store_true",
        help="Do not reseed between controllers (legacy).",
    )
    parser.add_argument(
        "--paper_plots",
        action="store_true",
        help="Write publication figures to {plot_dir}/paper/ after evaluation.",
    )
    parser.add_argument(
        "--no_recompile_cbf",
        action="store_true",
        help="Reuse existing acados slack CBF .so if present (faster iterative runs).",
    )
    parser.add_argument("--physics-hz", type=float, default=1000.0, help="Simulation physics rate (Hz).")
    parser.add_argument("--rl-hz", type=float, default=100.0, help="Outer controller/policy rate (Hz).")
    parser.add_argument(
        "--reward-aggregate",
        type=str,
        choices=("sum", "mean"),
        default="sum",
        help="How to aggregate inner-step rewards into one outer RL step.",
    )
    parser.add_argument(
        "--no-save-rollouts",
        action="store_true",
        help="Do not write rollout_data.pkl per scenario (default: save).",
    )
    args = parser.parse_args()

    # Validate early so user gets immediate feedback.
    _validate_decimation(args.physics_hz, args.rl_hz)

    scfg = cs._load_scenarios_config(args.scenarios_config)
    runs = scfg.get("comparison_runs")
    if not runs:
        runs = [
            {
                "name": "nominal",
                "description": "implicit single run",
                "controller_model_overrides": {},
                "quadrotor_overrides": {},
            }
        ]

    try:
        controllers = fr._resolve_controllers(args)
    except ValueError as e:
        parser.error(str(e))

    needs_rl = any(c in controllers for c in ["RL", "RL+CBF"])
    checkpoint = args.checkpoint or scfg.get("rl_policy", {}).get("checkpoint")
    if needs_rl and checkpoint is None:
        parser.error("--checkpoint required for RL controllers")

    model_folder_name, model_config_yaml = sim_plant._rl_model_folder_and_config_yaml(checkpoint)

    plot_cfg = scfg.get("plotting", {})
    if args.plot_dir is None:
        args.plot_dir = os.path.join(_REPO_ROOT, "comparison_plots", model_folder_name)
    args.plot_dir = os.path.abspath(os.path.expanduser(args.plot_dir))
    args.nested_comparison_plot_layout = True

    sim_plant._copy_run_configs(args.plot_dir, args.scenarios_config, model_config_yaml)
    print(f"Output directory (configs copied): {args.plot_dir}")
    print(
        f"Decimated timing: physics_hz={args.physics_hz:g}, rl_hz={args.rl_hz:g}, "
        f"reward_aggregate={args.reward_aggregate}",
    )

    if not args.save_plots and plot_cfg.get("save_plots", False):
        args.save_plots = True

    temp_files: List[str] = []
    all_summaries: List[dict] = []
    all_ep_data: Dict[str, Dict[str, Dict[str, dict]]] = {}
    try:
        for ri, run_cfg in enumerate(runs):
            rows, _, ep_map = _run_one_comparison_run_decimated(ri, run_cfg, scfg, args, temp_files)
            all_summaries.extend(rows)
            run_name = run_cfg.get("name", f"run_{ri}")
            all_ep_data[run_name] = ep_map
    finally:
        for p in temp_files:
            try:
                os.unlink(p)
            except OSError:
                pass

    n_seeds = max(1, int(args.n_seeds))
    col_w = 22
    W = 36 + col_w * len(controllers)
    print(f"\n{'='*W}")
    print("COMBINED SUMMARY (all comparison runs) [sim_plant_decimated]")
    print(f"{'='*W}")
    header = f"{'Run':<18}{'Scenario':<30}" + "".join(f"{c:>{col_w}}" for c in controllers)
    print(header)
    print("-" * W)
    for row in all_summaries:
        line = f"{row['run']:<18}{row['name']:<30}"
        for c in controllers:
            err = row.get(f"{c}_err", float("nan"))
            steps = row.get(f"{c}_steps", 0)
            if n_seeds > 1 and f"{c}_err_std" in row:
                es = row[f"{c}_err_std"]
                line += f"{err:>6.3f}±{es:.2f}m {steps:>5.0f}st  "
            else:
                line += f"{err:>7.3f}m {steps:>5.0f}st  "
        print(line)
    print(f"{'='*W}")

    csv_path = os.path.join(args.plot_dir, "summary.csv")
    fr.write_summary_csv(csv_path, all_summaries, controllers, n_seeds)
    print(f"Saved summary CSV to: {csv_path}")

    xlsx_path = os.path.join(args.plot_dir, "comparison_summary.xlsx")
    try:
        fr.write_comparison_summary_xlsx(
            xlsx_path,
            all_summaries,
            controllers,
            scfg,
            os.path.join(_REPO_ROOT, "template.xlsx"),
        )
        print(f"Saved comparison Excel to: {xlsx_path}")
    except ImportError as e:
        print(f"Skipping Excel export: {e}")
    except OSError as e:
        print(f"Excel export failed: {e}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.plot_dir, f"compare_fair_runs_sim_plant_decimated_log_{ts}.txt")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        f.write("compare log\n")
        f.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"scenarios_config: {args.scenarios_config}\n")
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"controllers: {controllers}\n")
        f.write(f"n_seeds: {n_seeds}\n")
        f.write(f"physics_hz: {args.physics_hz}\n")
        f.write(f"rl_hz: {args.rl_hz}\n")
        f.write(f"reward_aggregate: {args.reward_aggregate}\n\n")
        for row in all_summaries:
            f.write(f"{row['run']} / {row['name']}\n")
            for c in controllers:
                if n_seeds > 1 and f"{c}_err_std" in row:
                    f.write(
                        f"  {c}: err={row.get(f'{c}_err', float('nan')):.4f} ± {row[f'{c}_err_std']:.4f}, "
                        f"mae={row.get(f'{c}_mae', float('nan')):.4f} ± {row.get(f'{c}_mae_std', float('nan')):.4f}, "
                        f"rew={row.get(f'{c}_rew', float('nan')):.3f}, "
                        f"steps={row.get(f'{c}_steps', 0)}, "
                        f"wall={row.get(f'{c}_wall', float('nan')):.4f}s\n",
                    )
                else:
                    f.write(
                        f"  {c}: err={row.get(f'{c}_err', float('nan')):.4f}, "
                        f"mae={row.get(f'{c}_mae', float('nan')):.4f}, "
                        f"rew={row.get(f'{c}_rew', float('nan')):.3f}, "
                        f"steps={row.get(f'{c}_steps', 0)}, "
                        f"wall={row.get(f'{c}_wall', float('nan')):.4f}s\n",
                    )
            f.write("\n")
    print(f"Saved log to: {log_path}")

    if args.paper_plots:
        fr.generate_paper_plots(all_summaries, all_ep_data, scfg, args)


if __name__ == "__main__":
    main()
