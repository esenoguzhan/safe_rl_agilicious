#!/usr/bin/env python3
"""
Fair multi-run scenario comparison with per-run *simulation* plant parameters.

Same stochastic seeding and controller stack as `compare_scenarios_fair_runs.py`, but each
`comparison_runs` entry may set `quadrotor_overrides`: shallow-merged into the scenarios
YAML `quadrotor:` block for that run only (Flightmare / `quadrotor_dynamics`). Every
scenario in that run uses the same plant.

`controller_model_overrides` still only affects the Python QuadrotorModel YAML used by
MPC/CBF (and RL helpers), not the sim — same as the original fair-runs script.

Example `comparison_runs` (e.g. dedicated YAML or a copy of your scenarios config):

  comparison_runs:
    - name: nominal
      base_quadrotor_model_path: configs/quadrotor_model.yaml
      controller_model_overrides: {}
      quadrotor_overrides: {}

    - name: mass_mismatch_sim
      description: "Heavier sim mass; nominal controller model."
      base_quadrotor_model_path: configs/quadrotor_model.yaml
      controller_model_overrides: {}
      quadrotor_overrides:
        mass: 0.65

    - name: motor_tau_mismatch_sim
      description: "Slower motors in sim; nominal mass and controller model."
      controller_model_overrides: {}
      quadrotor_overrides:
        motor_tau: 0.1

Outputs (default ``--plot_dir``)::

  comparison_plots/<rl_model_folder_name>/
    <scenarios_yaml_basename>          # copy of the scenarios config used
    model_config.yaml                  # copy of RL training config.yaml next to checkpoint (if any)
    summary.csv
    comparison_summary.xlsx              # MAE, wall time, final error (needs openpyxl)
    compare_fair_runs_sim_plant_log_*.txt
    <comparison_runs name>/            # e.g. nominal, mass_mismatch_sim
      <scenario_name>/                 # sanitized scenario name
        rollout_data.pkl               # trajectories for all seeds/controllers (replay plots)
        metrics_<Controller>_*.png     # per-controller metric PNGs (with --save_plots)

Usage:
  python scripts/compare_scenarios_fair_runs_sim_plant.py \\
    --scenarios_config configs/config_scenarious_2_sim_plant.yaml \\
    --checkpoint models/.../best_model \\
    --cbf_config configs/cbf_config.yaml \\
    --mpc_config configs/mpc_config.yaml \\
    [--plot_dir path/to/output_root] \\
    [--n_seeds 3] [--paper_plots] [--skip_controllers MPC] [--no_recompile_cbf]
"""
from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import scripts.compare_scenarios as cs
import scripts.compare_scenarios_fair_runs as fr

ALL_CONTROLLERS = fr.ALL_CONTROLLERS


def _rl_model_folder_and_config_yaml(checkpoint: Optional[str]) -> Tuple[str, Optional[str]]:
    """Derive a short folder label and optional ``config.yaml`` path next to an RL checkpoint."""
    if not checkpoint:
        return "no_rl", None
    cp = os.path.abspath(os.path.expanduser(checkpoint))
    parent = os.path.dirname(cp)
    cur = parent
    for _ in range(8):
        cfg = os.path.join(cur, "config.yaml")
        if os.path.isfile(cfg):
            return os.path.basename(cur), cfg
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    base = os.path.basename(parent) if parent else "checkpoint"
    return base or "checkpoint", None


def _copy_run_configs(plot_dir: str, scenarios_config: str, model_config_yaml: Optional[str]) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    dst_sc = os.path.join(plot_dir, os.path.basename(scenarios_config))
    shutil.copy2(scenarios_config, dst_sc)
    if model_config_yaml and os.path.isfile(model_config_yaml):
        shutil.copy2(model_config_yaml, os.path.join(plot_dir, "model_config.yaml"))


def _scfg_with_quadrotor_overrides(scfg: Dict[str, Any], run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Deep copy ``scfg`` and merge ``run_cfg['quadrotor_overrides']`` into ``quadrotor``.

    Mass overrides are handled specially: they are *not* merged into the
    Flightmare env construction YAML. Doing so would rescale action
    denormalization (``act_mean_`` / ``act_std_``) in the C++ env constructor
    and hide the effect of mass mismatch. Instead we:

      - Snapshot the pre-override ``quadrotor.mass`` as ``_nominal_action_mass``
        so downstream Python ``act_mean`` / ``act_std`` stay pinned to the
        nominal training-time mass.
      - Store the override value as ``_sim_mass_override``; the fair-runs
        runner applies it post-construction via ``setEnvMasses`` so only the
        simulator dynamics change.

    All other overrides (e.g. ``motor_tau``) are merged into ``quadrotor`` as
    before.
    """
    run_scfg = copy.deepcopy(scfg)
    # Always record the nominal (pre-override) action mass so downstream
    # helpers can look it up unambiguously.
    base_q = run_scfg.get("quadrotor") or {}
    if "mass" in base_q:
        run_scfg["_nominal_action_mass"] = float(base_q["mass"])

    overrides = run_cfg.get("quadrotor_overrides")
    if not overrides:
        return run_scfg

    remaining_overrides = dict(overrides)
    if "mass" in remaining_overrides:
        run_scfg["_sim_mass_override"] = float(remaining_overrides.pop("mass"))

    if remaining_overrides:
        q = dict(run_scfg.get("quadrotor") or {})
        for k, v in remaining_overrides.items():
            q[k] = v
        run_scfg["quadrotor"] = q
    return run_scfg


def _run_one_comparison_run(
    run_idx: int,
    run_cfg: Dict[str, Any],
    scfg: Dict[str, Any],
    args: argparse.Namespace,
    temp_files: List[str],
):
    run_scfg = _scfg_with_quadrotor_overrides(scfg, run_cfg)
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
    return fr._run_one_comparison_run(run_idx, run_cfg, run_scfg, args, temp_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fair scenario comparison with per-run quadrotor_overrides for simulation "
            "(see module docstring)."
        ),
    )
    parser.add_argument(
        "--scenarios_config",
        type=str,
        default=os.path.join(_REPO_ROOT, "configs", "config_scenarious_2_sim_plant.yaml"),
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
    parser.add_argument(
        "--no-save-rollouts",
        action="store_true",
        help="Do not write rollout_data.pkl per scenario (default: save).",
    )
    args = parser.parse_args()

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

    model_folder_name, model_config_yaml = _rl_model_folder_and_config_yaml(checkpoint)

    plot_cfg = scfg.get("plotting", {})
    if args.plot_dir is None:
        args.plot_dir = os.path.join(_REPO_ROOT, "comparison_plots", model_folder_name)
    args.plot_dir = os.path.abspath(os.path.expanduser(args.plot_dir))
    args.nested_comparison_plot_layout = True

    _copy_run_configs(args.plot_dir, args.scenarios_config, model_config_yaml)
    print(f"Output directory (configs copied): {args.plot_dir}")

    if not args.save_plots and plot_cfg.get("save_plots", False):
        args.save_plots = True

    temp_files: List[str] = []
    all_summaries: List[dict] = []
    all_ep_data: Dict[str, Dict[str, Dict[str, dict]]] = {}
    try:
        for ri, run_cfg in enumerate(runs):
            rows, _, ep_map = _run_one_comparison_run(ri, run_cfg, scfg, args, temp_files)
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
    print("COMBINED SUMMARY (all comparison runs) [sim_plant]")
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
    log_path = os.path.join(args.plot_dir, f"compare_fair_runs_sim_plant_log_{ts}.txt")
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        f.write("compare_scenarios_fair_runs_sim_plant log\n")
        f.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"scenarios_config: {args.scenarios_config}\n")
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"controllers: {controllers}\n")
        f.write(f"n_seeds: {n_seeds}\n\n")
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
