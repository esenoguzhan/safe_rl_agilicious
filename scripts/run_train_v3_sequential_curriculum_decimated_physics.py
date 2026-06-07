#!/usr/bin/env python3
"""
Run scripts/train_v3_decimated_physics.py in seven sequential stages, each continuing
from the previous run (same curriculum as run_train_v3_sequential_curriculum.py).

  Stage 1 — Base YAML as-is (default: configs/single_stage_no_curriculum_cauchy.yaml).
  Stage 2 — Widen position spawn ranges (x,y ±8 m; z 0.5–8 m).
  Stage 3 — Widen initial linear/angular velocity spawn ranges (±8 m/s, ±10 rad/s).
  Stage 4 — Enable command delay domain randomization.
  Stage 5 — Enable mass domain randomization.
  Stage 6 — Enable motor time-constant (tau) domain randomization.
  Stage 7 — Enable wind/drag/force disturbances (env.disturbances.enable: true).

Later stages are cumulative: each stage's config is the base config with all prior
patches applied. Each stage after the first loads weights and VecNormalize via
--resume from the previous stage's run directory.

Child process output is not buffered: logs and progress bar print live.

Default decimation: ``--physics-hz 1000 --rl-hz 100`` (override as needed). To use
plain ``train_v3.py`` instead, pass ``--train-script scripts/train_v3.py`` (physics
flags are then omitted).

Usage (from repo root):

  python3 scripts/run_train_v3_sequential_curriculum_decimated_physics.py
  python3 scripts/run_train_v3_sequential_curriculum_decimated_physics.py --dry-run
  python3 scripts/run_train_v3_sequential_curriculum_decimated_physics.py --physics-hz 500 --rl-hz 50

Resume after a failure (e.g. restart at stage 3, using stage 2's output folder):

  python3 scripts/run_train_v3_sequential_curriculum_decimated_physics.py --start-from 3 \\
      --resume-from-run-dir models/PPO_32000000_seq7dec_s2_spawn_pos_ph1000_rl100
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import tempfile

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_BASE = os.path.join(_REPO_ROOT, "configs", "single_stage_no_curriculum_cauchy.yaml")
_TRAIN_V3_DECIMATED = os.path.join(_REPO_ROOT, "scripts", "train_v3_decimated_physics.py")
_TRAIN_V3_PLAIN = os.path.join(_REPO_ROOT, "scripts", "train_v3.py")
_MODELS_DIR = "models"
_NUM_STAGES = 7


def _sanitize_run_name_suffix(name: str) -> str:
    """Same rules as scripts/train_v3.py (safe folder suffix)."""
    if not name:
        return ""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name))


def _get_next_ppo_steps_dir(total_timesteps: int, name_suffix: str | None = None) -> str:
    """Match train_v3.get_next_ppo_steps_dir — first models/PPO_<steps>[_suffix] that does not exist."""
    if name_suffix:
        suf = _sanitize_run_name_suffix(name_suffix)
        base = (
            os.path.join(_MODELS_DIR, f"PPO_{total_timesteps}_{suf}")
            if suf
            else os.path.join(_MODELS_DIR, f"PPO_{total_timesteps}")
        )
    else:
        base = os.path.join(_MODELS_DIR, f"PPO_{total_timesteps}")
    run_dir = base
    i = 1
    while os.path.isdir(run_dir):
        i += 1
        run_dir = f"{base}_{i}"
    return run_dir


# Cumulative curriculum: patches applied in order after stage 1 (same as seq6 non-decimated runner).
_STAGE_RUN_NAMES_DECIMATED = (
    "seq7dec_s1_base",
    "seq7dec_s2_spawn_pos",
    "seq7dec_s3_spawn_vel",
    "seq7dec_s4_cmd_delay_dr",
    "seq7dec_s5_mass_dr",
    "seq7dec_s6_tau_dr",
    "seq7dec_s7_disturbances",
)
# Same stages when using plain train_v3.py (--train-script …/train_v3.py); distinct from seq6_s_*.
_STAGE_RUN_NAMES_PLAIN = (
    "seq7pl_s1_base",
    "seq7pl_s2_spawn_pos",
    "seq7pl_s3_spawn_vel",
    "seq7pl_s4_cmd_delay_dr",
    "seq7pl_s5_mass_dr",
    "seq7pl_s6_tau_dr",
    "seq7pl_s7_disturbances",
)

_SPAWN_POS_PATCH = {
    "env": {
        "spawn_ranges": {
            "pos_x": [-8.0, 8.0],
            "pos_y": [-8.0, 8.0],
            "pos_z": [0.5, 8.0],
        }
    }
}

_SPAWN_VEL_ANG_PATCH = {
    "env": {
        "spawn_ranges": {
            "vel_x": [-8.0, 8.0],
            "vel_y": [-8.0, 8.0],
            "vel_z": [-8.0, 8.0],
            "ang_vel_x": [-10.0, 10.0],
            "ang_vel_y": [-10.0, 10.0],
            "ang_vel_z": [-10.0, 10.0],
        }
    }
}

_DISTURBANCES_PATCH = {"env": {"disturbances": {"enable": True}}}

_COMMAND_DELAY_DR_PATCH = {
    "env": {"domain_randomization": {"randomize_command_delay": True}}
}

_MASS_DR_PATCH = {"env": {"domain_randomization": {"randomize_mass": True}}}

_TAU_DR_PATCH = {"env": {"domain_randomization": {"randomize_motor_tau": True}}}

_PATCHES_IN_ORDER = (
    _SPAWN_POS_PATCH,
    _SPAWN_VEL_ANG_PATCH,
    _COMMAND_DELAY_DR_PATCH,
    _MASS_DR_PATCH,
    _TAU_DR_PATCH,
    _DISTURBANCES_PATCH,
)


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _config_for_stage(base_cfg: dict, stage: int) -> dict:
    """stage is 1..NUM_STAGES; returns full merged config."""
    if stage < 1 or stage > _NUM_STAGES:
        raise ValueError(f"stage must be in 1..{_NUM_STAGES}")
    cfg = copy.deepcopy(base_cfg)
    for i in range(stage - 1):
        cfg = _deep_merge(cfg, _PATCHES_IN_ORDER[i])
    return cfg


def _run_name_physics_tag(physics_hz: float, rl_hz: float) -> str:
    """Short suffix for run names when using decimated trainer (folder uniqueness)."""
    p = int(round(physics_hz))
    r = int(round(rl_hz))
    return f"ph{p}_rl{r}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seven-stage sequential curriculum: train_v3_decimated_physics by default "
        "(or train_v3 via --train-script), resume chain — same patches as "
        "run_train_v3_sequential_curriculum.py."
    )
    parser.add_argument(
        "--base-config",
        default=_DEFAULT_BASE,
        help=f"Base YAML (default: {_DEFAULT_BASE}).",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        metavar="N",
        help="1-based stage index to start from (default: 1).",
    )
    parser.add_argument(
        "--resume-from-run-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="When --start-from > 1: run directory of the completed previous stage "
        "(contains best_model.zip or ppo_drone_final.zip). Required in that case.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional: passed to the training script as --seed for every stage.",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default=_TRAIN_V3_DECIMATED,
        help=f"Training script to spawn (default: {_TRAIN_V3_DECIMATED}). "
        f"Use {_TRAIN_V3_PLAIN} for non-decimated train_v3.",
    )
    parser.add_argument(
        "--physics-hz",
        type=float,
        default=1000.0,
        help="Only when using train_v3_decimated_physics.py: simulator rate (Hz).",
    )
    parser.add_argument(
        "--rl-hz",
        type=float,
        default=100.0,
        help="Only when using train_v3_decimated_physics.py: policy step rate (Hz).",
    )
    parser.add_argument(
        "--reward-aggregate",
        type=str,
        choices=("sum", "mean"),
        default="sum",
        help="Only when using train_v3_decimated_physics.py: inner reward aggregation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stages and commands only; do not train.",
    )
    args = parser.parse_args()

    train_script = os.path.abspath(args.train_script)
    if not os.path.isfile(train_script):
        print(f"Training script not found: {train_script}", file=sys.stderr)
        return 1
    use_decimated = os.path.basename(train_script) == "train_v3_decimated_physics.py"
    if use_decimated:
        ph, rlh = float(args.physics_hz), float(args.rl_hz)
        if ph <= 0.0 or rlh <= 0.0:
            print("--physics-hz and --rl-hz must be positive", file=sys.stderr)
            return 2
        ratio = ph / rlh
        if abs(ratio - round(ratio)) > 1e-6:
            print(
                f"physics_hz / rl_hz must be an integer; got {ph}/{rlh} = {ratio}",
                file=sys.stderr,
            )
            return 2
    physics_tag = _run_name_physics_tag(args.physics_hz, args.rl_hz) if use_decimated else ""

    if args.start_from < 1 or args.start_from > _NUM_STAGES:
        print(f"--start-from must be between 1 and {_NUM_STAGES}", file=sys.stderr)
        return 2

    if args.start_from > 1:
        if not args.resume_from_run_dir:
            print(
                "When --start-from > 1, you must pass --resume-from-run-dir "
                "pointing at the finished previous stage's run directory.",
                file=sys.stderr,
            )
            return 2
        prev = os.path.abspath(args.resume_from_run_dir)
        if not args.dry_run and not os.path.isdir(prev):
            print(f"Not a directory: {prev}", file=sys.stderr)
            return 2

    base_path = os.path.abspath(args.base_config)
    if not os.path.isfile(base_path):
        print(f"Missing base config: {base_path}", file=sys.stderr)
        return 1

    with open(base_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    os.chdir(_REPO_ROOT)

    print(
        "Stages:\n"
        "  1 — base config\n"
        "  2 — spawn position x,y ±8 m; z 0.5–8 m\n"
        "  3 — spawn vel/ang_vel ±8 / ±10\n"
        "  4 — command delay DR on\n"
        "  5 — mass DR on\n"
        "  6 — motor tau DR on\n"
        "  7 — disturbances on\n",
        flush=True,
    )
    print(f"Training script: {train_script}", flush=True)
    if use_decimated:
        print(
            f"Decimated physics: physics_hz={args.physics_hz} rl_hz={args.rl_hz} "
            f"reward_aggregate={args.reward_aggregate}",
            flush=True,
        )

    last_run_dir: str | None = None
    if args.start_from > 1:
        last_run_dir = os.path.abspath(args.resume_from_run_dir)

    _dry_resume_hint = (
        "<run directory printed after the immediately previous stage completes>"
    )

    for stage in range(1, _NUM_STAGES + 1):
        if stage < args.start_from:
            continue

        cfg = _config_for_stage(base_cfg, stage)
        names = _STAGE_RUN_NAMES_DECIMATED if use_decimated else _STAGE_RUN_NAMES_PLAIN
        base_run_name = names[stage - 1]
        run_name = f"{base_run_name}_{physics_tag}" if physics_tag else base_run_name

        fd, tmp_path = tempfile.mkstemp(
            suffix=".yaml",
            prefix=f"train_v3_decimated_seq{_NUM_STAGES}_stage{stage}_",
            text=True,
        )
        try:
            with os.fdopen(fd, "w") as tf:
                yaml.dump(cfg, tf, default_flow_style=False, sort_keys=False)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        resume_for_cmd: str | None = last_run_dir
        if args.dry_run and stage > 1 and last_run_dir is None:
            resume_for_cmd = _dry_resume_hint

        cmd = [
            sys.executable,
            train_script,
            "--config",
            tmp_path,
            "--run-name",
            run_name,
        ]
        if args.seed is not None:
            cmd.extend(["--seed", str(args.seed)])
        if resume_for_cmd is not None:
            cmd.extend(["--resume", resume_for_cmd])
        if use_decimated:
            cmd.extend(
                [
                    "--physics-hz",
                    str(args.physics_hz),
                    "--rl-hz",
                    str(args.rl_hz),
                    "--reward-aggregate",
                    str(args.reward_aggregate),
                ]
            )

        print(f"\n=== Stage {stage}/{_NUM_STAGES}: {run_name} ===", flush=True)
        if resume_for_cmd:
            print(f"  --resume {resume_for_cmd}", flush=True)
        print(" ".join(cmd), flush=True)

        if args.dry_run:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            ts = cfg.get("training", {}).get("total_timesteps", 100_000)
            pred = _get_next_ppo_steps_dir(int(ts), run_name)
            print(f"  (dry-run) would use run dir: {os.path.abspath(pred)}", flush=True)
            if stage == 1:
                last_run_dir = None
            else:
                last_run_dir = _dry_resume_hint
            continue

        total_timesteps = int(cfg.get("training", {}).get("total_timesteps", 100_000))
        predicted_run_dir = _get_next_ppo_steps_dir(total_timesteps, run_name)
        predicted_run_dir = os.path.abspath(predicted_run_dir)
        print(
            f"Streaming training logs below (run dir for next --resume: {predicted_run_dir})\n",
            flush=True,
        )

        try:
            proc = subprocess.run(
                cmd,
                cwd=_REPO_ROOT,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if proc.returncode != 0:
            print(
                f"Stage {stage} failed with exit code {proc.returncode}. "
                f"After fixing the issue, re-run with e.g.\n"
                f"  --start-from {stage} --resume-from-run-dir <previous_stage_run_dir>",
                file=sys.stderr,
            )
            return proc.returncode

        if not os.path.isdir(predicted_run_dir):
            print(
                f"Expected run directory missing after training: {predicted_run_dir}. "
                f"Use --resume-from-run-dir with the folder the training script actually created.",
                file=sys.stderr,
            )
            return 1
        last_run_dir = predicted_run_dir
        print(f"Next stage will --resume from: {last_run_dir}", flush=True)

    print(f"\nAll {_NUM_STAGES} stages finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
