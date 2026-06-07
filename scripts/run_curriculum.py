#!/usr/bin/env python3
"""
Run a YAML-defined curriculum: train scripts/train.py (or
train_core.py) in N sequential stages, each continuing from the previous run.

Stages are defined in a curriculum YAML (default: configs/curriculum.yaml), not
hardcoded.  Each stage applies a set of parameter *overrides* on top of the base
config; overrides are cumulative, so stage k's config is the base config merged
with the overrides of stages 1..k (deep merge).  Stage 1 typically has no
overrides (= base config as-is).  You can add arbitrary stages just by editing
the YAML — no code changes needed.

Curriculum YAML schema:

    run_name_prefix: curric                      # optional; used in run/model folder names
    base_config: configs/ppo_config_base.yaml    # optional; --base-config overrides it
    stages:
      - name: base                               # required; sanitized into the run name
        description: "Base config as-is"          # optional; printed in the stage summary
        # (no `overrides:` key  ->  no change)
      - name: extended_position
        description: "Widen position spawn ranges"
        overrides:                               # deep-merged onto the cumulative config
          env:
            spawn_ranges:
              pos_x: [-8.0, 8.0]
              pos_y: [-8.0, 8.0]
              pos_z: [0.5, 8.0]
      - ...

Each stage after the first loads weights and VecNormalize via --resume from the
previous stage's run directory.  Child process output is not buffered: logs and
the progress bar print live.

Default decimation: ``--physics-hz 1000 --rl-hz 100`` (override as needed). To use
the plain single-rate trainer instead, pass ``--train-script scripts/train_core.py``
(physics flags are then omitted).

Usage (from repo root):

  python3 scripts/run_curriculum.py
  python3 scripts/run_curriculum.py --dry-run
  python3 scripts/run_curriculum.py \\
      --curriculum configs/curriculum.yaml --physics-hz 500 --rl-hz 50

Resume after a failure (e.g. restart at stage 3, using stage 2's output folder):

  python3 scripts/run_curriculum.py --start-from 3 \\
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
_DEFAULT_BASE = os.path.join(_REPO_ROOT, "configs", "ppo_config_base.yaml")
_DEFAULT_CURRICULUM = os.path.join(_REPO_ROOT, "configs", "curriculum.yaml")
_TRAIN_MAIN = os.path.join(_REPO_ROOT, "scripts", "train.py")
_TRAIN_CORE = os.path.join(_REPO_ROOT, "scripts", "train_core.py")
_MODELS_DIR = "models"
_DEFAULT_RUN_NAME_PREFIX = "curric"


def _sanitize_run_name_suffix(name: str) -> str:
    """Same rules as scripts/train_core.py (safe folder suffix)."""
    if not name:
        return ""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name))


def _get_next_ppo_steps_dir(total_timesteps: int, name_suffix: str | None = None) -> str:
    """Match train_core.get_next_ppo_steps_dir — first models/PPO_<steps>[_suffix] that does not exist."""
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


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_curriculum(path: str) -> tuple[str, str | None, list[dict]]:
    """Parse a curriculum YAML.

    Returns ``(run_name_prefix, base_config_or_None, stages)`` where each stage is
    a dict with keys ``name`` (str), ``description`` (str) and ``overrides`` (dict).
    """
    with open(path, "r") as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(f"Curriculum file must be a mapping: {path}")
    raw_stages = doc.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError(f"Curriculum '{path}' must define a non-empty 'stages' list.")
    prefix = str(doc.get("run_name_prefix") or _DEFAULT_RUN_NAME_PREFIX)
    base_config = doc.get("base_config")
    stages: list[dict] = []
    for i, st in enumerate(raw_stages, start=1):
        if not isinstance(st, dict):
            raise ValueError(f"Stage {i} in '{path}' must be a mapping.")
        name = st.get("name")
        if not name:
            raise ValueError(f"Stage {i} in '{path}' is missing a 'name'.")
        overrides = st.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"Stage {i} ('{name}') 'overrides' must be a mapping.")
        stages.append(
            {
                "name": str(name),
                "description": str(st.get("description", "")),
                "overrides": overrides,
            }
        )
    return prefix, base_config, stages


def _config_for_stage(base_cfg: dict, stages: list[dict], stage: int) -> dict:
    """stage is 1..len(stages); returns the base config merged with overrides of stages 1..stage."""
    if stage < 1 or stage > len(stages):
        raise ValueError(f"stage must be in 1..{len(stages)}")
    cfg = copy.deepcopy(base_cfg)
    for i in range(stage):
        cfg = _deep_merge(cfg, stages[i]["overrides"])
    return cfg


def _run_name_physics_tag(physics_hz: float, rl_hz: float) -> str:
    """Short suffix for run names when using decimated trainer (folder uniqueness)."""
    p = int(round(physics_hz))
    r = int(round(rl_hz))
    return f"ph{p}_rl{r}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YAML-driven sequential curriculum: train.py (decimated physics) by default "
        "(or train_core.py via --train-script), resuming each stage from the previous one. "
        "Stages are read from a curriculum YAML."
    )
    parser.add_argument(
        "--curriculum",
        default=_DEFAULT_CURRICULUM,
        help=f"Curriculum YAML defining the stages (default: {_DEFAULT_CURRICULUM}).",
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Base YAML applied before any stage overrides. Overrides the curriculum's "
        f"'base_config' key; falls back to {_DEFAULT_BASE}.",
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
        default=_TRAIN_MAIN,
        help=f"Training script to spawn (default: {_TRAIN_MAIN}). "
        f"Use {_TRAIN_CORE} for non-decimated single-rate training.",
    )
    parser.add_argument(
        "--physics-hz",
        type=float,
        default=1000.0,
        help="Only when using train.py (decimated): simulator rate (Hz).",
    )
    parser.add_argument(
        "--rl-hz",
        type=float,
        default=100.0,
        help="Only when using train.py (decimated): policy step rate (Hz).",
    )
    parser.add_argument(
        "--reward-aggregate",
        type=str,
        choices=("sum", "mean"),
        default="sum",
        help="Only when using train.py (decimated): inner reward aggregation.",
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
    use_decimated = os.path.basename(train_script) == "train.py"
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

    curriculum_path = os.path.abspath(args.curriculum)
    if not os.path.isfile(curriculum_path):
        print(f"Missing curriculum file: {curriculum_path}", file=sys.stderr)
        return 1
    try:
        run_name_prefix, curr_base_config, stages = _load_curriculum(curriculum_path)
    except (ValueError, yaml.YAMLError) as exc:
        print(f"Invalid curriculum '{curriculum_path}': {exc}", file=sys.stderr)
        return 2
    num_stages = len(stages)

    if args.start_from < 1 or args.start_from > num_stages:
        print(f"--start-from must be between 1 and {num_stages}", file=sys.stderr)
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

    # Base config precedence: CLI --base-config > curriculum 'base_config' > default.
    base_choice = args.base_config or curr_base_config or _DEFAULT_BASE
    base_path = base_choice if os.path.isabs(base_choice) else os.path.join(_REPO_ROOT, base_choice)
    base_path = os.path.abspath(base_path)
    if not os.path.isfile(base_path):
        print(f"Missing base config: {base_path}", file=sys.stderr)
        return 1

    with open(base_path, "r") as f:
        base_cfg = yaml.safe_load(f)

    os.chdir(_REPO_ROOT)

    print(f"Curriculum: {curriculum_path}  ({num_stages} stages)", flush=True)
    print(f"Base config: {base_path}", flush=True)
    print("Stages:", flush=True)
    for idx, st in enumerate(stages, start=1):
        desc = f" — {st['description']}" if st["description"] else ""
        changes = "base config as-is" if not st["overrides"] else "overrides: " + ", ".join(sorted(st["overrides"].keys()))
        print(f"  {idx} — {st['name']}{desc}  ({changes})", flush=True)
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

    for stage in range(1, num_stages + 1):
        if stage < args.start_from:
            continue

        cfg = _config_for_stage(base_cfg, stages, stage)
        stage_name = _sanitize_run_name_suffix(stages[stage - 1]["name"])
        base_run_name = f"{run_name_prefix}_s{stage}_{stage_name}"
        run_name = f"{base_run_name}_{physics_tag}" if physics_tag else base_run_name

        fd, tmp_path = tempfile.mkstemp(
            suffix=".yaml",
            prefix=f"train_curriculum_stage{stage}_",
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

        print(f"\n=== Stage {stage}/{num_stages}: {run_name} ===", flush=True)
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

    print(f"\nAll {num_stages} stages finished.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
