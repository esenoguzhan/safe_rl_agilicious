#!/usr/bin/env python3
"""
Curriculum learning v4: start with everything on, then ramp up together.

  1. Narrow random goals, some linear vel (±2), some angular vel (±2), full orientation
  2. Ramp: increase goal/spawn range, linear vel, angular vel together (medium → full)
  3. Domain randomization (motor_tau, then mass)
  4. Fine-tune (tighter log_std, full domain rand)

Advancement: 50 eval episodes, success rate ≥ 45/50 (90%) stage 1, 80% thereafter.
Progress is saved to <output_dir>/curriculum_state.yaml.

Usage:
  python scripts/curriculum_train_v4.py [--config configs/curriculum/curriculum_v4_base.yaml] [--output_dir models/curriculum_v4] [--seed 0]
"""
import argparse
import copy
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config_loader import load_config
from scripts.curriculum_train import (
    _deep_merge,
    load_state,
    save_state,
    train_phase,
)

# Curriculum v4 defaults
EVAL_PATIENCE = 3
LOG_STD_START = -1.0
LOG_STD_END = -2.5
LOG_STD_WARMUP = 0.10
LOG_STD_DECAY = 0.40

# Narrow: stage 1
NARROW_POS_X = [-2.0, 2.0]
NARROW_POS_Y = [-2.0, 2.0]
NARROW_POS_Z = [3.0, 7.0]
NARROW_GOAL_RANGE = {"x": [-2.0, 2.0], "y": [-2.0, 2.0], "z": [3.0, 7.0]}

# Medium: stage 2a ramp
MEDIUM_POS_X = [-5.0, 5.0]
MEDIUM_POS_Y = [-5.0, 5.0]
MEDIUM_POS_Z = [2.0, 10.0]
MEDIUM_GOAL_RANGE = {"x": [-5.0, 5.0], "y": [-5.0, 5.0], "z": [2.0, 10.0]}

# Wide: stage 2b+
WIDE_POS_X = [-15.0, 15.0]
WIDE_POS_Y = [-15.0, 15.0]
WIDE_POS_Z = [1.0, 18.0]
WIDE_GOAL_RANGE = {"x": [-12.0, 12.0], "y": [-12.0, 12.0], "z": [1.0, 18.0]}

MAX_LIN_VEL = 8.0
MAX_ANG_VEL = 5.0


def _build_phase_configs(
    motor_tau: float,
    max_timesteps: int,
    reward_thr: float,
    length_thr: float,
):
    """Return the v4 curriculum phase list: everything on → ramp up → domain rand → finetune."""
    phases = []

    # -------------------------------------------------------------------------
    # 1. Start with everything on: narrow goals, some linear vel, some ang vel, full orientation
    # -------------------------------------------------------------------------
    LIN_VEL_1 = 2.0
    ANG_VEL_1 = 2.0
    phases.append({
        "name": "stage1_narrow_all_on",
        "stage": 1,
        "description": f"Narrow random goals, vel ±{LIN_VEL_1}, ang ±{ANG_VEL_1}, full ori, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": False,
                    "randomize_motor_tau": False,
                    "randomize_goal": True,
                    "goal_pos_range": copy.deepcopy(NARROW_GOAL_RANGE),
                },
                "spawn_ranges": {
                    "pos_x": copy.deepcopy(NARROW_POS_X),
                    "pos_y": copy.deepcopy(NARROW_POS_Y),
                    "pos_z": copy.deepcopy(NARROW_POS_Z),
                    "vel_x": [-LIN_VEL_1, LIN_VEL_1],
                    "vel_y": [-LIN_VEL_1, LIN_VEL_1],
                    "vel_z": [-LIN_VEL_1, LIN_VEL_1],
                    "ang_vel_x": [-ANG_VEL_1, ANG_VEL_1],
                    "ang_vel_y": [-ANG_VEL_1, ANG_VEL_1],
                    "ang_vel_z": [-ANG_VEL_1, ANG_VEL_1],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 2a. Ramp: medium goals/spawn, medium vel/ang
    # -------------------------------------------------------------------------
    LIN_VEL_2A = 4.0
    ANG_VEL_2A = 3.5
    phases.append({
        "name": "stage2_ramp_medium",
        "stage": 2,
        "description": f"Medium goals/spawn, vel ±{LIN_VEL_2A}, ang ±{ANG_VEL_2A}, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": False,
                    "randomize_motor_tau": False,
                    "randomize_goal": True,
                    "goal_pos_range": copy.deepcopy(MEDIUM_GOAL_RANGE),
                },
                "spawn_ranges": {
                    "pos_x": copy.deepcopy(MEDIUM_POS_X),
                    "pos_y": copy.deepcopy(MEDIUM_POS_Y),
                    "pos_z": copy.deepcopy(MEDIUM_POS_Z),
                    "vel_x": [-LIN_VEL_2A, LIN_VEL_2A],
                    "vel_y": [-LIN_VEL_2A, LIN_VEL_2A],
                    "vel_z": [-LIN_VEL_2A, LIN_VEL_2A],
                    "ang_vel_x": [-ANG_VEL_2A, ANG_VEL_2A],
                    "ang_vel_y": [-ANG_VEL_2A, ANG_VEL_2A],
                    "ang_vel_z": [-ANG_VEL_2A, ANG_VEL_2A],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 2b. Ramp: wide goals/spawn, full vel/ang
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage2_ramp_full",
        "stage": 2,
        "description": f"Wide goals/spawn, vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL}, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": False,
                    "randomize_motor_tau": False,
                    "randomize_goal": True,
                    "goal_pos_range": copy.deepcopy(WIDE_GOAL_RANGE),
                },
                "spawn_ranges": {
                    "pos_x": copy.deepcopy(WIDE_POS_X),
                    "pos_y": copy.deepcopy(WIDE_POS_Y),
                    "pos_z": copy.deepcopy(WIDE_POS_Z),
                    "vel_x": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "vel_y": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "vel_z": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "ang_vel_x": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_y": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_z": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 3a. Domain randomization: motor_tau
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage3_domain_motor_tau",
        "stage": 3,
        "description": f"Wide goals + motor_tau randomization, vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": False,
                    "randomize_motor_tau": True,
                    "randomize_goal": True,
                    "motor_tau_range": [0.02, 0.05],
                    "goal_pos_range": copy.deepcopy(WIDE_GOAL_RANGE),
                },
                "spawn_ranges": {
                    "pos_x": copy.deepcopy(WIDE_POS_X),
                    "pos_y": copy.deepcopy(WIDE_POS_Y),
                    "pos_z": copy.deepcopy(WIDE_POS_Z),
                    "vel_x": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "vel_y": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "vel_z": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "ang_vel_x": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_y": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_z": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 3b. Domain randomization: add mass
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage3_domain_mass",
        "stage": 3,
        "description": "Wide goals + full domain rand (motor_tau + mass), vel ±8, ang ±5",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": True,
                    "randomize_motor_tau": True,
                    "randomize_goal": True,
                    "mass_range": [0.65, 0.90],
                    "motor_tau_range": [0.02, 0.05],
                    "goal_pos_range": copy.deepcopy(WIDE_GOAL_RANGE),
                },
                "spawn_ranges": {
                    "pos_x": copy.deepcopy(WIDE_POS_X),
                    "pos_y": copy.deepcopy(WIDE_POS_Y),
                    "pos_z": copy.deepcopy(WIDE_POS_Z),
                    "vel_x": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "vel_y": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "vel_z": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "ang_vel_x": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_y": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_z": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 4. Fine-tune (tighter log_std, full domain rand)
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage4_finetune",
        "stage": 4,
        "description": "Fine-tune: anneal log_std for precision, full domain rand",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "log_std_start": -1.5,
        "log_std_end": -3.0,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": True,
                    "randomize_motor_tau": True,
                    "randomize_goal": True,
                    "mass_range": [0.65, 0.90],
                    "motor_tau_range": [0.02, 0.05],
                    "goal_pos_range": copy.deepcopy(WIDE_GOAL_RANGE),
                },
                "spawn_ranges": {
                    "pos_x": copy.deepcopy(WIDE_POS_X),
                    "pos_y": copy.deepcopy(WIDE_POS_Y),
                    "pos_z": copy.deepcopy(WIDE_POS_Z),
                    "vel_x": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "vel_y": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "vel_z": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "ang_vel_x": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_y": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_z": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # ---- Eval thresholds and phase defaults ----
    EVAL_REWARD_THR = 900
    EVAL_LENGTH_THR = 1000
    _log_std_defaults = {
        1: (LOG_STD_START, LOG_STD_END),
        2: (LOG_STD_START, LOG_STD_END),
        3: (LOG_STD_START, -1.5),
        4: (-1.5, -3.0),
    }
    N_EVAL_EPISODES = 50
    EVAL_SUCCESS_RATE_THR = 45 / 50   # 0.9 for stage 1
    EVAL_SUCCESS_RATE_THR_AFTER_STAGE1 = 0.8  # 80% for stage 2+
    _ent_coef_by_stage = {
        1: 0.01,
        2: 0.01,
        3: 0.005,
        4: 0.0001,
    }

    past_stage1 = False
    for p in phases:
        s = p["stage"]
        default_ls = _log_std_defaults.get(s, (LOG_STD_START, LOG_STD_END))
        ls_start = p.get("log_std_start", default_ls[0])
        ls_end = p.get("log_std_end", default_ls[1])
        p["n_eval_episodes"] = N_EVAL_EPISODES
        p["eval_print_every_time"] = True
        p.setdefault("eval_reward_threshold", EVAL_REWARD_THR)
        p.setdefault("eval_length_threshold", EVAL_LENGTH_THR)
        p["eval_success_rate_threshold"] = (
            EVAL_SUCCESS_RATE_THR_AFTER_STAGE1 if past_stage1 else EVAL_SUCCESS_RATE_THR
        )
        p.setdefault("eval_patience", EVAL_PATIENCE)
        p["log_std_start"] = ls_start
        p["log_std_end"] = ls_end
        p.setdefault("log_std_warmup", LOG_STD_WARMUP)
        p.setdefault("log_std_decay", LOG_STD_DECAY)
        p.setdefault("config_patch", {})
        p["config_patch"].setdefault("ppo", {})["ent_coef"] = _ent_coef_by_stage.get(s, 0.001)
        if s == 1:
            past_stage1 = True

    return phases


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum v4: narrow all-on → ramp (goals/vel/ang) → domain rand → finetune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/curriculum_v4",
        help="Root directory for all phase outputs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--motor_tau",
        type=float,
        default=0.033,
        help="Nominal motor time constant",
    )
    parser.add_argument(
        "--max_phase_steps",
        type=int,
        default=5_000_000,
        help="Max timesteps per phase before forced advance (default: 5M)",
    )
    parser.add_argument(
        "--reward_threshold",
        type=float,
        default=850,
        help="Episode reward threshold (training ref)",
    )
    parser.add_argument(
        "--length_threshold",
        type=float,
        default=1000,
        help="Episode length threshold (training ref)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/curriculum/curriculum_v4_base.yaml",
        help="Path to base config YAML. Relative to repo root.",
    )
    args = parser.parse_args()

    config_path = os.path.join(_REPO_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}")
        print("Curriculum v4 requires a base config YAML, e.g. configs/curriculum/curriculum_v4_base.yaml")
        sys.exit(1)
    base_cfg = load_config(config_path)
    print(f"Loaded curriculum v4 config from {args.config}")

    os.makedirs(args.output_dir, exist_ok=True)

    phases = _build_phase_configs(
        motor_tau=args.motor_tau,
        max_timesteps=args.max_phase_steps,
        reward_thr=args.reward_threshold,
        length_thr=args.length_threshold,
    )

    state = load_state(args.output_dir)
    completed = set(state.get("completed_phases", []))

    print("=" * 60)
    print("CURRICULUM V4 — narrow all-on → ramp (goals/vel/ang) → domain rand → finetune")
    print("=" * 60)
    print(f"Config           : {args.config}")
    print(f"Output directory : {args.output_dir}")
    print(f"Seed             : {args.seed}")
    print(f"Motor tau        : {args.motor_tau}")
    print(f"Max steps/phase  : {args.max_phase_steps:,}")
    print(f"Total phases     : {len(phases)}")
    if completed:
        print(f"Already done     : {len(completed)} phases")
    print("=" * 60)

    prev_dir = None
    for p in phases:
        if p["name"] in completed:
            candidate = os.path.join(args.output_dir, p["name"])
            if os.path.isdir(candidate):
                prev_dir = candidate

    for i, phase in enumerate(phases):
        if phase["name"] in completed:
            print(f"\n[{i+1}/{len(phases)}] {phase['name']} — SKIPPED (already completed)")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{len(phases)}] {phase['name']}")
        print(f"  {phase['description']}")
        print(f"{'=' * 60}")

        cfg = _deep_merge(base_cfg, phase["config_patch"])
        phase_dir = train_phase(
            phase, cfg, args.output_dir, args.seed, prev_model_dir=prev_dir
        )

        completed.add(phase["name"])
        state["completed_phases"] = list(completed)
        state["current_phase"] = phase["name"]
        save_state(args.output_dir, state)

        prev_dir = phase_dir

    print(f"\n{'=' * 60}")
    print("CURRICULUM V4 COMPLETE")
    print(f"Final model: {prev_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
