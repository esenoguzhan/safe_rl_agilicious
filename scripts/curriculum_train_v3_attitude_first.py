#!/usr/bin/env python3
"""
Curriculum learning v3: attitude-first then linear control.

  1. Fixed goal, fully random attitude (ori_scale=1.0), zero linear/angular velocity
  2. Fixed goal, introduce angular velocities with intermediate steps (ang ±0.5 → ±5)
  3. Random goals (narrow) with full angular velocity
  4. Wider random goals (wide spawn/goals), full angular velocity
  5. Include linear velocity with intermediate steps (vel ±2 → ±8)
  6. Domain randomization (motor_tau, then mass)
  7. Fine-tune (tighter log_std, full domain rand)

Advancement: 50 eval episodes, success rate ≥ 45/50 (90%); per-episode success = reward ≥ 900 and length ≥ 1000.
After stage 2 angular ramp, success threshold drops to 80% (40/50).
Progress is saved to <output_dir>/curriculum_state.yaml.

Usage:
  python scripts/curriculum_train_v3_attitude_first.py [--config configs/curriculum/curriculum_v3_base.yaml] [--output_dir models/curriculum_v3_attitude_first] [--seed 0]
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

# Curriculum v3 defaults (aligned with curriculum_v3_base.yaml / train pipeline)
EVAL_PATIENCE = 3
LOG_STD_START = -1.0
LOG_STD_END = -2.5
LOG_STD_WARMUP = 0.10
LOG_STD_DECAY = 0.40

# Narrow: for stages 3
NARROW_POS_X = [-2.0, 2.0]
NARROW_POS_Y = [-2.0, 2.0]
NARROW_POS_Z = [3.0, 7.0]
NARROW_GOAL_RANGE = {"x": [-2.0, 2.0], "y": [-2.0, 2.0], "z": [3.0, 7.0]}

# Wide: for stages 4+
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
    """Return the attitude-first curriculum phase list."""
    phases = []

    # -------------------------------------------------------------------------
    # 1. Fixed goal, fully random attitude, zero velocity
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage1_fixed_goal_random_attitude",
        "stage": 1,
        "description": f"Fixed goal [0,0,5], full random attitude (ori_scale=1), zero vel, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": False,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {"enabled": False},
                "goal_position": [0.0, 0.0, 5.0],
                "spawn_ranges": {
                    "pos_x": [-1.0, 1.0],
                    "pos_y": [-1.0, 1.0],
                    "pos_z": [4.0, 6.0],
                    "vel_x": [0.0, 0.0],
                    "vel_y": [0.0, 0.0],
                    "vel_z": [0.0, 0.0],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 2. Fixed goal, introduce angular velocities (fewer intermediate steps)
    # -------------------------------------------------------------------------
    _ang_steps = [1.0, 2.5, 5.0]
    for a in _ang_steps:
        phases.append({
            "name": f"stage2_ang_{int(a*10):02d}",
            "stage": 2,
            "description": f"Fixed goal, ori_scale=1, ang_vel ±{a}, zero linear vel, motor_tau={motor_tau}",
            "max_timesteps": max_timesteps,
            "reward_threshold": reward_thr,
            "length_threshold": length_thr,
            "eval_domain_rand": False,
            "config_patch": {
                "env": {
                    "quadrotor_dynamics": {"motor_tau": motor_tau},
                    "domain_randomization": {"enabled": False},
                    "goal_position": [0.0, 0.0, 5.0],
                    "spawn_ranges": {
                        "pos_x": [-1.0, 1.0],
                        "pos_y": [-1.0, 1.0],
                        "pos_z": [4.0, 6.0],
                        "vel_x": [0.0, 0.0],
                        "vel_y": [0.0, 0.0],
                        "vel_z": [0.0, 0.0],
                        "ang_vel_x": [-a, a],
                        "ang_vel_y": [-a, a],
                        "ang_vel_z": [-a, a],
                        "ori_scale": 1.0,
                    },
                },
            },
        })

    # -------------------------------------------------------------------------
    # 3. Random goals (narrow) with full angular velocity
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage3_narrow_goals_ang",
        "stage": 3,
        "description": f"Random goals (narrow), ori_scale=1, ang_vel ±{MAX_ANG_VEL}, zero linear vel, motor_tau={motor_tau}",
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
                    "vel_x": [0.0, 0.0],
                    "vel_y": [0.0, 0.0],
                    "vel_z": [0.0, 0.0],
                    "ang_vel_x": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_y": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_z": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 4. Wider random goals (full angular, zero linear vel)
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage4_wider_goals",
        "stage": 4,
        "description": f"Wide random goals, ori_scale=1, ang_vel ±{MAX_ANG_VEL}, zero linear vel, motor_tau={motor_tau}",
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
                    "vel_x": [0.0, 0.0],
                    "vel_y": [0.0, 0.0],
                    "vel_z": [0.0, 0.0],
                    "ang_vel_x": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_y": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_z": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 5. Include linear velocity (intermediate steps: ±2, ±4, ±6, ±8)
    # -------------------------------------------------------------------------
    _vel_steps = [2.0, 4.0, 6.0, 8.0]
    for v in _vel_steps:
        phases.append({
            "name": f"stage5_vel_{int(v):02d}",
            "stage": 5,
            "description": f"Wide goals, ori_scale=1, ang ±{MAX_ANG_VEL}, linear vel ±{v}, motor_tau={motor_tau}",
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
                        "vel_x": [-v, v],
                        "vel_y": [-v, v],
                        "vel_z": [-v, v],
                        "ang_vel_x": [-MAX_ANG_VEL, MAX_ANG_VEL],
                        "ang_vel_y": [-MAX_ANG_VEL, MAX_ANG_VEL],
                        "ang_vel_z": [-MAX_ANG_VEL, MAX_ANG_VEL],
                        "ori_scale": 1.0,
                    },
                },
            },
        })

    # -------------------------------------------------------------------------
    # 6a. Domain randomization: motor_tau
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage6_motor_tau_rand",
        "stage": 6,
        "description": f"Wide goals + motor_tau randomization, vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL}, motor_tau nominal={motor_tau}",
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
    # 6b. Domain randomization: add mass
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage6_mass_rand",
        "stage": 6,
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
    # 7. Fine-tune (tighter log_std, full domain rand)
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage7_finetune",
        "stage": 7,
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
        3: (LOG_STD_START, -2.0),
        4: (LOG_STD_START, -2.0),
        5: (LOG_STD_START, -1.8),
        6: (LOG_STD_START, -1.5),
        7: (-1.5, -3.0),
    }
    N_EVAL_EPISODES = 50
    EVAL_SUCCESS_RATE_THR = 45 / 50   # 0.9
    EVAL_SUCCESS_RATE_THR_AFTER_ANG = 0.8  # after stage2 angular ramp
    _ent_coef_by_stage = {
        1: 0.01, 2: 0.01, 3: 0.01,
        4: 0.01, 5: 0.01,   # higher entropy for wider-goals section
        6: 0.005, 7: 0.0001,
    }

    past_ang_ramp = False
    for p in phases:
        s = p["stage"]
        ls_start, ls_end = _log_std_defaults.get(s, (LOG_STD_START, LOG_STD_END))
        p["n_eval_episodes"] = N_EVAL_EPISODES
        p["eval_print_every_time"] = True  # print eval stats every eval (v3)
        p.setdefault("eval_reward_threshold", EVAL_REWARD_THR)
        p.setdefault("eval_length_threshold", EVAL_LENGTH_THR)
        p["eval_success_rate_threshold"] = (
            EVAL_SUCCESS_RATE_THR_AFTER_ANG if past_ang_ramp else EVAL_SUCCESS_RATE_THR
        )
        p.setdefault("eval_patience", EVAL_PATIENCE)
        p.setdefault("log_std_start", ls_start)
        p.setdefault("log_std_end", ls_end)
        p.setdefault("log_std_warmup", LOG_STD_WARMUP)
        p.setdefault("log_std_decay", LOG_STD_DECAY)
        p.setdefault("config_patch", {})
        p["config_patch"].setdefault("ppo", {})["ent_coef"] = _ent_coef_by_stage.get(s, 0.001)
        if s == 2 and p["name"] == "stage2_ang_50":
            past_ang_ramp = True

    return phases


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum v3 (attitude-first): fixed goal + random attitude → ang vel → narrow goals → wide goals → linear vel → domain rand → finetune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/curriculum_v3_attitude_first",
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
        default="configs/curriculum/curriculum_v3_base.yaml",
        help="Path to base config YAML (env, ppo, training). Relative to repo root.",
    )
    args = parser.parse_args()

    # Load curriculum v3 base config from YAML (required)
    config_path = os.path.join(_REPO_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}")
        print("Curriculum v3 requires a base config YAML, e.g. configs/curriculum/curriculum_v3_base.yaml")
        sys.exit(1)
    base_cfg = load_config(config_path)
    print(f"Loaded curriculum v3 config from {args.config}")

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
    print("CURRICULUM V3 (ATTITUDE-FIRST) — fixed goal + random attitude → ang vel → goals → linear vel → domain rand → finetune")
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
    print("CURRICULUM V3 (ATTITUDE-FIRST) COMPLETE")
    print(f"Final model: {prev_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
