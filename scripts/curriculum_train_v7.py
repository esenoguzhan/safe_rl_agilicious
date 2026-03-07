#!/usr/bin/env python3
"""
Curriculum v7: Academic best-practice drone control.

8 stages, rebalanced cauchy reward, crash penalty, bridging navigation stages.

  Stage 1 — Hover Stabilization:   fixed goal, mild tilt, no velocity, hover motors
  Stage 2 — Attitude Recovery:     full random attitude, mild velocity, motors-off start
  Stage 3 — Velocity Rejection:    full velocity/angular velocity range
  Stage 4 — Fly to Goal:           fixed goal, wide spawn ±8m (learn to fly)
  Stage 5 — Narrow Navigation:     random goals ±5m, spawn ±8m (goal-directed flight)
  Stage 6 — Wide Navigation:       random goals ±10m, spawn ±12m
  Stage 7 — Domain Randomization:  mass + motor_tau rand, goals ±15m
  Stage 8 — Fine-tune:             reduced exploration, lower lr

Advancement: eval-based success rate (90/85/80/75/70%% depending on stage).

Seed strategy:
  Training   — varies per run (--seed 0, 1, 2, …)
  Validation — fixed 7777 for comparable eval during training
  Test/Final — multiple fixed seeds (mean ± std after curriculum)

Usage:
  python scripts/curriculum_train_v7.py \\
      --config configs/curriculum/curriculum_v7_base.yaml \\
      --output_dir models/curriculum_v7 \\
      [--seed 0] [--no-final-eval]
"""
import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config_loader import load_config
from scripts.curriculum_train import (
    _deep_merge,
    FINAL_EVAL_SEEDS_DEFAULT,
    load_state,
    run_final_eval,
    save_state,
    train_phase,
)

EVAL_PATIENCE = 3
LOG_STD_START = -0.5
LOG_STD_END = -2.5
LOG_STD_WARMUP = 0.15
LOG_STD_DECAY = 0.50

MAX_ANG_VEL = 5.0
MAX_LIN_VEL = 8.0
OBS_NOISE_POS = 0.01
OBS_NOISE_VEL = 0.05

N_EVAL_EPISODES = 100
EVAL_REWARD_THR = 600
EVAL_LENGTH_THR = 1000


def _build_phase_configs(motor_tau, max_timesteps, reward_thr, length_thr):
    """Build the 6-stage v7 curriculum."""
    phases = []

    # ------------------------------------------------------------------
    # Stage 1 — Hover Stabilization
    # Fixed goal, mild tilt (ori_scale=0.3), no velocity, hover motor init.
    # ------------------------------------------------------------------
    phases.append({
        "name": "stage1_hover",
        "stage": 1,
        "description": (
            "Hover stabilization: fixed goal [0,0,5], ori_scale=0.3, "
            "no velocity, motor_init=hover"
        ),
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": False,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "motor_init": "hover",
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
                    "ori_scale": 0.3,
                },
            },
        },
    })

    # ------------------------------------------------------------------
    # Stage 2 — Attitude Recovery
    # Full random attitude, mild velocity/angular velocity, motors-off.
    # ------------------------------------------------------------------
    phases.append({
        "name": "stage2_recovery",
        "stage": 2,
        "description": (
            "Attitude recovery: full random ori, vel ±2, ang_vel ±3, "
            "motor_init=zero"
        ),
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": False,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "motor_init": "zero",
                "domain_randomization": {"enabled": False},
                "goal_position": [0.0, 0.0, 5.0],
                "spawn_ranges": {
                    "pos_x": [-2.0, 2.0],
                    "pos_y": [-2.0, 2.0],
                    "pos_z": [3.0, 7.0],
                    "vel_x": [-2.0, 2.0],
                    "vel_y": [-2.0, 2.0],
                    "vel_z": [-2.0, 2.0],
                    "ang_vel_x": [-3.0, 3.0],
                    "ang_vel_y": [-3.0, 3.0],
                    "ang_vel_z": [-3.0, 3.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # ------------------------------------------------------------------
    # Stage 3 — Velocity Rejection
    # Full velocity and angular velocity range, wider spawn.
    # ------------------------------------------------------------------
    phases.append({
        "name": "stage3_velocity",
        "stage": 3,
        "description": (
            f"Velocity rejection: vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL}, "
            "spawn ±4m, motor_init=zero"
        ),
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": False,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "motor_init": "zero",
                "domain_randomization": {"enabled": False},
                "goal_position": [0.0, 0.0, 5.0],
                "spawn_ranges": {
                    "pos_x": [-4.0, 4.0],
                    "pos_y": [-4.0, 4.0],
                    "pos_z": [2.0, 8.0],
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

    # ------------------------------------------------------------------
    # Stage 4 — Fly to Goal (BRIDGING)
    # Fixed goal but wide spawn — teaches the drone to actively fly.
    # ------------------------------------------------------------------
    phases.append({
        "name": "stage4_fly_to_goal",
        "stage": 4,
        "description": (
            "Fly to goal: fixed goal [0,0,5], spawn ±8m, "
            f"vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL}"
        ),
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": False,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "motor_init": "zero",
                "domain_randomization": {"enabled": False},
                "goal_position": [0.0, 0.0, 5.0],
                "spawn_ranges": {
                    "pos_x": [-8.0, 8.0],
                    "pos_y": [-8.0, 8.0],
                    "pos_z": [1.0, 10.0],
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

    # ------------------------------------------------------------------
    # Stage 5 — Narrow Navigation
    # Random goals ±5m, moderate spawn — learns goal-directed flight.
    # ------------------------------------------------------------------
    phases.append({
        "name": "stage5_narrow_nav",
        "stage": 5,
        "description": (
            "Narrow nav: random goals ±5m, spawn ±8m, "
            f"vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL}"
        ),
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "motor_init": "zero",
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": False,
                    "randomize_motor_tau": False,
                    "randomize_goal": True,
                    "goal_pos_range": {
                        "x": [-5.0, 5.0],
                        "y": [-5.0, 5.0],
                        "z": [2.0, 10.0],
                    },
                },
                "observation_noise": {
                    "position": OBS_NOISE_POS,
                    "velocity": OBS_NOISE_VEL,
                },
                "spawn_ranges": {
                    "pos_x": [-8.0, 8.0],
                    "pos_y": [-8.0, 8.0],
                    "pos_z": [1.0, 12.0],
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

    # ------------------------------------------------------------------
    # Stage 6 — Wide Navigation
    # Random goals ±10m, wide spawn, full perturbations.
    # ------------------------------------------------------------------
    phases.append({
        "name": "stage6_wide_nav",
        "stage": 6,
        "description": (
            "Wide nav: random goals ±10m, spawn ±12m, "
            f"vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL}"
        ),
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "motor_init": "zero",
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": False,
                    "randomize_motor_tau": False,
                    "randomize_goal": True,
                    "goal_pos_range": {
                        "x": [-10.0, 10.0],
                        "y": [-10.0, 10.0],
                        "z": [1.0, 15.0],
                    },
                },
                "observation_noise": {
                    "position": OBS_NOISE_POS,
                    "velocity": OBS_NOISE_VEL,
                },
                "spawn_ranges": {
                    "pos_x": [-12.0, 12.0],
                    "pos_y": [-12.0, 12.0],
                    "pos_z": [1.0, 15.0],
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

    # ------------------------------------------------------------------
    # Stage 7 — Domain Randomization
    # Full workspace, mass + motor_tau randomization, obs noise.
    # ------------------------------------------------------------------
    phases.append({
        "name": "stage7_domain_rand",
        "stage": 7,
        "description": (
            "Domain rand: goals ±15m, spawn ±18m, mass+motor_tau rand, "
            "observation noise"
        ),
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "motor_init": "zero",
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": True,
                    "randomize_motor_tau": True,
                    "randomize_goal": True,
                    "mass_range": [0.65, 0.90],
                    "motor_tau_range": [0.02, 0.05],
                    "goal_pos_range": {
                        "x": [-15.0, 15.0],
                        "y": [-15.0, 15.0],
                        "z": [1.0, 18.0],
                    },
                },
                "observation_noise": {
                    "position": OBS_NOISE_POS,
                    "velocity": OBS_NOISE_VEL,
                },
                "spawn_ranges": {
                    "pos_x": [-18.0, 18.0],
                    "pos_y": [-18.0, 18.0],
                    "pos_z": [1.0, 18.0],
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

    # ------------------------------------------------------------------
    # Stage 8 — Fine-tune
    # Same conditions as stage 7, tighter exploration, lower lr.
    # ------------------------------------------------------------------
    phases.append({
        "name": "stage8_finetune",
        "stage": 8,
        "description": (
            "Fine-tune: full DR, log_std -1.5→-3.0, ent_coef=0.0001, "
            "lr=2e-5"
        ),
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "log_std_start": -1.5,
        "log_std_end": -3.0,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "motor_init": "zero",
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": True,
                    "randomize_motor_tau": True,
                    "randomize_goal": True,
                    "mass_range": [0.65, 0.90],
                    "motor_tau_range": [0.02, 0.05],
                    "goal_pos_range": {
                        "x": [-15.0, 15.0],
                        "y": [-15.0, 15.0],
                        "z": [1.0, 18.0],
                    },
                },
                "observation_noise": {
                    "position": OBS_NOISE_POS,
                    "velocity": OBS_NOISE_VEL,
                },
                "spawn_ranges": {
                    "pos_x": [-18.0, 18.0],
                    "pos_y": [-18.0, 18.0],
                    "pos_z": [1.0, 18.0],
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

    # ------------------------------------------------------------------
    # Per-stage defaults: success rate, ent_coef, lr, log_std, etc.
    # ------------------------------------------------------------------
    _success_rate = {
        1: 0.95, 2: 0.95, 3: 0.95,
        4: 0.95, 5: 0.95, 6: 0.95,
        7: 0.95, 8: 0.95,
    }
    _ent_coef = {
        1: 0.01, 2: 0.01, 3: 0.005,
        4: 0.01, 5: 0.01, 6: 0.005,
        7: 0.005, 8: 0.0001,
    }
    _lr = {
        1: 3e-4, 2: 1e-4, 3: 1e-4,
        4: 1e-4, 5: 1e-4, 6: 5e-5,
        7: 5e-5, 8: 2e-5,
    }
    _log_std = {
        1: (LOG_STD_START, LOG_STD_END),
        2: (LOG_STD_START, -1.5),
        3: (LOG_STD_START, -2.0),
        4: (LOG_STD_START, -1.5),
        5: (LOG_STD_START, -1.5),
        6: (-0.5, -2.0),
        7: (-0.8, -2.0),
        8: (-1.5, -3.0),
    }
    _max_ts_mult = {
        1: 1.0, 2: 0.8, 3: 0.8,
        4: 1.5, 5: 1.5, 6: 1.5,
        7: 1.5, 8: 1.5,
    }

    for p in phases:
        s = p["stage"]
        ls = _log_std.get(s, (LOG_STD_START, LOG_STD_END))
        p["n_eval_episodes"] = N_EVAL_EPISODES
        p["eval_print_every_time"] = True
        p.setdefault("eval_reward_threshold", EVAL_REWARD_THR)
        p.setdefault("eval_length_threshold", EVAL_LENGTH_THR)
        p.setdefault("eval_success_rate_threshold", _success_rate.get(s, 0.80))
        p.setdefault("eval_patience", EVAL_PATIENCE)
        p.setdefault("log_std_start", ls[0])
        p.setdefault("log_std_end", ls[1])
        p["log_std_warmup"] = LOG_STD_WARMUP
        p["log_std_decay"] = LOG_STD_DECAY
        p["max_timesteps"] = int(max_timesteps * _max_ts_mult.get(s, 1.0))
        ppo_patch = p.setdefault("config_patch", {}).setdefault("ppo", {})
        ppo_patch["ent_coef"] = _ent_coef.get(s, 0.005)
        ppo_patch["learning_rate"] = _lr.get(s, 5e-5)

    return phases


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum v7: academic best-practice drone control (8 stages).",
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/curriculum_v7",
        help="Root directory for all phase outputs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--motor_tau", type=float, default=0.033,
        help="Nominal motor time constant",
    )
    parser.add_argument(
        "--max_phase_steps", type=int, default=10_000_000,
        help="Max timesteps per phase (scaled per stage, default 10M)",
    )
    parser.add_argument(
        "--reward_threshold", type=float, default=850,
        help="Episode reward threshold (training reference)",
    )
    parser.add_argument(
        "--length_threshold", type=float, default=1000,
        help="Episode length threshold (training reference)",
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/curriculum/curriculum_v7_base.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--final-eval-seeds", type=str, default=None,
        help="Comma-separated test seeds for final report",
    )
    parser.add_argument(
        "--no-final-eval", action="store_true",
        help="Skip multi-seed final evaluation",
    )
    args = parser.parse_args()

    config_path = (
        os.path.join(_REPO_ROOT, args.config)
        if not os.path.isabs(args.config)
        else args.config
    )
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}")
        sys.exit(1)
    base_cfg = load_config(config_path)
    print(f"Loaded curriculum v7 config from {args.config}")

    os.makedirs(args.output_dir, exist_ok=True)

    phases = _build_phase_configs(
        motor_tau=args.motor_tau,
        max_timesteps=args.max_phase_steps,
        reward_thr=args.reward_threshold,
        length_thr=args.length_threshold,
    )

    state = load_state(args.output_dir)
    completed = set(state.get("completed_phases", []))

    total_steps = sum(p["max_timesteps"] for p in phases)
    print("=" * 60)
    print("CURRICULUM V7 — Academic Best-Practice Drone Control (8 stages)")
    print("=" * 60)
    print(f"Config           : {args.config}")
    print(f"Output directory : {args.output_dir}")
    print(f"Seed             : {args.seed}")
    print(f"Motor tau        : {args.motor_tau}")
    print(f"Max steps/phase  : {args.max_phase_steps:,} (scaled per stage)")
    print(f"Total budget     : {total_steps:,} steps across {len(phases)} stages")
    if completed:
        print(f"Already done     : {len(completed)} phases")
    print("=" * 60)

    prev_dir = None
    last_phase_cfg = None
    last_phase = None
    for p in phases:
        if p["name"] in completed:
            candidate = os.path.join(args.output_dir, p["name"])
            if os.path.isdir(candidate):
                prev_dir = candidate

    for i, phase in enumerate(phases):
        if phase["name"] in completed:
            print(
                f"\n[{i+1}/{len(phases)}] {phase['name']} "
                f"— SKIPPED (already completed)"
            )
            continue

        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{len(phases)}] {phase['name']}")
        print(f"  {phase['description']}")
        print(f"  max_timesteps={phase['max_timesteps']:,}  "
              f"success_rate>={phase['eval_success_rate_threshold']:.0%}")
        print(f"{'=' * 60}")

        cfg = _deep_merge(base_cfg, phase["config_patch"])
        last_phase_cfg = cfg
        last_phase = phase
        phase_dir = train_phase(
            phase, cfg, args.output_dir, args.seed, prev_model_dir=prev_dir,
        )

        completed.add(phase["name"])
        state["completed_phases"] = list(completed)
        state["current_phase"] = phase["name"]
        save_state(args.output_dir, state)

        prev_dir = phase_dir

    print(f"\n{'=' * 60}")
    print("CURRICULUM V7 COMPLETE")
    print(f"Final model: {prev_dir}")
    print(f"{'=' * 60}")

    if (
        prev_dir
        and not args.no_final_eval
        and last_phase_cfg is not None
        and last_phase is not None
    ):
        final_seeds = FINAL_EVAL_SEEDS_DEFAULT
        if args.final_eval_seeds:
            final_seeds = [
                int(s.strip()) for s in args.final_eval_seeds.split(",")
            ]
        n_eval_episodes = last_phase.get("n_eval_episodes", N_EVAL_EPISODES)
        run_final_eval(
            prev_dir,
            last_phase_cfg,
            final_seeds,
            n_eval_episodes,
            eval_domain_rand=last_phase.get("eval_domain_rand", False),
        )


if __name__ == "__main__":
    main()
