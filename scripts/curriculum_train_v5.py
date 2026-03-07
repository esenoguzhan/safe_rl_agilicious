#!/usr/bin/env python3
"""
Curriculum learning v5: "Velocity Anchor" — master high-speed recovery and braking
before expanding navigation distance to avoid the reward cliff seen in V4.

  Stage 1 (Recovery Anchor): Fixed goal [0,0,5], zero linear vel, MAX angular vel (±5).
      Wide reward signal (cauchy_scale=0.5), ent_coef=0.01.
  Stage 2 (Velocity Ramp): Narrow spawn/goal (±2m), max ang ±5. Sub-phases: vel ±2, ±4, ±5, ±6, ±7, ±8.
      cauchy_scale=0.5, ent_coef=0.01.
  Stage 2.5 (Medium Nav): Goal range ±6m, cauchy_scale=0.5. Bridges 2m box and 12m world.
      PPO: n_steps=4096, vf_coef=1.0 for value stability.
  Stage 3a/3b (Intermediate Nav): ±8m then ±10m goal/spawn (no direct jump to ±12).
  Stage 3 (Navigation Expansion): WIDE_GOAL_RANGE (±12 goal / ±15 spawn), full velocities. 2x max_timesteps.
      PPO: n_steps=4096, vf_coef=1.0.
  Stage 4 (Decoupled DR): 4a Motor Tau only, 4b Full DR (Motor Tau + Mass).
      cauchy_scale=0.1 for precision, ent_coef=0.005.
  Stage 5 (Fine-tune): log_std -1.5 → -3.0, ent_coef=0.0001.

Eval: 90% success for all stages. Default 10M steps/phase. Progress: <output_dir>/curriculum_state.yaml.

Usage:
  python scripts/curriculum_train_v5.py [--config configs/curriculum/curriculum_v5_base.yaml] [--output_dir models/curriculum_v5] [--seed 0]
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

# V5: longer warmup/decay so agent explores before honing in
EVAL_PATIENCE = 3
LOG_STD_START = -1.0
LOG_STD_END = -2.5
LOG_STD_WARMUP = 0.15
LOG_STD_DECAY = 0.5

NARROW_POS_X = [-2.0, 2.0]
NARROW_POS_Y = [-2.0, 2.0]
NARROW_POS_Z = [3.0, 7.0]
NARROW_GOAL_RANGE = {"x": [-2.0, 2.0], "y": [-2.0, 2.0], "z": [3.0, 7.0]}

# Medium: ±6m, bridges 2m braking box and 12m navigation (Stage 2.5)
MEDIUM_POS_X = [-6.0, 6.0]
MEDIUM_POS_Y = [-6.0, 6.0]
MEDIUM_POS_Z = [2.0, 10.0]
MEDIUM_GOAL_RANGE = {"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [2.0, 10.0]}

WIDE_POS_X = [-15.0, 15.0]
WIDE_POS_Y = [-15.0, 15.0]
WIDE_POS_Z = [1.0, 18.0]
WIDE_GOAL_RANGE = {"x": [-12.0, 12.0], "y": [-12.0, 12.0], "z": [1.0, 18.0]}

MAX_LIN_VEL = 8.0
MAX_ANG_VEL = 5.0

# Anti-memorization: observation noise (calibratable). Written to env.observation_noise in config_patch;
# the training pipeline (e.g. wrap_env in curriculum_train.py) must apply it when present.
OBS_NOISE_POS = 0.01   # position obs noise std (indices 0:3)
OBS_NOISE_VEL = 0.05   # velocity obs noise std (lin_vel indices 7:10)

# Relaxed box ±4m for Stage 2a/2b (sweet spot for learning braking at 2–4 m/s)
RELAXED_BOX_POS_X = [-4.0, 4.0]
RELAXED_BOX_POS_Y = [-4.0, 4.0]
RELAXED_BOX_POS_Z = [2.0, 10.0]
RELAXED_BOX_GOAL_RANGE = {"x": [-4.0, 4.0], "y": [-4.0, 4.0], "z": [2.0, 10.0]}


def _build_phase_configs(
    motor_tau: float,
    max_timesteps: int,
    reward_thr: float,
    length_thr: float,
):
    """Return the V5 Velocity Anchor curriculum phase list."""
    phases = []

    # -------------------------------------------------------------------------
    # Stage 1: Recovery Anchor — fixed goal, zero linear vel, MAX angular vel
    # Wide reward (cauchy_scale=0.5), high entropy (0.01)
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage1_recovery_anchor",
        "stage": 1,
        "description": f"Recovery Anchor: fixed goal [0,0,5], zero lin vel, ang ±{MAX_ANG_VEL}, cauchy_scale=0.5",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": False,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {"enabled": False},
                "goal_position": [0.0, 0.0, 5.0],
                "custom_reward": {"cauchy_scale": 0.5},
                "observation_noise": {"position": OBS_NOISE_POS, "velocity": OBS_NOISE_VEL},
                "spawn_ranges": {
                    "pos_x": [-1.0, 1.0],
                    "pos_y": [-1.0, 1.0],
                    "pos_z": [4.0, 6.0],
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
    # Stage 2: Velocity Ramp — ent_coef=0.02 (entropy shock). 2a/2b: ±4m box; 2c–f: ±2m.
    # 2a: log_std_start=-0.5 to un-freeze policy. Observation noise to reduce memorization.
    # -------------------------------------------------------------------------
    _stage2_vels = [
        (2.0, "stage2a_vel2"),
        (4.0, "stage2b_vel4"),
        (5.0, "stage2c_vel5"),
        (6.0, "stage2d_vel6"),
        (7.0, "stage2e_vel7"),
        (8.0, "stage2f_vel8"),
    ]
    _stage2_relaxed_box = ("stage2a_vel2", "stage2b_vel4")  # ±4m for 2a/2b
    for vel_mag, phase_name in _stage2_vels:
        use_relaxed_box = phase_name in _stage2_relaxed_box
        pos_x = copy.deepcopy(RELAXED_BOX_POS_X if use_relaxed_box else NARROW_POS_X)
        pos_y = copy.deepcopy(RELAXED_BOX_POS_Y if use_relaxed_box else NARROW_POS_Y)
        pos_z = copy.deepcopy(RELAXED_BOX_POS_Z if use_relaxed_box else NARROW_POS_Z)
        goal_range = copy.deepcopy(RELAXED_BOX_GOAL_RANGE if use_relaxed_box else NARROW_GOAL_RANGE)
        box_label = "±4m" if use_relaxed_box else "±2m"
        phase_entry = {
            "name": phase_name,
            "stage": 2,
            "description": f"Braking ramp: {box_label} box, lin vel ±{vel_mag}, ang ±{MAX_ANG_VEL}, ent_coef=0.02",
            "max_timesteps": max_timesteps,
            "reward_threshold": reward_thr,
            "length_threshold": length_thr,
            "eval_domain_rand": True,
            "log_std_end": -1.5,
            "config_patch": {
                "env": {
                    "quadrotor_dynamics": {"motor_tau": motor_tau},
                    "domain_randomization": {
                        "enabled": True,
                        "randomize_mass": False,
                        "randomize_motor_tau": False,
                        "randomize_goal": True,
                        "goal_pos_range": goal_range,
                    },
                    "custom_reward": {"cauchy_scale": 0.5},
                    "observation_noise": {
                        "position": OBS_NOISE_POS,
                        "velocity": OBS_NOISE_VEL,
                    },
                    "spawn_ranges": {
                        "pos_x": pos_x,
                        "pos_y": pos_y,
                        "pos_z": pos_z,
                        "vel_x": [-vel_mag, vel_mag],
                        "vel_y": [-vel_mag, vel_mag],
                        "vel_z": [-vel_mag, vel_mag],
                        "ang_vel_x": [-MAX_ANG_VEL, MAX_ANG_VEL],
                        "ang_vel_y": [-MAX_ANG_VEL, MAX_ANG_VEL],
                        "ang_vel_z": [-MAX_ANG_VEL, MAX_ANG_VEL],
                        "ori_scale": 1.0,
                    },
                },
            },
        }
        if phase_name == "stage2a_vel2":
            phase_entry["log_std_start"] = -0.5
        phases.append(phase_entry)

    # -------------------------------------------------------------------------
    # Stage 2.5: Medium Nav — goal range ±6m, bridges 2m braking box and 12m world
    # cauchy_scale=0.5; n_steps=4096, vf_coef=1.0 to stabilize value function
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage2_5_medium_nav",
        "stage": 3,
        "description": f"Medium Nav: goal ±6m, vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL}, cauchy_scale=0.5, n_steps=4096",
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
                "custom_reward": {"cauchy_scale": 0.5},
                "observation_noise": {"position": OBS_NOISE_POS, "velocity": OBS_NOISE_VEL},
                "spawn_ranges": {
                    "pos_x": copy.deepcopy(MEDIUM_POS_X),
                    "pos_y": copy.deepcopy(MEDIUM_POS_Y),
                    "pos_z": copy.deepcopy(MEDIUM_POS_Z),
                    "vel_x": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "vel_y": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "vel_z": [-MAX_LIN_VEL, MAX_LIN_VEL],
                    "ang_vel_x": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_y": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ang_vel_z": [-MAX_ANG_VEL, MAX_ANG_VEL],
                    "ori_scale": 1.0,
                },
            },
            "ppo": {"n_steps": 4096, "vf_coef": 1.0},
        },
    })

    # -------------------------------------------------------------------------
    # Intermediate nav: ±8m then ±10m (avoid direct jump ±6 → ±12)
    # -------------------------------------------------------------------------
    _intermediate_nav = [
        (8.0, 8.0, "stage3a_nav_8", "Goal ±8m, spawn ±8m"),
        (10.0, 11.0, "stage3b_nav_10", "Goal ±10m, spawn ±11m"),
    ]
    for goal_half, spawn_half, phase_name, desc in _intermediate_nav:
        goal_range = {"x": [-goal_half, goal_half], "y": [-goal_half, goal_half], "z": [2.0, 10.0]}
        pos_x = [-spawn_half, spawn_half]
        pos_y = [-spawn_half, spawn_half]
        pos_z = [2.0, 10.0]
        phases.append({
            "name": phase_name,
            "stage": 3,
            "description": f"Nav: {desc}, vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL}, n_steps=4096",
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
                        "goal_pos_range": copy.deepcopy(goal_range),
                    },
                    "custom_reward": {"cauchy_scale": 0.5},
                    "observation_noise": {"position": OBS_NOISE_POS, "velocity": OBS_NOISE_VEL},
                    "spawn_ranges": {
                        "pos_x": copy.deepcopy(pos_x),
                        "pos_y": copy.deepcopy(pos_y),
                        "pos_z": copy.deepcopy(pos_z),
                        "vel_x": [-MAX_LIN_VEL, MAX_LIN_VEL],
                        "vel_y": [-MAX_LIN_VEL, MAX_LIN_VEL],
                        "vel_z": [-MAX_LIN_VEL, MAX_LIN_VEL],
                        "ang_vel_x": [-MAX_ANG_VEL, MAX_ANG_VEL],
                        "ang_vel_y": [-MAX_ANG_VEL, MAX_ANG_VEL],
                        "ang_vel_z": [-MAX_ANG_VEL, MAX_ANG_VEL],
                        "ori_scale": 1.0,
                    },
                },
                "ppo": {"n_steps": 4096, "vf_coef": 1.0},
            },
        })

    # -------------------------------------------------------------------------
    # Stage 3: Navigation Expansion — wide goals/spawn ±12/±15, full velocities
    # Double max_timesteps for more random-walk time to discover 12m goal.
    # n_steps=4096, vf_coef=1.0 to stabilize value function.
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage3_navigation_expansion",
        "stage": 3,
        "description": f"Navigation Expansion: wide goals/spawn, vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL} (2x steps)",
        "max_timesteps": max_timesteps * 2,
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
                "custom_reward": {"cauchy_scale": 0.5},
                "observation_noise": {"position": OBS_NOISE_POS, "velocity": OBS_NOISE_VEL},
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
            "ppo": {"n_steps": 4096, "vf_coef": 1.0},
        },
    })

    # -------------------------------------------------------------------------
    # Stage 4a: Decoupled DR — Motor Tau randomization only
    # cauchy_scale=0.1 for precision, ent_coef=0.005
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage4a_domain_motor_tau",
        "stage": 4,
        "description": f"Domain Rand: motor_tau only, wide goals, vel ±{MAX_LIN_VEL}, ang ±{MAX_ANG_VEL}, cauchy_scale=0.1",
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
                "custom_reward": {"cauchy_scale": 0.1},
                "observation_noise": {"position": OBS_NOISE_POS, "velocity": OBS_NOISE_VEL},
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
    # Stage 4b: Full DR — Motor Tau + Mass
    # cauchy_scale=0.1, ent_coef=0.005
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage4b_domain_full",
        "stage": 4,
        "description": "Domain Rand: motor_tau + mass, wide goals, cauchy_scale=0.1",
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
                "custom_reward": {"cauchy_scale": 0.1},
                "observation_noise": {"position": OBS_NOISE_POS, "velocity": OBS_NOISE_VEL},
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
    # Stage 5: Fine-tune — log_std -1.5 → -3.0, ent_coef=0.0001
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage5_finetune",
        "stage": 5,
        "description": "Fine-tune: log_std -1.5→-3.0, ent_coef=0.0001, cauchy_scale=0.1",
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
                "custom_reward": {"cauchy_scale": 0.1},
                "observation_noise": {"position": OBS_NOISE_POS, "velocity": OBS_NOISE_VEL},
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
        2: (LOG_STD_START, -1.5),   # keep min at -1.5 during braking master
        3: (LOG_STD_START, -2.0),
        4: (LOG_STD_START, -1.5),
        5: (-1.5, -3.0),
    }
    N_EVAL_EPISODES = 50
    EVAL_SUCCESS_RATE = 0.90  # 90% for all stages
    _ent_coef_by_stage = {
        1: 0.01,
        2: 0.02,   # Stage 2: entropy shock (anti-memorization)
        3: 0.005,
        4: 0.005,
        5: 0.0001,
    }
    # Stage 1 & 2: 1e-4 (recovery/braking need high adaptability). Stage 3 & 4: 1e-4 (overcome DR noise). Stage 5: 5e-5 (precision).
    _learning_rate_by_stage = {
        1: 1.0e-4,
        2: 1.0e-4,
        3: 1.0e-4,
        4: 1.0e-4,
        5: 5.0e-5,
    }

    for p in phases:
        s = p["stage"]
        default_ls = _log_std_defaults.get(s, (LOG_STD_START, LOG_STD_END))
        ls_start = p.get("log_std_start", default_ls[0])
        ls_end = p.get("log_std_end", default_ls[1])
        p["n_eval_episodes"] = N_EVAL_EPISODES
        p["eval_print_every_time"] = True
        p.setdefault("eval_reward_threshold", EVAL_REWARD_THR)
        p.setdefault("eval_length_threshold", EVAL_LENGTH_THR)
        p["eval_success_rate_threshold"] = EVAL_SUCCESS_RATE
        p.setdefault("eval_patience", EVAL_PATIENCE)
        p["log_std_start"] = ls_start
        p["log_std_end"] = ls_end
        p["log_std_warmup"] = LOG_STD_WARMUP
        p["log_std_decay"] = LOG_STD_DECAY
        p.setdefault("config_patch", {})
        ppo_patch = p["config_patch"].setdefault("ppo", {})
        ppo_patch["ent_coef"] = _ent_coef_by_stage.get(s, 0.001)
        ppo_patch["learning_rate"] = _learning_rate_by_stage.get(s, 3.0e-5)

    return phases


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum v5 (Velocity Anchor): recovery → braking → navigation → domain rand → finetune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/curriculum_v5",
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
        default=10_000_000,
        help="Max timesteps per phase before forced advance (default: 10M)",
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
        default="configs/curriculum/curriculum_v5_base.yaml",
        help="Path to base config YAML. Relative to repo root.",
    )
    args = parser.parse_args()

    config_path = os.path.join(_REPO_ROOT, args.config) if not os.path.isabs(args.config) else args.config
    if not os.path.isfile(config_path):
        print(f"Config not found: {config_path}")
        print("Curriculum v5 requires a base config YAML, e.g. configs/curriculum/curriculum_v5_base.yaml")
        sys.exit(1)
    base_cfg = load_config(config_path)
    print(f"Loaded curriculum v5 config from {args.config}")

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
    print("CURRICULUM V5 (VELOCITY ANCHOR) — recovery → braking → navigation → domain rand → finetune")
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
    print("CURRICULUM V5 COMPLETE")
    print(f"Final model: {prev_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
