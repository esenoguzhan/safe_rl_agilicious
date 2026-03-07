#!/usr/bin/env python3
"""
Curriculum learning v2: multi-stage progression for quadrotor control.

  1. Fixed motor_tau=0.033, fixed goal
  2. Randomized goals (narrow)
  3. Wider randomized goals (spawn x,y ±18 z [1,18]; goals x,y ±12 z [1,18])
  4. Include different orientations (ori_scale 0.5)
  4b–4d. Orientation ramp ori 0.6, 0.7, 0.8 (same vel ±1.5).
  5. Wider linear velocities: ±2, ±2.5, … ±8 (more intermediate steps).
  6. Angular velocities: smooth ramp — vel 6.5→…→8, ang ±0.1→…→±1, ori 0.5→…→0.92 over 12 steps; then ang ±2→±5 at vel ±8 ori 1.0
  7. Add motor lag (tau) randomization (vel ±8, ang ±5)
  8. Add mass randomization (full domain rand)
  9. Fine-tune (tighter log_std, same full domain rand)

From stage 3 onward: spawn x,y ±18 z [1,18]; randomized goals x,y ±12 z [1,18].
Advancement: 50 eval episodes, success rate ≥ 45/50 (90%); per-episode success = reward ≥ 900 and length ≥ 1000.
Progress is saved to <output_dir>/curriculum_state.yaml.

Usage:
  python scripts/curriculum_train_v2.py [--output_dir models/curriculum_v2] [--seed 0]
"""
import argparse
import copy
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse full training pipeline and callbacks from the original curriculum
from scripts.curriculum_train import (
    BASE_CONFIG,
    EVAL_PATIENCE,
    LOG_STD_DECAY,
    LOG_STD_END,
    LOG_STD_START,
    LOG_STD_WARMUP,
    _deep_merge,
    load_state,
    save_state,
    train_phase,
)

# Spawn: x,y ±18, z [1, 18]. Randomized goals: x,y ±12, z [1, 18]
WIDE_POS_X = [-15.0, 15.0]
WIDE_POS_Y = [-15.0, 15.0]
WIDE_POS_Z = [1.0, 18.0]
WIDE_GOAL_RANGE = {"x": [-12.0, 12.0], "y": [-12.0, 12.0], "z": [1.0, 18.0]}
# Max velocity ranges: linear ±8, angular ±5
MAX_LIN_VEL = 8.0
MAX_ANG_VEL = 5.0


def _build_phase_configs_v2(
    motor_tau: float,
    max_timesteps: int,
    reward_thr: float,
    length_thr: float,
):
    """Return the 9-stage curriculum phase list. Stages 3+ use wide ranges (x,y ±10, z [1,18])."""
    phases = []

    # -------------------------------------------------------------------------
    # 1. Fixed motor_tau=0.033, fixed goal
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage1_fixed_goal",
        "stage": 1,
        "description": f"Fixed goal, motor_tau={motor_tau}, easy spawn",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": False,
        "n_eval_episodes": 20,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {"enabled": False},
                "goal_position": [0.0, 0.0, 5.0],
                "spawn_ranges": {
                    "pos_x": [-1.0, 1.0],
                    "pos_y": [-1.0, 1.0],
                    "pos_z": [4.0, 6.0],
                    "vel_x": [-1.0, 1.0],
                    "vel_y": [-1.0, 1.0],
                    "vel_z": [-1.0, 1.0],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 2. Randomized goals (narrow)
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage2_random_goals",
        "stage": 2,
        "description": f"Random goals (narrow), motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": False,
                    "randomize_motor_tau": False,
                    "randomize_goal": True,
                    "goal_pos_range": {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [3.0, 7.0]},
                },
                "spawn_ranges": {
                    "pos_x": [-2.0, 2.0],
                    "pos_y": [-2.0, 2.0],
                    "pos_z": [3.0, 7.0],
                    "vel_x": [-1.0, 1.0],
                    "vel_y": [-1.0, 1.0],
                    "vel_z": [-1.0, 1.0],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 3. Wider randomized goals — full wide range (x,y ±10, z [1, 18])
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage3_wider_goals",
        "stage": 3,
        "description": f"Wide random goals (x,y ±10, z [1,18]), motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-1.5, 1.5],
                    "vel_y": [-1.5, 1.5],
                    "vel_z": [-1.5, 1.5],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 4. Include different orientations (wide range)
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage4_orientations",
        "stage": 4,
        "description": f"Wide goals + orientation diversity, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-1.5, 1.5],
                    "vel_y": [-1.5, 1.5],
                    "vel_z": [-1.5, 1.5],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 4b–4d. Ramp orientation diversity (same vel ±1.5) for smoother transition
    # -------------------------------------------------------------------------
    for ori in (0.6, 0.7, 0.8):
        phases.append({
            "name": f"stage4{'b' if ori == 0.6 else 'c' if ori == 0.7 else 'd'}_ori_{ori}",
            "stage": 4,
            "description": f"Wide goals + ori_scale {ori}, vel ±1.5, motor_tau={motor_tau}",
            "max_timesteps": max_timesteps,
            "reward_threshold": reward_thr,
            "length_threshold": length_thr,
            "eval_domain_rand": True,
            "n_eval_episodes": 20,
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
                        "vel_x": [-1.5, 1.5],
                        "vel_y": [-1.5, 1.5],
                        "vel_z": [-1.5, 1.5],
                        "ang_vel_x": [0.0, 0.0],
                        "ang_vel_y": [0.0, 0.0],
                        "ang_vel_z": [0.0, 0.0],
                        "ori_scale": ori,
                    },
                },
            },
        })

    # -------------------------------------------------------------------------
    # 5. Wider linear velocities — more intermediate steps (wide range)
    # -------------------------------------------------------------------------
    # 5a: vel ±2.0
    phases.append({
        "name": "stage5a_vel_2",
        "stage": 5,
        "description": f"Wide range + linear vel ±2.0, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-2.0, 2.0],
                    "vel_y": [-2.0, 2.0],
                    "vel_z": [-2.0, 2.0],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5a2: vel ±2.5
    phases.append({
        "name": "stage5a2_vel_2_5",
        "stage": 5,
        "description": f"Wide range + linear vel ±2.5, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-2.5, 2.5],
                    "vel_y": [-2.5, 2.5],
                    "vel_z": [-2.5, 2.5],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5b: vel ±3
    phases.append({
        "name": "stage5b_vel_3",
        "stage": 5,
        "description": f"Wide range + linear vel ±3, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-3.0, 3.0],
                    "vel_y": [-3.0, 3.0],
                    "vel_z": [-3.0, 3.0],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5b2: vel ±3.5
    phases.append({
        "name": "stage5b2_vel_3_5",
        "stage": 5,
        "description": f"Wide range + linear vel ±3.5, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-3.5, 3.5],
                    "vel_y": [-3.5, 3.5],
                    "vel_z": [-3.5, 3.5],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5c: vel ±4
    phases.append({
        "name": "stage5c_vel_4",
        "stage": 5,
        "description": f"Wide range + linear vel ±4, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-4.0, 4.0],
                    "vel_y": [-4.0, 4.0],
                    "vel_z": [-4.0, 4.0],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5c2: vel ±4.5
    phases.append({
        "name": "stage5c2_vel_4_5",
        "stage": 5,
        "description": f"Wide range + linear vel ±4.5, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-4.5, 4.5],
                    "vel_y": [-4.5, 4.5],
                    "vel_z": [-4.5, 4.5],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5d: vel ±5
    phases.append({
        "name": "stage5d_vel_5",
        "stage": 5,
        "description": f"Wide range + linear vel ±5, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-5.0, 5.0],
                    "vel_y": [-5.0, 5.0],
                    "vel_z": [-5.0, 5.0],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5d2: vel ±5.5
    phases.append({
        "name": "stage5d2_vel_5_5",
        "stage": 5,
        "description": f"Wide range + linear vel ±5.5, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-5.5, 5.5],
                    "vel_y": [-5.5, 5.5],
                    "vel_z": [-5.5, 5.5],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5e: vel ±6
    phases.append({
        "name": "stage5e_vel_6",
        "stage": 5,
        "description": f"Wide range + linear vel ±6, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-6.0, 6.0],
                    "vel_y": [-6.0, 6.0],
                    "vel_z": [-6.0, 6.0],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5e2: vel ±6.5
    phases.append({
        "name": "stage5e2_vel_6_5",
        "stage": 5,
        "description": f"Wide range + linear vel ±6.5, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-6.5, 6.5],
                    "vel_y": [-6.5, 6.5],
                    "vel_z": [-6.5, 6.5],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5f: vel ±7
    phases.append({
        "name": "stage5f_vel_7",
        "stage": 5,
        "description": f"Wide range + linear vel ±7, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-7.0, 7.0],
                    "vel_y": [-7.0, 7.0],
                    "vel_z": [-7.0, 7.0],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5f2: vel ±7.5
    phases.append({
        "name": "stage5f2_vel_7_5",
        "stage": 5,
        "description": f"Wide range + linear vel ±7.5, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-7.5, 7.5],
                    "vel_y": [-7.5, 7.5],
                    "vel_z": [-7.5, 7.5],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })
    # 5g: vel ±8 (max)
    phases.append({
        "name": "stage5g_vel_8",
        "stage": 5,
        "description": f"Wide range + linear vel ±8, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-8.0, 8.0],
                    "vel_y": [-8.0, 8.0],
                    "vel_z": [-8.0, 8.0],
                    "ang_vel_x": [0.0, 0.0],
                    "ang_vel_y": [0.0, 0.0],
                    "ang_vel_z": [0.0, 0.0],
                    "ori_scale": 0.5,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 6. Angular velocities — smooth ramp: bridge then (vel, ang, ori) co-ramp to ±1 / ±8 / 1.0
    # -------------------------------------------------------------------------
    # Bridge: first touch of angular at reduced vel, ori 0.5
    _ang_ramp = [
        (6.5, 0.10, 0.50),
        (6.0, 0.18, 0.53),
        (5.5, 0.26, 0.56),
        (5.0, 0.34, 0.60),
        (5.0, 0.42, 0.64),
        (5.5, 0.50, 0.68),
        (6.0, 0.58, 0.72),
        (6.0, 0.66, 0.76),
        (6.5, 0.74, 0.80),
        (7.0, 0.82, 0.84),
        (7.5, 0.90, 0.88),
        (8.0, 1.00, 0.92),
    ]
    for i, (v, a, o) in enumerate(_ang_ramp):
        if a >= 1.0:
            phase_name = "stage6a_ang_1"
        else:
            phase_name = f"stage6_ang_{int(a*100):02d}"
        phases.append({
            "name": phase_name,
            "stage": 6,
            "description": f"Wide range + vel ±{v}, ang_vel ±{a:.2f}, ori {o:.2f}, motor_tau={motor_tau}",
            "max_timesteps": max_timesteps,
            "reward_threshold": reward_thr,
            "length_threshold": length_thr,
            "eval_domain_rand": True,
            "n_eval_episodes": 20,
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
                        "ang_vel_x": [-a, a],
                        "ang_vel_y": [-a, a],
                        "ang_vel_z": [-a, a],
                        "ori_scale": o,
                    },
                },
            },
        })
    # 6b: ang_vel ±2 (vel ±8, ori 1.0)
    phases.append({
        "name": "stage6b_ang_2",
        "stage": 6,
        "description": f"Wide range + vel ±8, ang_vel ±2, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-8.0, 8.0],
                    "vel_y": [-8.0, 8.0],
                    "vel_z": [-8.0, 8.0],
                    "ang_vel_x": [-2.0, 2.0],
                    "ang_vel_y": [-2.0, 2.0],
                    "ang_vel_z": [-2.0, 2.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })
    # 6c: ang_vel ±3 (vel ±8)
    phases.append({
        "name": "stage6c_ang_3",
        "stage": 6,
        "description": f"Wide range + vel ±8, ang_vel ±3, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-8.0, 8.0],
                    "vel_y": [-8.0, 8.0],
                    "vel_z": [-8.0, 8.0],
                    "ang_vel_x": [-3.0, 3.0],
                    "ang_vel_y": [-3.0, 3.0],
                    "ang_vel_z": [-3.0, 3.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })
    # 6d: ang_vel ±4 (vel ±8)
    phases.append({
        "name": "stage6d_ang_4",
        "stage": 6,
        "description": f"Wide range + vel ±8, ang_vel ±4, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-8.0, 8.0],
                    "vel_y": [-8.0, 8.0],
                    "vel_z": [-8.0, 8.0],
                    "ang_vel_x": [-4.0, 4.0],
                    "ang_vel_y": [-4.0, 4.0],
                    "ang_vel_z": [-4.0, 4.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })
    # 6e: ang_vel ±5 (max, vel ±8)
    phases.append({
        "name": "stage6e_ang_5",
        "stage": 6,
        "description": f"Wide range + vel ±8, ang_vel ±5, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-8.0, 8.0],
                    "vel_y": [-8.0, 8.0],
                    "vel_z": [-8.0, 8.0],
                    "ang_vel_x": [-5.0, 5.0],
                    "ang_vel_y": [-5.0, 5.0],
                    "ang_vel_z": [-5.0, 5.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 7. Add motor lag (tau) randomization (wide range)
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage7_motor_tau_rand",
        "stage": 7,
        "description": f"Wide range + motor_tau randomization, nominal motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-8.0, 8.0],
                    "vel_y": [-8.0, 8.0],
                    "vel_z": [-8.0, 8.0],
                    "ang_vel_x": [-5.0, 5.0],
                    "ang_vel_y": [-5.0, 5.0],
                    "ang_vel_z": [-5.0, 5.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 8. Add mass randomization (wide range)
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage8_mass_rand",
        "stage": 8,
        "description": "Wide range + full domain rand: goals + motor_tau + mass",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
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
                    "vel_x": [-8.0, 8.0],
                    "vel_y": [-8.0, 8.0],
                    "vel_z": [-8.0, 8.0],
                    "ang_vel_x": [-5.0, 5.0],
                    "ang_vel_y": [-5.0, 5.0],
                    "ang_vel_z": [-5.0, 5.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -------------------------------------------------------------------------
    # 9. Fine-tune (wide range, full domain rand, tighter log_std)
    # -------------------------------------------------------------------------
    phases.append({
        "name": "stage9_finetune",
        "stage": 9,
        "description": "Fine-tune: anneal log_std for precision, wide range + full domain rand",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
        "log_std_start": -1.5,
        "log_std_end": -3.0,
        "eval_patience": 3,
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
                    "vel_x": [-8.0, 8.0],
                    "vel_y": [-8.0, 8.0],
                    "vel_z": [-8.0, 8.0],
                    "ang_vel_x": [-5.0, 5.0],
                    "ang_vel_y": [-5.0, 5.0],
                    "ang_vel_z": [-5.0, 5.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # ---- Eval thresholds: fixed 1000 steps and >= 900 reward for every stage ----
    EVAL_REWARD_THR = 900
    EVAL_LENGTH_THR = 1000
    _log_std_defaults = {
        1: (LOG_STD_START, LOG_STD_END),
        2: (LOG_STD_START, LOG_STD_END),
        3: (LOG_STD_START, -2.0),
        4: (LOG_STD_START, -2.0),
        5: (LOG_STD_START, -1.8),
        6: (LOG_STD_START, -1.5),
        7: (LOG_STD_START, -1.5),
        8: (LOG_STD_START, -1.5),
        9: (-1.5, -3.0),
    }
    # Advancement: 50 eval episodes; success rate 90% up to stage6_ang_90, then 80%
    N_EVAL_EPISODES = 50
    EVAL_SUCCESS_RATE_THR = 45 / 50   # 0.9
    EVAL_SUCCESS_RATE_THR_AFTER_ANG90 = 0.8  # 40/50 after angular 0.9 stage
    # Entropy coefficient by stage: 1-3 → 0.01, 4-5 → 0.005, 6 → ramp 0.01→0.02, 7-8 → 0.005, 9 → 0.0001
    _ent_coef_by_stage = {
        1: 0.01, 2: 0.01, 3: 0.01,
        4: 0.005, 5: 0.005,
        7: 0.005, 8: 0.005,
        9: 0.0001,
    }
    stage6_phases = [i for i, p in enumerate(phases) if p["stage"] == 6]
    n_stage6 = len(stage6_phases)

    past_ang_90 = False
    stage6_idx = 0
    for i, p in enumerate(phases):
        s = p["stage"]
        ls_start, ls_end = _log_std_defaults.get(s, (LOG_STD_START, LOG_STD_END))
        p["n_eval_episodes"] = N_EVAL_EPISODES
        p.setdefault("eval_reward_threshold", EVAL_REWARD_THR)
        p.setdefault("eval_length_threshold", EVAL_LENGTH_THR)
        p["eval_success_rate_threshold"] = (
            EVAL_SUCCESS_RATE_THR_AFTER_ANG90 if past_ang_90 else EVAL_SUCCESS_RATE_THR
        )
        p.setdefault("eval_patience", EVAL_PATIENCE)
        p.setdefault("log_std_start", ls_start)
        p.setdefault("log_std_end", ls_end)
        p.setdefault("log_std_warmup", LOG_STD_WARMUP)
        p.setdefault("log_std_decay", LOG_STD_DECAY)

        if s == 6:
            # Stage 6: ramp ent_coef from 0.01 to 0.02
            t = stage6_idx / max(1, n_stage6 - 1)
            ent_coef = 0.01 + (0.02 - 0.01) * t
            stage6_idx += 1
        else:
            ent_coef = _ent_coef_by_stage.get(s, 0.001)
        p.setdefault("config_patch", {})
        p["config_patch"].setdefault("ppo", {})["ent_coef"] = ent_coef

        if p["name"] == "stage6_ang_90":
            past_ang_90 = True

    return phases


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum v2: 9-stage quadrotor (fixed tau → goals → ori → vel → ang_vel → motor_tau rand → mass rand → finetune). Wide: x,y ±10, z [1,18].",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/curriculum_v2",
        help="Root directory for all phase outputs (default: models/curriculum_v2)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--motor_tau",
        type=float,
        default=0.033,
        help="Nominal motor time constant (real hardware value)",
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
        help="Episode reward threshold (training ref) for early phase advancement",
    )
    parser.add_argument(
        "--length_threshold",
        type=float,
        default=1000,
        help="Episode length threshold (training ref) for early phase advancement",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    phases = _build_phase_configs_v2(
        motor_tau=args.motor_tau,
        max_timesteps=args.max_phase_steps,
        reward_thr=args.reward_threshold,
        length_thr=args.length_threshold,
    )

    state = load_state(args.output_dir)
    completed = set(state.get("completed_phases", []))

    print("=" * 60)
    print("CURRICULUM V2 — spawn x,y ±18 z [1,18], goals x,y ±12 z [1,18], vel ±8 ang ±5")
    print("=" * 60)
    print(f"Output directory : {args.output_dir}")
    print(f"Seed             : {args.seed}")
    print(f"Motor tau        : {args.motor_tau}")
    print(f"Reward threshold : {args.reward_threshold}")
    print(f"Length threshold : {args.length_threshold}")
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

        cfg = _deep_merge(BASE_CONFIG, phase["config_patch"])
        phase_dir = train_phase(
            phase, cfg, args.output_dir, args.seed, prev_model_dir=prev_dir
        )

        completed.add(phase["name"])
        state["completed_phases"] = list(completed)
        state["current_phase"] = phase["name"]
        save_state(args.output_dir, state)

        prev_dir = phase_dir

    print(f"\n{'=' * 60}")
    print("CURRICULUM V2 COMPLETE")
    print(f"Final model: {prev_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
