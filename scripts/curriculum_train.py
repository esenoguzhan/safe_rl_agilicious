#!/usr/bin/env python3
"""
Automated curriculum learning for quadrotor drone control.

Progresses through 4 stages, each advancing only when the policy meets
reward and episode-length thresholds:

  Stage 1 – Fixed goal, low motor lag (motor_tau ≈ 0.001)
  Stage 2 – Fixed goal, motor_tau linearly increased toward the real value
  Stage 3 – Randomized goals, fixed (real) motor_tau
  Stage 4 – Randomized goals + randomized mass & motor_tau

Progress is saved to <output_dir>/curriculum_state.yaml so interrupted runs
can be resumed with the same command.

Usage:
  python scripts/curriculum_train.py [--output_dir models/curriculum] [--seed 0]
"""
import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config_loader import prepare_env_run_dir, get_vec_env_config_string
from scripts.context import flightmare_context
from scripts.custom_reward_wrapper import CustomRewardWrapper
from scripts.env_wrapper import (
    FlightlibVecEnv,
    VecMaxEpisodeSteps,
    DomainRandomizationWrapper,
    ActionHistoryWrapper,
)
from scripts.record_episode_statistics import VecRecordEpisodeStatistics
from scripts.train import _get_QuadrotorEnv_v1, _pack_spawn_ranges
from stable_baselines3.common.evaluation import evaluate_policy

# ---------------------------------------------------------------------------
# Curriculum defaults — override via --config or CLI args
# ---------------------------------------------------------------------------

MOTOR_TAU = 0.033

REWARD_THRESHOLD = 850
LENGTH_THRESHOLD = 1000
PATIENCE = 5            # consecutive eval windows above threshold (legacy, unused)
CHECK_FREQ = 8192        # steps between training threshold checks (legacy, unused)
EVAL_PATIENCE = 5 #3       # consecutive eval passes above threshold before advancing

MAX_PHASE_TIMESTEPS = 3_000_000
EVAL_FREQ = 12_500
SAVE_INTERVAL = 100_000
LOG_STD_START = -1.0        # floor at stage entry (std ≈ 0.37)
LOG_STD_END   = -2.5        # floor at end of anneal (std ≈ 0.08)
LOG_STD_WARMUP = 0.10       # fraction of stage steps: hold start floor (explore)
LOG_STD_DECAY  = 0.40       # fraction of stage steps: linear ramp to end floor
EVAL_SEED = 7777            # fixed seed for reproducible eval episodes

# ---------------------------------------------------------------------------
# Env / PPO base config (matches drone_ppo_default.yaml structure)
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "env": {
        "max_episode_steps": 1000,
        "vec_env": {
            "seed": 1,
            "scene_id": 0,
            "num_envs": 8,
            "num_threads": 2,
            "render": False,
        },
        "quadrotor_env": {"sim_dt": 0.02, "max_t": 20.0},
        "quadrotor_dynamics": {
            "mass": 0.774,
            "arm_l": 0.125,
            "motor_omega_min": 150.0,
            "motor_omega_max": 2800.0,
            "motor_tau": 0.033,
            "thrust_map": [1.562522e-6, 0.0, 0.0],
            "kappa": 0.022,
            "omega_max": [10.0, 10.0, 4.0],
        },
        "motor_init": "zero",
        "goal_position": [0.0, 0.0, 5.0],
        "world_box": [-20, 20, -20, 20, 0, 20],
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
        "action_history_len": 5,
        "domain_randomization": {
            "enabled": False,
            "randomize_mass": False,
            "randomize_motor_tau": False,
            "randomize_goal": False,
            "mass_range": [0.65, 0.90],
            "motor_tau_range": [0.02, 0.05],
            "goal_pos_range": {"x": [-2.0, 2.0], "y": [-2.0, 2.0], "z": [2.0, 8.0]},
        },
        "custom_reward": {
            "enabled": True,
            "mode": "cauchy",
            "x_goal": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "act_goal": [0, 0, 0, 0],
            "rew_act_rate_weight": [0.005, 0.005, 0.005, 0.005],
            "cauchy_scale": 0.1,
            "rew_state_weight": [2, 2, 3, 0.2, 0.2, 0.2, 0.2, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "rew_act_weight": [0.001, 0.001, 0.001, 0.001],
            "rew_exponential": True,
        },
    },
    "ppo": {
        "learning_rate": 3.0e-5,
        "n_steps": 2048,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.001,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "log_std_init": -1.0,
        },
    },
    "training": {
        "normalize_obs": True,
        "normalize_reward": True,
        "seed": 0,
    },
}


# ===================================================================
# Phase definitions
# ===================================================================

def _build_phase_configs(
    motor_tau: float,
    max_timesteps: int,
    reward_thr: float,
    length_thr: float,
):
    """Return an ordered list of (name, config_patch) dicts for the full curriculum."""

    phases = []

    # -- Stage 1: fixed goal, real motor lag, easy spawn (small ori/vel so VecNorm learns variance) --
    phases.append({
        "name": "stage1_fixed_goal",
        "stage": 1,
        "description": f"Fixed goal, motor_tau={motor_tau}, easy spawn, ori_scale=0.05",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr,
        "length_threshold": length_thr,
        "eval_domain_rand": False,
        "n_eval_episodes": 20,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {"enabled": False},
                "spawn_ranges": {
                    "pos_x": [-1.0, 1.0],
                    "pos_y": [-1.0, 1.0],
                    "pos_z": [4.0, 6.0],
                    "vel_x": [-0.5, 0.5],
                    "vel_y": [-0.5, 0.5],
                    "vel_z": [-0.5, 0.5],
                    "ori_scale": 0.05,
                },
            },
        },
    })

    # -- Stage 2: gradually increase orientation, velocity, ang_vel -----
    ori_ramp = [
        # (ori_scale, spawn_range, vel_range, ang_vel_range)
        (0.30, 2.0, 1.5, 0.0),
        (0.40, 3.0, 1.5, 0.0),
        (0.50, 4.0, 2.0, 1.0),
        (0.70, 5.0, 2.0, 2.0),
        (1.00, 5.0, 2.0, 3.0),
    ]
    for ori_val, sp_r, vel_r, ang_r in ori_ramp:
        sp_z_lo = max(1.0, 4.0 - 3.0 * ori_val)
        sp_z_hi = min(10.0, 6.0 + 4.0 * ori_val)
        ang_cfg = {}
        if ang_r > 0:
            ang_cfg = {
                "ang_vel_x": [-ang_r, ang_r],
                "ang_vel_y": [-ang_r, ang_r],
                "ang_vel_z": [-ang_r, ang_r],
            }
        phases.append({
            "name": f"stage2_ori_{int(ori_val*100):03d}",
            "stage": 2,
            "description": (f"Fixed goal, motor_tau={motor_tau}, "
                            f"ori={ori_val:.2f}, vel=±{vel_r}, ang=±{ang_r}"),
            "max_timesteps": max_timesteps,
            "reward_threshold": reward_thr,
            "length_threshold": length_thr,
            "eval_domain_rand": False,
            "n_eval_episodes": 20,
            "config_patch": {
                "env": {
                    "quadrotor_dynamics": {"motor_tau": motor_tau},
                    "domain_randomization": {"enabled": False},
                    "spawn_ranges": {
                        "pos_x": [-sp_r, sp_r],
                        "pos_y": [-sp_r, sp_r],
                        "pos_z": [sp_z_lo, sp_z_hi],
                        "vel_x": [-vel_r, vel_r],
                        "vel_y": [-vel_r, vel_r],
                        "vel_z": [-vel_r, vel_r],
                        "ori_scale": float(ori_val),
                        **ang_cfg,
                    },
                },
            },
        })

    # -- Stage 3a: narrow random goals, full orientation ---------------
    phases.append({
        "name": "stage3a_narrow_goals",
        "stage": 3,
        "description": f"Narrow random goals, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr * 0.9,
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
                    "pos_x": [-5.0, 5.0],
                    "pos_y": [-5.0, 5.0],
                    "pos_z": [1.0, 10.0],
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

    # -- Stage 3b: medium random goals ---------------------------------
    phases.append({
        "name": "stage3b_medium_goals",
        "stage": 3,
        "description": f"Medium random goals, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr * 0.85,
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
                    "goal_pos_range": {"x": [-2.0, 2.0], "y": [-2.0, 2.0], "z": [2.0, 8.0]},
                },
                "spawn_ranges": {
                    "pos_x": [-5.0, 5.0],
                    "pos_y": [-5.0, 5.0],
                    "pos_z": [1.0, 10.0],
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

    # -- Stage 3c: wide random goals -----------------------------------
    phases.append({
        "name": "stage3c_wide_goals",
        "stage": 3,
        "description": f"Wide random goals, motor_tau={motor_tau}",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr * 0.8,
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
                    "goal_pos_range": {"x": [-5.0, 5.0], "y": [-5.0, 5.0], "z": [1.0, 10.0]},
                },
                "spawn_ranges": {
                    "pos_x": [-8.0, 8.0],
                    "pos_y": [-8.0, 8.0],
                    "pos_z": [1.0, 15.0],
                    "vel_x": [-3.0, 3.0],
                    "vel_y": [-3.0, 3.0],
                    "vel_z": [-3.0, 3.0],
                    "ang_vel_x": [-5.0, 5.0],
                    "ang_vel_y": [-5.0, 5.0],
                    "ang_vel_z": [-5.0, 5.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -- Stage 4a: mass randomization only -----------------------------
    phases.append({
        "name": "stage4a_mass_rand",
        "stage": 4,
        "description": "Wide goals + mass randomization",
        "max_timesteps": max_timesteps,
        "reward_threshold": reward_thr * 0.8,
        "length_threshold": length_thr * 0.9,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
        "config_patch": {
            "env": {
                "quadrotor_dynamics": {"motor_tau": motor_tau},
                "domain_randomization": {
                    "enabled": True,
                    "randomize_mass": True,
                    "randomize_motor_tau": False,
                    "randomize_goal": True,
                    "mass_range": [0.65, 0.90],
                    "goal_pos_range": {"x": [-5.0, 5.0], "y": [-5.0, 5.0], "z": [1.0, 10.0]},
                },
                "spawn_ranges": {
                    "pos_x": [-8.0, 8.0],
                    "pos_y": [-8.0, 8.0],
                    "pos_z": [1.0, 15.0],
                    "vel_x": [-3.0, 3.0],
                    "vel_y": [-3.0, 3.0],
                    "vel_z": [-3.0, 3.0],
                    "ang_vel_x": [-5.0, 5.0],
                    "ang_vel_y": [-5.0, 5.0],
                    "ang_vel_z": [-5.0, 5.0],
                    "ori_scale": 1.0,
                },
            },
        },
    })

    # -- Stage 4b: full domain randomization ---------------------------
    _full_dr_patch = {
        "env": {
            "quadrotor_dynamics": {"motor_tau": motor_tau},
            "domain_randomization": {
                "enabled": True,
                "randomize_mass": True,
                "randomize_motor_tau": True,
                "randomize_goal": True,
                "mass_range": [0.65, 0.90],
                "motor_tau_range": [0.02, 0.05],
                "goal_pos_range": {"x": [-5.0, 5.0], "y": [-5.0, 5.0], "z": [1.0, 10.0]},
            },
            "spawn_ranges": {
                "pos_x": [-8.0, 8.0],
                "pos_y": [-8.0, 8.0],
                "pos_z": [1.0, 15.0],
                "vel_x": [-3.0, 3.0],
                "vel_y": [-3.0, 3.0],
                "vel_z": [-3.0, 3.0],
                "ang_vel_x": [-5.0, 5.0],
                "ang_vel_y": [-5.0, 5.0],
                "ang_vel_z": [-5.0, 5.0],
                "ori_scale": 1.0,
            },
        },
    }
    phases.append({
        "name": "stage4b_full_domain_rand",
        "stage": 4,
        "description": "Wide goals + mass & motor_tau randomization",
        "reward_threshold": reward_thr * 0.75,
        "length_threshold": length_thr * 0.9,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
        "config_patch": copy.deepcopy(_full_dr_patch),
    })

    # -- Stage 5: precision fine-tuning with tighter anneal -------------
    phases.append({
        "name": "stage5_finetune",
        "stage": 5,
        "description": "Fine-tune: anneal log_std -1.5 → -3.0 for precision",
        "reward_threshold": reward_thr * 0.75,
        "length_threshold": length_thr * 0.9,
        "eval_reward_threshold": 600,
        "eval_length_threshold": 800,
        "eval_patience": 3,
        "eval_domain_rand": True,
        "n_eval_episodes": 20,
        "log_std_start": -1.5,
        "log_std_end": -3.0,
        "config_patch": copy.deepcopy(_full_dr_patch),
    })

    # -- Assign defaults: eval thresholds, max_timesteps, annealing ----
    _eval_defaults = {
        1: (700, 900),
        2: (600, 800),
        3: (400, 650),
        4: (250, 450),
        5: (600, 800),
    }
    _max_step_mult = {1: 1, 2: 1, 3: 2, 4: 3, 5: 2}
    _log_std_defaults = {
        1: (LOG_STD_START, LOG_STD_END),
        2: (LOG_STD_START, LOG_STD_END),
        3: (LOG_STD_START, -2.0),
        4: (LOG_STD_START, -1.5),
        5: (-1.5, -3.0),
    }
    for p in phases:
        s = p["stage"]
        rthr, lthr = _eval_defaults.get(s, (400, 650))
        ls_start, ls_end = _log_std_defaults.get(s, (LOG_STD_START, LOG_STD_END))
        p.setdefault("eval_reward_threshold", rthr)
        p.setdefault("eval_length_threshold", lthr)
        p.setdefault("eval_success_rate_threshold", 0.7)
        p.setdefault("eval_patience", EVAL_PATIENCE)
        p.setdefault("log_std_start", ls_start)
        p.setdefault("log_std_end", ls_end)
        p.setdefault("log_std_warmup", LOG_STD_WARMUP)
        p.setdefault("log_std_decay", LOG_STD_DECAY)
        p["max_timesteps"] = max_timesteps * _max_step_mult.get(s, 1)

    return phases


# ===================================================================
# Deep-merge helper
# ===================================================================

def _deep_merge(base, patch):
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# Obs layout: [pos_err(3), quat(4), lin_vel(3), ang_vel(3)] = 13; optional action history follows
VECNORM_MIN_VAR = 0.01  # minimum variance for all obs dims to avoid extreme scaling / normalization collapse
VECNORM_EPSILON = 1e-4  # epsilon in VecNormalize so std >= sqrt(epsilon) even when variance collapses


def _clamp_vecnorm_obs_variance(venv, min_var=VECNORM_MIN_VAR):
    """Clamp all observation variances to at least min_var to avoid extreme scaling and distribution shift.
    Call after loading VecNormalize from a previous stage (e.g. after ori_scale=0 or long hover phases)."""
    if not hasattr(venv, "obs_rms") or venv.obs_rms is None:
        return
    var = venv.obs_rms.var
    if var.size > 0:
        np.maximum(var, min_var, out=var)


# ===================================================================
# Env creation (mirrors train.py but self-contained)
# ===================================================================

_MOTOR_INIT_MODES = {"zero": 0, "hover": 1}


def make_env(cfg, num_envs_override=None):
    """Create FlightlibVecEnv from a config dict."""
    cfg = copy.deepcopy(cfg)
    if num_envs_override is not None:
        cfg["env"]["vec_env"]["num_envs"] = num_envs_override
        cfg["env"]["vec_env"]["num_threads"] = min(num_envs_override, 2)

    QuadrotorEnv_v1 = _get_QuadrotorEnv_v1()

    run_dir = prepare_env_run_dir(cfg)
    if run_dir:
        with flightmare_context(run_dir):
            impl = QuadrotorEnv_v1()
    else:
        vec_config_str = get_vec_env_config_string(cfg)
        impl = QuadrotorEnv_v1(vec_config_str, False)

    env_cfg = cfg.get("env", {})

    mode = _MOTOR_INIT_MODES.get(env_cfg.get("motor_init", "zero"), 0)
    impl.setMotorInitMode(mode)

    goal = env_cfg.get("goal_position")
    if goal is not None:
        goals = np.array([goal] * impl.getNumOfEnvs(), dtype=np.float32)
        impl.setEnvGoalPositions(goals)

    spawn = env_cfg.get("spawn_ranges")
    if spawn is not None:
        impl.setSpawnRanges(_pack_spawn_ranges(spawn))

    wb = env_cfg.get("world_box")
    if wb is not None:
        impl.setWorldBox(np.array(wb, dtype=np.float32))

    return FlightlibVecEnv(impl)


def wrap_env(env, cfg, *, add_domain_rand=True, add_record_stats=True):
    """Apply the standard wrapper stack on top of a raw FlightlibVecEnv."""
    env_cfg = cfg.get("env", {})

    dr_cfg = env_cfg.get("domain_randomization", {})
    if add_domain_rand and dr_cfg.get("enabled", False):
        env = DomainRandomizationWrapper(env, dr_cfg)

    cr_cfg = env_cfg.get("custom_reward")
    if cr_cfg and cr_cfg.get("enabled", False):
        env = CustomRewardWrapper(env, cr_cfg)

    max_steps = env_cfg.get("max_episode_steps")
    if max_steps is not None:
        env = VecMaxEpisodeSteps(env, max_steps)

    ahl = env_cfg.get("action_history_len", 0)
    if ahl > 0:
        env = ActionHistoryWrapper(env, ahl)

    if add_record_stats:
        env = VecRecordEpisodeStatistics(env, deque_size=100)

    return env


# ===================================================================
# Callbacks
# ===================================================================

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization  # noqa: E402


class ThresholdCallback(BaseCallback):
    """Stop training once reward and episode-length thresholds are met
    for *patience* consecutive checks."""

    def __init__(self, reward_thr, length_thr, check_freq=8192, patience=20,
                 verbose=1):
        super().__init__(verbose)
        self.reward_thr = reward_thr
        self.length_thr = length_thr
        self.check_freq = check_freq
        self.patience = patience
        self._streak = 0
        self.threshold_met = False

    def _on_step(self):
        if self.n_calls % self.check_freq != 0:
            return True

        ep_buf = self.model.ep_info_buffer
        if len(ep_buf) < 10:
            return True

        mean_r = np.mean([ep["r"] for ep in ep_buf])
        mean_l = np.mean([ep["l"] for ep in ep_buf])

        if mean_r >= self.reward_thr and mean_l >= self.length_thr:
            self._streak += 1
            if self.verbose:
                print(f"  [Curriculum] threshold check {self._streak}/{self.patience}: "
                      f"rew={mean_r:.1f} >= {self.reward_thr}, "
                      f"len={mean_l:.0f} >= {self.length_thr}")
        else:
            if self._streak > 0 and self.verbose:
                print(f"  [Curriculum] streak reset (rew={mean_r:.1f}, len={mean_l:.0f})")
            self._streak = 0

        if self._streak >= self.patience:
            self.threshold_met = True
            if self.verbose:
                print(f"  [Curriculum] THRESHOLD MET — advancing to next phase")
            return False
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """Sync VecNormalize stats from train -> eval env, and provide a
    helper to persist them alongside checkpoints/best models."""

    def __init__(self, save_path, eval_env, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_env = eval_env

    def _on_step(self):
        if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            sync_envs_normalization(self.training_env, self.eval_env)
        return True

    def save_vecnormalize(self, suffix=""):
        if isinstance(self.training_env, VecNormalize):
            fname = f"vecnormalize{suffix}.pkl"
            self.training_env.save(os.path.join(self.save_path, fname))


class CheckpointWithNormCallback(CheckpointCallback):
    """CheckpointCallback that also saves VecNormalize alongside each checkpoint."""

    def __init__(self, save_freq, save_path, name_prefix, vecnorm_cb):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)
        self._vecnorm_cb = vecnorm_cb

    def _on_step(self):
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0 and self._vecnorm_cb is not None:
            self._vecnorm_cb.save_vecnormalize(f"_{self.num_timesteps}_steps")
        return result


def _unwrap_to_flightlib(env):
    """Walk the wrapper chain to find the FlightlibVecEnv (has set_seed)."""
    cur = env
    while cur is not None:
        if hasattr(cur, "set_seed") and hasattr(cur, "_impl"):
            return cur
        cur = getattr(cur, "venv", None)
    return None


class EvalWithNormCallback(EvalCallback):
    """EvalCallback that saves VecNormalize alongside best_model.zip,
    reseeds both Python and C++ RNGs for reproducible evaluations, and
    checks eval-based thresholds (reward/length or success-rate) for curriculum advancement."""

    def __init__(self, eval_env, best_model_save_path, log_path,
                 eval_freq, n_eval_episodes, deterministic, vecnorm_cb,
                 eval_seed=None,
                 eval_reward_thr=None, eval_length_thr=None,
                 eval_patience=EVAL_PATIENCE,
                 eval_success_rate_thr=None):
        super().__init__(
            eval_env, best_model_save_path=best_model_save_path,
            log_path=log_path, eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes, deterministic=deterministic,
        )
        self._vecnorm_cb = vecnorm_cb
        self._eval_seed = eval_seed
        self._eval_reward_thr = eval_reward_thr
        self._eval_length_thr = eval_length_thr
        self._eval_patience = eval_patience
        self._eval_success_rate_thr = eval_success_rate_thr
        self._eval_streak = 0
        self.threshold_met = False
        self._base_eval_env = _unwrap_to_flightlib(eval_env)

    def _on_step(self):
        is_eval_step = (self.eval_freq > 0
                        and self.n_calls % self.eval_freq == 0)

        if is_eval_step and self._eval_seed is not None:
            np.random.seed(self._eval_seed)
            if self._base_eval_env is not None:
                self._base_eval_env.set_seed(self._eval_seed)

        if is_eval_step and self._eval_success_rate_thr is not None:
            # Run eval with per-episode returns to compute success rate (one eval run)
            episode_rewards, episode_lengths = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic, return_episode_rewards=True,
            )
            self.last_mean_reward = float(np.mean(episode_rewards))
            if not hasattr(self, "evaluations_length"):
                self.evaluations_length = []
            self.evaluations_length.append(episode_lengths)

            if self.last_mean_reward > self.best_mean_reward:
                self.best_mean_reward = self.last_mean_reward
                self.model.save(os.path.join(self.best_model_save_path, "best_model.zip"))
                if self._vecnorm_cb is not None:
                    self._vecnorm_cb.save_vecnormalize()

            # Success = episode reached length_thr and reward_thr
            episode_lengths = np.array(episode_lengths)
            episode_rewards = np.array(episode_rewards)
            success = ((episode_lengths >= self._eval_length_thr)
                       & (episode_rewards >= self._eval_reward_thr))
            success_rate = float(np.mean(success))

            cur_std = float(self.model.policy.log_std.exp().mean())
            if success_rate >= self._eval_success_rate_thr:
                self._eval_streak += 1
                print(f"  [Curriculum] EVAL pass {self._eval_streak}/"
                      f"{self._eval_patience}: success_rate={success_rate:.0%}>="
                      f"{self._eval_success_rate_thr:.0%} "
                      f"(mean_rew={self.last_mean_reward:.1f}, mean_len={np.mean(episode_lengths):.0f}, std={cur_std:.3f})")
            else:
                if self._eval_streak > 0:
                    print(f"  [Curriculum] EVAL streak reset "
                          f"(success_rate={success_rate:.0%}, "
                          f"mean_rew={self.last_mean_reward:.1f}, std={cur_std:.3f})")
                self._eval_streak = 0

            if self._eval_streak >= self._eval_patience:
                self.threshold_met = True
                print("  [Curriculum] EVAL THRESHOLD MET — advancing")
                return False
            return True

        prev_best = self.best_mean_reward
        result = super()._on_step()

        if self.best_mean_reward > prev_best and self._vecnorm_cb is not None:
            self._vecnorm_cb.save_vecnormalize()

        if is_eval_step and self._eval_reward_thr is not None and self._eval_success_rate_thr is None:
            mean_rew = self.last_mean_reward
            mean_len = 0.0
            if hasattr(self, "evaluations_length") and self.evaluations_length:
                mean_len = float(np.mean(self.evaluations_length[-1]))

            cur_std = float(self.model.policy.log_std.exp().mean())
            if (mean_rew >= self._eval_reward_thr
                    and mean_len >= self._eval_length_thr):
                self._eval_streak += 1
                print(f"  [Curriculum] EVAL pass {self._eval_streak}/"
                      f"{self._eval_patience}: eval_rew={mean_rew:.1f}>="
                      f"{self._eval_reward_thr}, eval_len={mean_len:.0f}>="
                      f"{self._eval_length_thr} (std={cur_std:.3f})")
            else:
                if self._eval_streak > 0:
                    print(f"  [Curriculum] EVAL streak reset "
                          f"(eval_rew={mean_rew:.1f}, "
                          f"eval_len={mean_len:.0f}, std={cur_std:.3f})")
                self._eval_streak = 0

            if self._eval_streak >= self._eval_patience:
                self.threshold_met = True
                print("  [Curriculum] EVAL THRESHOLD MET — advancing")
                return False

        return result


class ClampLogStdCallback(BaseCallback):
    """Anneal the log_std floor from ``log_std_start`` to ``log_std_end``
    over a fraction of the stage's training budget.

    Schedule (fraction of ``total_timesteps``):
        [0, warmup)          : floor = log_std_start   (explore new stage)
        [warmup, warmup+decay): linear ramp to log_std_end
        [warmup+decay, 1.0]  : floor = log_std_end     (exploit precise mean)
    """

    def __init__(self, log_std_start=-1.0, log_std_end=-2.5,
                 warmup_frac=0.10, decay_frac=0.40,
                 total_timesteps=3_000_000, verbose=0):
        super().__init__(verbose)
        self.log_std_start = log_std_start
        self.log_std_end = log_std_end
        self.warmup_frac = warmup_frac
        self.decay_frac = decay_frac
        self.total_timesteps = total_timesteps
        self._current_floor = log_std_start

    def _get_floor(self):
        progress = min(self.num_timesteps / max(self.total_timesteps, 1), 1.0)
        if progress < self.warmup_frac:
            return self.log_std_start
        anneal_end = self.warmup_frac + self.decay_frac
        if progress < anneal_end:
            t = (progress - self.warmup_frac) / max(self.decay_frac, 1e-8)
            return self.log_std_start + t * (self.log_std_end - self.log_std_start)
        return self.log_std_end

    def _on_rollout_start(self):
        self._current_floor = self._get_floor()
        with torch.no_grad():
            self.model.policy.log_std.clamp_(min=self._current_floor)

    def _on_step(self):
        return True


# ===================================================================
# Curriculum state persistence (for resuming interrupted runs)
# ===================================================================

def _state_path(output_dir):
    return os.path.join(output_dir, "curriculum_state.yaml")


def load_state(output_dir):
    p = _state_path(output_dir)
    if os.path.isfile(p):
        with open(p) as f:
            return yaml.safe_load(f)
    return {"completed_phases": [], "current_phase": None}


def save_state(output_dir, state):
    with open(_state_path(output_dir), "w") as f:
        yaml.dump(state, f, default_flow_style=False, sort_keys=False)


# ===================================================================
# Main training loop
# ===================================================================

def train_phase(phase, cfg, output_dir, seed, prev_model_dir=None):
    """Train a single curriculum phase.  Returns the phase output dir."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    phase_dir = os.path.join(output_dir, phase["name"])
    os.makedirs(phase_dir, exist_ok=True)

    cfg = copy.deepcopy(cfg)
    cfg["paths"] = {"log_dir": phase_dir, "save_dir": phase_dir}

    with open(os.path.join(phase_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    set_random_seed(seed)
    np.random.seed(seed)

    training_cfg = cfg.get("training", {})
    normalize_obs = training_cfg.get("normalize_obs", True)
    normalize_reward = training_cfg.get("normalize_reward", False)
    use_vecnorm = normalize_obs or normalize_reward

    vecnorm_pkl = None
    if prev_model_dir:
        p = os.path.join(prev_model_dir, "vecnormalize.pkl")
        if os.path.isfile(p):
            vecnorm_pkl = p

    # ---- training env ----
    train_raw = make_env(cfg)
    train_env = wrap_env(train_raw, cfg, add_domain_rand=True, add_record_stats=True)
    if use_vecnorm:
        if vecnorm_pkl:
            train_env = VecNormalize.load(vecnorm_pkl, train_env)
            train_env.training = True
            train_env.norm_reward = normalize_reward
            _clamp_vecnorm_obs_variance(train_env)
            print(f"  Loaded VecNormalize from {vecnorm_pkl} (all obs var clamped to >={VECNORM_MIN_VAR})")
        else:
            train_env = VecNormalize(
                train_env, norm_obs=normalize_obs,
                norm_reward=normalize_reward, clip_obs=10.0,
                epsilon=VECNORM_EPSILON,
            )

    # ---- eval env ----
    eval_domain_rand = phase.get("eval_domain_rand", False)
    eval_raw = make_env(cfg, num_envs_override=1)
    eval_env = wrap_env(eval_raw, cfg,
                        add_domain_rand=eval_domain_rand,
                        add_record_stats=False)
    if use_vecnorm:
        if vecnorm_pkl:
            eval_env = VecNormalize.load(vecnorm_pkl, eval_env)
            _clamp_vecnorm_obs_variance(eval_env)
        else:
            eval_env = VecNormalize(
                eval_env, norm_obs=normalize_obs,
                norm_reward=False, clip_obs=10.0,
                epsilon=VECNORM_EPSILON,
            )
        eval_env.training = False
        eval_env.norm_reward = False

    # ---- PPO model ----
    ppo_cfg = copy.deepcopy(cfg.get("ppo", {}))
    policy_kwargs = ppo_cfg.pop(
        "policy_kwargs",
        {"net_arch": dict(pi=[256, 256], vf=[256, 256]), "log_std_init": -1.0},
    )
    ppo_kwargs = {k: v for k, v in ppo_cfg.items()}

    model_zip = None
    if prev_model_dir:
        for name in ("best_model.zip", "ppo_drone_final.zip"):
            c = os.path.join(prev_model_dir, name)
            if os.path.isfile(c):
                model_zip = c
                break

    if model_zip:
        model = PPO.load(model_zip, env=train_env, seed=seed,
                         tensorboard_log=phase_dir, **ppo_kwargs)
        print(f"  Resumed model from {model_zip}")
    else:
        model = PPO(
            "MlpPolicy", train_env, verbose=1, seed=seed,
            tensorboard_log=phase_dir,
            policy_kwargs=policy_kwargs, **ppo_kwargs,
        )

    # ---- log_std annealing ----
    ls_start = phase.get("log_std_start", LOG_STD_START)
    ls_end = phase.get("log_std_end", LOG_STD_END)
    ls_warmup = phase.get("log_std_warmup", LOG_STD_WARMUP)
    ls_decay = phase.get("log_std_decay", LOG_STD_DECAY)
    max_ts = phase["max_timesteps"]

    with torch.no_grad():
        model.policy.log_std.fill_(ls_start)
    # Clear optimizer state for log_std so Adam momentum from previous stage doesn't pull it back
    log_std_param = model.policy.log_std
    if log_std_param in model.policy.optimizer.state:
        del model.policy.optimizer.state[log_std_param]
    print(f"  Reset log_std → {ls_start} (std={np.exp(ls_start):.3f}), cleared optimizer state for log_std")

    # ---- callbacks ----
    vecnorm_cb = SaveVecNormalizeCallback(phase_dir, eval_env)

    n_eval_episodes = phase.get("n_eval_episodes", 20)
    eval_seed = EVAL_SEED

    eval_cb = EvalWithNormCallback(
        eval_env,
        best_model_save_path=phase_dir,
        log_path=phase_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        vecnorm_cb=vecnorm_cb,
        eval_seed=eval_seed,
        eval_reward_thr=phase.get("eval_reward_threshold"),
        eval_length_thr=phase.get("eval_length_threshold"),
        eval_patience=phase.get("eval_patience", EVAL_PATIENCE),
        eval_success_rate_thr=phase.get("eval_success_rate_threshold"),
    )

    callbacks = [
        vecnorm_cb,
        ClampLogStdCallback(
            log_std_start=ls_start, log_std_end=ls_end,
            warmup_frac=ls_warmup, decay_frac=ls_decay,
            total_timesteps=max_ts,
        ),
        CheckpointWithNormCallback(
            save_freq=SAVE_INTERVAL,
            save_path=phase_dir,
            name_prefix="ppo_drone",
            vecnorm_cb=vecnorm_cb,
        ),
        eval_cb,
    ]

    # ---- train ----
    eval_rthr = phase.get("eval_reward_threshold", "N/A")
    eval_lthr = phase.get("eval_length_threshold", "N/A")
    print(f"  Training up to {max_ts:,} steps "
          f"(eval_rew_thr={eval_rthr}, eval_len_thr={eval_lthr})")
    print(f"  log_std anneal: {ls_start}→{ls_end} "
          f"(warmup {ls_warmup:.0%}, decay {ls_decay:.0%}, "
          f"hold last {1-ls_warmup-ls_decay:.0%})")
    t0 = time.time()
    model.learn(total_timesteps=max_ts, callback=callbacks, progress_bar=True)
    elapsed = time.time() - t0

    reason = "eval_threshold_met" if eval_cb.threshold_met else "max_timesteps"
    print(f"  Phase finished ({reason}) in {elapsed:.0f}s, "
          f"total_steps={model.num_timesteps}")

    model.save(os.path.join(phase_dir, "ppo_drone_final"))
    if use_vecnorm:
        vn_path = os.path.join(phase_dir, "vecnormalize.pkl")
        if not os.path.isfile(vn_path):
            train_env.save(vn_path)
        train_env.save(os.path.join(phase_dir, "vecnormalize_final.pkl"))

    train_env.close()
    eval_env.close()

    return phase_dir


def main():
    parser = argparse.ArgumentParser(
        description="Automated curriculum learning for quadrotor control",
    )
    parser.add_argument("--output_dir", type=str, default="models/curriculum",
                        help="Root directory for all phase outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motor_tau", type=float, default=MOTOR_TAU,
                        help="Motor time constant (real hardware value)")
    parser.add_argument("--max_phase_steps", type=int, default=MAX_PHASE_TIMESTEPS,
                        help="Max timesteps per phase before forced advance")
    parser.add_argument("--reward_threshold", type=float, default=REWARD_THRESHOLD,
                        help="Episode reward threshold for phase advancement")
    parser.add_argument("--length_threshold", type=float, default=LENGTH_THRESHOLD,
                        help="Episode length threshold for phase advancement")
    args = parser.parse_args()

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
    print("AUTOMATED CURRICULUM LEARNING")
    print("=" * 60)
    print(f"Output directory : {args.output_dir}")
    print(f"Seed             : {args.seed}")
    print(f"Motor tau        : {args.motor_tau}")
    print(f"Reward threshold : {args.reward_threshold} (training ref)")
    print(f"Length threshold : {args.length_threshold} (training ref)")
    print(f"Max steps/phase  : {args.max_phase_steps:,} (base, scaled per stage)")
    print(f"Eval patience    : {EVAL_PATIENCE} consecutive evals")
    print(f"log_std anneal   : {LOG_STD_START}→{LOG_STD_END} "
          f"(warmup {LOG_STD_WARMUP:.0%}, decay {LOG_STD_DECAY:.0%})")
    print(f"Total phases     : {len(phases)}")
    if completed:
        print(f"Already done     : {len(completed)} phases")
    print("=" * 60)

    prev_dir = None
    # find the latest completed phase directory to use as resume point
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
            phase, cfg, args.output_dir, args.seed, prev_model_dir=prev_dir,
        )

        completed.add(phase["name"])
        state["completed_phases"] = list(completed)
        state["current_phase"] = phase["name"]
        save_state(args.output_dir, state)

        prev_dir = phase_dir

    print(f"\n{'=' * 60}")
    print("CURRICULUM COMPLETE")
    print(f"Final model: {prev_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
