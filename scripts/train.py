#!/usr/bin/env python3
"""
PPO training for drone control using flightlib and Stable-Baselines3.
All parameters are configured via a single YAML file.
Creates a run folder under models/ named by training steps (e.g. PPO_2000000, PPO_2000000_2)
and saves best model, checkpoints, TensorBoard logs, and config there.
Usage: python train.py --config configs/drone_ppo_default.yaml [--seed 0]
"""
import argparse
import copy
import os
import sys

import numpy as np
import yaml

# Add project root for imports if needed
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config_loader import load_config, prepare_env_run_dir, get_vec_env_config_string
from scripts.context import flightmare_context
from scripts.custom_reward_wrapper import CustomRewardWrapper
from scripts.env_wrapper import FlightlibVecEnv, VecMaxEpisodeSteps
from scripts.record_episode_statistics import VecRecordEpisodeStatistics

MODELS_DIR = "models"


def get_next_ppo_steps_dir(total_timesteps):
    """Return models/PPO_<steps> or models/PPO_<steps>_2, etc., whichever does not exist yet."""
    base = os.path.join(MODELS_DIR, f"PPO_{total_timesteps}")
    run_dir = base
    i = 1
    while os.path.isdir(run_dir):
        i += 1
        run_dir = f"{base}_{i}"
    return run_dir


def _ensure_flightgym_path():
    """If flightgym/flightlib not found, add flightlib source and build dirs to sys.path (editable install)."""
    import glob
    flightlib_dir = os.path.join(_REPO_ROOT, "flightmare", "flightlib")
    if not os.path.isdir(flightlib_dir):
        return
    # Editable install often leaves .so in build/lib.* or build/temp.*
    build_dir = os.path.join(flightlib_dir, "build")
    for pattern in ["lib.*", "lib"]:
        for path in glob.glob(os.path.join(build_dir, pattern)):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
    if flightlib_dir not in sys.path:
        sys.path.insert(0, flightlib_dir)


def _get_QuadrotorEnv_v1():
    """Import QuadrotorEnv_v1 from flightgym or flightlib (build-dependent)."""
    try:
        from flightgym import QuadrotorEnv_v1
        return QuadrotorEnv_v1
    except ModuleNotFoundError:
        pass
    try:
        from flightlib import QuadrotorEnv_v1
        return QuadrotorEnv_v1
    except ModuleNotFoundError:
        pass
    _ensure_flightgym_path()
    try:
        from flightgym import QuadrotorEnv_v1
        return QuadrotorEnv_v1
    except ModuleNotFoundError:
        pass
    try:
        from flightlib import QuadrotorEnv_v1
        return QuadrotorEnv_v1
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Could not import 'flightgym' or 'flightlib'. Install the flightlib Python binding:\n"
            "  export FLIGHTMARE_PATH=/path/to/safe_rl_agilicious/flightmare\n"
            "  cd $FLIGHTMARE_PATH/flightlib && pip install .\n"
            "Use the same Python for pip and for running (e.g. python -m pip install .).\n"
            "See scripts/README.md for full setup."
        ) from None


def _make_env(cfg):
    """Create FlightlibVecEnv, optionally inside flightmare_context and with VecNormalize."""
    QuadrotorEnv_v1 = _get_QuadrotorEnv_v1()

    run_dir = prepare_env_run_dir(cfg)
    if run_dir:
        with flightmare_context(run_dir):
            impl = QuadrotorEnv_v1()
    else:
        vec_config_str = get_vec_env_config_string(cfg)
        impl = QuadrotorEnv_v1(vec_config_str, False)

    env = FlightlibVecEnv(impl)
    return env


def main():
    parser = argparse.ArgumentParser(description="PPO training for drone control (flightlib + SB3)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    total_timesteps = cfg.get("training", {}).get("total_timesteps", 100_000)
    run_dir = get_next_ppo_steps_dir(total_timesteps)
    os.makedirs(run_dir, exist_ok=True)
    paths = {**cfg.get("paths", {}), "log_dir": run_dir, "save_dir": run_dir}
    cfg["paths"] = paths

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print("Run directory:", run_dir)

    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 0)
    np.random.seed(seed)

    env = _make_env(cfg)
    env_cfg = cfg.get("env", {})
    custom_reward_cfg = env_cfg.get("custom_reward")
    if custom_reward_cfg and custom_reward_cfg.get("enabled", False):
        env = CustomRewardWrapper(env, custom_reward_cfg)
    max_episode_steps = env_cfg.get("max_episode_steps")
    if max_episode_steps is not None:
        env = VecMaxEpisodeSteps(env, max_episode_steps)
    training_cfg = cfg.get("training", {})
    normalize_obs = training_cfg.get("normalize_obs", True)
    normalize_reward = training_cfg.get("normalize_reward", False)
    record_deque_size = training_cfg.get("record_episode_statistics_deque_size", 100)
    env = VecRecordEpisodeStatistics(env, deque_size=record_deque_size)

    if normalize_obs or normalize_reward:
        from stable_baselines3.common.vec_env import VecNormalize
        env = VecNormalize(env, norm_obs=normalize_obs, norm_reward=normalize_reward, clip_obs=10.0)

    ppo_cfg = cfg.get("ppo", {})
    policy_kwargs = ppo_cfg.get("policy_kwargs") or {"net_arch": [dict(pi=[128, 128], vf=[128, 128])]}
    ppo_kwargs = {k: v for k, v in ppo_cfg.items() if k != "policy_kwargs"}

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)

    class SyncVecNormalizeCallback(BaseCallback):
        """Sync training env VecNormalize stats to eval env so evaluation uses same normalization."""

        def __init__(self, eval_env, verbose=0):
            super().__init__(verbose)
            self.eval_env = eval_env

        def _on_step(self):
            train_env = self.training_env
            if hasattr(train_env, "obs_rms") and train_env.obs_rms is not None and hasattr(self.eval_env, "obs_rms"):
                self.eval_env.obs_rms = train_env.obs_rms
            return True

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=seed,
        tensorboard_log=run_dir,
        policy_kwargs=policy_kwargs,
        **ppo_kwargs,
    )

    save_interval = training_cfg.get("save_interval", 50_000)
    eval_freq = training_cfg.get("eval_freq", 10_000)

    cfg_eval = copy.deepcopy(cfg)
    if "env" not in cfg_eval:
        cfg_eval["env"] = {}
    if "vec_env" not in cfg_eval["env"]:
        cfg_eval["env"]["vec_env"] = {}
    cfg_eval["env"]["vec_env"]["num_envs"] = 1
    cfg_eval["env"]["vec_env"]["num_threads"] = 1
    eval_env = _make_env(cfg_eval)
    if custom_reward_cfg and custom_reward_cfg.get("enabled", False):
        eval_env = CustomRewardWrapper(eval_env, custom_reward_cfg)
    if max_episode_steps is not None:
        eval_env = VecMaxEpisodeSteps(eval_env, max_episode_steps)
    if normalize_obs or normalize_reward:
        from stable_baselines3.common.vec_env import VecNormalize
        eval_env = VecNormalize(eval_env, norm_obs=normalize_obs, norm_reward=False, clip_obs=10.0)

    callbacks = [
        SyncVecNormalizeCallback(eval_env),
        CheckpointCallback(
            save_freq=save_interval,
            save_path=run_dir,
            name_prefix="ppo_drone",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=eval_freq,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(run_dir, "ppo_drone_final"))
    if normalize_obs or normalize_reward:
        env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    env.close()
    eval_env.close()
    print("Training finished. Best model, checkpoints, logs and TensorBoard in:", run_dir)


if __name__ == "__main__":
    main()
