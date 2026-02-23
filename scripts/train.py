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
from scripts.env_wrapper import FlightlibVecEnv, VecMaxEpisodeSteps, DomainRandomizationWrapper, ActionHistoryWrapper
from scripts.record_episode_statistics import VecRecordEpisodeStatistics

MODELS_DIR = "models"
EVAL_SEED = 7777  # fixed seed for reproducible eval episodes (match curriculum_train.py)
VECNORM_EPSILON = 1e-4  # epsilon in VecNormalize so std >= sqrt(epsilon) even when variance collapses
VECNORM_MIN_VAR = 0.01  # minimum variance for all obs dims when loading VecNormalize (avoids extreme scaling)


def _clamp_vecnorm_obs_variance(venv, min_var=VECNORM_MIN_VAR):
    """Clamp all observation variances to at least min_var. Call after VecNormalize.load."""
    if not hasattr(venv, "obs_rms") or venv.obs_rms is None:
        return
    var = venv.obs_rms.var
    if var.size > 0:
        np.maximum(var, min_var, out=var)


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


_MOTOR_INIT_MODES = {"zero": 0, "hover": 1}


def _resolve_resume_path(resume: str):
    """Return (run_dir, model_zip) from a --resume argument.

    Accepts either a directory (looks for best_model.zip inside) or a .zip path.
    """
    if os.path.isdir(resume):
        run_dir = resume
        for name in ("best_model.zip", "ppo_drone_final.zip"):
            candidate = os.path.join(run_dir, name)
            if os.path.isfile(candidate):
                return run_dir, candidate
        raise FileNotFoundError(
            f"No best_model.zip or ppo_drone_final.zip found in {run_dir}"
        )
    if resume.endswith(".zip") and os.path.isfile(resume):
        return os.path.dirname(resume), resume
    raise FileNotFoundError(f"Cannot resolve --resume path: {resume}")


def _pack_spawn_ranges(spawn_cfg):
    """Pack spawn_ranges YAML config into a flat 19-element float32 vector for C++."""
    def _r(key, default):
        return spawn_cfg.get(key, default)
    return np.array([
        _r("pos_x", [-1.0, 1.0])[0], _r("pos_x", [-1.0, 1.0])[1],
        _r("pos_y", [-1.0, 1.0])[0], _r("pos_y", [-1.0, 1.0])[1],
        _r("pos_z", [4.0, 6.0])[0],  _r("pos_z", [4.0, 6.0])[1],
        _r("vel_x", [-1.0, 1.0])[0], _r("vel_x", [-1.0, 1.0])[1],
        _r("vel_y", [-1.0, 1.0])[0], _r("vel_y", [-1.0, 1.0])[1],
        _r("vel_z", [-1.0, 1.0])[0], _r("vel_z", [-1.0, 1.0])[1],
        _r("ang_vel_x", [0.0, 0.0])[0], _r("ang_vel_x", [0.0, 0.0])[1],
        _r("ang_vel_y", [0.0, 0.0])[0], _r("ang_vel_y", [0.0, 0.0])[1],
        _r("ang_vel_z", [0.0, 0.0])[0], _r("ang_vel_z", [0.0, 0.0])[1],
        _r("ori_scale", 1.0),
    ], dtype=np.float32)


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

    motor_init = cfg.get("env", {}).get("motor_init", "zero")
    mode = _MOTOR_INIT_MODES.get(motor_init, 0)
    impl.setMotorInitMode(mode)

    goal_pos = cfg.get("env", {}).get("goal_position")
    if goal_pos is not None:
        goals = np.array([[goal_pos[0], goal_pos[1], goal_pos[2]]] * impl.getNumOfEnvs(),
                         dtype=np.float32)
        impl.setEnvGoalPositions(goals)

    spawn_cfg = cfg.get("env", {}).get("spawn_ranges")
    if spawn_cfg is not None:
        impl.setSpawnRanges(_pack_spawn_ranges(spawn_cfg))

    world_box = cfg.get("env", {}).get("world_box")
    if world_box is not None:
        impl.setWorldBox(np.array(world_box, dtype=np.float32))

    env = FlightlibVecEnv(impl)
    return env


def main():
    parser = argparse.ArgumentParser(description="PPO training for drone control (flightlib + SB3)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from a previous run directory or model .zip path. "
             "Loads policy weights and VecNormalize stats from that run.",
    )
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

    domain_rand_cfg = env_cfg.get("domain_randomization", {})
    if domain_rand_cfg.get("enabled", False):
        env = DomainRandomizationWrapper(env, domain_rand_cfg)
        active = []
        if domain_rand_cfg.get("randomize_mass", False):
            active.append(f"mass={domain_rand_cfg.get('mass_range')}")
        if domain_rand_cfg.get("randomize_motor_tau", False):
            active.append(f"motor_tau={domain_rand_cfg.get('motor_tau_range')}")
        if domain_rand_cfg.get("randomize_goal", False):
            active.append(f"goal_pos={domain_rand_cfg.get('goal_pos_range')}")
        print(f"Domain randomization enabled: {', '.join(active) or 'none'}")

    custom_reward_cfg = env_cfg.get("custom_reward")
    if custom_reward_cfg and custom_reward_cfg.get("enabled", False):
        env = CustomRewardWrapper(env, custom_reward_cfg)
        print(f"Custom reward enabled, mode: {custom_reward_cfg.get('mode', 'weighted_exp')}")
    max_episode_steps = env_cfg.get("max_episode_steps")
    if max_episode_steps is not None:
        env = VecMaxEpisodeSteps(env, max_episode_steps)

    action_history_len = env_cfg.get("action_history_len", 0)
    if action_history_len > 0:
        env = ActionHistoryWrapper(env, action_history_len)
        print(f"Action history enabled: last {action_history_len} actions appended to obs "
              f"(obs_dim: {env.observation_space.shape[0]})")

    training_cfg = cfg.get("training", {})
    normalize_obs = training_cfg.get("normalize_obs", True)
    normalize_reward = training_cfg.get("normalize_reward", False)
    record_deque_size = training_cfg.get("record_episode_statistics_deque_size", 100)
    env = VecRecordEpisodeStatistics(env, deque_size=record_deque_size)

    ppo_cfg = cfg.get("ppo", {})
    policy_kwargs = ppo_cfg.get("policy_kwargs") or {"net_arch": dict(pi=[128, 128], vf=[128, 128])}
    ppo_kwargs = {k: v for k, v in ppo_cfg.items() if k != "policy_kwargs"}

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

    set_random_seed(seed)

    use_vecnorm = normalize_obs or normalize_reward

    class SaveVecNormalizeCallback(BaseCallback):
        """Save VecNormalize stats alongside checkpoints and best model, and sync to eval env."""

        def __init__(self, save_path, eval_env, verbose=0):
            super().__init__(verbose)
            self.save_path = save_path
            self.eval_env = eval_env

        def _on_step(self):
            if use_vecnorm:
                sync_envs_normalization(self.training_env, self.eval_env)
            return True

        def save_vecnormalize(self, suffix=""):
            if not use_vecnorm:
                return
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
        """EvalCallback that saves VecNormalize alongside best_model.zip.
        Uses fixed eval seed (np.random + C++ env) for reproducible evaluations."""

        def __init__(self, eval_env, best_model_save_path, log_path,
                     eval_freq, n_eval_episodes, deterministic, vecnorm_cb,
                     eval_seed=EVAL_SEED):
            super().__init__(
                eval_env, best_model_save_path=best_model_save_path,
                log_path=log_path, eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes, deterministic=deterministic,
            )
            self._vecnorm_cb = vecnorm_cb
            self._eval_seed = eval_seed
            self._base_eval_env = _unwrap_to_flightlib(eval_env)

        def _on_step(self):
            is_eval_step = (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0)
            if is_eval_step and self._eval_seed is not None:
                np.random.seed(self._eval_seed)
                if self._base_eval_env is not None:
                    self._base_eval_env.set_seed(self._eval_seed)
            prev_best = self.best_mean_reward
            result = super()._on_step()
            if self.best_mean_reward > prev_best and self._vecnorm_cb is not None:
                self._vecnorm_cb.save_vecnormalize()
            return result

    # ------------------------------------------------------------------
    # Resume from previous run or create fresh model
    # ------------------------------------------------------------------
    resume_path = args.resume
    vecnorm_pkl_loaded = None

    if resume_path:
        resume_dir, model_zip = _resolve_resume_path(resume_path)
        vecnorm_pkl = os.path.join(resume_dir, "vecnormalize.pkl")

        if use_vecnorm and os.path.isfile(vecnorm_pkl):
            vecnorm_pkl_loaded = vecnorm_pkl
            env = VecNormalize.load(vecnorm_pkl, env)
            _clamp_vecnorm_obs_variance(env)
            env.training = True
            env.norm_reward = normalize_reward
            print(f"Resumed VecNormalize stats from {vecnorm_pkl} (all obs var clamped to >={VECNORM_MIN_VAR})")
        elif use_vecnorm:
            env = VecNormalize(env, norm_obs=normalize_obs, norm_reward=normalize_reward, clip_obs=10.0, epsilon=VECNORM_EPSILON)
            print("Warning: --resume specified but no vecnormalize.pkl found; starting fresh normalization")

        model = PPO.load(model_zip, env=env, seed=seed, tensorboard_log=run_dir,
                         **ppo_kwargs)
        print(f"Resumed model from {model_zip}")
    else:
        if use_vecnorm:
            env = VecNormalize(env, norm_obs=normalize_obs, norm_reward=normalize_reward, clip_obs=10.0, epsilon=VECNORM_EPSILON)

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
    if domain_rand_cfg.get("enabled", False) and domain_rand_cfg.get("randomize_goal", False):
        eval_env = DomainRandomizationWrapper(eval_env, domain_rand_cfg)
    if custom_reward_cfg and custom_reward_cfg.get("enabled", False):
        eval_env = CustomRewardWrapper(eval_env, custom_reward_cfg)
    if max_episode_steps is not None:
        eval_env = VecMaxEpisodeSteps(eval_env, max_episode_steps)
    if action_history_len > 0:
        eval_env = ActionHistoryWrapper(eval_env, action_history_len)
    if use_vecnorm:
        if vecnorm_pkl_loaded:
            eval_env = VecNormalize.load(vecnorm_pkl_loaded, eval_env)
            _clamp_vecnorm_obs_variance(eval_env)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            eval_env = VecNormalize(eval_env, norm_obs=normalize_obs, norm_reward=False, clip_obs=10.0, epsilon=VECNORM_EPSILON)

    vecnorm_cb = SaveVecNormalizeCallback(run_dir, eval_env)

    eval_cfg = cfg.get("evaluation", {})
    n_eval_episodes = eval_cfg.get("n_episodes", 5)

    callbacks = [
        vecnorm_cb,
        CheckpointWithNormCallback(
            save_freq=save_interval,
            save_path=run_dir,
            name_prefix="ppo_drone",
            vecnorm_cb=vecnorm_cb,
        ),
        EvalWithNormCallback(
            eval_env,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            vecnorm_cb=vecnorm_cb,
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(run_dir, "ppo_drone_final"))
    if use_vecnorm:
        env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    env.close()
    eval_env.close()
    print("Training finished. Best model, checkpoints, logs and TensorBoard in:", run_dir)


if __name__ == "__main__":
    main()
