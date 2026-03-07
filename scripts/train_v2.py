#!/usr/bin/env python3
"""
PPO training v2: observation noise and three-layer seed strategy.

Seed strategy:
  Layer        | Seed strategy              | Purpose
  -------------|----------------------------|----------------------------------------------------------
  Training     | Varying (e.g. --seed 1,2,3)| Robustness; avoid "getting lucky" with one initialization.
  Validation   | Fixed (EVAL_SEED=7777)     | Reproducible benchmark during training (same n scenarios).
  Test/Final   | Multiple fixed seeds       | Final report: mean ± std over 5–10 seeds (e.g. 100,200,...).

- Observation noise: Gaussian on position (obs 0:3) and velocity (obs 7:10). Config: env.observation_noise.
- Training env reseed: every N episode terminations (config training.reseed_every_n_episodes, default 1; 0=off). Use --reseed-every-n-episodes to override.
- Use --seed N for training (vary N across runs). Use --no-final-eval to skip the multi-seed final test.

Usage: python train_v2.py --config configs/drone_ppo_default.yaml [--seed 0]
"""
import argparse
import copy
import os
import sys

import numpy as np
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config_loader import load_config, prepare_env_run_dir, get_vec_env_config_string
from scripts.context import flightmare_context
from scripts.custom_reward_wrapper import CustomRewardWrapper
from scripts.env_wrapper import (
    FlightlibVecEnv,
    VecMaxEpisodeSteps,
    DomainRandomizationWrapper,
    ObservationNoiseWrapper,
    ActionHistoryWrapper,
)
from scripts.record_episode_statistics import VecRecordEpisodeStatistics

MODELS_DIR = "models"
EVAL_SEED = 7777  # Validation: fixed benchmark seed so every eval uses the same n scenarios (comparable metrics).
FINAL_EVAL_SEEDS = [100, 200, 300, 400, 500]  # Test/Final: multiple fixed seeds for mean ± std report.
OBS_NOISE_POS_DEFAULT = 0.01
OBS_NOISE_VEL_DEFAULT = 0.05
VECNORM_EPSILON = 1e-4
VECNORM_MIN_VAR = 0.01


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
            "Could not import 'flightgym' or 'flightlib'. Install the flightlib Python binding.\n"
            "See scripts/README.md for full setup."
        ) from None


_MOTOR_INIT_MODES = {"zero": 0, "hover": 1}


def _resolve_resume_path(resume: str):
    """Return (run_dir, model_zip) from a --resume argument."""
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
    """Create FlightlibVecEnv, optionally inside flightmare_context."""
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


def _wrap_env(env, cfg, *, add_obs_noise=True):
    """Apply domain rand, observation noise, custom reward, max steps, action history, record stats."""
    env_cfg = cfg.get("env", {})

    domain_rand_cfg = env_cfg.get("domain_randomization", {})
    if domain_rand_cfg.get("enabled", False):
        env = DomainRandomizationWrapper(env, domain_rand_cfg)

    if add_obs_noise:
        obs_noise_cfg = env_cfg.get("observation_noise")
        if isinstance(obs_noise_cfg, dict) and (
            obs_noise_cfg.get("position", 0) > 0 or obs_noise_cfg.get("velocity", 0) > 0
        ):
            env = ObservationNoiseWrapper(env, obs_noise_cfg)
            print(f"Observation noise: position={obs_noise_cfg.get('position', 0)}, velocity={obs_noise_cfg.get('velocity', 0)}")
        else:
            # Default noise when not in config
            default_noise = {"position": OBS_NOISE_POS_DEFAULT, "velocity": OBS_NOISE_VEL_DEFAULT}
            env = ObservationNoiseWrapper(env, default_noise)
            print(f"Observation noise (default): position={OBS_NOISE_POS_DEFAULT}, velocity={OBS_NOISE_VEL_DEFAULT}")

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
        print(f"Action history enabled: last {action_history_len} actions (obs_dim: {env.observation_space.shape[0]})")

    return env


def _unwrap_to_flightlib(env):
    """Walk the wrapper chain to find the FlightlibVecEnv (has set_seed)."""
    cur = env
    while cur is not None:
        if hasattr(cur, "set_seed") and hasattr(cur, "_impl"):
            return cur
        cur = getattr(cur, "venv", None)
    return None


class EpisodeReseedWrapper:
    """VecEnv wrapper: count episode terminations and reseed the base env every N episodes.
    Use reseed_every_n_episodes=1 for a new seed every episode; 0 to disable."""
    def __init__(self, venv, run_seed, reseed_every_n_episodes):
        self.venv = venv
        self._run_seed = run_seed
        self._reseed_every = int(reseed_every_n_episodes)
        self._episodes_finished = 0
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        n = getattr(venv, "num_envs", getattr(venv, "_num_envs", None))
        self.num_envs = self._num_envs = n

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._episodes_finished += int(np.sum(dones))
        if self._reseed_every > 0 and self._episodes_finished > 0 and self._episodes_finished % self._reseed_every == 0:
            base = _unwrap_to_flightlib(self)
            if base is not None:
                new_seed = self._run_seed + self._episodes_finished
                np.random.seed(new_seed)
                base.set_seed(new_seed)
        return obs, rewards, dones, infos

    def reset(self):
        return self.venv.reset()

    def seed(self, seed=None):
        """SB3 BaseVecEnv calls env.seed(seed); forward to inner env."""
        if hasattr(self.venv, "seed"):
            return self.venv.seed(seed)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="PPO training v2: observation noise + eval seed randomization (flightlib + SB3)"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from a previous run directory or model .zip path.",
    )
    parser.add_argument(
        "--no-obs-noise",
        action="store_true",
        help="Disable observation noise wrapper (default: enabled with config or defaults)",
    )
    parser.add_argument(
        "--final-eval-seeds",
        type=str,
        default=None,
        help="Comma-separated test seeds for final report, e.g. '100,200,300,400,500' (default: 5 seeds)",
    )
    parser.add_argument(
        "--no-final-eval",
        action="store_true",
        help="Skip multi-seed final evaluation after training",
    )
    parser.add_argument(
        "--reseed-every-n-episodes",
        type=int,
        default=None,
        help="Reseed training env every N episode terminations (default: from config, 1=every episode; 0=disabled)",
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
    # Training layer: use run seed so different runs (--seed 1, 2, 3, ...) see different scenarios.
    if "env" not in cfg:
        cfg["env"] = {}
    if "vec_env" not in cfg["env"]:
        cfg["env"]["vec_env"] = {}
    cfg["env"]["vec_env"]["seed"] = seed

    env = _make_env(cfg)
    env_cfg = cfg.get("env", {})

    domain_rand_cfg = env_cfg.get("domain_randomization", {})
    if domain_rand_cfg.get("enabled", False):
        active = []
        if domain_rand_cfg.get("randomize_mass", False):
            active.append(f"mass={domain_rand_cfg.get('mass_range')}")
        if domain_rand_cfg.get("randomize_motor_tau", False):
            active.append(f"motor_tau={domain_rand_cfg.get('motor_tau_range')}")
        if domain_rand_cfg.get("randomize_goal", False):
            active.append(f"goal_pos={domain_rand_cfg.get('goal_pos_range')}")
        print(f"Domain randomization enabled: {', '.join(active) or 'none'}")

    add_obs_noise = not args.no_obs_noise
    env = _wrap_env(env, cfg, add_obs_noise=add_obs_noise)

    training_cfg = cfg.get("training", {})
    normalize_obs = training_cfg.get("normalize_obs", True)
    normalize_reward = training_cfg.get("normalize_reward", False)
    reseed_every_n_episodes = args.reseed_every_n_episodes if args.reseed_every_n_episodes is not None else training_cfg.get("reseed_every_n_episodes", 1)
    if reseed_every_n_episodes > 0:
        env = EpisodeReseedWrapper(env, seed, reseed_every_n_episodes)
        print(f"Training env: reseed every {reseed_every_n_episodes} episode(s) (run_seed + episode_count)")
    record_deque_size = training_cfg.get("record_episode_statistics_deque_size", 100)
    env = VecRecordEpisodeStatistics(env, deque_size=record_deque_size)

    ppo_cfg = cfg.get("ppo", {})
    policy_kwargs = ppo_cfg.get("policy_kwargs") or {"net_arch": dict(pi=[128, 128], vf=[128, 128])}
    ppo_kwargs = {k: v for k, v in ppo_cfg.items() if k != "policy_kwargs"}

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
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

    class EvalWithNormCallback(EvalCallback):
        """EvalCallback that saves VecNormalize alongside best_model.zip.
        Validation layer: uses fixed eval_seed (e.g. 7777) for a consistent benchmark every eval."""

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
    # Resume or fresh model
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
            print(f"Resumed VecNormalize stats from {vecnorm_pkl} (obs var clamped to >={VECNORM_MIN_VAR})")
        elif use_vecnorm:
            env = VecNormalize(env, norm_obs=normalize_obs, norm_reward=normalize_reward, clip_obs=10.0, epsilon=VECNORM_EPSILON)
            print("Warning: --resume specified but no vecnormalize.pkl found; starting fresh normalization")

        model = PPO.load(model_zip, env=env, seed=seed, tensorboard_log=run_dir, **ppo_kwargs)
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
    eval_env = _wrap_env(eval_env, cfg_eval, add_obs_noise=add_obs_noise)

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
            eval_seed=EVAL_SEED,
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(run_dir, "ppo_drone_final"))
    if use_vecnorm:
        env.save(os.path.join(run_dir, "vecnormalize.pkl"))

    # Test/Final layer: evaluate on multiple fixed seeds for mean ± std report.
    if not args.no_final_eval:
        final_seeds = FINAL_EVAL_SEEDS
        if args.final_eval_seeds:
            final_seeds = [int(s.strip()) for s in args.final_eval_seeds.split(",")]
        base_eval = _unwrap_to_flightlib(eval_env)
        mean_rewards = []
        for s in final_seeds:
            np.random.seed(s)
            if base_eval is not None:
                base_eval.set_seed(s)
            mean_r, _ = evaluate_policy(
                model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
            )
            mean_rewards.append(mean_r)
        mean_rewards = np.array(mean_rewards)
        print("\nFinal evaluation (test layer, multiple fixed seeds):")
        for s, r in zip(final_seeds, mean_rewards):
            print(f"  Seed {s}: mean_reward = {r:.2f}")
        print(f"  Across seeds: mean = {mean_rewards.mean():.2f} ± {mean_rewards.std():.2f}")
        results_path = os.path.join(run_dir, "final_eval_results.txt")
        with open(results_path, "w") as f:
            f.write("seed,mean_reward\n")
            for s, r in zip(final_seeds, mean_rewards):
                f.write(f"{s},{r}\n")
            f.write(f"\nmean,std\n{mean_rewards.mean()},{mean_rewards.std()}\n")
        print(f"  Results written to {results_path}")

    env.close()
    eval_env.close()
    print("Training finished. Best model, checkpoints, logs and TensorBoard in:", run_dir)


if __name__ == "__main__":
    main()
