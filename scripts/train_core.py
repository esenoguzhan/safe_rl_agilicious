#!/usr/bin/env python3
"""
PPO trainer core: success-based LR/entropy schedule and early stopping.

Reused by train.py (decimated trainer) via imports.

Success (per episode): full length (1000 steps) and undiscounted return > 950.

- Learning rate and entropy coefficient follow eval success bands (see
  v3_lr_ent_for_success_rate): discovery [0,25%) lr=5e-4 ent=0.01; [25%,50%) lr=3e-4
  ent=0.008; [50%,75%) lr=1e-4 ent=0.005; [75%,100%] lr=1e-4 ent=0.001 (fine-tuning).
  Fresh runs start at discovery; each success-schedule eval updates lr/ent from the
  current eval success rate.
- Training stops as soon as three consecutive evals each have success rate >= 80%.

Also logs train/success_rate_recent (training deque).

Usage: python scripts/train_core.py --config configs/ppo_config_base.yaml [--seed 0]
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
from scripts.env_wrapper import (
    FlightlibVecEnv,
    VecMaxEpisodeSteps,
    DomainRandomizationWrapper,
    ActionHistoryWrapper,
    ObservationNoiseWrapper,
)
from scripts.record_episode_statistics import VecRecordEpisodeStatistics

MODELS_DIR = "models"
EVAL_SEED = 7777  # fixed seed for reproducible eval episodes (match curriculum_train.py)
VECNORM_EPSILON = 1e-4  # epsilon in VecNormalize so std >= sqrt(epsilon) even when variance collapses
VECNORM_MIN_VAR = 0.01  # minimum variance for all obs dims when loading VecNormalize (avoids extreme scaling)

# Episode success for logging and train_core (full horizon + return threshold)
SUCCESS_EPISODE_LENGTH = 1000
SUCCESS_MIN_REWARD = 1860.0
SUCCESS_STOP_THRESHOLD = 0.80  # early stop after 3 consecutive evals at/above this rate

# PPO lr / ent_coef by eval success rate band (train_core success-schedule evals)
V3_LR_ENT_DISCOVERY = (5e-4, 0.01)


def v3_lr_ent_for_success_rate(rate: float) -> tuple:
    """Map eval success rate in [0,1] to (learning_rate, ent_coef)."""
    r = float(rate)
    if r < 0.25:
        return (5e-4, 0.01)
    if r < 0.50:
        return (3e-4, 0.005)
    if r < 0.75:
        return (1e-4, 0.002)
    return (1e-4, 0.001)


def episode_success(ep_length, ep_return):
    """True if the episode ran full length and exceeded the return threshold."""
    return int(ep_length) == SUCCESS_EPISODE_LENGTH and float(ep_return) > SUCCESS_MIN_REWARD


def unwrap_vec_record_episode_statistics(venv):
    """Return VecRecordEpisodeStatistics if present under VecNormalize / wrapper chain."""
    seen = set()
    cur = venv
    for _ in range(32):
        if cur is None or id(cur) in seen:
            break
        seen.add(id(cur))
        if cur.__class__.__name__ == "VecRecordEpisodeStatistics":
            return cur
        cur = getattr(cur, "venv", None)
    return None


def _clamp_vecnorm_obs_variance(venv, min_var=VECNORM_MIN_VAR):
    """Clamp all observation variances to at least min_var. Call after VecNormalize.load."""
    if not hasattr(venv, "obs_rms") or venv.obs_rms is None:
        return
    var = venv.obs_rms.var
    if var.size > 0:
        np.maximum(var, min_var, out=var)


def _sanitize_run_name_suffix(name: str) -> str:
    """Allow safe folder suffix: alnum, dot, underscore, hyphen."""
    if not name:
        return ""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name))


def get_next_ppo_steps_dir(total_timesteps, name_suffix=None):
    """Return models/PPO_<steps>[_<suffix>] or ..._2, etc., whichever does not exist yet."""
    if name_suffix:
        suf = _sanitize_run_name_suffix(name_suffix)
        base = os.path.join(MODELS_DIR, f"PPO_{total_timesteps}_{suf}") if suf else os.path.join(MODELS_DIR, f"PPO_{total_timesteps}")
    else:
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
        "--run-name",
        type=str,
        default=None,
        help="Optional label: save under models/PPO_<total_timesteps>_<run_name> (still gets _2, _3 if folder exists).",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from a previous run directory or model .zip path. "
             "Loads policy weights and VecNormalize stats from that run.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    total_timesteps = cfg.get("training", {}).get("total_timesteps", 100_000)
    run_dir = get_next_ppo_steps_dir(total_timesteps, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    paths = {**cfg.get("paths", {}), "log_dir": run_dir, "save_dir": run_dir}
    cfg["paths"] = paths

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print("Run directory:", run_dir)
    print(
        "[train_core] Success = episode length 1000 and return > 950; "
        "lr/ent by eval band: [0,25%) 5e-4/0.01, [25%,50%) 3e-4/0.008, "
        "[50%,75%) 1e-4/0.005, [75%,1] 1e-4/0.001; "
        "stop after 3 consecutive evals with success ≥80%."
    )

    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 0)
    np.random.seed(seed)

    env = _make_env(cfg)
    env_cfg = cfg.get("env", {})

    # Wrapper order matches curriculum_train.wrap_env (inner → outer):
    #   FlightlibVecEnv → ObservationNoise → CustomReward → VecMaxEpisodeSteps
    #   → DomainRandomization → ActionHistory → VecRecordEpisodeStatistics → VecNormalize
    # DomainRandomizationWrapper MUST be outside VecMaxEpisodeSteps (goal/mass/tau on any done).

    obs_noise_cfg = env_cfg.get("observation_noise")
    if isinstance(obs_noise_cfg, dict) and (
        obs_noise_cfg.get("position", 0) > 0 or obs_noise_cfg.get("velocity", 0) > 0
    ):
        env = ObservationNoiseWrapper(env, obs_noise_cfg)
        print(
            f"Observation noise enabled: pos σ={obs_noise_cfg.get('position', 0)}, "
            f"vel σ={obs_noise_cfg.get('velocity', 0)}"
        )

    custom_reward_cfg = env_cfg.get("custom_reward")
    if custom_reward_cfg and custom_reward_cfg.get("enabled", False):
        env = CustomRewardWrapper(env, custom_reward_cfg)
        print(f"Custom reward enabled, mode: {custom_reward_cfg.get('mode', 'weighted_exp')}")

    max_episode_steps = env_cfg.get("max_episode_steps")
    if max_episode_steps is not None:
        env = VecMaxEpisodeSteps(env, max_episode_steps)

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
    # Discovery phase hyperparameters (YAML lr/ent_coef overridden for this schedule)
    if not args.resume:
        ppo_kwargs["learning_rate"] = V3_LR_ENT_DISCOVERY[0]
        ppo_kwargs["ent_coef"] = V3_LR_ENT_DISCOVERY[1]

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
    from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
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
        """EvalCallback that saves best_model.zip + vecnormalize.pkl ONLY when the eval
        success rate (per ``episode_success``) strictly improves over the best so far.

        Mean-reward improvement is intentionally ignored as a save trigger; the parent's
        automatic best-model saving is disabled and replaced with success-rate gating.
        Uses a fixed eval seed (np.random + C++ env) for reproducible evaluations.
        """

        def __init__(self, eval_env, best_model_save_path, log_path,
                     eval_freq, n_eval_episodes, deterministic, vecnorm_cb,
                     eval_seed=EVAL_SEED):
            super().__init__(
                eval_env, best_model_save_path=None,
                log_path=log_path, eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes, deterministic=deterministic,
            )
            self._best_save_path = best_model_save_path
            self._vecnorm_cb = vecnorm_cb
            self._eval_seed = eval_seed
            self._base_eval_env = _unwrap_to_flightlib(eval_env)
            self.best_success_rate = -1.0
            self.last_eval_success_rate = None  # set after each EvalCallback rollout (for schedule)
            if self._best_save_path is not None:
                os.makedirs(self._best_save_path, exist_ok=True)

        def _on_step(self):
            self.last_eval_success_rate = None
            is_eval_step = (self.eval_freq > 0 and self.n_calls % self.eval_freq == 0)
            if is_eval_step and self._eval_seed is not None:
                np.random.seed(self._eval_seed)
                if self._base_eval_env is not None:
                    self._base_eval_env.set_seed(self._eval_seed)
            result = super()._on_step()
            if is_eval_step and self.evaluations_results and self.evaluations_length:
                ep_rewards = self.evaluations_results[-1]
                ep_lengths = self.evaluations_length[-1]
                n = max(len(ep_rewards), 1)
                successes = sum(
                    1 for L, R in zip(ep_lengths, ep_rewards) if episode_success(L, R)
                )
                success_rate = successes / float(n)
                self.last_eval_success_rate = success_rate
                if success_rate > self.best_success_rate:
                    prev = self.best_success_rate
                    self.best_success_rate = success_rate
                    if self._best_save_path is not None:
                        self.model.save(os.path.join(self._best_save_path, "best_model"))
                    if self._vecnorm_cb is not None:
                        self._vecnorm_cb.save_vecnormalize()
                    if self.verbose > 0:
                        print(
                            f"[train_core] Eval success rate {success_rate:.1%} > previous best "
                            f"{max(prev, 0.0):.1%} → saved best_model.zip"
                            + (" + vecnormalize.pkl" if self._vecnorm_cb is not None else "")
                        )
                elif self.verbose > 0:
                    print(
                        f"[train_core] Eval success rate {success_rate:.1%} ≤ best "
                        f"{self.best_success_rate:.1%} → not saving"
                    )
                self.logger.record("eval/best_save_success_rate", self.best_success_rate)
            return result

    class SuccessRateLoggingCallback(BaseCallback):
        """Log recent training success rate (episode_success) to TensorBoard."""

        def __init__(self, log_freq, window=100, verbose=0):
            super().__init__(verbose)
            self.log_freq = int(log_freq)
            self.window = int(window)

        def _on_step(self):
            if self.log_freq <= 0 or self.n_calls % self.log_freq != 0:
                return True
            stats = unwrap_vec_record_episode_statistics(self.training_env)
            if stats is None or not stats.return_queue or not stats.length_queue:
                return True
            rq = list(stats.return_queue)
            lq = list(stats.length_queue)
            n = min(len(rq), len(lq), self.window)
            if n <= 0:
                return True
            succ = sum(1 for i in range(-n, 0) if episode_success(lq[i], rq[i]))
            self.logger.record("train/success_rate_recent", succ / n)
            return True

    class SuccessScheduleV3Callback(BaseCallback):
        """Set LR/ent from eval success band each schedule eval; stop after 3 evals >= SUCCESS_STOP_THRESHOLD.

        Uses the same eval rollout as ``EvalWithNormCallback`` (must run after it in the callback list).
        """

        def __init__(self, eval_with_norm_cb, eval_freq, verbose=0):
            super().__init__(verbose)
            self._eval_cb = eval_with_norm_cb
            self.eval_freq = int(eval_freq)
            self._consecutive_at_stop_thr = 0
            self._lr = None
            self._ent = None

        def _on_training_start(self) -> None:
            self._lr = float(self.model.lr_schedule(self.model._current_progress_remaining))
            ec = self.model.ent_coef
            self._ent = float(ec.item() if hasattr(ec, "item") else ec)

        def _on_step(self) -> bool:
            if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
                return True
            rate = self._eval_cb.last_eval_success_rate
            if rate is None:
                return True
            self.logger.record("eval/success_rate", rate)
            target_lr, target_ent = v3_lr_ent_for_success_rate(rate)
            if target_lr != self._lr or target_ent != self._ent:
                self._lr = target_lr
                self._ent = target_ent
                self.model.lr_schedule = get_schedule_fn(self._lr)
                self.model.ent_coef = self._ent
                if self.verbose > 0:
                    print(
                        f"[train_core] Eval success {rate:.1%} → band lr={self._lr:g} ent_coef={self._ent:g}"
                    )
            if rate >= SUCCESS_STOP_THRESHOLD:
                self._consecutive_at_stop_thr += 1
            else:
                self._consecutive_at_stop_thr = 0
            self.logger.record("eval/consecutive_ge_stop_thr", self._consecutive_at_stop_thr)
            if self._consecutive_at_stop_thr >= 3:
                print(
                    f"[train_core] Stopping: {self._consecutive_at_stop_thr} consecutive evals "
                    f"with success rate >= {SUCCESS_STOP_THRESHOLD:.0%}."
                )
                return False
            return True

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
    if isinstance(obs_noise_cfg, dict) and (
        obs_noise_cfg.get("position", 0) > 0 or obs_noise_cfg.get("velocity", 0) > 0
    ):
        eval_env = ObservationNoiseWrapper(eval_env, obs_noise_cfg)
    if custom_reward_cfg and custom_reward_cfg.get("enabled", False):
        eval_env = CustomRewardWrapper(eval_env, custom_reward_cfg)
    if max_episode_steps is not None:
        eval_env = VecMaxEpisodeSteps(eval_env, max_episode_steps)
    if domain_rand_cfg.get("enabled", False):
        eval_env = DomainRandomizationWrapper(eval_env, domain_rand_cfg)
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

    eval_with_norm_cb = EvalWithNormCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        vecnorm_cb=vecnorm_cb,
    )
    callbacks = [
        vecnorm_cb,
        CheckpointWithNormCallback(
            save_freq=save_interval,
            save_path=run_dir,
            name_prefix="ppo_drone",
            vecnorm_cb=vecnorm_cb,
        ),
        eval_with_norm_cb,
        SuccessRateLoggingCallback(log_freq=eval_freq, window=min(100, record_deque_size)),
        SuccessScheduleV3Callback(
            eval_with_norm_cb,
            eval_freq=eval_freq,
            verbose=1,
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
