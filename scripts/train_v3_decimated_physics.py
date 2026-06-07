#!/usr/bin/env python3
"""
PPO training with high-rate Flightmare physics and lower-rate RL control.

Runs the underlying VecEnv at ``sim_dt = 1/physics_hz`` and applies each RL
action for ``n_substeps = physics_hz / rl_hz`` inner steps (zero-order hold),
so the policy is called at ``rl_hz`` while the simulator integrates at
``physics_hz``.

This script does not modify ``train_v3.py``; it reuses its env factory and
constants via imports and mirrors the v3 success / LR schedule / callbacks.

Usage:
  python scripts/train_v3_decimated_physics.py \\
      --config configs/single_stage_no_curriculum_cauchy.yaml \\
      --physics-hz 1000 --rl-hz 100

YAML: ``env.quadrotor_env.sim_dt`` is overridden to ``1/physics_hz`` unless it
already matches. If ``env.trajectory`` exists, ``trajectory.sim_dt`` is set to
the same value for consistency.

Wrapper order (inner → outer): FlightlibVecEnv → VecPhysicsDecimationWrapper
→ (same chain as train_v3: observation noise, custom reward, max steps, …).
"""
from __future__ import annotations

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

from scripts.config_loader import load_config
from scripts.custom_reward_wrapper import CustomRewardWrapper
from scripts.env_wrapper import (
    VecMaxEpisodeSteps,
    DomainRandomizationWrapper,
    ActionHistoryWrapper,
    ObservationNoiseWrapper,
)
from scripts.record_episode_statistics import VecRecordEpisodeStatistics
from scripts.train_v3 import (
    EVAL_SEED,
    SUCCESS_MIN_REWARD,
    SUCCESS_STOP_THRESHOLD,
    V3_LR_ENT_DISCOVERY,
    VECNORM_EPSILON,
    VECNORM_MIN_VAR,
    _clamp_vecnorm_obs_variance,
    _make_env,
    _resolve_resume_path,
    get_next_ppo_steps_dir,
    unwrap_vec_record_episode_statistics,
    v3_lr_ent_for_success_rate,
)


class VecPhysicsDecimationWrapper:
    """
    Hold each RL action for ``n_substeps`` inner VecEnv steps (ZOH).

    Rewards are aggregated per RL step (sum or mean). If any inner step sets
    ``done[i]``, the RL step reports ``done[i]=True`` and attaches
    ``terminal_observation`` from the *first* inner step where that env
    terminated (Flightmare auto-resets on done).

    Optional AgiSim-style command/actuator delay: when ``command_delay_s > 0``
    (or randomization is enabled), a per-env ring buffer of past RL actions is
    maintained at physics-tick resolution. At each inner substep we write the
    current RL action at the buffer head and read back the action from
    ``delay_substeps`` ticks earlier, then forward that delayed sample to the
    inner VecEnv. With ``command_delay_s == 0`` and no randomization the
    wrapper takes a fast path that is identical to the original behaviour.
    """

    def __init__(
        self,
        venv,
        n_substeps: int,
        aggregate_reward: str = "sum",
        sim_dt_s: float | None = None,
        command_delay_s: float = 0.0,
        randomize_command_delay: bool = False,
        command_delay_range_s: tuple | list | None = None,
        rng_seed: int = 0,
    ):
        if int(n_substeps) < 1:
            raise ValueError("n_substeps must be >= 1")
        self.venv = venv
        self.n_substeps = int(n_substeps)
        self._aggregate_reward = aggregate_reward
        self._num_envs = int(getattr(venv, "num_envs", venv._num_envs))
        self._pending_actions: np.ndarray | None = None

        nominal_delay = float(command_delay_s)
        if nominal_delay < 0.0:
            raise ValueError("command_delay_s must be >= 0")

        range_substeps = None
        if randomize_command_delay:
            if command_delay_range_s is None or len(command_delay_range_s) != 2:
                raise ValueError(
                    "command_delay_range_s must be a [lo, hi] pair when "
                    "randomize_command_delay=True"
                )
            lo_s, hi_s = float(command_delay_range_s[0]), float(command_delay_range_s[1])
            if lo_s < 0.0 or hi_s < lo_s:
                raise ValueError(
                    f"Invalid command_delay_range_s={command_delay_range_s}; "
                    "need 0 <= lo <= hi"
                )
            if sim_dt_s is None or float(sim_dt_s) <= 0.0:
                raise ValueError(
                    "sim_dt_s must be a positive float when "
                    "randomize_command_delay=True"
                )
            range_substeps = (
                int(round(lo_s / float(sim_dt_s))),
                int(round(hi_s / float(sim_dt_s))),
            )

        nominal_substeps = 0
        if nominal_delay > 0.0:
            if sim_dt_s is None or float(sim_dt_s) <= 0.0:
                raise ValueError(
                    "sim_dt_s must be a positive float when command_delay_s > 0"
                )
            nominal_substeps = int(round(nominal_delay / float(sim_dt_s)))

        self._sim_dt_s = float(sim_dt_s) if sim_dt_s is not None else None
        self._nominal_delay_s = nominal_delay
        self._randomize_command_delay = bool(randomize_command_delay)
        self._delay_range_substeps = range_substeps
        self._delay_rng = np.random.RandomState(int(rng_seed))

        self._delay_active = bool(self._randomize_command_delay or nominal_substeps > 0)

        if self._delay_active:
            act_dim = int(self.venv.action_space.shape[0])
            self._delay_substeps = np.full(self._num_envs, nominal_substeps, dtype=np.int64)
            if self._randomize_command_delay:
                lo_n, hi_n = self._delay_range_substeps
                self._delay_substeps[:] = self._delay_rng.randint(
                    lo_n, hi_n + 1, size=self._num_envs
                )
            max_substeps = int(self._delay_substeps.max())
            if self._delay_range_substeps is not None:
                max_substeps = max(max_substeps, int(self._delay_range_substeps[1]))
            self._buf_len = max(max_substeps + 1, 1)
            self._action_buffer = np.zeros(
                (self._num_envs, self._buf_len, act_dim), dtype=np.float32
            )
            self._head = 0
            self._env_index = np.arange(self._num_envs)
        else:
            self._delay_substeps = None
            self._buf_len = 0
            self._action_buffer = None
            self._head = 0
            self._env_index = np.arange(self._num_envs)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def observation_space(self):
        return self.venv.observation_space

    @property
    def action_space(self):
        return self.venv.action_space

    @property
    def command_delay_active(self) -> bool:
        return self._delay_active

    @property
    def command_delay_substeps(self) -> np.ndarray | None:
        if self._delay_substeps is None:
            return None
        return self._delay_substeps.copy()

    def set_command_delays(self, delays_seconds: np.ndarray) -> None:
        """Override per-env command delay (seconds). Mirrors setEnvMasses-style API
        so eval scripts can pin a deterministic delay per env."""
        if not self._delay_active:
            raise RuntimeError(
                "VecPhysicsDecimationWrapper was constructed without command "
                "delay support; cannot set per-env delays."
            )
        if self._sim_dt_s is None or self._sim_dt_s <= 0.0:
            raise RuntimeError(
                "sim_dt_s is not set; cannot convert delays to substeps."
            )
        delays = np.asarray(delays_seconds, dtype=np.float64).reshape(-1)
        if delays.size != self._num_envs:
            raise ValueError(
                f"delays_seconds has size {delays.size}, expected {self._num_envs}"
            )
        if np.any(delays < 0.0):
            raise ValueError("delays_seconds must be non-negative")
        new_substeps = np.rint(delays / self._sim_dt_s).astype(np.int64)
        max_substeps = int(new_substeps.max())
        if max_substeps >= self._buf_len:
            new_buf_len = max_substeps + 1
            new_buf = np.zeros(
                (self._num_envs, new_buf_len, self._action_buffer.shape[2]),
                dtype=np.float32,
            )
            self._action_buffer = new_buf
            self._buf_len = new_buf_len
            self._head = 0
        self._delay_substeps = new_substeps

    def reset(self, **kwargs):
        out = self.venv.reset(**kwargs)
        if self._delay_active:
            self._action_buffer.fill(0.0)
            self._head = 0
            if self._randomize_command_delay and self._delay_range_substeps is not None:
                lo_n, hi_n = self._delay_range_substeps
                self._delay_substeps[:] = self._delay_rng.randint(
                    lo_n, hi_n + 1, size=self._num_envs
                )
        return out[0] if isinstance(out, tuple) else out

    def step_async(self, actions: np.ndarray):
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_async was not called before step_wait")
        # Local writable copy: when an env dies mid-RL-step we will zero its
        # row to prevent the pre-reset policy action from being written into
        # the freshly-drained ring buffer during the remaining substeps.
        actions = self._pending_actions.copy()
        self._pending_actions = None

        r_acc = np.zeros(self._num_envs, dtype=np.float32)
        seen_done = np.zeros(self._num_envs, dtype=bool)
        terminal_snap: list[np.ndarray | None] = [None] * self._num_envs

        obs_last = None
        infos_last: list | None = None

        for _ in range(self.n_substeps):
            if self._delay_active:
                write_idx = self._head % self._buf_len
                self._action_buffer[:, write_idx, :] = actions
                read_idx = (self._head - self._delay_substeps) % self._buf_len
                inner_actions = self._action_buffer[self._env_index, read_idx, :]
                self._head += 1
            else:
                inner_actions = actions

            self.venv.step_async(inner_actions)
            obs, rew, dones, infos = self.venv.step_wait()
            obs_last = obs
            infos_last = infos
            r_acc += np.asarray(rew, dtype=np.float32).reshape(self._num_envs)
            for i in range(self._num_envs):
                if bool(dones[i]) and not seen_done[i]:
                    seen_done[i] = True
                    tobs = infos[i].get("terminal_observation")
                    if tobs is None:
                        tobs = np.asarray(obs[i], dtype=np.float32).copy()
                    else:
                        tobs = np.asarray(tobs, dtype=np.float32).copy()
                    terminal_snap[i] = tobs
                    # AgiSim-style empty cmd_queue_ on reset:
                    #   1. zero env i's ring buffer so remaining substeps of
                    #      this RL step read hover-thrust (drain stale queue);
                    #   2. zero actions[i] for the rest of the loop so the
                    #      pre-reset policy action stops being written into
                    #      the ring (otherwise it would re-surface via the
                    #      delayed reads when delay <= n_substeps);
                    #   3. resample the per-env delay so the new episode
                    #      flies its final delay from substep 0.
                    if self._delay_active:
                        self._action_buffer[i] = 0.0
                        actions[i] = 0.0
                        if (
                            self._randomize_command_delay
                            and self._delay_range_substeps is not None
                        ):
                            lo_n, hi_n = self._delay_range_substeps
                            self._delay_substeps[i] = int(
                                self._delay_rng.randint(lo_n, hi_n + 1)
                            )

        assert obs_last is not None and infos_last is not None

        if self._aggregate_reward == "mean":
            r_acc = r_acc / float(self.n_substeps)

        infos_out = []
        for i in range(self._num_envs):
            d = dict(infos_last[i])
            if seen_done[i]:
                d["terminal_observation"] = terminal_snap[i]
                d.pop("episode", None)
            infos_out.append(d)

        return obs_last, r_acc, seen_done.copy(), infos_out

    def step(self, actions: np.ndarray):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if hasattr(self.venv, "close"):
            self.venv.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        if hasattr(self.venv, "env_is_wrapped"):
            return self.venv.env_is_wrapped(wrapper_class, indices=indices)
        return (False,) * self._num_envs

    def __getattr__(self, name):
        return getattr(self.venv, name)


def _apply_physics_dt_to_cfg(cfg: dict, physics_hz: float):
    """Set env.quadrotor_env.sim_dt and optional trajectory.sim_dt to 1/physics_hz.

    Returns the previous ``sim_dt`` from YAML if present, else ``None``.
    """
    dt = 1.0 / float(physics_hz)
    env = cfg.setdefault("env", {})
    qe = env.setdefault("quadrotor_env", {})
    prev = qe.get("sim_dt")
    qe["sim_dt"] = dt
    traj = env.get("trajectory")
    if isinstance(traj, dict):
        traj["sim_dt"] = dt
    return float(prev) if prev is not None else None


def main():
    parser = argparse.ArgumentParser(
        description="PPO (train_v3 schedule) with decimated high-rate Flightmare physics"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument(
        "--physics-hz",
        type=float,
        default=1000.0,
        help="Simulator integration rate (Hz); sim_dt = 1/physics_hz",
    )
    parser.add_argument(
        "--rl-hz",
        type=float,
        default=100.0,
        help="Policy / outer VecEnv step rate (Hz); must divide physics_hz evenly",
    )
    parser.add_argument(
        "--reward-aggregate",
        type=str,
        choices=("sum", "mean"),
        default="sum",
        help="How to combine inner rewards over one RL step (default: sum)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional folder suffix under models/PPO_<steps>_<run_name>",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from run directory or model .zip (same as train_v3)",
    )
    args = parser.parse_args()

    ph = float(args.physics_hz)
    rlh = float(args.rl_hz)
    if ph <= 0.0 or rlh <= 0.0:
        raise SystemExit("--physics-hz and --rl-hz must be positive")
    ratio = ph / rlh
    n_substeps = int(round(ratio))
    if abs(ratio - n_substeps) > 1e-6:
        raise SystemExit(
            f"physics_hz / rl_hz must be an integer; got {ph}/{rlh} = {ratio}"
        )
    if n_substeps < 1:
        raise SystemExit("n_substeps computed as < 1")

    cfg = load_config(args.config)
    total_timesteps = cfg.get("training", {}).get("total_timesteps", 100_000)
    run_dir = get_next_ppo_steps_dir(total_timesteps, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    paths = {**cfg.get("paths", {}), "log_dir": run_dir, "save_dir": run_dir}
    cfg["paths"] = paths

    physics_dt = 1.0 / ph
    prev_sim_dt = _apply_physics_dt_to_cfg(cfg, ph)
    cfg["decimated_physics"] = {
        "physics_hz": ph,
        "rl_hz": rlh,
        "n_substeps": n_substeps,
        "sim_dt": physics_dt,
        "reward_aggregate": args.reward_aggregate,
        "previous_yaml_sim_dt": prev_sim_dt,
    }

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print("Run directory:", run_dir)
    print(
        f"[train_v3_decimated] physics={ph:g} Hz (sim_dt={physics_dt:g} s), "
        f"RL={rlh:g} Hz, n_substeps={n_substeps}, reward={args.reward_aggregate}"
    )
    if prev_sim_dt is not None and abs(float(prev_sim_dt) - physics_dt) > 1e-12:
        print(
            f"[train_v3_decimated] YAML sim_dt was {prev_sim_dt}; overridden to {physics_dt:g}"
        )

    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 0)
    np.random.seed(seed)

    env_cfg = cfg.get("env", {})
    domain_rand_cfg = env_cfg.get("domain_randomization", {})

    quad_env_cfg = env_cfg.get("quadrotor_env", {}) or {}
    command_delay_s = float(quad_env_cfg.get("command_delay", 0.0) or 0.0)
    dr_enabled = bool(domain_rand_cfg.get("enabled", False))
    randomize_command_delay = bool(
        dr_enabled and domain_rand_cfg.get("randomize_command_delay", False)
    )
    command_delay_range_s = domain_rand_cfg.get("command_delay_range")

    env = _make_env(cfg)
    env = VecPhysicsDecimationWrapper(
        env,
        n_substeps=n_substeps,
        aggregate_reward=args.reward_aggregate,
        sim_dt_s=physics_dt,
        command_delay_s=command_delay_s,
        randomize_command_delay=randomize_command_delay,
        command_delay_range_s=command_delay_range_s,
        rng_seed=int(seed),
    )
    if env.command_delay_active:
        if randomize_command_delay:
            print(
                f"[train_v3_decimated] Command delay enabled: nominal="
                f"{command_delay_s:g}s, randomized per episode in "
                f"{list(command_delay_range_s)} s "
                f"(={int(round(command_delay_range_s[0]/physics_dt))}–"
                f"{int(round(command_delay_range_s[1]/physics_dt))} substeps "
                f"@ {physics_dt:g}s)"
            )
        else:
            print(
                f"[train_v3_decimated] Command delay enabled: "
                f"{command_delay_s:g}s (={int(round(command_delay_s/physics_dt))} "
                f"substeps @ {physics_dt:g}s)"
            )

    success_episode_length = int(env_cfg.get("max_episode_steps") or 1000)

    def episode_success_local(ep_length, ep_return):
        return (
            int(ep_length) == success_episode_length
            and float(ep_return) > SUCCESS_MIN_REWARD
        )

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

    if domain_rand_cfg.get("enabled", False):
        env = DomainRandomizationWrapper(env, domain_rand_cfg)
        active = []
        if domain_rand_cfg.get("randomize_mass", False):
            active.append(f"mass={domain_rand_cfg.get('mass_range')}")
        if domain_rand_cfg.get("randomize_motor_tau", False):
            active.append(f"motor_tau={domain_rand_cfg.get('motor_tau_range')}")
        if domain_rand_cfg.get("randomize_goal", False):
            active.append(f"goal_pos={domain_rand_cfg.get('goal_pos_range')}")
        if randomize_command_delay:
            active.append(f"command_delay={list(command_delay_range_s)}s")
        print(f"Domain randomization enabled: {', '.join(active) or 'none'}")

    action_history_len = env_cfg.get("action_history_len", 0)
    if action_history_len > 0:
        env = ActionHistoryWrapper(env, action_history_len)
        print(
            f"Action history enabled: last {action_history_len} actions appended to obs "
            f"(obs_dim: {env.observation_space.shape[0]})"
        )

    training_cfg = cfg.get("training", {})
    normalize_obs = training_cfg.get("normalize_obs", True)
    normalize_reward = training_cfg.get("normalize_reward", False)
    record_deque_size = training_cfg.get("record_episode_statistics_deque_size", 100)
    env = VecRecordEpisodeStatistics(env, deque_size=record_deque_size)

    ppo_cfg = cfg.get("ppo", {})
    policy_kwargs = ppo_cfg.get("policy_kwargs") or {"net_arch": dict(pi=[128, 128], vf=[128, 128])}
    ppo_kwargs = {k: v for k, v in ppo_cfg.items() if k != "policy_kwargs"}
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
        def __init__(self, save_freq, save_path, name_prefix, vecnorm_cb):
            super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)
            self._vecnorm_cb = vecnorm_cb

        def _on_step(self):
            result = super()._on_step()
            if self.n_calls % self.save_freq == 0 and self._vecnorm_cb is not None:
                self._vecnorm_cb.save_vecnormalize(f"_{self.num_timesteps}_steps")
            return result

    def _unwrap_to_flightlib(env_inner):
        cur = env_inner
        while cur is not None:
            if hasattr(cur, "set_seed") and hasattr(cur, "_impl"):
                return cur
            cur = getattr(cur, "venv", None)
        return None

    class EvalWithNormCallback(EvalCallback):
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
            self.last_eval_success_rate = None
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
                    1 for L, R in zip(ep_lengths, ep_rewards) if episode_success_local(L, R)
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
                            f"[train_v3_decimated] Eval success rate {success_rate:.1%} > previous best "
                            f"{max(prev, 0.0):.1%} → saved best_model.zip"
                            + (" + vecnormalize.pkl" if self._vecnorm_cb is not None else "")
                        )
                elif self.verbose > 0:
                    print(
                        f"[train_v3_decimated] Eval success rate {success_rate:.1%} ≤ best "
                        f"{self.best_success_rate:.1%} → not saving"
                    )
                self.logger.record("eval/best_save_success_rate", self.best_success_rate)
            return result

    class SuccessRateLoggingCallback(BaseCallback):
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
            succ = sum(1 for i in range(-n, 0) if episode_success_local(lq[i], rq[i]))
            self.logger.record("train/success_rate_recent", succ / n)
            return True

    class SuccessScheduleV3Callback(BaseCallback):
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
                        f"[train_v3_decimated] Eval success {rate:.1%} → band lr={self._lr:g} "
                        f"ent_coef={self._ent:g}"
                    )
            if rate >= SUCCESS_STOP_THRESHOLD:
                self._consecutive_at_stop_thr += 1
            else:
                self._consecutive_at_stop_thr = 0
            self.logger.record("eval/consecutive_ge_stop_thr", self._consecutive_at_stop_thr)
            if self._consecutive_at_stop_thr >= 3:
                print(
                    f"[train_v3_decimated] Stopping: {self._consecutive_at_stop_thr} consecutive evals "
                    f"with success rate >= {SUCCESS_STOP_THRESHOLD:.0%}."
                )
                return False
            return True

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
            print(
                f"Resumed VecNormalize stats from {vecnorm_pkl} "
                f"(all obs var clamped to >={VECNORM_MIN_VAR})"
            )
        elif use_vecnorm:
            env = VecNormalize(
                env, norm_obs=normalize_obs, norm_reward=normalize_reward,
                clip_obs=10.0, epsilon=VECNORM_EPSILON,
            )
            print("Warning: --resume but no vecnormalize.pkl; starting fresh normalization")

        model = PPO.load(model_zip, env=env, seed=seed, tensorboard_log=run_dir, **ppo_kwargs)
        print(f"Resumed model from {model_zip}")
    else:
        if use_vecnorm:
            env = VecNormalize(
                env, norm_obs=normalize_obs, norm_reward=normalize_reward,
                clip_obs=10.0, epsilon=VECNORM_EPSILON,
            )

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
    _apply_physics_dt_to_cfg(cfg_eval, ph)

    eval_env = _make_env(cfg_eval)
    eval_env = VecPhysicsDecimationWrapper(
        eval_env,
        n_substeps=n_substeps,
        aggregate_reward=args.reward_aggregate,
        sim_dt_s=physics_dt,
        command_delay_s=command_delay_s,
        randomize_command_delay=False,
        command_delay_range_s=None,
        rng_seed=int(seed) + 1,
    )
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
            eval_env = VecNormalize(
                eval_env, norm_obs=normalize_obs, norm_reward=False,
                clip_obs=10.0, epsilon=VECNORM_EPSILON,
            )

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

    print(
        "[train_v3_decimated] Success = episode length "
        f"{success_episode_length} and return > {SUCCESS_MIN_REWARD}; "
        "same LR/entropy schedule as train_v3."
    )

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(run_dir, "ppo_drone_final"))
    if use_vecnorm:
        env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    env.close()
    eval_env.close()
    print("Training finished. Best model, checkpoints, logs and TensorBoard in:", run_dir)


if __name__ == "__main__":
    main()
