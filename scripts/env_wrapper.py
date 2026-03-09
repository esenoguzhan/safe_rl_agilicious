"""
Gymnasium-compatible VecEnv wrapper for flightlib QuadrotorEnv_v1.
- Action clipping to [-1, 1] before C++ step to prevent crashes.
- terminal_observation in infos on done (cache obs before step).
- episode {"r", "l"} in infos when an episode ends.
"""
import warnings

import numpy as np
import gymnasium as gym


class FlightlibVecEnv:
    """
    Vectorized environment wrapping flightgym.QuadrotorEnv_v1 with Gymnasium/SB3 semantics.
    Use gymnasium.spaces; clip actions; provide terminal_observation and episode in infos.
    """

    def __init__(self, impl):
        self._impl = impl
        self._num_envs = impl.getNumOfEnvs()
        obs_dim = impl.getObsDim()
        act_dim = impl.getActDim()

        self._observation_space = gym.spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self._action_space = gym.spaces.Box(
            low=np.ones(act_dim, dtype=np.float32) * -1.0,
            high=np.ones(act_dim, dtype=np.float32) * 1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

        self._obs = np.zeros((self._num_envs, obs_dim), dtype=np.float32)
        self._reward = np.zeros(self._num_envs, dtype=np.float32)
        self._done = np.zeros(self._num_envs, dtype=bool)
        self._extra_info_names = impl.getExtraInfoNames()
        self._extra_info = np.zeros(
            (self._num_envs, len(self._extra_info_names)), dtype=np.float32
        )

        self._has_get_obs = hasattr(impl, "getObs")

        self._episode_rewards = [[] for _ in range(self._num_envs)]
        self.reset_infos = [{} for _ in range(self._num_envs)]
        self.render_mode = None  # SB3 expects this attribute
        self._pending_actions = None  # for step_async / step_wait

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def observation_space(self) -> gym.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._action_space

    def set_seed(self, seed: int) -> None:
        self._impl.setSeed(seed)

    def seed(self, seed=None):
        """SB3 VecEnv expects seed(seed); returns list of seeds for compatibility."""
        if seed is not None:
            self._impl.setSeed(seed)
        return [seed] if seed is not None else None

    def reset(self):
        self._reward.fill(0.0)
        self._impl.reset(self._obs)
        self.reset_infos = [{} for _ in range(self._num_envs)]
        return self._obs.copy()

    def step_async(self, actions: np.ndarray):
        """SB3 VecEnv: store actions for step_wait."""
        self._pending_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)

    def step_wait(self):
        """SB3 VecEnv: run step with pending actions and return (obs, rewards, dones, infos)."""
        if self._pending_actions is None:
            raise RuntimeError("step_async was not called before step_wait")
        actions = self._pending_actions
        self._pending_actions = None

        ok = self._impl.step(actions, self._obs, self._reward, self._done, self._extra_info)
        if not ok:
            warnings.warn(
                "C++ step() returned False — dimension check likely failed. "
                "obs/reward/done may not have been updated.",
                RuntimeWarning,
                stacklevel=2,
            )

        terminal_obs = None
        any_done = np.any(self._done)
        if any_done:
            terminal_obs = np.zeros_like(self._obs)
            self._impl.getTerminalObs(terminal_obs)

        if self._has_get_obs:
            self._impl.getObs(self._obs)

        infos = [{} for _ in range(self._num_envs)]
        for i in range(self._num_envs):
            if self._extra_info_names:
                infos[i]["extra_info"] = {
                    self._extra_info_names[j]: self._extra_info[i, j]
                    for j in range(len(self._extra_info_names))
                }
            self._episode_rewards[i].append(float(self._reward[i]))
            if self._done[i]:
                infos[i]["terminal_observation"] = terminal_obs[i].copy()
                infos[i]["episode"] = {
                    "r": sum(self._episode_rewards[i]),
                    "l": len(self._episode_rewards[i]),
                }
                self._episode_rewards[i].clear()

        return self._obs.copy(), self._reward.copy(), self._done.copy(), infos

    def step(self, actions: np.ndarray):
        """Synchronous step (e.g. for direct use); SB3 uses step_async + step_wait."""
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        self._impl.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        """SB3 expects this for EvalCallback / evaluate_policy; we use no Monitor."""
        return (False,) * self._num_envs

    def setQuadState(self, state):
        """Set exact state [pos(3), quat(4), vel(3), omega(3)] for env 0, return fresh obs."""
        self._impl.setQuadState(0, np.asarray(state, dtype=np.float32))
        self._impl.getObs(self._obs)
        return self._obs.copy()

    def connect_unity(self):
        """Optional: connect to Unity for rendering."""
        return self._impl.connectUnity()

    def disconnect_unity(self):
        self._impl.disconnectUnity()


class VecMaxEpisodeSteps:
    """
    VecEnv wrapper that truncates each env after max_steps per episode.
    Sets done=True and terminal_observation in info so outer wrappers (e.g. VecRecordEpisodeStatistics) record the episode.
    """

    def __init__(self, venv, max_steps):
        self.venv = venv
        self._max_steps = int(max_steps)
        self._num_envs = getattr(venv, "num_envs", venv._num_envs)
        self._episode_steps = np.zeros(self._num_envs, dtype=np.int32)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self.venv.observation_space

    @property
    def action_space(self):
        return self.venv.action_space

    def reset(self, **kwargs):
        self._episode_steps.fill(0)
        out = self.venv.reset(**kwargs)
        return out[0] if isinstance(out, tuple) else out

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self._episode_steps += 1
        for i in range(self._num_envs):
            if not dones[i] and self._episode_steps[i] >= self._max_steps:
                dones[i] = True
                infos[i]["terminal_observation"] = obs[i].copy()
                infos[i]["TimeLimit.truncated"] = True
            if dones[i]:
                self._episode_steps[i] = 0
        return obs, rewards, dones, infos

    def step(self, actions):
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


class ObservationNoiseWrapper:
    """
    VecEnv wrapper that adds zero-mean Gaussian noise to position (obs 0:3)
    and linear velocity (obs 7:10) to reduce memorization and improve robustness.
    Config: {"position": std_pos, "velocity": std_vel} (floats, std in same units as obs).

    Uses its own independent RandomState so it never interferes with the
    global numpy RNG (important for reproducible eval with fixed seeds).
    """

    # Observation layout: [pos_error(3), q(4), lin_vel(3), omega(3)]
    POS_SLICE = slice(0, 3)
    VEL_SLICE = slice(7, 10)

    def __init__(self, venv, noise_cfg, rng_seed=42):
        self.venv = venv
        self._num_envs = venv.num_envs
        self._std_pos = float(noise_cfg.get("position", 0.0))
        self._std_vel = float(noise_cfg.get("velocity", 0.0))
        self._rng = np.random.RandomState(rng_seed)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self.venv.observation_space

    @property
    def action_space(self):
        return self.venv.action_space

    def _add_noise(self, obs):
        if obs.size == 0:
            return obs
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        n_envs = obs.shape[0]
        if self._std_pos > 0:
            obs[:, self.POS_SLICE] += self._rng.normal(
                0, self._std_pos, size=(n_envs, 3)
            ).astype(np.float32)
        if self._std_vel > 0:
            obs[:, self.VEL_SLICE] += self._rng.normal(
                0, self._std_vel, size=(n_envs, 3)
            ).astype(np.float32)
        return obs

    def reset(self, **kwargs):
        out = self.venv.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        return self._add_noise(np.asarray(obs, dtype=np.float32))

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = self._add_noise(obs)
        if np.any(dones):
            for i in np.where(dones)[0]:
                if "terminal_observation" in infos[i]:
                    tobs = np.asarray(infos[i]["terminal_observation"], dtype=np.float32)
                    noised = self._add_noise(tobs.reshape(1, -1))
                    infos[i]["terminal_observation"] = noised[0]
        return obs, rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        self.venv.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        if hasattr(self.venv, "env_is_wrapped"):
            return self.venv.env_is_wrapped(wrapper_class, indices=indices)
        return (False,) * self._num_envs

    def __getattr__(self, name):
        return getattr(self.venv, name)


class DomainRandomizationWrapper:
    """
    VecEnv wrapper that randomizes mass, motor_tau, and goal position
    per-env at each episode boundary (crash, OOB, or truncation).

    IMPORTANT: This wrapper must sit OUTSIDE VecMaxEpisodeSteps so that
    truncation dones also trigger goal re-randomization.  Otherwise goals
    stay fixed for the entire training run and the agent never learns
    navigation to arbitrary targets.

    Requires the underlying FlightlibVecEnv._impl to expose
    setEnvMasses / setEnvMotorTauInvs / setEnvGoalPositions / getObs.
    """

    def __init__(self, venv, domain_rand_cfg):
        self.venv = venv
        self._num_envs = venv.num_envs
        self._impl = venv._impl
        self._obs_dim = venv.observation_space.shape[0]

        self._mass_range = domain_rand_cfg.get("mass_range") if domain_rand_cfg.get("randomize_mass", False) else None
        self._motor_tau_range = domain_rand_cfg.get("motor_tau_range") if domain_rand_cfg.get("randomize_motor_tau", False) else None
        self._goal_pos_range = domain_rand_cfg.get("goal_pos_range") if domain_rand_cfg.get("randomize_goal", False) else None

        self._current_masses = np.zeros(self._num_envs, dtype=np.float32)
        self._current_tau_invs = np.zeros(self._num_envs, dtype=np.float32)
        self._current_goals = np.zeros((self._num_envs, 3), dtype=np.float32)

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self.venv.observation_space

    @property
    def action_space(self):
        return self.venv.action_space

    def _randomize_goal(self, env_ids):
        """Randomize goal positions for the given env indices."""
        if self._goal_pos_range is None or len(env_ids) == 0:
            return
        x_range = self._goal_pos_range.get("x", [0.0, 0.0])
        y_range = self._goal_pos_range.get("y", [0.0, 0.0])
        z_range = self._goal_pos_range.get("z", [3.0, 7.0])
        for i in env_ids:
            self._current_goals[i, 0] = np.random.uniform(x_range[0], x_range[1])
            self._current_goals[i, 1] = np.random.uniform(y_range[0], y_range[1])
            self._current_goals[i, 2] = np.random.uniform(z_range[0], z_range[1])
        self._impl.setEnvGoalPositions(self._current_goals)

    def _randomize_dynamics(self, env_ids):
        """Randomize mass and motor_tau for the given env indices."""
        if len(env_ids) == 0:
            return
        if self._mass_range is not None:
            lo, hi = self._mass_range
            new_masses = np.random.uniform(lo, hi, size=len(env_ids)).astype(np.float32)
            self._current_masses[env_ids] = new_masses
            self._impl.setEnvMasses(self._current_masses)
            for i in env_ids:
                self._impl.reinitHoverMotor(int(i))
        if self._motor_tau_range is not None:
            lo, hi = self._motor_tau_range
            new_taus = np.random.uniform(lo, hi, size=len(env_ids))
            new_tau_invs = (1.0 / new_taus).astype(np.float32)
            self._current_tau_invs[env_ids] = new_tau_invs
            self._impl.setEnvMotorTauInvs(self._current_tau_invs)

    def _randomize_envs(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self._num_envs)
        if len(env_ids) == 0:
            return
        self._randomize_dynamics(env_ids)
        self._randomize_goal(env_ids)

    def _refresh_obs(self, obs, env_ids):
        """Re-read C++ obs for env_ids so pos_err reflects the new goal."""
        if len(env_ids) == 0:
            return
        fresh = np.zeros((self._num_envs, self._obs_dim), dtype=np.float32)
        self._impl.getObs(fresh)
        for i in env_ids:
            obs[i, :self._obs_dim] = fresh[i]

    def reset(self, **kwargs):
        self._randomize_envs()
        return self.venv.reset(**kwargs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        done_ids = np.where(dones)[0]
        if len(done_ids) > 0:
            truncated = np.array([
                infos[i].get("TimeLimit.truncated", False) for i in done_ids
            ])
            crash_ids = done_ids[~truncated]
            trunc_ids = done_ids[truncated]

            # Crashes / OOB: randomize everything (C++ already auto-reset)
            if len(crash_ids) > 0:
                self._randomize_dynamics(crash_ids)
                self._randomize_goal(crash_ids)

            # Truncation: only re-randomize goal (drone keeps flying)
            if len(trunc_ids) > 0:
                self._randomize_goal(trunc_ids)

            # Refresh obs so pos_err uses the new goal immediately
            self._refresh_obs(obs, done_ids)
        return obs, rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        self.venv.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        if hasattr(self.venv, "env_is_wrapped"):
            return self.venv.env_is_wrapped(wrapper_class, indices=indices)
        return (False,) * self._num_envs

    def __getattr__(self, name):
        return getattr(self.venv, name)


class ActionHistoryWrapper:
    """
    VecEnv wrapper that augments observations with the last N actions.
    Observation becomes [obs, a_{t-N}, a_{t-N+1}, ..., a_{t-1}].
    On episode reset the action history is zeroed out.
    """

    def __init__(self, venv, n_actions):
        self.venv = venv
        self._n_actions = int(n_actions)
        self._num_envs = venv.num_envs

        self._inner_obs_dim = venv.observation_space.shape[0]
        self._act_dim = venv.action_space.shape[0]
        aug_dim = self._inner_obs_dim + self._n_actions * self._act_dim

        self._observation_space = gym.spaces.Box(
            low=np.full(aug_dim, -np.inf, dtype=np.float32),
            high=np.full(aug_dim, np.inf, dtype=np.float32),
            shape=(aug_dim,),
            dtype=np.float32,
        )
        self._action_space = venv.action_space

        self._act_buffer = np.zeros(
            (self._num_envs, self._n_actions, self._act_dim), dtype=np.float32
        )

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def inner_obs_dim(self):
        return self._inner_obs_dim

    def _augment_obs(self, obs):
        flat_hist = self._act_buffer.reshape(self._num_envs, -1)
        return np.concatenate([obs, flat_hist], axis=1).astype(np.float32)

    def reset(self, **kwargs):
        self._act_buffer.fill(0.0)
        obs = self.venv.reset(**kwargs)
        return self._augment_obs(obs)

    def step_async(self, actions):
        self._act_buffer = np.roll(self._act_buffer, -1, axis=1)
        self._act_buffer[:, -1, :] = actions
        self.venv.step_async(actions)

    def _augment_single(self, obs_1d, env_id):
        flat_hist = self._act_buffer[env_id].reshape(-1)
        return np.concatenate([obs_1d, flat_hist]).astype(np.float32)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for i in range(self._num_envs):
            if dones[i]:
                if "terminal_observation" in infos[i]:
                    infos[i]["terminal_observation"] = self._augment_single(
                        infos[i]["terminal_observation"], i
                    )
                self._act_buffer[i].fill(0.0)
        return self._augment_obs(obs), rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        self.venv.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        if hasattr(self.venv, "env_is_wrapped"):
            return self.venv.env_is_wrapped(wrapper_class, indices=indices)
        return (False,) * self._num_envs

    def __getattr__(self, name):
        return getattr(self.venv, name)
