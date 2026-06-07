"""
Trajectory-tracking VecEnv wrapper.

Converts the hover-style env into a trajectory-tracking env by:
  1. Updating the C++ goal position each timestep to follow a reference trajectory.
  2. Converting obs vel/quat/omega from absolute values to trajectory-relative errors
     so the policy always sees a "drive errors to zero" problem (same as hover).
  3. Optionally setting per-env x_goal on a CustomRewardWrapper so the reward
     penalises deviation from the desired trajectory state, not from zero.
  4. Optionally appending N future trajectory waypoints (relative position offsets)
     to the observation, giving the policy look-ahead information about path curvature.

Supports two trajectory sources (``env.trajectory.source`` in YAML):
  - ``"csv"`` (default): load a single trajectory CSV, shared by all envs.
  - ``"random"``: generate a new random trajectory per env on each episode reset.

Wrapper placement: sits ABOVE DomainRandomizationWrapper in the chain.
"""

import os

import gymnasium as gym
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Vectorised quaternion helpers (wxyz convention, pure numpy, no CasADi)
# ---------------------------------------------------------------------------

def _quat_inverse_batch(q):
    """Inverse of unit quaternions.  q: (..., 4) wxyz -> (..., 4) wxyz."""
    qi = q.copy()
    qi[..., 1:] *= -1.0
    return qi


def _quat_multiply_batch(q1, q2):
    """Hamilton product q1 * q2.  Both (..., 4) wxyz -> (..., 4) wxyz."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _quat_normalize_batch(q):
    """Normalise quaternions to unit length.  q: (..., 4) -> (..., 4)."""
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.clip(n, 1e-8, None)
    return q / n


# Obs layout: [pos_err(3), quat(4), lin_vel(3), ang_vel(3)] = 13
_POS_SLICE = slice(0, 3)
_ORI_SLICE = slice(3, 7)
_VEL_SLICE = slice(7, 10)
_OMEGA_SLICE = slice(10, 13)


def _load_trajectory(csv_path, sim_dt):
    """Load a trajectory CSV and resample to *sim_dt* if needed.

    Returns arrays of shape (T, D) for pos, vel, quat, omega.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Trajectory CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"t", "p_x", "p_y", "p_z", "q_w", "q_x", "q_y", "q_z",
                "v_x", "v_y", "v_z", "w_x", "w_y", "w_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory CSV missing columns: {missing}")

    t = df["t"].values.astype(np.float64)
    pos = df[["p_x", "p_y", "p_z"]].values.astype(np.float32)
    quat = df[["q_w", "q_x", "q_y", "q_z"]].values.astype(np.float32)
    vel = df[["v_x", "v_y", "v_z"]].values.astype(np.float32)
    omega = df[["w_x", "w_y", "w_z"]].values.astype(np.float32)

    traj_dt = float(np.median(np.diff(t)))
    if abs(traj_dt - sim_dt) / max(sim_dt, 1e-9) > 0.05:
        t_new = np.arange(t[0], t[-1], sim_dt)
        pos = interp1d(t, pos, axis=0, kind="linear", fill_value="extrapolate")(t_new).astype(np.float32)
        vel = interp1d(t, vel, axis=0, kind="linear", fill_value="extrapolate")(t_new).astype(np.float32)
        omega = interp1d(t, omega, axis=0, kind="linear", fill_value="extrapolate")(t_new).astype(np.float32)
        quat_interp = interp1d(t, quat, axis=0, kind="linear", fill_value="extrapolate")(t_new).astype(np.float32)
        quat = _quat_normalize_batch(quat_interp)
        t = t_new
        print(f"Trajectory resampled from dt={traj_dt:.4f} to sim_dt={sim_dt:.4f}  "
              f"({len(t)} steps, {t[-1]:.2f}s)")
    else:
        print(f"Trajectory loaded: {len(t)} steps, dt={traj_dt:.4f}s, duration={t[-1]:.2f}s")

    return t, pos, vel, quat, omega


class TrajectoryTrackingWrapper:
    """VecEnv wrapper that turns a hover env into a trajectory-tracking env.

    Trajectory arrays are stored per-env as ``(num_envs, T, D)`` tensors.
    All envs share the same ``T`` (trajectory length).

    Parameters
    ----------
    venv : VecEnv-like
        Inner vectorised environment (must expose ``_impl`` via attribute chain).
    trajectory_cfg : dict
        ``env.trajectory`` section from YAML.
        Required: ``sim_dt`` (float).
        For CSV mode (default): ``csv_path`` (str).
        For random mode: ``source: "random"`` plus generation parameters
        (see ``RandomTrajectoryGenerator``).
        Optional: ``loop`` (bool, default True),
        ``future_goals`` (int, default 0), ``future_spacing`` (int, default 1).
    reward_wrapper : CustomRewardWrapper | None
        If provided, per-env ``_x_goal`` is set each step so the reward
        penalises deviation from the trajectory's desired vel/quat/omega.
    """

    def __init__(self, venv, trajectory_cfg, reward_wrapper=None):
        self.venv = venv
        self._num_envs = venv.num_envs
        self._inner_obs_dim = venv.observation_space.shape[0]
        self._reward_wrapper = reward_wrapper
        self._cfg = trajectory_cfg

        sim_dt = float(trajectory_cfg["sim_dt"])
        self._loop = trajectory_cfg.get("loop", True)
        self._source = trajectory_cfg.get("source", "csv")
        self._shared_trajectory = bool(trajectory_cfg.get("shared_trajectory", False))

        N = self._num_envs
        self._env_idx = np.arange(N)
        self._generator = None
        self._frozen_traj = np.zeros(N, dtype=bool)
        self._shared_done_count = 0
        self._shared_refresh_interval = N

        if self._source == "random":
            from scripts.trajectory_generator import RandomTrajectoryGenerator
            self._generator = RandomTrajectoryGenerator(trajectory_cfg)
            self._rng = np.random.default_rng(
                trajectory_cfg.get("seed", None))

            p0, v0, q0, o0 = self._generator.generate(self._rng)
            T = len(p0)
            self._traj_len = T
            self._traj_pos = np.zeros((N, T, 3), dtype=np.float32)
            self._traj_vel = np.zeros((N, T, 3), dtype=np.float32)
            self._traj_quat = np.zeros((N, T, 4), dtype=np.float32)
            self._traj_omega = np.zeros((N, T, 3), dtype=np.float32)

            if self._shared_trajectory:
                self._traj_pos[:] = p0[np.newaxis]
                self._traj_vel[:] = v0[np.newaxis]
                self._traj_quat[:] = q0[np.newaxis]
                self._traj_omega[:] = o0[np.newaxis]
                print(f"  Random trajectories (SHARED): {N} envs x {T} steps"
                      f" — refresh every {N} dones")
            else:
                self._traj_pos[0], self._traj_vel[0] = p0, v0
                self._traj_quat[0], self._traj_omega[0] = q0, o0
                for i in range(1, N):
                    p, v, q, o = self._generator.generate(self._rng)
                    self._traj_pos[i], self._traj_vel[i] = p, v
                    self._traj_quat[i], self._traj_omega[i] = q, o
                print(f"  Random trajectories: {N} envs x {T} steps")
        else:
            csv_path = trajectory_cfg["csv_path"]
            _, pos, vel, quat, omega = _load_trajectory(csv_path, sim_dt)
            T = len(pos)
            self._traj_len = T
            # Broadcast single trajectory to all envs
            self._traj_pos = np.tile(pos[np.newaxis], (N, 1, 1))
            self._traj_vel = np.tile(vel[np.newaxis], (N, 1, 1))
            self._traj_quat = np.tile(quat[np.newaxis], (N, 1, 1))
            self._traj_omega = np.tile(omega[np.newaxis], (N, 1, 1))

        # Future goals config
        self._n_future = int(trajectory_cfg.get("future_goals", 0))
        self._future_spacing = max(1, int(trajectory_cfg.get("future_spacing", 1)))
        self._future_dim = self._n_future * 3

        if self._n_future > 0:
            aug_dim = self._inner_obs_dim + self._future_dim
            self._observation_space = gym.spaces.Box(
                low=np.full(aug_dim, -np.inf, dtype=np.float32),
                high=np.full(aug_dim, np.inf, dtype=np.float32),
                shape=(aug_dim,),
                dtype=np.float32,
            )
            print(f"  Future goals: {self._n_future} waypoints, "
                  f"spacing={self._future_spacing} steps, "
                  f"obs dim {self._inner_obs_dim} -> {aug_dim}")
        else:
            self._observation_space = venv.observation_space

        self._step_counts = np.zeros(self._num_envs, dtype=np.int64)
        self._force_done = np.zeros(self._num_envs, dtype=bool)

        self._impl = self._find_impl(venv)
        self._obs_buf = np.zeros((self._num_envs, self._impl.getObsDim()), dtype=np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_impl(env):
        """Walk the wrapper chain to find the C++ impl."""
        cur = env
        for _ in range(64):
            if hasattr(cur, "_impl"):
                return cur._impl
            cur = getattr(cur, "venv", None)
            if cur is None:
                break
        raise AttributeError("Cannot find _impl in the wrapper chain")

    def _regenerate_trajectory(self, env_id):
        """Generate a new random trajectory for a single env."""
        if self._generator is None or self._frozen_traj[env_id]:
            return
        p, v, q, o = self._generator.generate(self._rng)
        self._traj_pos[env_id] = p
        self._traj_vel[env_id] = v
        self._traj_quat[env_id] = q
        self._traj_omega[env_id] = o

    def _regenerate_shared_trajectory(self):
        """Generate one trajectory and broadcast to all envs (skipped when frozen)."""
        if self._generator is None or np.any(self._frozen_traj):
            return
        p, v, q, o = self._generator.generate(self._rng)
        self._traj_pos[:] = p[np.newaxis]
        self._traj_vel[:] = v[np.newaxis]
        self._traj_quat[:] = q[np.newaxis]
        self._traj_omega[:] = o[np.newaxis]

    def set_eval_trajectory(self, env_id, pos, vel, quat, omega):
        """Set a specific trajectory for *env_id* and freeze it (skip regeneration)."""
        self._traj_pos[env_id] = pos
        self._traj_vel[env_id] = vel
        self._traj_quat[env_id] = quat
        self._traj_omega[env_id] = omega
        self._frozen_traj[env_id] = True

    def unfreeze_trajectories(self):
        """Re-enable trajectory regeneration for all envs."""
        self._frozen_traj[:] = False

    def _clamp_steps(self):
        """Handle trajectory end: loop or flag forced done."""
        over = self._step_counts >= self._traj_len
        if not np.any(over):
            return
        if self._loop:
            self._step_counts[over] %= self._traj_len
        else:
            self._step_counts[over] = self._traj_len - 1
            self._force_done[over] = True

    def _set_goals_for_steps(self, steps):
        """Push trajectory positions at *steps* as C++ goal."""
        goals = self._traj_pos[self._env_idx, steps]  # (N, 3)
        self._impl.setEnvGoalPositions(goals.astype(np.float32).copy())

    def _set_reward_x_goal(self, steps):
        """Update CustomRewardWrapper._x_goal with per-env desired state."""
        if self._reward_wrapper is None:
            return
        idx = self._env_idx
        x_goal = np.zeros((self._num_envs, 13), dtype=np.float32)
        x_goal[:, _POS_SLICE] = 0.0
        x_goal[:, _ORI_SLICE] = self._traj_quat[idx, steps]
        x_goal[:, _VEL_SLICE] = self._traj_vel[idx, steps]
        x_goal[:, _OMEGA_SLICE] = self._traj_omega[idx, steps]
        self._reward_wrapper.set_x_goal(x_goal)

    def _compute_future_goals(self, steps):
        """Compute relative position offsets to N future waypoints.

        Returns shape (num_envs, n_future * 3).
        """
        idx = self._env_idx
        current_pos = self._traj_pos[idx, steps]  # (N, 3)
        out = np.zeros((self._num_envs, self._future_dim), dtype=np.float32)
        for k in range(self._n_future):
            future_step = steps + (k + 1) * self._future_spacing
            if self._loop:
                future_step = future_step % self._traj_len
            else:
                future_step = np.clip(future_step, 0, self._traj_len - 1)
            offset = self._traj_pos[idx, future_step] - current_pos
            out[:, k * 3:(k + 1) * 3] = offset
        return out

    def _augment_obs(self, obs, steps):
        """Append future goal offsets to observation if configured."""
        if self._n_future <= 0:
            return obs
        future = self._compute_future_goals(steps)
        return np.concatenate([obs, future], axis=1).astype(np.float32)

    def _augment_single(self, obs_1d, env_id, step):
        """Append future goals for a single env (used for terminal_observation)."""
        if self._n_future <= 0:
            return obs_1d
        current_pos = self._traj_pos[env_id, step]
        parts = [obs_1d]
        for k in range(self._n_future):
            fs = step + (k + 1) * self._future_spacing
            if self._loop:
                fs = fs % self._traj_len
            else:
                fs = min(fs, self._traj_len - 1)
            parts.append(self._traj_pos[env_id, fs] - current_pos)
        return np.concatenate(parts).astype(np.float32)

    def _convert_obs_to_errors(self, obs, steps, env_mask=None, obs_rows=None):
        """Convert absolute vel/quat/omega in *obs* to trajectory-relative errors.

        Operates in-place on the first 13 dims of obs.
        *env_mask*: which env indices to use for trajectory lookup.
        *obs_rows*: which rows of obs to modify (defaults to env_mask).
                    Useful when obs is a small temporary array (e.g. terminal obs).

        *steps* is either the full step_counts array (length == num_envs,
        indexed by env_mask) or a pre-extracted array aligned 1:1 with
        env_mask/obs_rows (used for terminal observations).
        """
        if env_mask is not None and len(env_mask) == 0:
            return
        if env_mask is None:
            env_mask = np.arange(self._num_envs)
        if obs_rows is None:
            obs_rows = env_mask

        if isinstance(steps, np.ndarray) and len(steps) == self._num_envs:
            s = steps[env_mask]
        else:
            s = steps

        obs[obs_rows, _VEL_SLICE.start:_VEL_SLICE.stop] -= self._traj_vel[env_mask, s]
        obs[obs_rows, _OMEGA_SLICE.start:_OMEGA_SLICE.stop] -= self._traj_omega[env_mask, s]

        q_actual = obs[obs_rows, _ORI_SLICE.start:_ORI_SLICE.stop].copy()
        q_desired = self._traj_quat[env_mask, s]
        q_err = _quat_multiply_batch(
            _quat_inverse_batch(q_desired.astype(np.float32)),
            q_actual.astype(np.float32),
        )
        q_err = _quat_normalize_batch(q_err)
        obs[obs_rows, _ORI_SLICE.start:_ORI_SLICE.stop] = q_err

    # ------------------------------------------------------------------
    # VecEnv interface
    # ------------------------------------------------------------------

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self.venv.action_space

    def reset(self, **kwargs):
        self._step_counts.fill(0)
        self._force_done.fill(False)

        if self._generator is not None:
            if self._shared_trajectory:
                self._regenerate_shared_trajectory()
                self._shared_done_count = 0
            else:
                for i in range(self._num_envs):
                    self._regenerate_trajectory(i)

        self._set_goals_for_steps(self._step_counts)
        self._set_reward_x_goal(self._step_counts)

        obs = self.venv.reset(**kwargs)
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        self._impl.getObs(self._obs_buf)
        obs[:, :self._impl.getObsDim()] = self._obs_buf

        self._convert_obs_to_errors(obs, self._step_counts)
        obs = self._augment_obs(obs, self._step_counts)
        return obs

    def step_async(self, actions):
        self._step_counts += 1
        self._clamp_steps()
        self._set_goals_for_steps(self._step_counts)
        self._set_reward_x_goal(self._step_counts)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        obs = np.asarray(obs, dtype=np.float32)

        if np.any(self._force_done):
            for i in np.where(self._force_done)[0]:
                if not dones[i]:
                    dones[i] = True
                    infos[i]["terminal_observation"] = obs[i].copy()
                    infos[i]["trajectory_end"] = True
            self._force_done.fill(False)

        done_ids = np.where(dones)[0]

        self._convert_obs_to_errors(obs, self._step_counts)

        for i in done_ids:
            if "terminal_observation" in infos[i]:
                tobs = infos[i]["terminal_observation"].copy().reshape(1, -1)
                step_arr = np.array([self._step_counts[i]])
                self._convert_obs_to_errors(
                    tobs, step_arr,
                    env_mask=np.array([i]), obs_rows=np.array([0]))
                infos[i]["terminal_observation"] = self._augment_single(
                    tobs[0], i, self._step_counts[i])

        if len(done_ids) > 0:
            self._step_counts[done_ids] = 0
            if self._generator is not None:
                if self._shared_trajectory:
                    self._shared_done_count += len(done_ids)
                    if self._shared_done_count >= self._shared_refresh_interval:
                        self._regenerate_shared_trajectory()
                        self._shared_done_count = 0
                else:
                    for i in done_ids:
                        self._regenerate_trajectory(i)
            self._set_goals_for_steps(self._step_counts)
            self._impl.getObs(self._obs_buf)
            for i in done_ids:
                obs[i, :self._impl.getObsDim()] = self._obs_buf[i]
            self._convert_obs_to_errors(obs, self._step_counts, env_mask=done_ids)

        obs = self._augment_obs(obs, self._step_counts)
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
