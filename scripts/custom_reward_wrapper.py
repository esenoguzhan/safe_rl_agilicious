"""
Custom reward wrapper: overwrites C++ env reward when enabled.

Four reward modes (set via YAML `env.custom_reward.mode`):

  "weighted_exp" (legacy):
    dist = state_weight @ (obs - x_goal)^2 + act_weight @ (act - act_goal)^2
    rew = exp(-dist)  [if rew_exponential]  or  rew = -dist

  "cauchy" (recommended for hover):
    Same dist as weighted_exp, but a dual-scale Cauchy mixture:
      rew_wide  = 1 / (1 + cauchy_scale_wide  * dist)   # navigable signal at large error
      rew_sharp = 1 / (1 + cauchy_scale_sharp * dist)   # precision near goal
      rew = cauchy_wide_weight * rew_wide + cauchy_sharp_weight * rew_sharp
    Always positive, bounded, coupled; wide term keeps gradient informative far from goal.

  "sum_of_exp":
    rew = w_pos  * exp(-k_pos  * ||pos_err||^2)  + ...  per group
    Each term provides independent gradients; total reward in [0, sum_of_weights].

  "path_aligned" (recommended for trajectory tracking):
    Decomposes position error into cross-track (perpendicular to path) and
    along-track (progress along path). Uses reference velocity as path tangent.
    All proximity terms use Cauchy kernels (heavy-tailed, always informative).
    rew = w_ct / (1 + k_ct * cross_track^2)
        + w_prog * dot(vel, tangent) / k_prog
        + w_ori / (1 + k_ori * ori_err)
        + w_omega / (1 + k_omega * ||omega_err||^2)
        - action_rate_penalty

Optional L1 position penalty (all modes): rew -= l1_pos_penalty * (|ex|+|ey|+|ez|).
Provides component-wise linear gradient toward zero error. Set l1_pos_penalty > 0 to enable.

Optional Cauchy-only additive term (YAML ``goal_approach_weight``):
  Normalized closing direction: d_hat = pos_err / max(||pos_err||, eps) with
  pos_err = obs[..., :3] - x_goal[..., :3]. World velocity is
  v_world = obs[..., 7:10] + x_goal[..., 7:10] so it matches both hover
  (absolute vel in obs) and trajectory tracking (vel error + reference in x_goal).
  rew += goal_approach_weight * (v_world · d_hat)

All parameters from YAML env.custom_reward.
"""
import numpy as np

DEFAULT_X_GOAL = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_ACT_GOAL = [0.0, 0.0, 0.0, 0.0]

# obs layout: [pos_err(3), quat(4), lin_vel(3), ang_vel(3)] = 13
_POS_SLICE = slice(0, 3)
_ORI_SLICE = slice(3, 7)
_VEL_SLICE = slice(7, 10)
_OMEGA_SLICE = slice(10, 13)

DEFAULT_SUM_OF_EXP_TERMS = {
    "position":    {"weight": 0.4,  "scale": 0.5},
    "orientation": {"weight": 0.3,  "scale": 1.0},
    "velocity":    {"weight": 0.15, "scale": 0.5},
    "ang_velocity":{"weight": 0.1,  "scale": 0.1},
    "action":      {"weight": 0.05, "scale": 0.1},
}

DEFAULT_PATH_ALIGNED_TERMS = {
    "cross_track":  {"weight": 0.4,  "scale": 2.0},
    "progress":     {"weight": 0.3},
    "orientation":  {"weight": 0.2,  "scale": 1.0},
    "ang_velocity": {"weight": 0.1,  "scale": 0.5},
}


def _parse_list(cfg, key, default, length):
    raw = cfg.get(key, default)
    if raw is None:
        return np.array(default, dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 1:
        return np.full(length, float(arr.flat[0]), dtype=np.float32)
    if arr.size != length:
        raise ValueError(f"{key} must have length {length} or 1, got {arr.size}")
    return arr.flat[:length].copy() if arr.ndim > 1 else arr


def _l1_norm_position_error(pos_err):
    """Per-env L1 norm of 3D position error: |ex| + |ey| + |ez|."""
    return np.sum(np.abs(pos_err), axis=1)


def _goal_approach_normalized(obs, x_goal, eps):
    """Per-env dot(v_world, pos_err / ||pos_err||). See module docstring."""
    pos_err = obs[:, _POS_SLICE] - x_goal[..., _POS_SLICE]
    pe_norm = np.linalg.norm(pos_err, axis=1, keepdims=True)
    d_hat = pos_err / np.maximum(pe_norm, eps)
    vel_world = obs[:, _VEL_SLICE] + x_goal[..., _VEL_SLICE]
    return np.sum(vel_world * d_hat, axis=1)


class CustomRewardWrapper:
    """
    VecEnv wrapper that overwrites step rewards with a configurable reward function.
    Modes: "weighted_exp", "cauchy", "sum_of_exp", "path_aligned".
    """

    def __init__(self, venv, custom_reward_cfg):
        self.venv = venv
        cfg = custom_reward_cfg or {}
        self.enabled = cfg.get("enabled", False)
        if not self.enabled:
            return

        obs_dim = getattr(venv, "observation_space", None)
        if hasattr(obs_dim, "shape"):
            obs_dim = obs_dim.shape[0]
        else:
            obs_dim = 13
        act_dim = getattr(venv, "action_space", None)
        if hasattr(act_dim, "shape"):
            act_dim = act_dim.shape[0]
        else:
            act_dim = 4

        self._mode = cfg.get("mode", "weighted_exp")
        self._act_goal = _parse_list(cfg, "act_goal", DEFAULT_ACT_GOAL, act_dim)
        self._rew_act_rate_weight = _parse_list(cfg, "rew_act_rate_weight", [0.0] * act_dim, act_dim)
        self._pending_actions = None
        self._prev_actions = None

        self._crash_penalty = float(cfg.get("crash_penalty", 0.0))
        self._max_pos_error = float(cfg.get("max_pos_error", 0.0))
        self._l1_pos_penalty = float(cfg.get("l1_pos_penalty", 0.0))

        self._goal_approach_weight = float(cfg.get("goal_approach_weight", 0.0))
        self._goal_approach_eps = float(cfg.get("goal_approach_eps", 1e-6))
        if self._mode != "cauchy":
            self._goal_approach_weight = 0.0

        if self._mode == "sum_of_exp":
            self._init_sum_of_exp(cfg, obs_dim)
        elif self._mode == "path_aligned":
            self._init_path_aligned(cfg, obs_dim)
        else:
            self._init_weighted_exp(cfg, obs_dim, act_dim)
            if self._mode == "cauchy":
                self._cauchy_scale_wide = float(cfg.get("cauchy_scale_wide", 0.02))
                self._cauchy_scale_sharp = float(cfg.get("cauchy_scale_sharp", 2.0))
                self._cauchy_wide_weight = float(cfg.get("cauchy_wide_weight", 0.6))
                self._cauchy_sharp_weight = float(cfg.get("cauchy_sharp_weight", 0.4))

    def _init_weighted_exp(self, cfg, obs_dim, act_dim):
        self._x_goal = _parse_list(cfg, "x_goal", DEFAULT_X_GOAL, obs_dim)
        self._rew_state_weight = _parse_list(
            cfg, "rew_state_weight",
            [0.1] * 3 + [0.2] * 4 + [0.01] * 6,
            obs_dim,
        )
        self._rew_act_weight = _parse_list(cfg, "rew_act_weight", [0.001] * act_dim, act_dim)
        self._rew_exponential = cfg.get("rew_exponential", False)

    def _init_sum_of_exp(self, cfg, obs_dim):
        terms_cfg = cfg.get("terms", DEFAULT_SUM_OF_EXP_TERMS)
        self._x_goal = _parse_list(cfg, "x_goal", DEFAULT_X_GOAL, obs_dim)

        def _t(name):
            t = terms_cfg.get(name, DEFAULT_SUM_OF_EXP_TERMS.get(name, {"weight": 0.0, "scale": 1.0}))
            return float(t.get("weight", 0.0)), float(t.get("scale", 1.0))

        self._pos_w, self._pos_k = _t("position")
        self._ori_w, self._ori_k = _t("orientation")
        self._vel_w, self._vel_k = _t("velocity")
        self._omega_w, self._omega_k = _t("ang_velocity")
        self._act_w, self._act_k = _t("action")

    def _init_path_aligned(self, cfg, obs_dim):
        self._x_goal = _parse_list(cfg, "x_goal", DEFAULT_X_GOAL, obs_dim)
        terms_cfg = cfg.get("terms", DEFAULT_PATH_ALIGNED_TERMS)

        def _t(name):
            t = terms_cfg.get(name, DEFAULT_PATH_ALIGNED_TERMS.get(
                name, {"weight": 0.0, "scale": 1.0}))
            return float(t.get("weight", 0.0)), float(t.get("scale", 1.0))

        self._ct_w, self._ct_k = _t("cross_track")
        self._prog_w, self._prog_k = _t("progress")
        self._ori_w, self._ori_k = _t("orientation")
        self._omega_w, self._omega_k = _t("ang_velocity")

        self._along_track_k = float(cfg.get("along_track_scale", 0.0))

    def _compute_reward(self, obs, actions):
        if self._mode == "sum_of_exp":
            return self._compute_sum_of_exp(obs, actions)
        if self._mode == "cauchy":
            return self._compute_cauchy(obs, actions)
        if self._mode == "path_aligned":
            return self._compute_path_aligned(obs, actions)
        return self._compute_weighted_exp(obs, actions)

    def _compute_weighted_exp(self, obs, actions):
        state_error = obs - self._x_goal
        act_error = actions - self._act_goal
        dist = np.sum(self._rew_state_weight * state_error * state_error, axis=1) + np.sum(
            self._rew_act_weight * act_error * act_error, axis=1
        )
        # Replace quaternion component L2 with sign-invariant orientation error:
        # 1 - (q · q_goal)^2, which is identical for q and -q.
        ori_old = np.sum(
            self._rew_state_weight[_ORI_SLICE]
            * state_error[:, _ORI_SLICE]
            * state_error[:, _ORI_SLICE],
            axis=1,
        )
        ori_new = np.sum(self._rew_state_weight[_ORI_SLICE]) * self._quat_ori_error(obs)
        dist += (ori_new - ori_old)
        if self._prev_actions is not None:
            act_delta = actions - self._prev_actions
            dist += np.sum(self._rew_act_rate_weight * act_delta * act_delta, axis=1)
        rew = -dist
        if self._rew_exponential:
            rew = np.exp(rew)
        if self._l1_pos_penalty > 0:
            pos_err = obs[:, _POS_SLICE] - self._x_goal[..., _POS_SLICE]
            rew -= self._l1_pos_penalty * _l1_norm_position_error(pos_err)
        return rew.astype(np.float32)

    def _compute_cauchy(self, obs, actions):
        state_error = obs - self._x_goal
        act_error = actions - self._act_goal
        dist = np.sum(self._rew_state_weight * state_error * state_error, axis=1) + np.sum(
            self._rew_act_weight * act_error * act_error, axis=1
        )
        # Replace quaternion component L2 with sign-invariant orientation error:
        # 1 - (q · q_goal)^2, which is identical for q and -q.
        ori_old = np.sum(
            self._rew_state_weight[_ORI_SLICE]
            * state_error[:, _ORI_SLICE]
            * state_error[:, _ORI_SLICE],
            axis=1,
        )
        ori_new = np.sum(self._rew_state_weight[_ORI_SLICE]) * self._quat_ori_error(obs)
        dist += (ori_new - ori_old)
        if self._prev_actions is not None:
            act_delta = actions - self._prev_actions
            dist += np.sum(self._rew_act_rate_weight * act_delta * act_delta, axis=1)
        rew_wide = 1.0 / (1.0 + self._cauchy_scale_wide * dist)
        rew_sharp = 1.0 / (1.0 + self._cauchy_scale_sharp * dist)
        rew = (
            self._cauchy_wide_weight * rew_wide
            + self._cauchy_sharp_weight * rew_sharp
        )
        if self._goal_approach_weight != 0.0:
            rew = rew + self._goal_approach_weight * _goal_approach_normalized(
                obs, self._x_goal, self._goal_approach_eps,
            )
        if self._l1_pos_penalty > 0:
            pos_err = obs[:, _POS_SLICE] - self._x_goal[..., _POS_SLICE]
            rew -= self._l1_pos_penalty * _l1_norm_position_error(pos_err)
        return rew.astype(np.float32)

    def _compute_sum_of_exp(self, obs, actions):
        pos_err = obs[:, _POS_SLICE] - self._x_goal[..., _POS_SLICE]
        vel = obs[:, _VEL_SLICE] - self._x_goal[..., _VEL_SLICE]
        omega = obs[:, _OMEGA_SLICE] - self._x_goal[..., _OMEGA_SLICE]
        act_err = actions - self._act_goal
        ori_err_scalar = self._quat_ori_error(obs)

        rew = (
            self._pos_w * np.exp(-self._pos_k * np.sum(pos_err * pos_err, axis=1))
            + self._ori_w * np.exp(-self._ori_k * ori_err_scalar)
            + self._vel_w * np.exp(-self._vel_k * np.sum(vel * vel, axis=1))
            + self._omega_w * np.exp(-self._omega_k * np.sum(omega * omega, axis=1))
            + self._act_w * np.exp(-self._act_k * np.sum(act_err * act_err, axis=1))
        )

        if self._prev_actions is not None:
            act_delta = actions - self._prev_actions
            rew -= np.sum(self._rew_act_rate_weight * act_delta * act_delta, axis=1)

        if self._l1_pos_penalty > 0:
            rew -= self._l1_pos_penalty * _l1_norm_position_error(pos_err)

        return rew.astype(np.float32)

    def _compute_path_aligned(self, obs, actions):
        pos = obs[:, _POS_SLICE]
        vel = obs[:, _VEL_SLICE]
        pos_ref = self._x_goal[..., _POS_SLICE]
        vel_ref = self._x_goal[..., _VEL_SLICE]
        omega = obs[:, _OMEGA_SLICE]
        omega_ref = self._x_goal[..., _OMEGA_SLICE]

        # Path tangent from reference velocity
        vel_ref_norm = np.linalg.norm(vel_ref, axis=-1, keepdims=True)
        tangent = vel_ref / (vel_ref_norm + 1e-6)

        # Decompose position error into along-track and cross-track
        pos_err = pos - pos_ref
        along_track_dist = np.sum(pos_err * tangent, axis=1, keepdims=True)
        cross_track_err = pos_err - along_track_dist * tangent
        cross_track_sq = np.sum(cross_track_err ** 2, axis=1)

        # Progress: velocity projected onto path tangent, normalised by scale
        progress = np.sum(vel * tangent, axis=1) / (self._prog_k + 1e-8)

        # Cauchy kernels: w / (1 + k * err^2)  — heavy-tailed, always informative
        rew = self._ct_w / (1.0 + self._ct_k * cross_track_sq)

        if self._along_track_k > 0:
            rew += self._ct_w / (1.0 + self._along_track_k * along_track_dist.ravel() ** 2)

        rew += self._prog_w * progress

        rew += self._ori_w / (1.0 + self._ori_k * self._quat_ori_error(obs))

        omega_err = omega - omega_ref
        rew += self._omega_w / (1.0 + self._omega_k * np.sum(omega_err ** 2, axis=1))

        if self._prev_actions is not None:
            act_delta = actions - self._prev_actions
            rew -= np.sum(self._rew_act_rate_weight * act_delta * act_delta, axis=1)

        if self._l1_pos_penalty > 0:
            rew -= self._l1_pos_penalty * _l1_norm_position_error(pos_err)

        return rew.astype(np.float32)

    def set_x_goal(self, goal):
        """Set x_goal. Accepts shape (obs_dim,) or (num_envs, obs_dim)."""
        self._x_goal = np.asarray(goal, dtype=np.float32)

    def set_act_goal(self, goal):
        """Set act_goal. Accepts shape (act_dim,) or (num_envs, act_dim)."""
        self._act_goal = np.asarray(goal, dtype=np.float32)

    def _quat_ori_error(self, obs):
        """Sign-invariant quaternion orientation error: 1 - (q·q_goal)^2.

        Supports both a single shared goal (shape (4,)) and per-env goals
        (shape (N, 4)) for trajectory tracking.
        """
        q = obs[:, _ORI_SLICE].astype(np.float32)
        q_goal = self._x_goal[..., _ORI_SLICE].astype(np.float32)

        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm = np.clip(q_norm, 1e-8, None)
        qn = q / q_norm

        if q_goal.ndim == 1:
            qg_norm = np.linalg.norm(q_goal)
            if qg_norm < 1e-8:
                qg = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                qg = q_goal / qg_norm
            dot = np.sum(qn * qg[None, :], axis=1)
        else:
            qg_norm = np.linalg.norm(q_goal, axis=1, keepdims=True)
            qg_norm = np.clip(qg_norm, 1e-8, None)
            qg = q_goal / qg_norm
            dot = np.sum(qn * qg, axis=1)

        return 1.0 - np.square(dot)

    @property
    def num_envs(self):
        return self.venv.num_envs

    @property
    def observation_space(self):
        return self.venv.observation_space

    @property
    def action_space(self):
        return self.venv.action_space

    def reset(self, **kwargs):
        self._pending_actions = None
        self._prev_actions = None
        out = self.venv.reset(**kwargs)
        return out[0] if isinstance(out, tuple) else out

    def step_async(self, actions):
        if self.enabled:
            self._pending_actions = np.asarray(actions, dtype=np.float32)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if self.enabled and self._pending_actions is not None:
            reward_obs = obs.copy()
            for i in range(len(dones)):
                if dones[i] and "terminal_observation" in infos[i]:
                    reward_obs[i] = infos[i]["terminal_observation"]

            # Out-of-bounds early termination: if position error exceeds
            # max_pos_error, treat as a crash (done + penalty).
            if self._max_pos_error > 0:
                for i in range(len(dones)):
                    if not dones[i]:
                        pos_err_sq = float(np.sum(reward_obs[i, _POS_SLICE] ** 2))
                        if pos_err_sq > self._max_pos_error ** 2:
                            dones[i] = True
                            infos[i]["terminal_observation"] = reward_obs[i].copy()
                            infos[i]["oob_termination"] = True

            rewards = self._compute_reward(reward_obs, self._pending_actions)

            # Crash penalty: any done seen here is from C++ terminal state
            # (ground crash) or out-of-bounds, not from episode truncation
            # (which is handled by VecMaxEpisodeSteps above this wrapper).
            if self._crash_penalty != 0.0:
                for i in range(len(dones)):
                    if dones[i]:
                        rewards[i] = self._crash_penalty

            prev = self._pending_actions.copy()
            if np.any(dones):
                for i in range(len(dones)):
                    if dones[i]:
                        prev[i] = 0.0
            self._prev_actions = prev
            self._pending_actions = None
        return obs, rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if hasattr(self.venv, "close"):
            self.venv.close()

    def seed(self, seed=None):
        if hasattr(self.venv, "seed"):
            return self.venv.seed(seed)
        return None

    def set_seed(self, seed):
        if hasattr(self.venv, "set_seed"):
            self.venv.set_seed(seed)

    def __getattr__(self, name):
        return getattr(self.venv, name)
