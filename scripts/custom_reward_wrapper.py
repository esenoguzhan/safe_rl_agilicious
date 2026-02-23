"""
Custom reward wrapper: overwrites C++ env reward when enabled.

Three reward modes (set via YAML `env.custom_reward.mode`):

  "weighted_exp" (legacy):
    dist = state_weight @ (obs - x_goal)^2 + act_weight @ (act - act_goal)^2
    rew = exp(-dist)  [if rew_exponential]  or  rew = -dist

  "cauchy" (recommended):
    Same dist as weighted_exp, but:  rew = 1 / (1 + dist)
    Always positive [0,1], bounded, coupled, and gradient never vanishes.

  "sum_of_exp":
    rew = w_pos  * exp(-k_pos  * ||pos_err||^2)  + ...  per group
    Each term provides independent gradients; total reward in [0, sum_of_weights].

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


class CustomRewardWrapper:
    """
    VecEnv wrapper that overwrites step rewards with a configurable reward function.
    Modes: "weighted_exp", "cauchy" (recommended), "sum_of_exp".
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

        if self._mode == "sum_of_exp":
            self._init_sum_of_exp(cfg, obs_dim)
        else:
            self._init_weighted_exp(cfg, obs_dim, act_dim)
            if self._mode == "cauchy":
                self._cauchy_scale = float(cfg.get("cauchy_scale", 1.0))

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

    def _compute_reward(self, obs, actions):
        if self._mode == "sum_of_exp":
            return self._compute_sum_of_exp(obs, actions)
        if self._mode == "cauchy":
            return self._compute_cauchy(obs, actions)
        return self._compute_weighted_exp(obs, actions)

    def _compute_weighted_exp(self, obs, actions):
        state_error = obs - self._x_goal
        act_error = actions - self._act_goal
        dist = np.sum(self._rew_state_weight * state_error * state_error, axis=1) + np.sum(
            self._rew_act_weight * act_error * act_error, axis=1
        )
        if self._prev_actions is not None:
            act_delta = actions - self._prev_actions
            dist += np.sum(self._rew_act_rate_weight * act_delta * act_delta, axis=1)
        rew = -dist
        if self._rew_exponential:
            rew = np.exp(rew)
        return rew.astype(np.float32)

    def _compute_cauchy(self, obs, actions):
        state_error = obs - self._x_goal
        act_error = actions - self._act_goal
        dist = np.sum(self._rew_state_weight * state_error * state_error, axis=1) + np.sum(
            self._rew_act_weight * act_error * act_error, axis=1
        )
        if self._prev_actions is not None:
            act_delta = actions - self._prev_actions
            dist += np.sum(self._rew_act_rate_weight * act_delta * act_delta, axis=1)
        rew = 1.0 / (1.0 + self._cauchy_scale * dist)
        return rew.astype(np.float32)

    def _compute_sum_of_exp(self, obs, actions):
        pos_err = obs[:, _POS_SLICE] - self._x_goal[_POS_SLICE]
        ori_err = obs[:, _ORI_SLICE] - self._x_goal[_ORI_SLICE]
        vel = obs[:, _VEL_SLICE] - self._x_goal[_VEL_SLICE]
        omega = obs[:, _OMEGA_SLICE] - self._x_goal[_OMEGA_SLICE]
        act_err = actions - self._act_goal

        rew = (
            self._pos_w * np.exp(-self._pos_k * np.sum(pos_err * pos_err, axis=1))
            + self._ori_w * np.exp(-self._ori_k * np.sum(ori_err * ori_err, axis=1))
            + self._vel_w * np.exp(-self._vel_k * np.sum(vel * vel, axis=1))
            + self._omega_w * np.exp(-self._omega_k * np.sum(omega * omega, axis=1))
            + self._act_w * np.exp(-self._act_k * np.sum(act_err * act_err, axis=1))
        )

        if self._prev_actions is not None:
            act_delta = actions - self._prev_actions
            rew -= np.sum(self._rew_act_rate_weight * act_delta * act_delta, axis=1)

        return rew.astype(np.float32)

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
            rewards = self._compute_reward(reward_obs, self._pending_actions)
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
