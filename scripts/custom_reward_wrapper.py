"""
Custom reward wrapper: overwrites C++ env reward with quadratic state/action cost when enabled.
Reward: rew = -dist, dist = sum(rew_state_weight * (obs - x_goal)^2) + sum(rew_act_weight * (act - act_goal)^2).
Optional: rew = exp(rew) for positive bounded reward.
All parameters from YAML env.custom_reward.
"""
import numpy as np

# Default goal (matches flightlib quadrotor_env: pos 0,0,5; zero ori, vel, ang_vel)
DEFAULT_X_GOAL = [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_ACT_GOAL = [0.0, 0.0, 0.0, 0.0]


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
    VecEnv wrapper that overwrites step rewards with custom quadratic cost when enabled.
    rew = -dist, dist = rew_state_weight @ (obs - x_goal)^2 + rew_act_weight @ (act - act_goal)^2.
    Optional: rew = exp(rew).
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
            obs_dim = 12
        act_dim = getattr(venv, "action_space", None)
        if hasattr(act_dim, "shape"):
            act_dim = act_dim.shape[0]
        else:
            act_dim = 4
        self._x_goal = _parse_list(cfg, "x_goal", DEFAULT_X_GOAL, obs_dim)
        self._act_goal = _parse_list(cfg, "act_goal", DEFAULT_ACT_GOAL, act_dim)
        self._rew_state_weight = _parse_list(
            cfg, "rew_state_weight",
            [0.1] * 3 + [0.2] * 3 + [0.01] * 6,  # pos, ori, lin_vel, ang_vel
            obs_dim,
        )
        self._rew_act_weight = _parse_list(cfg, "rew_act_weight", [0.001] * act_dim, act_dim)
        self._rew_exponential = cfg.get("rew_exponential", False)
        self._pending_actions = None

    @property
    def num_envs(self):
        return self.venv.num_envs

    @property
    def observation_space(self):
        return self.venv.observation_space

    @property
    def action_space(self):
        return self.venv.action_space

    def _compute_reward(self, obs, actions):
        state_error = obs - self._x_goal
        act_error = actions - self._act_goal
        dist = np.sum(self._rew_state_weight * state_error * state_error, axis=1) + np.sum(
            self._rew_act_weight * act_error * act_error, axis=1
        )
        rew = -dist
        if self._rew_exponential:
            rew = np.exp(rew)
        return rew.astype(np.float32)

    def reset(self, **kwargs):
        self._pending_actions = None
        out = self.venv.reset(**kwargs)
        return out[0] if isinstance(out, tuple) else out

    def step_async(self, actions):
        if self.enabled:
            self._pending_actions = np.asarray(actions, dtype=np.float32)
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        if self.enabled and self._pending_actions is not None:
            rewards = self._compute_reward(obs, self._pending_actions)
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
