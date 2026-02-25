"""
CBF wrapper for VecEnv: filters RL policy actions through a CBF safety filter (position barriers)
before applying them to the environment. Use only at evaluation/deployment, not during training.

Observation layout (Flightmare): [pos_error(3), q(4), v(3), omega(3)] = 13.
State for CBF: [p(3), q(4), v(3), omega(3)] with p = goal_pos - pos_error.
Action: env expects normalized [-1, 1]; CBF works in motor thrusts [N], so we denormalize -> filter -> normalize.
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np

from scripts.cbf_filter import CBFFilter
from scripts.quadrotor_model import QuadrotorModel, STATE_DIM

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


class CBFWrapper:
    """
    VecEnv wrapper that runs the RL action through a CBF filter (position barriers) before stepping.
    Requires goal_position to reconstruct state from observation (pos_error -> p = goal - pos_error).
    Uses act_mean / act_std to convert between normalized actions and motor thrusts [N].
    """

    def __init__(
        self,
        venv,
        cbf_config_path: Optional[Union[str, Path]] = None,
        goal_position: Optional[Union[list, tuple, np.ndarray]] = None,
        act_mean: Optional[np.ndarray] = None,
        act_std: Optional[np.ndarray] = None,
    ):
        """
        venv: VecEnv with step_async / step_wait and observation shape (n_envs, 13).
        cbf_config_path: path to cbf_config.yaml (default: configs/cbf_config.yaml).
        goal_position: (3,) or (n_envs, 3) goal position for state reconstruction. If None, uses [0,0,5].
        act_mean, act_std: (4,) normalization for actions. If None, computed from CBF's quadrotor model
            as act_mean = (-mass*g)/4 * ones(4), act_std = (-mass*2*g)/4 * ones(4).
        """
        self.venv = venv
        self._num_envs = venv.num_envs
        self._cbf = CBFFilter(config_path=cbf_config_path)
        model = self._cbf.model

        if goal_position is not None:
            goal_position = np.asarray(goal_position, dtype=np.float64)
            if goal_position.ndim == 1:
                goal_position = np.tile(goal_position.ravel()[:3], (self._num_envs, 1))
            self._goal_pos = np.asarray(goal_position, dtype=np.float64)[: self._num_envs]
            if self._goal_pos.shape[0] < self._num_envs:
                self._goal_pos = np.tile(self._goal_pos[0], (self._num_envs, 1))
        else:
            self._goal_pos = np.tile(np.array([0.0, 0.0, 5.0], dtype=np.float64), (self._num_envs, 1))

        if act_mean is not None and act_std is not None:
            self._act_mean = np.asarray(act_mean, dtype=np.float32).ravel()[:4]
            self._act_std = np.asarray(act_std, dtype=np.float32).ravel()[:4]
            if self._act_mean.size < 4:
                self._act_mean = np.resize(self._act_mean, 4)
            if self._act_std.size < 4:
                self._act_std = np.resize(self._act_std, 4)
        else:
            mass = model.mass
            g = -model.gravity  # positive magnitude
            self._act_mean = np.full(4, (mass * g) / 4.0, dtype=np.float32)
            self._act_std = np.full(4, (mass * 2 * g) / 4.0, dtype=np.float32)

        self._pending_actions = None

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
        out = self.venv.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        self._last_obs = np.asarray(obs, dtype=np.float32)
        return out[0] if isinstance(out, tuple) else out

    def step_async(self, actions: np.ndarray):
        """Store RL actions; filtering happens in step_wait using cached _last_obs."""
        self._pending_actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_async was not called before step_wait")
        actions = self._pending_actions
        self._pending_actions = None

        obs = getattr(self, "_last_obs", None)
        if obs is None:
            safe_actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
            self.venv.step_async(safe_actions)
            obs, rewards, dones, infos = self.venv.step_wait()
            self._last_obs = obs
            return obs, rewards, dones, infos

        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        n_envs = min(actions.shape[0], self._num_envs)
        safe_actions = np.clip(actions.copy(), -1.0, 1.0).astype(np.float32)

        for i in range(n_envs):
            ob = obs[i].ravel()[:13]
            goal_i = self._goal_pos[i]
            state = self._cbf.model.state_from_observation(ob, goal_pos=goal_i)
            u_rl = actions[i, :4].astype(np.float64) * self._act_std.astype(np.float64) + self._act_mean.astype(np.float64)
            u_safe = self._cbf.filter(state, u_rl)
            anorm = (u_safe.astype(np.float32) - self._act_mean) / (self._act_std + 1e-8)
            safe_actions[i, :4] = np.clip(anorm, -1.0, 1.0)

        self.venv.step_async(safe_actions)
        obs, rewards, dones, infos = self.venv.step_wait()
        self._last_obs = obs
        return obs, rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if hasattr(self.venv, "close"):
            self.venv.close()

    def __getattr__(self, name):
        return getattr(self.venv, name)


def wrap_vec_env_with_cbf(
    venv,
    cbf_config_path: Optional[Union[str, Path]] = None,
    goal_position: Optional[Union[list, tuple, np.ndarray]] = None,
    act_mean: Optional[np.ndarray] = None,
    act_std: Optional[np.ndarray] = None,
) -> CBFWrapper:
    """Wrap a VecEnv with CBFWrapper. reset() caches obs so the first step is filtered correctly."""
    wrapper = CBFWrapper(
        venv,
        cbf_config_path=cbf_config_path,
        goal_position=goal_position,
        act_mean=act_mean,
        act_std=act_std,
    )
    return wrapper
