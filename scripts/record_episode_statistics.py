"""
Record episode statistics: returns, lengths, optional custom trackers (accumulate/queue).
Compatible with Gymnasium and SB3 VecEnv (step_async/step_wait).
"""
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np


class RecordEpisodeStatistics(gym.Wrapper):
    """Keep track of episode length and returns per instantiated env.

    Optionally track custom stats via add_tracker (accumulate or queue).
    """

    def __init__(self, env, deque_size=None, **kwargs):
        super().__init__(env)
        self.deque_size = deque_size
        self.t0 = time.time()
        self.episode_return = 0.0
        self.episode_length = 0
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.episode_stats = {}
        self.accumulated_stats = {}
        self.queued_stats = {}

    def add_tracker(self, name, init_value, mode="accumulate"):
        """Add a stat to track.

        Modes:
            accumulate: rolling sum (e.g. total constraint violations).
            queue: finite storage per episode (e.g. returns, lengths).
        """
        self.episode_stats[name] = init_value
        if mode == "accumulate":
            self.accumulated_stats[name] = init_value
        elif mode == "queue":
            self.queued_stats[name] = deque(maxlen=self.deque_size)
        else:
            raise ValueError("Tracker mode must be 'accumulate' or 'queue'.")

    def reset(self, **kwargs):
        self.episode_return = 0.0
        self.episode_length = 0
        for key in self.episode_stats:
            self.episode_stats[key] *= 0
        out = self.env.reset(**kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        info = out[1] if isinstance(out, tuple) and len(out) > 1 else {}
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            observation, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            observation, reward, done, info = out
            terminated = truncated = done

        self.episode_return += reward
        self.episode_length += 1
        for key in self.episode_stats:
            if key in info:
                self.episode_stats[key] += info[key]
        if done:
            info["episode"] = {
                "r": self.episode_return,
                "l": self.episode_length,
                "t": round(time.time() - self.t0, 6),
            }
            self.return_queue.append(self.episode_return)
            self.length_queue.append(self.episode_length)
            for key in self.episode_stats:
                info["episode"][key] = deepcopy(self.episode_stats[key])
                if key in self.accumulated_stats:
                    self.accumulated_stats[key] += deepcopy(self.episode_stats[key])
                if key in self.queued_stats:
                    self.queued_stats[key].append(deepcopy(self.episode_stats[key]))
                self.episode_stats[key] *= 0
            self.episode_return = 0.0
            self.episode_length = 0
        if len(out) == 5:
            return observation, reward, terminated, truncated, info
        return observation, reward, done, info


class VecRecordEpisodeStatistics:
    """Vectorized wrapper that records episodic statistics (returns, lengths, optional trackers).

    Works with any VecEnv that implements step_async, step_wait, reset, and returns
    infos as a list of dicts (one per env). Exposes return_queue, length_queue for logging.
    """

    def __init__(self, venv, deque_size=None, **kwargs):
        self.venv = venv
        self.deque_size = deque_size
        self.t0 = time.time()
        self._num_envs = getattr(venv, "num_envs", venv._num_envs)
        self.episode_return = np.zeros(self._num_envs)
        self.episode_length = np.zeros(self._num_envs, dtype=int)
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.episode_stats = {}
        self.accumulated_stats = {}
        self.queued_stats = {}

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def observation_space(self):
        return self.venv.observation_space

    @property
    def action_space(self):
        return self.venv.action_space

    def add_tracker(self, name, init_value, mode="accumulate"):
        """Add a stat to track (accumulate or queue)."""
        self.episode_stats[name] = [deepcopy(init_value) for _ in range(self._num_envs)]
        if mode == "accumulate":
            self.accumulated_stats[name] = deepcopy(init_value)
        elif mode == "queue":
            self.queued_stats[name] = deque(maxlen=self.deque_size)
        else:
            raise ValueError("Tracker mode must be 'accumulate' or 'queue'.")

    def reset(self, **kwargs):
        self.episode_return = np.zeros(self._num_envs)
        self.episode_length = np.zeros(self._num_envs, dtype=int)
        for key in self.episode_stats:
            for i in range(self._num_envs):
                self.episode_stats[key][i] *= 0
        out = self.venv.reset(**kwargs)
        return out[0] if isinstance(out, tuple) else out

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for i in range(self._num_envs):
            self.episode_return[i] += rewards[i]
            self.episode_length[i] += 1
            for key in self.episode_stats:
                if key in infos[i]:
                    self.episode_stats[key][i] += infos[i][key]
            if dones[i]:
                infos[i]["episode"] = {
                    "r": float(self.episode_return[i]),
                    "l": int(self.episode_length[i]),
                    "t": round(time.time() - self.t0, 6),
                }
                self.return_queue.append(float(self.episode_return[i]))
                self.length_queue.append(int(self.episode_length[i]))
                self.episode_return[i] = 0.0
                self.episode_length[i] = 0
                for key in self.episode_stats:
                    infos[i]["episode"][key] = deepcopy(self.episode_stats[key][i])
                    if key in self.accumulated_stats:
                        self.accumulated_stats[key] += deepcopy(self.episode_stats[key][i])
                    if key in self.queued_stats:
                        self.queued_stats[key].append(deepcopy(self.episode_stats[key][i]))
                    self.episode_stats[key][i] *= 0
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
