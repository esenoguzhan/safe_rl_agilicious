#!/usr/bin/env python3
"""
Shared RL policy logic (PPO + VecNormalize + FlightLib-style obs/thrust map).
Used by rl_feedthrough_node.py (rospy) and rl_feedthrough_rosbridge_client.py (roslibpy).
No ROS imports — pass numpy arrays or dicts matching rosbridge JSON shape.
"""
import csv
import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

OBS_BASE_DIM = 13
ACTION_HISTORY_LEN = 5
ACTION_DIM = 4


def obs_total_dim(action_history_len, action_dim=ACTION_DIM):
    # type: (int, int) -> int
    return OBS_BASE_DIM + int(action_history_len) * int(action_dim)


OBS_TOTAL_DIM = obs_total_dim(ACTION_HISTORY_LEN)


def model_run_dir(model_path):
    # type: (str) -> str
    if not model_path:
        return ""
    return os.path.dirname(os.path.abspath(model_path))


def action_history_len_from_model_config(model_path, default=ACTION_HISTORY_LEN):
    # type: (str, int) -> int
    """Read ``env.action_history_len`` from ``config.yaml`` beside the checkpoint."""
    run_dir = model_run_dir(model_path)
    if not run_dir:
        return int(default)
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        logger.warning(
            "config.yaml not found next to model (%s); action_history_len=%d",
            cfg_path,
            default,
        )
        return int(default)
    try:
        import yaml
    except ImportError:
        logger.warning(
            "PyYAML not installed; cannot read %s; action_history_len=%d",
            cfg_path,
            default,
        )
        return int(default)
    try:
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to parse %s: %s; action_history_len=%d", cfg_path, e, default)
        return int(default)
    env = data.get("env") or {}
    val = env.get("action_history_len")
    if val is None:
        logger.warning(
            "env.action_history_len missing in %s; action_history_len=%d",
            cfg_path,
            default,
        )
        return int(default)
    hist_len = int(val)
    logger.info(
        "action_history_len=%d from %s",
        hist_len,
        cfg_path,
    )
    return hist_len


def resolve_action_history_len(model_path, explicit_len=None):
    # type: (str, Optional[int]) -> int
    if explicit_len is not None:
        return int(explicit_len)
    return action_history_len_from_model_config(model_path)


try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class _ObsDummyEnv(gym.Env):
        def __init__(self, observation_dim):
            super().__init__()
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (int(observation_dim),), dtype=np.float32
            )
            self.action_space = spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, True, False, {}

except ImportError as _e:
    gym = None  # type: ignore
    spaces = None  # type: ignore
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore
    VecNormalize = None  # type: ignore
    _ObsDummyEnv = None  # type: ignore
    IMPORT_ERROR = _e
else:
    IMPORT_ERROR = None


def default_paths_under_repo(script_dir):
    # type: (str) -> Tuple[str, str]
    root_guess = os.path.normpath(os.path.join(script_dir, "..", "..", "..", "..", ".."))
    model = os.path.join(
        root_guess,
        "PPO_50000000_seq6dec_s6_tau_dr_ph500_rl50",
        "best_model.zip",
    )
    vnorm = os.path.join(
        root_guess,
        "PPO_50000000_seq6dec_s6_tau_dr_ph500_rl50",
        "vecnormalize.pkl",
    )
    return model, vnorm


def quat_wxyz_from_pose_dict(pose):
    # type: (dict) -> np.ndarray
    o = pose["orientation"]
    return np.array(
        [float(o["w"]), float(o["x"]), float(o["y"]), float(o["z"])],
        dtype=np.float32,
    )


def vec3_from_dict(d):
    # type: (dict) -> np.ndarray
    return np.array([float(d["x"]), float(d["y"]), float(d["z"])], dtype=np.float32)


def quat_wxyz_hemisphere(q):
    # type: (np.ndarray) -> np.ndarray
    q = np.asarray(q, dtype=np.float32).reshape(4)
    if q[0] < 0.0:
        q = -q
    return q


def rot_world_from_body_wxyz(q):
    # type: (np.ndarray) -> np.ndarray
    """Build the world-from-body rotation matrix from a wxyz quaternion."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    w, x, y, z = q[0], q[1], q[2], q[3]
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def resolve_act_affine(mass_kg, gravity_z, act_mean_param, act_std_param):
    # type: (float, float, Any, Any) -> Tuple[np.ndarray, np.ndarray]
    if act_mean_param is not None and act_std_param is not None:
        mu = np.asarray(act_mean_param, dtype=np.float64).reshape(-1)
        sig = np.asarray(act_std_param, dtype=np.float64).reshape(-1)
        if mu.size != 4 or sig.size != 4:
            raise ValueError("act_mean and act_std must each have length 4")
        return mu, sig

    if act_mean_param is not None or act_std_param is not None:
        logger.warning(
            "Ignoring partial act_mean/act_std override; provide both or neither."
        )

    m = mass_kg
    gz = gravity_z
    mu_scalar = (-m * gz) / 4.0
    sig_scalar = (-m * 2.0 * gz) / 4.0
    mu = np.full(4, mu_scalar, dtype=np.float64)
    sig = np.full(4, sig_scalar, dtype=np.float64)
    return mu, sig


class RlFeedthroughCore(object):
    """Policy + observation + thrust mapping (no transport layer)."""

    def __init__(
        self,
        model_path,
        vecnormalize_path,
        device="auto",
        action_dim=ACTION_DIM,
        action_history_len=None,
        use_single_rotor_thrust=True,
        collective_m_s2=9.81,
        quad_mass_kg=0.774,
        gravity_z=-9.81,
        act_mean=None,
        act_std=None,
        enforce_quat_hemisphere=True,
        fixed_goal_xyz=None,
        pos_err_sign=+1,
        motor_perm=(0, 1, 2, 3),
        use_body_velocity=False,
        action_lpf_alpha=1.0,
        action_clip=1.0,
        action_history_init=0.0,
        torque_scale=1.0,
        thrust_bias=0.0,
        noise_pos=0.0,
        noise_vel=0.0,
        noise_omega=0.0,
        noise_quat=0.0,
        noise_seed=None,
    ):
        self.model_path = model_path or ""
        self.vecnormalize_path = vecnormalize_path or ""
        self.device = device
        self.action_dim = int(action_dim)
        self.action_history_len = resolve_action_history_len(
            self.model_path, action_history_len
        )
        self.obs_total_dim = obs_total_dim(self.action_history_len, self.action_dim)
        self.use_single_rotor_thrust = use_single_rotor_thrust
        self.collective_m_s2 = float(collective_m_s2)
        self.enforce_quat_hemisphere = enforce_quat_hemisphere
        self.fixed_goal_xyz = fixed_goal_xyz
        self.pos_err_sign = +1 if int(pos_err_sign) >= 0 else -1
        if len(motor_perm) != 4 or sorted(motor_perm) != [0, 1, 2, 3]:
            raise ValueError(
                "motor_perm must be a permutation of (0,1,2,3); got %r" % (motor_perm,)
            )
        self.motor_perm = tuple(int(i) for i in motor_perm)
        self.use_body_velocity = bool(use_body_velocity)
        self.action_lpf_alpha = float(np.clip(action_lpf_alpha, 0.0, 1.0))
        self.action_clip = float(np.clip(action_clip, 0.0, 1.0))
        self.torque_scale = float(np.clip(torque_scale, 0.0, 1.0))
        self.thrust_bias = float(thrust_bias)

        self.noise_pos = float(max(0.0, noise_pos))
        self.noise_vel = float(max(0.0, noise_vel))
        self.noise_omega = float(max(0.0, noise_omega))
        self.noise_quat = float(max(0.0, noise_quat))
        self._rng = np.random.default_rng(noise_seed)

        self._act_mean, self._act_std = resolve_act_affine(
            quad_mass_kg, gravity_z, act_mean, act_std
        )

        self._action_history_init_value = float(action_history_init)
        self._action_hist = np.full(
            (self.action_history_len, self.action_dim),
            self._action_history_init_value,
            dtype=np.float32,
        )
        self._last_action = None  # type: Optional[np.ndarray]
        self._model = None  # type: Any
        self._vecnorm = None  # type: Any
        self._telemetry_dict = None  # type: Optional[dict]

    def set_telemetry_dict(self, msg):
        self._telemetry_dict = msg

    def load_policy(self):
        # type: () -> bool
        if IMPORT_ERROR is not None:
            logger.error("Missing RL deps: %s", IMPORT_ERROR)
            return False
        if not self.model_path or not os.path.isfile(self.model_path):
            logger.warning("model_path missing or not found: %r", self.model_path)
            return False
        try:
            assert _ObsDummyEnv is not None and DummyVecEnv is not None
            obs_dim = self.obs_total_dim
            venv = DummyVecEnv([lambda d=obs_dim: _ObsDummyEnv(d)])

            if self.vecnormalize_path and os.path.isfile(self.vecnormalize_path):
                self._vecnorm = VecNormalize.load(self.vecnormalize_path, venv)
                self._vecnorm.training = False
                self._vecnorm.norm_reward = False
                logger.info("Loaded VecNormalize from %s", self.vecnormalize_path)
            else:
                logger.warning("VecNormalize file missing: %r", self.vecnormalize_path)
                self._vecnorm = None

            self._model = PPO.load(self.model_path, device=self.device)
            logger.info("Loaded PPO from %s", self.model_path)
            return True
        except Exception as e:
            logger.exception("Failed to load policy: %s", e)
            self._model = None
            self._vecnorm = None
            return False

    def _apply_obs_noise(self, p, q, v, w):
        # type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        """Inject Gaussian noise into the raw state (p, q, v, w) before the
        observation is assembled. Models the behaviour of a noisy state
        estimator (Vicon / VIO + IMU) on real hardware so the policy can be
        stress-tested in sim against perturbations it didn't see during
        training. Position/velocity/angular-velocity noise are i.i.d. zero-mean
        Gaussian. Attitude noise is a small random rotation: axis sampled
        uniformly on the unit sphere, angle drawn from N(0, noise_quat) rad.
        """
        if self.noise_pos > 0.0:
            p = p + self._rng.normal(0.0, self.noise_pos, size=3).astype(np.float32)
        if self.noise_vel > 0.0:
            v = v + self._rng.normal(0.0, self.noise_vel, size=3).astype(np.float32)
        if self.noise_omega > 0.0:
            w = w + self._rng.normal(0.0, self.noise_omega, size=3).astype(np.float32)
        if self.noise_quat > 0.0:
            angle = float(self._rng.normal(0.0, self.noise_quat))
            axis = self._rng.normal(0.0, 1.0, size=3)
            an = float(np.linalg.norm(axis))
            if an > 1e-8:
                axis = (axis / an).astype(np.float64)
                half = 0.5 * angle
                dw = float(np.cos(half))
                ds = float(np.sin(half))
                dq = np.array([dw, ds * axis[0], ds * axis[1], ds * axis[2]],
                              dtype=np.float64)
                q64 = q.astype(np.float64)
                qw, qx, qy, qz = q64
                # Hamilton product dq ⊗ q
                new_w = dq[0] * qw - dq[1] * qx - dq[2] * qy - dq[3] * qz
                new_x = dq[0] * qx + dq[1] * qw + dq[2] * qz - dq[3] * qy
                new_y = dq[0] * qy - dq[1] * qz + dq[2] * qw + dq[3] * qx
                new_z = dq[0] * qz + dq[1] * qy - dq[2] * qx + dq[3] * qw
                q = np.array([new_w, new_x, new_y, new_z], dtype=np.float32)
                qn = float(np.linalg.norm(q))
                if qn > 1e-8:
                    q = q / qn
        return p, q, v, w

    def goal_position(self, fixed_goal_xyz=None):
        # type: (Any) -> np.ndarray
        fg = fixed_goal_xyz if fixed_goal_xyz is not None else self.fixed_goal_xyz
        if fg is not None and len(fg) == 3:
            return np.array(fg, dtype=np.float32)
        if self._telemetry_dict is not None:
            try:
                ref = self._telemetry_dict["reference"]["pose"]["position"]
                return np.array(
                    [float(ref["x"]), float(ref["y"]), float(ref["z"])],
                    dtype=np.float32,
                )
            except (KeyError, TypeError):
                pass
        return np.zeros(3, dtype=np.float32)

    def build_observation_from_state_dict(self, state_dict, fixed_goal_xyz=None):
        # type: (dict, Any) -> np.ndarray
        """state_dict: rosbridge-style agiros_msgs/QuadState."""
        p = vec3_from_dict(state_dict["pose"]["position"])
        q = quat_wxyz_from_pose_dict(state_dict["pose"])
        v = vec3_from_dict(state_dict["velocity"]["linear"])
        w = vec3_from_dict(state_dict["velocity"]["angular"])
        p, q, v, w = self._apply_obs_noise(p, q, v, w)
        if self.enforce_quat_hemisphere:
            q = quat_wxyz_hemisphere(q)
        g = self.goal_position(fixed_goal_xyz)
        pos_err = (self.pos_err_sign * (g - p)).astype(np.float32)
        if self.use_body_velocity:
            R = rot_world_from_body_wxyz(q)
            v = (R.T @ v.astype(np.float64)).astype(np.float32)
        base = np.concatenate([pos_err, q, v, w], axis=0)
        hist = self._action_hist.reshape(-1)
        obs = np.concatenate([base, hist], axis=0)
        return obs.astype(np.float32)

    def build_observation_from_ros_quadstate(self, state_msg, fixed_goal_xyz=None):
        # type: (Any, Any) -> np.ndarray
        """Use geometry_msgs fields from a rospy QuadState message."""
        p = np.array(
            [
                state_msg.pose.position.x,
                state_msg.pose.position.y,
                state_msg.pose.position.z,
            ],
            dtype=np.float32,
        )
        q = np.array(
            [
                state_msg.pose.orientation.w,
                state_msg.pose.orientation.x,
                state_msg.pose.orientation.y,
                state_msg.pose.orientation.z,
            ],
            dtype=np.float32,
        )
        v = np.array(
            [
                state_msg.velocity.linear.x,
                state_msg.velocity.linear.y,
                state_msg.velocity.linear.z,
            ],
            dtype=np.float32,
        )
        w = np.array(
            [
                state_msg.velocity.angular.x,
                state_msg.velocity.angular.y,
                state_msg.velocity.angular.z,
            ],
            dtype=np.float32,
        )
        p, q, v, w = self._apply_obs_noise(p, q, v, w)
        if self.enforce_quat_hemisphere:
            q = quat_wxyz_hemisphere(q)
        g = self.goal_position(fixed_goal_xyz)
        pos_err = (self.pos_err_sign * (g - p)).astype(np.float32)
        if self.use_body_velocity:
            R = rot_world_from_body_wxyz(q)
            v = (R.T @ v.astype(np.float64)).astype(np.float32)
        base = np.concatenate([pos_err, q, v, w], axis=0)
        hist = self._action_hist.reshape(-1)
        return np.concatenate([base, hist], axis=0).astype(np.float32)

    def predict_action(self, obs):
        # type: (np.ndarray) -> np.ndarray
        if self._model is None:
            return np.zeros(self.action_dim, dtype=np.float32)
        row = obs.reshape(1, -1).astype(np.float32)
        if self._vecnorm is not None:
            row = self._vecnorm.normalize_obs(row)
        act, _ = self._model.predict(row, deterministic=True)
        a = np.asarray(act, dtype=np.float32).reshape(-1)

        if self.action_clip < 1.0:
            a = np.clip(a, -self.action_clip, self.action_clip)

        if 0.0 < self.action_lpf_alpha < 1.0:
            if self._last_action is None:
                self._last_action = a.copy()
            else:
                a = self.action_lpf_alpha * a + (1.0 - self.action_lpf_alpha) * self._last_action
                self._last_action = a.copy()
        else:
            self._last_action = a.copy()

        # Decompose into collective + differential modes, soften the
        # differential (roll/pitch/yaw torques), bias the collective, then
        # recombine. Keeps total thrust ≈ same while reducing attitude gain —
        # which is the bang-bang failure mode we hit in agilicious.
        if self.torque_scale != 1.0 or self.thrust_bias != 0.0:
            mean = float(a.mean())
            collective = mean + self.thrust_bias
            diff = a - mean
            a = collective + self.torque_scale * diff
            a = np.clip(a, -1.0, 1.0).astype(np.float32)

        return a

    def reset_runtime_state(self):
        # type: () -> None
        """Re-initialise the per-step LPF and action history buffers.
        Use this when a safety watchdog disengages: when the policy
        re-engages, we want it to start from a clean slate rather than
        continuing the trajectory it was on when it diverged.
        """
        self._last_action = None
        self._action_hist = np.full(
            (self.action_history_len, self.action_dim),
            self._action_history_init_value,
            dtype=np.float32,
        )

    def push_action_history(self, action):
        # type: (np.ndarray) -> None
        a = np.clip(
            np.asarray(action, dtype=np.float32).reshape(-1)[: self.action_dim],
            -1.0,
            1.0,
        )
        if a.size < self.action_dim:
            a = np.pad(a, (0, self.action_dim - a.size))
        self._action_hist = np.roll(self._action_hist, -1, axis=0)
        self._action_hist[-1] = a

    def action_to_command_dict(self, t_sec, action):
        # type: (float, np.ndarray) -> dict
        """agiros_msgs/Command as dict for rosbridge / JSON."""
        import time as time_mod

        now = time_mod.time()
        secs = int(now)
        nsecs = int((now - secs) * 1e9)
        msg = {
            "header": {"stamp": {"secs": secs, "nsecs": nsecs}, "frame_id": ""},
            "t": float(t_sec),
            "is_single_rotor_thrust": self.use_single_rotor_thrust,
            "collective_thrust": 0.0,
            "bodyrates": {"x": 0.0, "y": 0.0, "z": 0.0},
            "thrusts": [0.0, 0.0, 0.0, 0.0],
        }
        if self.use_single_rotor_thrust:
            msg["is_single_rotor_thrust"] = True
            a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
            a4 = a[:4] if a.size >= 4 else np.pad(a, (0, 4 - a.size))
            thrusts = a4 * self._act_std + self._act_mean
            # Rotors cannot pull; negative T from the affine map clips to 0 N.
            thrusts = np.maximum(thrusts, 0.0)
            # Re-index policy thrust outputs into agilicious motor order.
            permuted = [float(thrusts[self.motor_perm[i]]) for i in range(4)]
            msg["thrusts"] = permuted
        else:
            msg["is_single_rotor_thrust"] = False
            msg["collective_thrust"] = float(self.collective_m_s2)
        return msg


# ---------------------------------------------------------------------------
# Shared helpers used by both rosbridge entrypoints
# ---------------------------------------------------------------------------


def resolve_policy_paths(model_arg, vnorm_arg, default_model="", default_vnorm=""):
    # type: (str, str, str, str) -> Tuple[str, str]
    """Resolve ``--model-path`` / ``--vecnormalize-path`` into concrete files.

    Accepts a directory (looks for ``best_model.zip`` and ``vecnormalize.pkl``
    inside it) or an explicit ``best_model.zip``; in the latter case the
    companion ``vecnormalize.pkl`` in the same directory is auto-discovered
    when ``--vecnormalize-path`` is not given.
    """
    model_path = (model_arg or "").strip()
    vnorm_path = (vnorm_arg or "").strip()

    if model_path:
        if os.path.isdir(model_path):
            run_dir = os.path.abspath(model_path)
            candidate_model = os.path.join(run_dir, "best_model.zip")
            if os.path.isfile(candidate_model):
                model_path = candidate_model
        else:
            run_dir = os.path.dirname(os.path.abspath(model_path))

        if not vnorm_path:
            candidate_vnorm = os.path.join(run_dir, "vecnormalize.pkl")
            if os.path.isfile(candidate_vnorm):
                vnorm_path = candidate_vnorm
    else:
        if default_model and os.path.isfile(default_model):
            model_path = default_model
        if not vnorm_path and default_vnorm and os.path.isfile(default_vnorm):
            vnorm_path = default_vnorm

    return model_path, vnorm_path


_CORE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIRNAME = "recordings"


def default_recordings_dir():
    # type: () -> str
    """``scripts/recordings`` — a dedicated folder next to this module for CSV
    traces, so runs no longer scatter files under ``/tmp``."""
    return os.path.join(_CORE_DIR, RECORDINGS_DIRNAME)


def default_trace_csv_path(tag="rl_feedthrough"):
    # type: (str) -> str
    """Auto-generated ``scripts/recordings/<tag>_trace_<YYYYMMDD_HHMMSS>.csv``."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(default_recordings_dir(), "{}_trace_{}.csv".format(tag, ts))


class GoalState(object):
    """Thread-safe holder for the live world-frame goal.

    The control loop reads via :meth:`get` once per step; any number of
    background threads (stdin reader, future ROS topic) update via :meth:`set`.
    ``on_change`` lets the script log changes and propagate the goal into the
    policy core (so the next step picks up the new ``fixed_goal_xyz``).
    """

    def __init__(self, initial_xyz=None, on_change=None):
        # type: (Any, Optional[Callable[[list, str], None]]) -> None
        self._lock = threading.Lock()
        self._xyz = list(initial_xyz) if initial_xyz is not None else None
        self._on_change = on_change

    def set(self, xyz, source="?"):
        # type: (Any, str) -> None
        new_xyz = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
        with self._lock:
            self._xyz = new_xyz
        if self._on_change is not None:
            try:
                self._on_change(list(new_xyz), source)
            except Exception as e:
                logger.warning("GoalState.on_change raised: %s", e)

    def get(self):
        # type: () -> Optional[list]
        with self._lock:
            return list(self._xyz) if self._xyz is not None else None


def start_stdin_goal_reader(goal_state, get_state_fn, enabled=True):
    # type: (GoalState, Callable[[], Any], bool) -> bool
    """Spawn a daemon thread that parses goal commands from ``sys.stdin``.

    Lines understood (case-insensitive):
        - ``X Y Z``           — set new goal (commas or whitespace separated)
        - ``snap``            — snap goal to current drone pose
        - ``show``            — log the current goal
        - ``q``/``quit``      — ignored (use Ctrl-C to quit)

    ``get_state_fn`` returns the latest state dict (or ``None``) for ``snap``.
    Auto-disabled when ``stdin`` is not a TTY.
    """
    if not enabled:
        return False
    if not sys.stdin.isatty():
        logger.info("stdin goals: stdin is not a TTY, reader disabled.")
        return False

    def reader():
        try:
            for line in sys.stdin:
                s = line.strip()
                if not s:
                    continue
                low = s.lower()
                if low in ("q", "quit", "exit"):
                    logger.info("stdin: %r -> ignored (use Ctrl-C to quit).", s)
                    continue
                if low == "show":
                    logger.info("stdin: current goal = %s", goal_state.get())
                    continue
                if low == "snap":
                    st_now = get_state_fn()
                    if st_now is None:
                        logger.warning("stdin: 'snap' but no state received yet.")
                        continue
                    try:
                        p = st_now["pose"]["position"]
                        goal_state.set(
                            [float(p["x"]), float(p["y"]), float(p["z"])],
                            "stdin:snap",
                        )
                    except (KeyError, TypeError) as e:
                        logger.warning("stdin: 'snap' failed: %s", e)
                    continue
                parts = s.replace(",", " ").split()
                if len(parts) < 3:
                    logger.warning(
                        "stdin: need 'x y z' (or 'snap'/'show'), got %r", s
                    )
                    continue
                try:
                    xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
                except ValueError:
                    logger.warning("stdin: cannot parse floats in %r", s)
                    continue
                goal_state.set(xyz, "stdin")
        except Exception as e:
            logger.warning("stdin goal reader stopped: %s", e)

    threading.Thread(target=reader, daemon=True).start()
    logger.info(
        "stdin goals ENABLED: type 'X Y Z <Enter>' to retarget, "
        "'snap' for current pose, 'show' to print current goal."
    )
    return True


class StateCsvTracer(object):
    """Append-only per-step state/action/thrust CSV trace (async writer).

    Records goal, full state, position error, raw policy action and published
    thrusts on every control step. Designed for offline plotting after a run
    (trajectories, attitude, action profiles, thrust distribution). The CBF
    entrypoint passes extra columns/values through ``extra_columns`` /
    ``extra_values`` for CBF-specific telemetry.

    All disk I/O (row formatting, ``writerow``, ``flush``) runs on a dedicated
    background thread. :meth:`write` only snapshots a few references and drops
    them on an unbounded queue, so the caller's control loop is never blocked
    on the filesystem — this keeps the command-publishing cadence tight so the
    autopilot doesn't fall back to another controller while we save data.

    The ``phase`` column tags each row with the pipeline stage it was captured
    in (``pre_engage`` before we take over, ``freefall``, ``engaged``), so the
    baseline trajectory recorded *before* we start overwriting ROS commands can
    be separated from the policy-controlled trajectory when plotting.
    """

    BASE_COLUMNS = (
        "wall_time", "step_idx", "t_sec", "phase",
        "gx", "gy", "gz",
        "px", "py", "pz",
        "vx", "vy", "vz",
        "wx", "wy", "wz",
        "qw", "qx", "qy", "qz",
        "ex", "ey", "ez",
        "a0", "a1", "a2", "a3",
        "thr0", "thr1", "thr2", "thr3", "thr_sum",
    )

    _SENTINEL = object()

    def __init__(self, path, extra_columns=(), flush_every=50):
        # type: (str, Any, int) -> None
        self.path = path
        self.extra_columns = tuple(extra_columns)
        self._flush_every = max(1, int(flush_every))
        self._fh = None  # type: Any
        self._writer = None  # type: Any
        self._row_count = 0
        self._dropped = 0
        # Unbounded queue: ``put_nowait`` never blocks the control loop.
        self._queue = queue.Queue()  # type: queue.Queue
        self._thread = threading.Thread(
            target=self._run, name="StateCsvTracer", daemon=True
        )
        self._thread.start()

    @property
    def row_count(self):
        return self._row_count

    @property
    def dropped(self):
        return self._dropped

    def write(self, step_idx, t_sec, goal, state_dict, obs, action, cmd,
              extra_values=(), phase="engaged"):
        """Queue a row for the writer thread (non-blocking).

        ``state_dict`` is the rosbridge-style QuadState dict. The referenced
        objects (``state_dict``, ``obs``, ``action``, ``cmd``) must not be
        mutated in place after this call, since formatting is deferred to the
        writer thread; the callers here always hand over freshly built objects.
        """
        item = (
            time.time(), step_idx, t_sec, phase, goal,
            state_dict, obs, action, cmd, tuple(extra_values),
        )
        try:
            self._queue.put_nowait(item)
        except queue.Full:  # pragma: no cover - queue is unbounded
            self._dropped += 1

    def _run(self):
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            try:
                self._write_row(item)
            except Exception as e:  # keep the writer thread alive on bad rows
                logger.warning("StateCsvTracer: dropping row: %s", e)
                self._dropped += 1
        self._close_file()

    def _ensure_open(self):
        if self._fh is not None:
            return
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        new_file = (not os.path.isfile(self.path)
                    or os.path.getsize(self.path) == 0)
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.writer(self._fh)
        if new_file:
            self._writer.writerow(list(self.BASE_COLUMNS) + list(self.extra_columns))
            self._fh.flush()

    def _write_row(self, item):
        (wall_time, step_idx, t_sec, phase, goal,
         state_dict, obs, action, cmd, extra_values) = item
        self._ensure_open()
        p = state_dict["pose"]["position"]
        v = state_dict["velocity"]["linear"]
        w = state_dict["velocity"]["angular"]
        q = state_dict["pose"]["orientation"]
        e = obs[0:3]
        thrusts = cmd.get("thrusts", [0.0, 0.0, 0.0, 0.0])
        g = goal if goal is not None else (float("nan"),) * 3
        row = [
            "%.6f" % wall_time, int(step_idx), "%.6f" % float(t_sec), phase,
            "%.6f" % float(g[0]), "%.6f" % float(g[1]), "%.6f" % float(g[2]),
            "%.6f" % float(p["x"]), "%.6f" % float(p["y"]), "%.6f" % float(p["z"]),
            "%.6f" % float(v["x"]), "%.6f" % float(v["y"]), "%.6f" % float(v["z"]),
            "%.6f" % float(w["x"]), "%.6f" % float(w["y"]), "%.6f" % float(w["z"]),
            "%.6f" % float(q["w"]), "%.6f" % float(q["x"]),
            "%.6f" % float(q["y"]), "%.6f" % float(q["z"]),
            "%.6f" % float(e[0]), "%.6f" % float(e[1]), "%.6f" % float(e[2]),
            "%.6f" % float(action[0]), "%.6f" % float(action[1]),
            "%.6f" % float(action[2]), "%.6f" % float(action[3]),
            "%.6f" % float(thrusts[0]), "%.6f" % float(thrusts[1]),
            "%.6f" % float(thrusts[2]), "%.6f" % float(thrusts[3]),
            "%.6f" % float(sum(thrusts)),
        ]
        for val in extra_values:
            row.append(val)
        self._writer.writerow(row)
        self._row_count += 1
        if self._row_count % self._flush_every == 0:
            self._fh.flush()

    def _close_file(self):
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None
            self._writer = None

    def close(self):
        """Flush the queue and stop the writer thread (blocks until drained)."""
        if self._thread is None:
            return
        self._queue.put(self._SENTINEL)
        self._thread.join()
        self._thread = None
