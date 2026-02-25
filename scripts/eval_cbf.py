#!/usr/bin/env python3
"""
CBF-specific evaluation: compare Raw RL vs CBF-filtered trajectories by running
the same episode twice in the simulator—first with raw policy actions, then with
CBF-filtered actions (same seed for same initial state when supported). Plots
trajectory views (x-y, y-z, z-x), states, raw vs filtered thrusts, and world/CBF limits.

Usage:
  python scripts/eval_cbf.py --config configs/drone_ppo_default.yaml \\
    --checkpoint models/curriculum_2/stage5_finetune/best_model \\
    [--episodes 5] [--save_plots] [--plot_dir ...]
"""
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

from scripts.config_loader import load_config, prepare_env_run_dir, get_vec_env_config_string
from scripts.context import flightmare_context
from scripts.custom_reward_wrapper import CustomRewardWrapper
from scripts.env_wrapper import FlightlibVecEnv, ActionHistoryWrapper
from scripts.cbf_filter import CBFFilter
from scripts.quadrotor_model import POS, VEL, QuadrotorModel

# State part of observation (pos_error 3 + quat 4 + vel 3 + omega 3); policy may see state + action history.
STATE_OBS_DIM = 13


def _ensure_flightgym_path():
    import glob
    flightlib_dir = os.path.join(_REPO_ROOT, "flightmare", "flightlib")
    if not os.path.isdir(flightlib_dir):
        return
    build_dir = os.path.join(flightlib_dir, "build")
    for pattern in ["lib.*", "lib"]:
        for path in glob.glob(os.path.join(build_dir, pattern)):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
    if flightlib_dir not in sys.path:
        sys.path.insert(0, flightlib_dir)


def _get_QuadrotorEnv_v1():
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
        try:
            from flightlib import QuadrotorEnv_v1
            return QuadrotorEnv_v1
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Could not import 'flightgym' or 'flightlib'. See scripts/README.md."
            ) from None


_MOTOR_INIT_MODES = {"zero": 0, "hover": 1}


def _pack_spawn_ranges(spawn_cfg):
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
        goals = np.array([[goal_pos[0], goal_pos[1], goal_pos[2]]] * impl.getNumOfEnvs(), dtype=np.float32)
        impl.setEnvGoalPositions(goals)
    spawn_cfg = cfg.get("env", {}).get("spawn_ranges")
    if spawn_cfg is not None:
        impl.setSpawnRanges(_pack_spawn_ranges(spawn_cfg))
    world_box = cfg.get("env", {}).get("world_box")
    if world_box is not None:
        impl.setWorldBox(np.array(world_box, dtype=np.float32))
    return FlightlibVecEnv(impl)


def _get_raw_obs(env, obs):
    """Return observation in physical units. If env uses VecNormalize, unnormalize; else return obs."""
    obs = np.asarray(obs, dtype=np.float64)
    if obs.ndim == 2:
        obs = obs[0]
    obs = obs.ravel()
    if getattr(env, "obs_rms", None) is None:
        return obs
    eps = getattr(env, "epsilon", 1e-8)
    var = np.asarray(env.obs_rms.var, dtype=np.float64).ravel()
    mean = np.asarray(env.obs_rms.mean, dtype=np.float64).ravel()
    n = min(obs.size, var.size, mean.size)
    return (obs[:n] * np.sqrt(var[:n] + eps) + mean[:n]).astype(np.float64)


def _set_env_seed(env, seed):
    """Set seed on the underlying Flightlib env so reset() is deterministic. Unwraps venv until set_seed/seed is found."""
    if seed is None:
        return
    e = env
    while e is not None:
        if hasattr(e, "set_seed"):
            e.set_seed(seed)
            return
        if hasattr(e, "seed"):
            e.seed(seed)
            return
        e = getattr(e, "venv", None)


def _draw_world_box_2d(ax, world_box, dims_xy):
    """Draw world box in 2D. dims_xy is ('x','y') or ('y','z') or ('z','x')."""
    if world_box is None or len(world_box) < 6:
        return
    x_min, x_max = world_box[0], world_box[1]
    y_min, y_max = world_box[2], world_box[3]
    z_min, z_max = world_box[4], world_box[5]
    dim_map = {"x": (0, x_min, x_max), "y": (1, y_min, y_max), "z": (2, z_min, z_max)}
    _, a_lo, a_hi = dim_map[dims_xy[0]]
    _, b_lo, b_hi = dim_map[dims_xy[1]]
    ax.axhline(b_lo, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axhline(b_hi, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(a_lo, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(a_hi, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlim(a_lo - 0.5, a_hi + 0.5)
    ax.set_ylim(b_lo - 0.5, b_hi + 0.5)


def _draw_cbf_barriers_2d(ax, barriers, dims_xy, z_fix=None):
    """
    Draw CBF barrier boundaries in 2D. barriers: list of dict with 'n' (3,) and 'q'.
    dims_xy is ('x','y'), ('y','z'), or ('z','x'). For 3D barriers we fix the missing coord.
    """
    if not barriers:
        return
    dim_idx = {"x": 0, "y": 1, "z": 2}
    ia = dim_idx[dims_xy[0]]
    ib = dim_idx[dims_xy[1]]
    ic = 3 - ia - ib
    for b in barriers:
        n = np.asarray(b["n"], dtype=np.float64).ravel()[:3]
        q = float(b["q"])
        if abs(n[ic]) < 1e-10:
            if abs(n[ib]) < 1e-10:
                if abs(n[ia]) > 1e-10:
                    ax.axvline(-q / n[ia], color="red", linewidth=1, alpha=0.8)
            else:
                p_a = np.linspace(-50, 50, 100)
                p_b = (-q - n[ia] * p_a) / (n[ib] + 1e-12)
                ax.plot(p_a, p_b, color="red", linewidth=1, alpha=0.8)
        else:
            if z_fix is None:
                z_fix = 5.0
            p_fix = z_fix
            rhs = -q - n[ic] * p_fix
            if abs(n[ib]) < 1e-10:
                if abs(n[ia]) > 1e-10:
                    ax.axvline(rhs / n[ia], color="red", linewidth=0.8, alpha=0.6)
            else:
                p_a = np.linspace(-50, 50, 100)
                p_b = (rhs - n[ia] * p_a) / (n[ib] + 1e-12)
                ax.plot(p_a, p_b, color="red", linewidth=0.8, alpha=0.6)


def _load_cbf_barriers(config_path=None):
    path = os.path.join(_REPO_ROOT, "configs", "cbf_config.yaml") if config_path is None else config_path
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cbf = cfg.get("cbf", {})
    return cbf.get("position_barriers", [])


def _quat_to_tilt_deg(qw, qx, qy, qz):
    body_z = np.clip(1.0 - 2.0 * (qx**2 + qy**2), -1.0, 1.0)
    return np.degrees(np.arccos(body_z))


def plot_cbf_episode(
    pos_actual,
    pos_raw,
    obs_list,
    raw_actions,
    safe_actions,
    t_axis,
    world_box,
    cbf_barriers,
    goal_pos,
    save_path=None,
    episode_idx=0,
    qp_failed=None,
):
    """Plot trajectory comparisons, states, raw vs filtered actions, world and CBF limits.
    qp_failed: optional list of bool per step (True = CBF QP was infeasible, raw RL used)."""
    import matplotlib.pyplot as plt

    pos_actual = np.array(pos_actual)
    pos_raw = np.array(pos_raw)
    obs = np.array(obs_list)
    raw_act = np.array(raw_actions)
    safe_act = np.array(safe_actions)
    n_steps = len(raw_act)
    t_act = t_axis[:n_steps] if len(t_axis) > n_steps else t_axis
    dt = (t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 0.02
    if qp_failed is not None:
        qp_failed = np.asarray(qp_failed, dtype=bool).ravel()[:n_steps]
    else:
        qp_failed = np.zeros(n_steps, dtype=bool)

    n_rows = 4
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 14))
    fig.suptitle(f"CBF Eval — Episode {episode_idx}: Raw RL vs CBF-filtered trajectory")

    # Row 0: Trajectory x-y, y-z, z-x
    for col, (dims, title) in enumerate([
        (("x", "y"), "x vs y"),
        (("y", "z"), "y vs z"),
        (("z", "x"), "z vs x"),
    ]):
        ax = axes[0, col]
        ia = {"x": 0, "y": 1, "z": 2}[dims[0]]
        ib = {"x": 0, "y": 1, "z": 2}[dims[1]]
        ax.plot(pos_actual[:, ia], pos_actual[:, ib], color="C0", linewidth=2, label="CBF filtered (actual)")
        ax.plot(pos_raw[:, ia], pos_raw[:, ib], color="C1", linewidth=1.2, alpha=0.8, linestyle="--", label="Raw RL (actual)")
        if goal_pos is not None:
            ax.scatter([goal_pos[ia]], [goal_pos[ib]], color="green", s=80, marker="*", zorder=5, label="Goal")
        _draw_world_box_2d(ax, world_box, dims)
        z_fix = float(np.median(pos_actual[:, 2])) if len(pos_actual) else 5.0
        _draw_cbf_barriers_2d(ax, cbf_barriers, dims, z_fix=z_fix)
        ax.set_xlabel(dims[0])
        ax.set_ylabel(dims[1])
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    # Row 1: Position vs time, orientation, linear velocity
    ax = axes[1, 0]
    n_actual, n_raw = len(pos_actual), len(pos_raw)
    t_actual = t_axis if len(t_axis) == n_actual else np.arange(n_actual) * dt
    t_raw = np.arange(n_raw) * dt
    ax.plot(t_actual, pos_actual[:, 0], color="C0", label="x (CBF)")
    ax.plot(t_raw, pos_raw[:, 0], color="C1", linestyle="--", alpha=0.8, label="x (raw)")
    ax.plot(t_actual, pos_actual[:, 1], color="C2", label="y (CBF)")
    ax.plot(t_raw, pos_raw[:, 1], color="C3", linestyle="--", alpha=0.8, label="y (raw)")
    ax.plot(t_actual, pos_actual[:, 2], color="C4", label="z (CBF)")
    ax.plot(t_raw, pos_raw[:, 2], color="C5", linestyle="--", alpha=0.8, label="z (raw)")
    ax.set_ylabel("Position (m)")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_axis, obs[:, 3], label="qw")
    ax.plot(t_axis, obs[:, 4], label="qx")
    ax.plot(t_axis, obs[:, 5], label="qy")
    ax.plot(t_axis, obs[:, 6], label="qz")
    ax.set_ylabel("Quaternion")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    tilt = _quat_to_tilt_deg(obs[:, 3], obs[:, 4], obs[:, 5], obs[:, 6])
    ax2.plot(t_axis, tilt, color="k", linewidth=1, alpha=0.6, label="tilt (deg)")
    ax2.set_ylabel("Tilt (deg)")

    ax = axes[1, 2]
    ax.plot(t_axis, obs[:, 7], label="vx")
    ax.plot(t_axis, obs[:, 8], label="vy")
    ax.plot(t_axis, obs[:, 9], label="vz")
    ax.set_ylabel("Linear vel (m/s)")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)

    # Row 2: Raw vs filtered motor thrusts
    ax = axes[2, 0]
    for i in range(4):
        ax.plot(t_act, raw_act[:, i], linestyle="--", alpha=0.7, label=f"raw_{i}")
    ax.set_ylabel("Raw RL thrust (N)")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    for i in range(4):
        ax.plot(t_act, safe_act[:, i], label=f"safe_{i}")
    ax.set_ylabel("CBF filtered thrust (N)")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    diff = safe_act - raw_act
    for i in range(4):
        ax.plot(t_act, diff[:, i], label=f"Δ_{i}")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Thrust diff (safe - raw) (N)")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)

    # Row 3: Angular velocity, deviation, 3D trajectory
    ax = axes[3, 0]
    ax.plot(t_axis, obs[:, 10], label="wx")
    ax.plot(t_axis, obs[:, 11], label="wy")
    ax.plot(t_axis, obs[:, 12], label="wz")
    ax.set_ylabel("Angular vel (rad/s)")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    min_len = min(len(pos_actual), len(pos_raw))
    if min_len > 0:
        dev = np.linalg.norm(pos_actual[:min_len] - pos_raw[:min_len], axis=1)
        t_dev = t_axis[:min_len] if len(t_axis) >= min_len else np.arange(min_len) * dt
        ax.plot(t_dev, dev, color="C0", label="|pos_CBF - pos_raw| (m)")
    # Shade time intervals where QP was infeasible (raw RL used)
    if np.any(qp_failed) and len(t_act) > 0:
        for i in range(len(qp_failed)):
            if qp_failed[i]:
                t0 = t_act[i] - 0.5 * dt if i > 0 else 0.0
                t1 = t_act[i] + 0.5 * dt if i + 1 < len(t_act) else t_act[i] + dt
                ax.axvspan(t0, t1, color="red", alpha=0.25, label="QP fallback" if i == 0 else None)
        ax_qp = ax.twinx()
        ax_qp.fill_between(t_act, 0, np.where(qp_failed, 1, 0), color="red", alpha=0.4, step="post", label="QP failed")
        ax_qp.set_ylim(-0.05, 1.2)
        ax_qp.set_ylabel("QP fallback", color="red", fontsize=7)
        ax_qp.tick_params(axis="y", labelcolor="red", labelsize=6)
    ax.set_ylabel("Position error (m)")
    ax.set_title("Deviation: CBF vs raw trajectory (red = QP infeasible, raw RL used)")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)

    axes[3, 2].remove()
    ax3d = fig.add_subplot(n_rows, n_cols, 11, projection="3d")
    ax3d.plot(pos_actual[:, 0], pos_actual[:, 1], pos_actual[:, 2], color="C0", linewidth=2, label="CBF filtered")
    ax3d.plot(pos_raw[:, 0], pos_raw[:, 1], pos_raw[:, 2], color="C1", linewidth=1, alpha=0.7, linestyle="--", label="Raw RL (actual)")
    if goal_pos is not None:
        ax3d.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]], color="green", s=100, marker="*")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("3D trajectory")
    ax3d.legend(loc="upper right", fontsize=7)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="CBF evaluation: raw RL vs CBF-filtered trajectories")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model zip or folder")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--debug", action="store_true", help="Print state, u_raw, u_safe, u_cbf, slack, barrier h every step for CBF run")
    parser.add_argument("--debug_steps", type=int, default=None, help="If --debug, only print first N steps (default: all)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_val = copy.deepcopy(cfg)
    if "env" not in cfg_val:
        cfg_val["env"] = {}
    if "vec_env" not in cfg_val["env"]:
        cfg_val["env"]["vec_env"] = {}
    cfg_val["env"]["vec_env"]["num_envs"] = 1
    cfg_val["env"]["vec_env"]["num_threads"] = 1
    if args.goal is not None:
        cfg_val["env"]["goal_position"] = list(args.goal)

    n_episodes = args.episodes
    max_episode_steps = cfg_val.get("evaluation", {}).get("max_episode_steps", 250)
    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 0)
    if seed is not None:
        np.random.seed(seed)

    env = _make_env(cfg_val)
    env_cfg = cfg.get("env", {})
    if env_cfg.get("custom_reward", {}).get("enabled", False):
        env = CustomRewardWrapper(env, env_cfg["custom_reward"])
    action_history_len = env_cfg.get("action_history_len", 0)
    if action_history_len > 0:
        env = ActionHistoryWrapper(env, action_history_len)

    qd = env_cfg.get("quadrotor_dynamics", {})
    mass = float(qd.get("mass", 0.774))
    g = 9.81
    act_mean = np.full(4, (mass * g) / 4.0, dtype=np.float64)
    act_std = np.full(4, (mass * 2 * g) / 4.0, dtype=np.float64)
    sim_dt = env_cfg.get("quadrotor_env", {}).get("sim_dt", 0.02)
    goal_pos = np.array(cfg_val.get("env", {}).get("goal_position", [0.0, 0.0, 5.0]), dtype=np.float64)
    world_box = env_cfg.get("world_box")
    cbf_barriers = _load_cbf_barriers()

    cbf = CBFFilter()
    model_quad = cbf.model  # for state_from_observation when running with CBF

    if cfg.get("training", {}).get("normalize_obs", True):
        vecnorm_path = os.path.join(os.path.dirname(args.checkpoint), "vecnormalize.pkl")
        if os.path.isfile(vecnorm_path):
            from stable_baselines3.common.vec_env import VecNormalize
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False

    from stable_baselines3 import PPO
    model = PPO.load(args.checkpoint, env=env)
    deterministic = cfg.get("evaluation", {}).get("deterministic", True)

    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    plot_dir = args.plot_dir if args.plot_dir is not None else run_dir
    if args.save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    act_dim = env.action_space.shape[0]

    for ep in range(n_episodes):
        ep_seed = (seed + ep) if seed is not None else None
        if ep_seed is not None:
            np.random.seed(ep_seed)
        _set_env_seed(env, ep_seed)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = np.asarray(obs[0]).ravel()
        obs_raw = _get_raw_obs(env, obs)

        # ----- Run 1: raw RL only (no CBF), record trajectory from simulator -----
        pos_raw = []
        state = model_quad.state_from_observation(obs_raw[:STATE_OBS_DIM], goal_pos=goal_pos)
        pos_raw.append(state[POS].copy())
        done = False
        steps = 0
        while not done and steps < max_episode_steps:
            action, _ = model.predict(obs.reshape(1, -1) if obs.ndim == 1 else obs, deterministic=deterministic)
            action = action.ravel()
            obs, rewards, dones, infos = env.step(action.reshape(1, -1))
            obs = np.asarray(obs[0]).ravel()
            obs_raw = _get_raw_obs(env, obs)
            state = model_quad.state_from_observation(obs_raw[:STATE_OBS_DIM], goal_pos=goal_pos)
            pos_raw.append(state[POS].copy())
            done = bool(dones[0])
            steps += 1
            if infos and getattr(infos[0], "__contains__", None) and "episode" in infos[0]:
                break

        # ----- Run 2: with CBF (same initial state via same seed) -----
        if ep_seed is not None:
            np.random.seed(ep_seed)
        _set_env_seed(env, ep_seed)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = np.asarray(obs[0]).ravel()
        obs_raw = _get_raw_obs(env, obs)

        pos_actual = []
        obs_list = []
        raw_actions = []
        safe_actions = []
        qp_failed_steps = []
        state = model_quad.state_from_observation(obs_raw[:STATE_OBS_DIM], goal_pos=goal_pos)
        pos_actual.append(state[POS].copy())
        obs_list.append(obs_raw[:STATE_OBS_DIM].copy())
        done = False
        steps = 0
        debug = getattr(args, "debug", False)
        debug_steps = getattr(args, "debug_steps", None)
        if debug and ep == 0:
            print("\n--- CBF run debug (episode 0): state, u_raw, u_safe, u_cbf, slack, barrier h (safe iff h>=0) ---\n")
        while not done and steps < max_episode_steps:
            action, _ = model.predict(obs.reshape(1, -1) if obs.ndim == 1 else obs, deterministic=deterministic)
            action = action.ravel()
            u_raw_norm = action[:4].astype(np.float64)
            u_raw = u_raw_norm * act_std + act_mean
            u_safe = cbf.filter(state, u_raw)
            qp_failed_steps.append(cbf.last_qp_failed)
            safe_norm = (u_safe.astype(np.float64) - act_mean) / (act_std + 1e-8)
            safe_norm = np.clip(safe_norm, -1.0, 1.0).astype(np.float32)
            safe_action = np.zeros(act_dim, dtype=np.float32)
            safe_action[:4] = safe_norm
            if act_dim > 4:
                safe_action[4:] = action[4:]

            raw_actions.append(u_raw.copy())
            safe_actions.append(u_safe.copy())

            if debug and ep == 0 and (debug_steps is None or steps < debug_steps):
                t = steps * sim_dt
                p, q, v, om = state[POS], state[3:7], state[7:10], state[10:13]
                h_vals = [cbf.barriers[i].h(p, v) for i in range(len(cbf.barriers))]
                print(f"  step={steps} t={t:.3f}s")
                print(f"    state: p=[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]  q=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
                print(f"           v=[{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]  omega=[{om[0]:.4f}, {om[1]:.4f}, {om[2]:.4f}]")
                print(f"    u_raw (N):  [{u_raw[0]:.4f}, {u_raw[1]:.4f}, {u_raw[2]:.4f}, {u_raw[3]:.4f}]")
                print(f"    u_safe (N): [{u_safe[0]:.4f}, {u_safe[1]:.4f}, {u_safe[2]:.4f}, {u_safe[3]:.4f}]")
                print(f"    u_safe-u_raw: [{u_safe[0]-u_raw[0]:.4f}, {u_safe[1]-u_raw[1]:.4f}, {u_safe[2]-u_raw[2]:.4f}, {u_safe[3]-u_raw[3]:.4f}]")
                u_cbf = cbf.last_u_cbf
                slack = cbf.last_slack
                if u_cbf is not None:
                    print(f"    u_cbf (N):  [{u_cbf[0]:.4f}, {u_cbf[1]:.4f}, {u_cbf[2]:.4f}, {u_cbf[3]:.4f}]")
                    if slack is not None:
                        print(f"    slack:      {slack}")
                    else:
                        print(f"    slack:      (no slack)")
                    print(f"    QP:         solved")
                else:
                    print(f"    u_cbf:      (QP failed)")
                    reason = getattr(cbf, "last_qp_failure_reason", None)
                    if reason:
                        print(f"    QP:         {reason}")
                print(f"    barrier h (safe if >=0): " + ", ".join(f"{cbf.barriers[i].name}={h_vals[i]:.4f}" for i in range(len(cbf.barriers))))
                viol = [getattr(cbf.barriers[i], "_name", f"h{i}") for i in range(len(h_vals)) if h_vals[i] < 0]
                if viol:
                    print(f"    *** VIOLATION: h < 0 for {viol}")
                print()

            obs, rewards, dones, infos = env.step(safe_action.reshape(1, -1))
            obs = np.asarray(obs[0]).ravel()
            obs_raw = _get_raw_obs(env, obs)
            state_next = model_quad.state_from_observation(obs_raw[:STATE_OBS_DIM], goal_pos=goal_pos)
            if debug and ep == 0 and (debug_steps is None or steps < debug_steps):
                p_next = state_next[POS]
                v_next = state_next[VEL]
                h_next = [cbf.barriers[i].h(p_next, v_next) for i in range(len(cbf.barriers))]
                print(f"    after step: p_next=[{p_next[0]:.4f}, {p_next[1]:.4f}, {p_next[2]:.4f}]")
                print(f"    barrier h at next state: " + ", ".join(f"{cbf.barriers[i].name}={h_next[i]:.4f}" for i in range(len(cbf.barriers))))
                if any(hi < 0 for hi in h_next):
                    print(f"    *** NEXT STATE VIOLATION: left safe set after applying u_safe")
                print()
            state = state_next
            pos_actual.append(state[POS].copy())
            obs_list.append(obs_raw[:STATE_OBS_DIM].copy())
            done = bool(dones[0])
            steps += 1
            if infos and getattr(infos[0], "__contains__", None) and "episode" in infos[0]:
                break

        obs_list = [np.asarray(o, dtype=np.float64).ravel()[:STATE_OBS_DIM] for o in obs_list]

        pos_raw = np.array(pos_raw)
        pos_actual = list(pos_actual)  # keep as list for plot_cbf_episode
        t_axis = np.arange(len(pos_actual)) * sim_dt
        save_path = os.path.join(plot_dir, f"cbf_eval_episode_{ep}.png") if args.save_plots else None
        plot_cbf_episode(
            pos_actual,
            pos_raw,
            obs_list,
            raw_actions,
            safe_actions,
            t_axis,
            world_box,
            cbf_barriers,
            goal_pos,
            save_path=save_path,
            episode_idx=ep,
            qp_failed=qp_failed_steps,
        )

    env.close()
    print("CBF eval done. Plots: Raw RL (actual from sim) vs CBF filtered (actual from sim); same seed for same initial state.")


if __name__ == "__main__":
    main()
