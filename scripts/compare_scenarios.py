#!/usr/bin/env python3
"""
Deterministic scenario comparison: RL vs RL+CBF vs MPC vs MPC+Con.

Each scenario in the config specifies an exact 13-dim initial state
[pos(3), quat(4), vel(3), omega(3)].  By default, the same scenario
also fixes the stochastic stream: C++ disturbance RNG (wind / OU /
force noise) and ObservationNoiseWrapper RNG are reseeded before each
rollout so RL, MPC, etc. see identical noise for fair comparison.
Use --no_sync_stochastic_seeds for the legacy behavior.

Usage:
  python scripts/compare_scenarios.py \
    --scenarios_config configs/scenarios_config.yaml \
    --checkpoint models/.../best_model \
    [--cbf_config configs/cbf_config.yaml] \
    [--mpc_config configs/mpc_config.yaml] \
    [--controllers RL RL+CBF MPC MPC+Con] \
    [--save_plots] [--plot_dir scenario_plots]
"""
import argparse
import glob
import os
import sys
import time
from datetime import datetime

import numpy as np
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config_loader import load_config, prepare_env_run_dir, write_env_configs
from scripts.context import flightmare_context
from scripts.custom_reward_wrapper import CustomRewardWrapper
from scripts.env_wrapper import (
    FlightlibVecEnv,
    ActionHistoryWrapper,
    ObservationNoiseWrapper,
)
from scripts.cbf_filter import CBFFilter
from scripts.quadrotor_model import POS, ATT, VEL, OME, QuadrotorModel, STATE_DIM
from scripts.mpc_controller import MPCController

STATE_OBS_DIM = 13

ALL_CONTROLLERS = ["RL", "RL+CBF", "MPC", "MPC+Con"]

CTRL_STYLE = {
    "RL":      {"color": "C1", "ls": "-"},
    "RL+CBF":  {"color": "C0", "ls": "-"},
    "MPC":     {"color": "C3", "ls": "--"},
    "MPC+Con": {"color": "C2", "ls": "--"},
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_scenarios_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_env_cfg(scfg):
    """Build a drone_ppo_default-style env config from the scenarios config."""
    env = {}
    env["max_episode_steps"] = scfg.get("max_episode_steps", 1000)

    env["vec_env"] = {"num_envs": 1, "num_threads": 1}
    env["quadrotor_env"] = {
        "sim_dt": scfg.get("sim_dt", 0.02),
        "max_t": scfg.get("max_episode_steps", 1000) * scfg.get("sim_dt", 0.02),
    }

    quad = scfg.get("quadrotor", {})
    if quad:
        # disturbances belong in env.disturbances / YAML root, not quadrotor_dynamics
        quad = {k: v for k, v in quad.items() if k != "disturbances"}
        env["quadrotor_dynamics"] = quad
    dist = scfg.get("disturbances")
    if dist is not None:
        env["disturbances"] = dist

    env["motor_init"] = scfg.get("motor_init", "hover")
    env["goal_position"] = list(scfg.get("goal", [0.0, 0.0, 5.0]))

    # Spawn ranges don't matter (we override via setQuadState), but need valid values
    env["spawn_ranges"] = {
        "pos_x": [-1.0, 1.0], "pos_y": [-1.0, 1.0], "pos_z": [3.0, 5.0],
        "vel_x": [0.0, 0.0], "vel_y": [0.0, 0.0], "vel_z": [0.0, 0.0],
        "ang_vel_x": [0.0, 0.0], "ang_vel_y": [0.0, 0.0], "ang_vel_z": [0.0, 0.0],
        "ori_scale": 0.0,
    }

    wb = scfg.get("world_box")
    if wb is not None:
        env["world_box"] = list(wb)

    rl_cfg = scfg.get("rl_policy", {})
    cr = rl_cfg.get("custom_reward")
    if cr is not None:
        env["custom_reward"] = cr
    env["action_history_len"] = rl_cfg.get("action_history_len", 0)

    base = {"env": env}
    base["training"] = {
        "normalize_obs": rl_cfg.get("normalize_obs", True),
        "seed": 0,
    }
    base["evaluation"] = {
        "deterministic": rl_cfg.get("deterministic", True),
        "max_episode_steps": scfg.get("max_episode_steps", 1000),
    }
    return base


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

_MOTOR_INIT_MODES = {"zero": 0, "hover": 1}


def _ensure_flightgym_path():
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
    for mod in ["flightgym", "flightlib"]:
        try:
            return getattr(__import__(mod), "QuadrotorEnv_v1")
        except (ModuleNotFoundError, AttributeError):
            pass
    _ensure_flightgym_path()
    for mod in ["flightgym", "flightlib"]:
        try:
            return getattr(__import__(mod), "QuadrotorEnv_v1")
        except (ModuleNotFoundError, AttributeError):
            pass
    raise ModuleNotFoundError("Could not import flightgym or flightlib.")


def _pack_spawn_ranges(spawn_cfg):
    def _r(key, default):
        return spawn_cfg.get(key, default)
    return np.array([
        _r("pos_x", [-1, 1])[0], _r("pos_x", [-1, 1])[1],
        _r("pos_y", [-1, 1])[0], _r("pos_y", [-1, 1])[1],
        _r("pos_z", [3, 5])[0],  _r("pos_z", [3, 5])[1],
        _r("vel_x", [0, 0])[0],  _r("vel_x", [0, 0])[1],
        _r("vel_y", [0, 0])[0],  _r("vel_y", [0, 0])[1],
        _r("vel_z", [0, 0])[0],  _r("vel_z", [0, 0])[1],
        _r("ang_vel_x", [0, 0])[0], _r("ang_vel_x", [0, 0])[1],
        _r("ang_vel_y", [0, 0])[0], _r("ang_vel_y", [0, 0])[1],
        _r("ang_vel_z", [0, 0])[0], _r("ang_vel_z", [0, 0])[1],
        _r("ori_scale", 0.0),
    ], dtype=np.float32)


def _make_env(cfg):
    QuadrotorEnv_v1 = _get_QuadrotorEnv_v1()
    run_dir = prepare_env_run_dir(cfg)
    if not run_dir:
        import tempfile
        run_dir = tempfile.mkdtemp(prefix="scenario_env_")
        write_env_configs(cfg, run_dir)
    with flightmare_context(run_dir):
        impl = QuadrotorEnv_v1()
    motor_init = cfg.get("env", {}).get("motor_init", "hover")
    impl.setMotorInitMode(_MOTOR_INIT_MODES.get(motor_init, 1))
    goal_pos = cfg.get("env", {}).get("goal_position")
    if goal_pos is not None:
        goals = np.array([goal_pos[:3]], dtype=np.float32)
        impl.setEnvGoalPositions(goals)
    spawn_cfg = cfg.get("env", {}).get("spawn_ranges")
    if spawn_cfg is not None:
        impl.setSpawnRanges(_pack_spawn_ranges(spawn_cfg))
    world_box = cfg.get("env", {}).get("world_box")
    if world_box is not None:
        impl.setWorldBox(np.array(world_box, dtype=np.float32))
    return FlightlibVecEnv(impl)


def _get_raw_obs(env, obs):
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


def _find_obs_noise_wrapper(env):
    """Walk wrapper chain for ObservationNoiseWrapper (outer → inner)."""
    seen = set()
    cur = env
    for _ in range(48):
        if cur is None or id(cur) in seen:
            return None
        seen.add(id(cur))
        if cur.__class__.__name__ == "ObservationNoiseWrapper":
            return cur
        cur = getattr(cur, "venv", None)
    return None


def _scenario_rollout_seed(
    base_seed: int,
    scenario_index: int,
    seed_trial: int = 0,
) -> int:
    """Distinct 32-bit seed per scenario and optional seed trial.

    Does *not* depend on comparison-run index: nominal, mass mismatch, motor tau
    mismatch, etc. all use the same disturbance + observation-noise stream for a
    given scenario so controllers are evaluated under identical stochastic conditions.
    Only ``seed_trial`` (``--n_seeds``) changes the realization for mean±std reporting.
    """
    s = (
        int(base_seed)
        + int(scenario_index) * 100_003
        + int(seed_trial) * 100_000
    ) & 0xFFFFFFFF
    return s


def _apply_rollout_seeding(env, base_env, rollout_seed: int) -> None:
    """Reseed C++ disturbances and obs-noise RNG before env.reset()."""
    if hasattr(base_env, "seed_disturbance"):
        base_env.seed_disturbance(int(rollout_seed))
    else:
        print(
            "  Warning: FlightlibVecEnv has no seed_disturbance — rebuild "
            "flightlib (pip install -e flightmare/flightlib) for matched "
            "disturbance RNG across controllers.",
        )
    onw = _find_obs_noise_wrapper(env)
    if onw is not None:
        onw.reset_noise_rng(int(rollout_seed))


# ---------------------------------------------------------------------------
# State setup
# ---------------------------------------------------------------------------

def _build_state_13d(scenario, default_pos=None):
    """Build the 13-dim state vector from a scenario dict."""
    pos = np.array(scenario.get("position", default_pos or [0, 0, 3]), dtype=np.float64)
    quat = np.array(scenario.get("quaternion", [1, 0, 0, 0]), dtype=np.float64)
    quat /= np.linalg.norm(quat)
    vel = np.array(scenario.get("velocity", [0, 0, 0]), dtype=np.float64)
    omega = np.array(scenario.get("angular_velocity", [0, 0, 0]), dtype=np.float64)
    return np.concatenate([pos, quat, vel, omega]).astype(np.float32)


def _reset_with_state(env, base_env, state_13d, goal_pos, act_hist_len,
                      rollout_seed=None):
    """Reset env and set exact initial state. Returns (obs_for_policy, raw_obs_13d).

    If rollout_seed is set, reseed disturbance + observation-noise RNGs first
    so every controller (RL, MPC, …) sees the same stochastic realization
    for this scenario.
    """
    if rollout_seed is not None:
        _apply_rollout_seeding(env, base_env, rollout_seed)

    env.reset()

    base_env.setQuadState(state_13d)

    goals = np.array([goal_pos[:3]], dtype=np.float32)
    base_env._impl.setEnvGoalPositions(goals)
    base_env._impl.getObs(base_env._obs)

    raw_obs = base_env._obs[0].copy()

    obs_for_policy = raw_obs.reshape(1, -1).copy()
    if act_hist_len > 0:
        zeros = np.zeros((1, act_hist_len * 4), dtype=np.float32)
        obs_for_policy = np.concatenate([obs_for_policy, zeros], axis=1)

    if hasattr(env, "normalize_obs"):
        obs_for_policy = env.normalize_obs(obs_for_policy)

    return obs_for_policy[0].ravel().astype(np.float32), raw_obs


# ---------------------------------------------------------------------------
# Rollout functions
# ---------------------------------------------------------------------------

def _run_rl_scenario(env, base_env, model_policy, model_quad, goal_pos,
                     act_mean, act_std, max_steps, state_13d,
                     act_hist_len, deterministic, rollout_seed=None):
    obs, raw_obs = _reset_with_state(
        env, base_env, state_13d, goal_pos, act_hist_len, rollout_seed)
    state = model_quad.state_from_observation(raw_obs[:STATE_OBS_DIM], goal_pos=goal_pos)

    positions = [state[POS].copy()]
    obs_list = [raw_obs[:STATE_OBS_DIM].copy()]
    actions, rewards = [], []

    done, steps = False, 0
    while not done and steps < max_steps:
        action, _ = model_policy.predict(
            obs.reshape(1, -1), deterministic=deterministic)
        action = action.ravel()
        u_raw = action[:4].astype(np.float64) * act_std + act_mean
        actions.append(u_raw.copy())

        obs_out, reward, dones, infos = env.step(action.reshape(1, -1))
        obs = np.asarray(obs_out[0]).ravel()
        obs_raw = _get_raw_obs(env, obs)
        state = model_quad.state_from_observation(obs_raw[:STATE_OBS_DIM], goal_pos=goal_pos)
        positions.append(state[POS].copy())
        obs_list.append(obs_raw[:STATE_OBS_DIM].copy())
        rewards.append(float(reward[0]))
        done = bool(dones[0])
        steps += 1

    return {"positions": np.array(positions), "obs": np.array(obs_list),
            "actions": np.array(actions), "rewards": np.array(rewards),
            "steps": steps}


def _run_cbf_scenario(env, base_env, model_policy, cbf, model_quad, goal_pos,
                      act_mean, act_std, max_steps, state_13d,
                      act_hist_len, deterministic, rollout_seed=None):
    obs, raw_obs = _reset_with_state(
        env, base_env, state_13d, goal_pos, act_hist_len, rollout_seed)
    state = model_quad.state_from_observation(raw_obs[:STATE_OBS_DIM], goal_pos=goal_pos)

    positions = [state[POS].copy()]
    obs_list = [raw_obs[:STATE_OBS_DIM].copy()]
    actions, rewards = [], []
    act_dim = env.action_space.shape[0]

    done, steps, qp_fail = False, 0, 0
    while not done and steps < max_steps:
        action, _ = model_policy.predict(
            obs.reshape(1, -1), deterministic=deterministic)
        action = action.ravel()
        u_raw = action[:4].astype(np.float64) * act_std + act_mean

        u_safe = cbf.filter(state, u_raw)
        if cbf.last_qp_failed:
            qp_fail += 1

        safe_norm = np.clip(
            (u_safe.astype(np.float64) - act_mean) / (act_std + 1e-8),
            -1.0, 1.0).astype(np.float32)
        safe_action = np.zeros(act_dim, dtype=np.float32)
        safe_action[:4] = safe_norm
        if act_dim > 4:
            safe_action[4:] = action[4:]

        actions.append(u_safe.copy())
        obs_out, reward, dones, infos = env.step(safe_action.reshape(1, -1))
        obs = np.asarray(obs_out[0]).ravel()
        obs_raw = _get_raw_obs(env, obs)
        state = model_quad.state_from_observation(obs_raw[:STATE_OBS_DIM], goal_pos=goal_pos)
        positions.append(state[POS].copy())
        obs_list.append(obs_raw[:STATE_OBS_DIM].copy())
        rewards.append(float(reward[0]))
        done = bool(dones[0])
        steps += 1

    if qp_fail:
        print(f"    CBF QP failures: {qp_fail}/{steps}")

    return {"positions": np.array(positions), "obs": np.array(obs_list),
            "actions": np.array(actions), "rewards": np.array(rewards),
            "steps": steps, "qp_failures": int(qp_fail)}


def _run_mpc_scenario(env, base_env, mpc, model_quad, goal_pos,
                      act_mean, act_std, max_steps, state_13d,
                      act_hist_len, rollout_seed=None):
    obs, raw_obs = _reset_with_state(
        env, base_env, state_13d, goal_pos, act_hist_len, rollout_seed)
    state = model_quad.state_from_observation(
        raw_obs[:STATE_OBS_DIM].astype(np.float64), goal_pos=goal_pos)
    mpc.reset(state)

    positions = [state[POS].copy()]
    obs_list = [raw_obs[:STATE_OBS_DIM].copy()]
    actions, rewards, solve_times, solver_statuses = [], [], [], []

    done, steps = False, 0
    while not done and steps < max_steps:
        u_mpc = mpc.solve(state, goal_pos)
        solve_times.append(mpc.last_solve_time_ms)
        solver_statuses.append(int(mpc.last_status))

        u_norm = np.clip(
            (u_mpc - act_mean) / (act_std + 1e-8), -1.0, 1.0).astype(np.float32)

        actions.append(u_mpc.copy())
        obs_out, reward, dones, infos = env.step(u_norm.reshape(1, -1))
        obs = np.asarray(obs_out[0]).ravel()
        obs_raw = _get_raw_obs(env, obs)
        state = model_quad.state_from_observation(
            obs_raw[:STATE_OBS_DIM].astype(np.float64), goal_pos=goal_pos)
        positions.append(state[POS].copy())
        obs_list.append(obs_raw[:STATE_OBS_DIM].copy())
        rewards.append(float(reward[0]))
        done = bool(dones[0])
        steps += 1

    return {"positions": np.array(positions), "obs": np.array(obs_list),
            "actions": np.array(actions), "rewards": np.array(rewards),
            "steps": steps, "solve_times": np.array(solve_times),
            "solver_statuses": np.array(solver_statuses, dtype=np.int32)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _quat_to_tilt_deg(qw, qx, qy, qz):
    body_z = np.clip(1.0 - 2.0 * (qx**2 + qy**2), -1.0, 1.0)
    return np.degrees(np.arccos(body_z))


def _draw_barriers_2d(ax, barriers, dims_xy, z_fix=None):
    if not barriers:
        return
    dim_idx = {"x": 0, "y": 1, "z": 2}
    ia, ib = dim_idx[dims_xy[0]], dim_idx[dims_xy[1]]
    ic = 3 - ia - ib
    for b in barriers:
        n = np.asarray(b["n"], dtype=np.float64).ravel()[:3]
        q = float(b["q"])
        kw = dict(color="red", linewidth=1, alpha=0.7)
        if abs(n[ic]) < 1e-10:
            if abs(n[ib]) < 1e-10 and abs(n[ia]) > 1e-10:
                ax.axvline(-q / n[ia], **kw)
            elif abs(n[ib]) > 1e-10:
                pa = np.linspace(-50, 50, 200)
                pb = (-q - n[ia] * pa) / (n[ib] + 1e-12)
                ax.plot(pa, pb, **kw)


def _barriers_to_axis_bounds(barriers):
    bounds = {}
    axis_names = {0: "x", 1: "y", 2: "z"}
    for b in barriers:
        n = np.asarray(b["n"], dtype=np.float64).ravel()[:3]
        q = float(b["q"])
        nonzero = [i for i in range(3) if abs(n[i]) > 1e-10]
        if len(nonzero) != 1:
            continue
        idx = nonzero[0]
        boundary = -q / n[idx]
        name = axis_names[idx]
        bounds.setdefault(name, [None, None])
        if n[idx] > 0:
            lo = bounds[name][0]
            bounds[name][0] = boundary if lo is None else min(lo, boundary)
        else:
            hi = bounds[name][1]
            bounds[name][1] = boundary if hi is None else max(hi, boundary)
    return {k: tuple(v) for k, v in bounds.items()}


def plot_scenario(datasets, sim_dt, barriers, goal_pos, scenario_name,
                  save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 20))
    fig.suptitle(scenario_name, fontsize=14, fontweight="bold", y=0.99)

    legend_handles = [
        Line2D([0], [0], color=CTRL_STYLE[lbl]["color"],
               ls=CTRL_STYLE[lbl]["ls"], lw=2, label=lbl)
        for lbl, _ in datasets
    ]
    legend_handles.append(
        Line2D([0], [0], color="green", marker="*", ls="None",
               markersize=10, label="Goal"))
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=len(datasets) + 1, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 0.97))

    # Row 0: trajectory projections
    for col, (dims, title) in enumerate([
        (("x", "y"), "x vs y"), (("y", "z"), "y vs z"), (("z", "x"), "z vs x"),
    ]):
        ax = axes[0, col]
        ia = {"x": 0, "y": 1, "z": 2}[dims[0]]
        ib = {"x": 0, "y": 1, "z": 2}[dims[1]]
        for lbl, d in datasets:
            s = CTRL_STYLE[lbl]
            pos = d["positions"]
            ax.plot(pos[:, ia], pos[:, ib], color=s["color"], lw=2, ls=s["ls"])
        if goal_pos is not None:
            ax.scatter([goal_pos[ia]], [goal_pos[ib]], color="green",
                       s=100, marker="*", zorder=5)
        _draw_barriers_2d(ax, barriers, dims,
                          z_fix=goal_pos[2] if goal_pos is not None else None)
        bnd = _barriers_to_axis_bounds(barriers) if barriers else {}
        for dk, set_lim in [(dims[0], ax.set_xlim), (dims[1], ax.set_ylim)]:
            if dk in bnd:
                lo, hi = bnd[dk]
                if lo is not None and hi is not None:
                    margin = 0.15 * (hi - lo)
                    set_lim(lo - margin, hi + margin)
        ax.set_xlabel(dims[0]); ax.set_ylabel(dims[1])
        ax.set_title(title); ax.grid(True, alpha=0.3); ax.axis("equal")

    # Row 1: position vs time, tilt, speed
    ax = axes[1, 0]
    for i, coord in enumerate(["x", "y", "z"]):
        for lbl, d in datasets:
            s = CTRL_STYLE[lbl]
            pos = d["positions"]
            t = np.arange(len(pos)) * sim_dt
            ax.plot(t, pos[:, i], color=f"C{i}", lw=1.5, ls=s["ls"])
    if goal_pos is not None:
        for i, c in enumerate(["C0", "C1", "C2"]):
            ax.axhline(goal_pos[i], color=c, lw=0.7, ls=":", alpha=0.5)
    ax.set_ylabel("Position (m)"); ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3); ax.set_title("Position vs time")

    ax = axes[1, 1]
    for lbl, d in datasets:
        s = CTRL_STYLE[lbl]
        obs = d["obs"]
        t = np.arange(len(obs)) * sim_dt
        tilt = _quat_to_tilt_deg(obs[:, 3], obs[:, 4], obs[:, 5], obs[:, 6])
        ax.plot(t, tilt, color=s["color"], lw=1.5, ls=s["ls"])
    ax.set_ylabel("Tilt (deg)"); ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3); ax.set_title("Tilt angle")

    ax = axes[1, 2]
    for lbl, d in datasets:
        s = CTRL_STYLE[lbl]
        obs = d["obs"]
        t = np.arange(len(obs)) * sim_dt
        speed = np.linalg.norm(obs[:, 7:10], axis=1)
        ax.plot(t, speed, color=s["color"], lw=1.5, ls=s["ls"])
    ax.set_ylabel("|v| (m/s)"); ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3); ax.set_title("Linear speed")

    # Row 2: total thrust, angular velocity, cumulative reward
    ax = axes[2, 0]
    for lbl, d in datasets:
        s = CTRL_STYLE[lbl]
        act = d["actions"]
        if len(act) > 0:
            t = np.arange(len(act)) * sim_dt
            ax.plot(t, np.sum(act, axis=1), color=s["color"], lw=1, ls=s["ls"])
    ax.set_ylabel("Total thrust (N)"); ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3); ax.set_title("Total thrust")

    ax = axes[2, 1]
    for lbl, d in datasets:
        s = CTRL_STYLE[lbl]
        obs = d["obs"]
        t = np.arange(len(obs)) * sim_dt
        ax.plot(t, np.linalg.norm(obs[:, 10:13], axis=1),
                color=s["color"], lw=1, ls=s["ls"])
    ax.set_ylabel("|omega| (rad/s)"); ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3); ax.set_title("Angular velocity")

    ax = axes[2, 2]
    for lbl, d in datasets:
        s = CTRL_STYLE[lbl]
        rew = d["rewards"]
        if len(rew) > 0:
            t = np.arange(len(rew)) * sim_dt
            ax.plot(t, np.cumsum(rew), color=s["color"], lw=1.5, ls=s["ls"])
    ax.set_ylabel("Cumulative reward"); ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3); ax.set_title("Cumulative reward")

    # Row 3: position error, per-motor thrust, 3D trajectory
    ax = axes[3, 0]
    if goal_pos is not None:
        parts = []
        for lbl, d in datasets:
            s = CTRL_STYLE[lbl]
            pos = d["positions"]
            t = np.arange(len(pos)) * sim_dt
            err = np.linalg.norm(pos - goal_pos, axis=1)
            ax.plot(t, err, color=s["color"], lw=1.5, ls=s["ls"])
            parts.append(f"{lbl}: {err[-1]:.3f}m")
        ax.set_title("Final err  " + "  ".join(parts))
    ax.set_ylabel("Pos error (m)"); ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    for lbl, d in datasets:
        s = CTRL_STYLE[lbl]
        act = d["actions"]
        if len(act) > 0:
            t = np.arange(len(act)) * sim_dt
            for m in range(min(act.shape[1], 4)):
                ax.plot(t, act[:, m], color=s["color"], lw=0.7, ls=s["ls"],
                        alpha=0.6)
    ax.set_ylabel("Motor thrust (N)"); ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3); ax.set_title("Per-motor thrust")

    axes[3, 2].remove()
    ax3d = fig.add_subplot(n_rows, n_cols, 12, projection="3d")
    for lbl, d in datasets:
        s = CTRL_STYLE[lbl]
        pos = d["positions"]
        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                  color=s["color"], lw=2, ls=s["ls"])
    if goal_pos is not None:
        ax3d.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]],
                     color="green", s=120, marker="*", zorder=5)
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax3d.set_title("3D trajectory")

    plt.tight_layout(h_pad=3.0, w_pad=2.5, rect=[0, 0, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deterministic scenario comparison: MPC vs RL+CBF")
    parser.add_argument("--scenarios_config", type=str,
                        default=os.path.join(_REPO_ROOT, "configs", "scenarios_config.yaml"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--cbf_config", type=str, default=None)
    parser.add_argument("--mpc_config", type=str, default=None)
    parser.add_argument("--controllers", nargs="+", default=None,
                        choices=ALL_CONTROLLERS,
                        help="Controllers to run (default: all)")
    parser.add_argument("--save_plots", action="store_true", default=False)
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument(
        "--comparison_base_seed",
        type=int,
        default=None,
        help="Base seed for per-scenario disturbance + obs-noise streams "
             "(default: scenarios YAML comparison_base_seed or 7777).",
    )
    parser.add_argument(
        "--no_sync_stochastic_seeds",
        action="store_true",
        help="Do not reseed between controllers (legacy: different noise per run).",
    )
    args = parser.parse_args()

    scfg = _load_scenarios_config(args.scenarios_config)

    controllers = args.controllers or ALL_CONTROLLERS
    needs_rl = any(c in controllers for c in ["RL", "RL+CBF"])
    needs_mpc = any(c in controllers for c in ["MPC", "MPC+Con"])

    goal_pos = np.array(scfg.get("goal", [0, 0, 5]), dtype=np.float64)
    max_steps = scfg.get("max_episode_steps", 1000)
    sim_dt = scfg.get("sim_dt", 0.02)
    barriers = scfg.get("position_barriers", [])
    scenarios = scfg.get("scenarios", [])
    plot_cfg = scfg.get("plotting", {})
    save_plots = args.save_plots or plot_cfg.get("save_plots", False)
    plot_dir = args.plot_dir or plot_cfg.get("plot_dir", "scenario_plots")
    log_dir = plot_dir
    rl_cfg = scfg.get("rl_policy", {})
    act_hist_len = rl_cfg.get("action_history_len", 0)
    deterministic = rl_cfg.get("deterministic", True)

    comparison_base_seed = args.comparison_base_seed
    if comparison_base_seed is None:
        comparison_base_seed = int(scfg.get("comparison_base_seed", 7777))

    if not scenarios:
        print("No scenarios defined in config."); return

    checkpoint = args.checkpoint or rl_cfg.get("checkpoint")
    if needs_rl and checkpoint is None:
        parser.error("--checkpoint required for RL controllers")

    # Quadrotor dynamics
    qd = scfg.get("quadrotor", {})
    mass = float(qd.get("mass", 0.774))
    g = 9.81
    act_mean = np.full(4, (mass * g) / 4.0, dtype=np.float64)
    act_std = np.full(4, (mass * 2 * g) / 4.0, dtype=np.float64)

    # Build environment
    print("Creating environment ...")
    cfg = _build_env_cfg(scfg)
    base_env = _make_env(cfg)
    env = base_env

    obs_noise_cfg = scfg.get("observation_noise")
    if isinstance(obs_noise_cfg, dict) and (
        obs_noise_cfg.get("position", 0) > 0 or obs_noise_cfg.get("velocity", 0) > 0
    ):
        env = ObservationNoiseWrapper(env, obs_noise_cfg)
        print(f"  Observation noise: pos={obs_noise_cfg.get('position', 0)}, "
              f"vel={obs_noise_cfg.get('velocity', 0)}")

    cr = rl_cfg.get("custom_reward")
    if cr is not None and cr.get("enabled", False):
        env = CustomRewardWrapper(env, cr)

    if act_hist_len > 0:
        env = ActionHistoryWrapper(env, act_hist_len)

    if not args.no_sync_stochastic_seeds:
        print(
            f"  Stochastic sync: comparison_base_seed={comparison_base_seed} "
            "(identical disturbance + obs-noise per scenario for all controllers)",
        )
    else:
        print("  Stochastic sync: disabled (--no_sync_stochastic_seeds)")

    # CBF filter
    cbf, model_quad = None, None
    if "RL+CBF" in controllers or needs_rl:
        print("Building CBF filter ...")
        cbf_kwargs = {}
        if args.cbf_config is not None:
            cbf_kwargs["config_path"] = args.cbf_config
        cbf = CBFFilter(**cbf_kwargs)
        model_quad = cbf.model

    if model_quad is None:
        model_quad = QuadrotorModel()

    # RL policy
    model_policy = None
    if needs_rl:
        print("Loading RL policy ...")
        if rl_cfg.get("normalize_obs", True):
            vecnorm_path = os.path.join(os.path.dirname(checkpoint),
                                        "vecnormalize.pkl")
            if os.path.isfile(vecnorm_path):
                from stable_baselines3.common.vec_env import VecNormalize
                env = VecNormalize.load(vecnorm_path, env)
                env.training = False
                env.norm_reward = False
        from stable_baselines3 import PPO
        model_policy = PPO.load(checkpoint, env=env)

    # MPC controllers
    mpc_free, mpc_con, model_quad_mpc = None, None, None
    if needs_mpc:
        bnd = _barriers_to_axis_bounds(barriers)
        pos_min = np.array([
            bnd.get("x", (-20, 20))[0] or -20,
            bnd.get("y", (-20, 20))[0] or -20,
            bnd.get("z", (0, 20))[0] or 0,
        ])
        pos_max = np.array([
            bnd.get("x", (-20, 20))[1] or 20,
            bnd.get("y", (-20, 20))[1] or 20,
            bnd.get("z", (0, 20))[1] or 20,
        ])
        model_quad_mpc = QuadrotorModel()

        if "MPC" in controllers:
            print("Building MPC solver (free) ...")
            mpc_free = MPCController(
                mpc_config_path=args.mpc_config, constrained=False,
                solver_label="free")
        if "MPC+Con" in controllers:
            print("Building MPC solver (constrained) ...")
            mpc_con = MPCController(
                mpc_config_path=args.mpc_config, pos_min=pos_min,
                pos_max=pos_max, constrained=True, solver_label="con")

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    else:
        os.makedirs(log_dir, exist_ok=True)

    # ---- Run scenarios ----
    col_w = 16
    summary_rows = []
    scenario_details = []

    for si, scenario in enumerate(scenarios):
        name = scenario.get("name", f"Scenario {si}")
        state_13d = _build_state_13d(scenario)

        print(f"\n{'='*70}")
        print(f"Scenario {si}: {name}")
        print(f"  pos={state_13d[:3]}  quat={state_13d[3:7]}  "
              f"vel={state_13d[7:10]}  omega={state_13d[10:13]}")
        print(f"{'='*70}")

        ep_data = {}
        rollout_seed = None
        if not args.no_sync_stochastic_seeds:
            rollout_seed = _scenario_rollout_seed(comparison_base_seed, si)

        for ctrl in controllers:
            print(f"  Running {ctrl} ...")
            t0 = time.perf_counter()

            if ctrl == "RL":
                ep_data[ctrl] = _run_rl_scenario(
                    env, base_env, model_policy, model_quad, goal_pos,
                    act_mean, act_std, max_steps, state_13d,
                    act_hist_len, deterministic, rollout_seed)
            elif ctrl == "RL+CBF":
                ep_data[ctrl] = _run_cbf_scenario(
                    env, base_env, model_policy, cbf, model_quad, goal_pos,
                    act_mean, act_std, max_steps, state_13d,
                    act_hist_len, deterministic, rollout_seed)
            elif ctrl == "MPC":
                ep_data[ctrl] = _run_mpc_scenario(
                    env, base_env, mpc_free, model_quad_mpc, goal_pos,
                    act_mean, act_std, max_steps, state_13d,
                    act_hist_len, rollout_seed)
            elif ctrl == "MPC+Con":
                ep_data[ctrl] = _run_mpc_scenario(
                    env, base_env, mpc_con, model_quad_mpc, goal_pos,
                    act_mean, act_std, max_steps, state_13d,
                    act_hist_len, rollout_seed)

            wall = time.perf_counter() - t0
            d = ep_data[ctrl]
            final_err = float(np.linalg.norm(d["positions"][-1] - goal_pos))
            total_rew = float(np.sum(d["rewards"]))
            print(f"    steps={d['steps']}, reward={total_rew:.2f}, "
                  f"final_err={final_err:.4f}m, wall={wall:.1f}s")

        row = {"name": name}
        detail = {"name": name, "controllers": {}}
        for ctrl in controllers:
            d = ep_data[ctrl]
            row[f"{ctrl}_err"] = float(np.linalg.norm(d["positions"][-1] - goal_pos))
            row[f"{ctrl}_rew"] = float(np.sum(d["rewards"]))
            row[f"{ctrl}_steps"] = d["steps"]
            cinfo = {
                "final_err": row[f"{ctrl}_err"],
                "total_reward": row[f"{ctrl}_rew"],
                "steps": int(d["steps"]),
            }
            if ctrl == "RL+CBF":
                cinfo["qp_failures"] = int(d.get("qp_failures", 0))
            if ctrl.startswith("MPC"):
                st = np.asarray(d.get("solve_times", []), dtype=np.float64)
                ss = np.asarray(d.get("solver_statuses", []), dtype=np.int32)
                if st.size > 0:
                    cinfo["solve_ms_mean"] = float(np.mean(st))
                    cinfo["solve_ms_max"] = float(np.max(st))
                    cinfo["solve_ms_min"] = float(np.min(st))
                if ss.size > 0:
                    cinfo["solver_nonzero_status_count"] = int(np.sum(ss != 0))
                    cinfo["solver_last_status"] = int(ss[-1])
            detail["controllers"][ctrl] = cinfo
        summary_rows.append(row)
        scenario_details.append(detail)

        save_path = os.path.join(
            plot_dir, f"scenario_{si}.png") if save_plots else None
        plot_datasets = [(c, ep_data[c]) for c in controllers]
        plot_scenario(plot_datasets, sim_dt, barriers, goal_pos, name,
                      save_path=save_path)

    env.close()

    # ---- Summary table ----
    W = 30 + col_w * len(controllers)
    print(f"\n{'='*W}")
    print("SCENARIO COMPARISON SUMMARY")
    print(f"{'='*W}")

    header = f"{'Scenario':<30}" + "".join(f"{c:>{col_w}}" for c in controllers)
    print(header)
    print("-" * W)
    for row in summary_rows:
        line = f"{row['name']:<30}"
        for c in controllers:
            err = row.get(f"{c}_err", float("nan"))
            steps = row.get(f"{c}_steps", 0)
            line += f"{err:>7.3f}m {steps:>4}st  "
        print(line)

    print(f"{'='*W}")
    print(f"  Goal: {goal_pos}")
    print(f"  Controllers: {controllers}")
    print(f"  max_steps: {max_steps}, sim_dt: {sim_dt}")
    print(f"{'='*W}")

    # ---- Save textual run log (errors/solve-time/status) ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"compare_scenarios_log_{ts}.txt")
    with open(log_path, "w") as f:
        f.write("Scenario comparison run log\n")
        f.write(f"timestamp: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"scenarios_config: {args.scenarios_config}\n")
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"cbf_config: {args.cbf_config}\n")
        f.write(f"mpc_config: {args.mpc_config}\n")
        f.write(f"goal: {goal_pos}\n")
        f.write(f"controllers: {controllers}\n")
        f.write(f"max_steps: {max_steps}, sim_dt: {sim_dt}\n\n")

        for row in summary_rows:
            f.write(f"Scenario: {row['name']}\n")
            for c in controllers:
                err = row.get(f"{c}_err", float("nan"))
                rew = row.get(f"{c}_rew", float("nan"))
                steps = row.get(f"{c}_steps", 0)
                f.write(f"  {c}: err={err:.4f}m, reward={rew:.3f}, steps={steps}\n")
            f.write("\n")

        f.write("Per-controller detailed runtime stats\n")
        f.write("-" * 60 + "\n")
        for si, detail in enumerate(scenario_details):
            f.write(f"Scenario {si}: {detail['name']}\n")
            for c in controllers:
                cinfo = detail["controllers"].get(c, {})
                line = (
                    f"  {c}: final_err={cinfo.get('final_err', float('nan')):.4f}m, "
                    f"total_reward={cinfo.get('total_reward', float('nan')):.3f}, "
                    f"steps={cinfo.get('steps', 0)}"
                )
                if "qp_failures" in cinfo:
                    line += f", qp_failures={cinfo['qp_failures']}"
                if "solve_ms_mean" in cinfo:
                    line += (
                        f", solve_ms(mean/min/max)="
                        f"{cinfo['solve_ms_mean']:.3f}/"
                        f"{cinfo['solve_ms_min']:.3f}/"
                        f"{cinfo['solve_ms_max']:.3f}"
                    )
                if "solver_nonzero_status_count" in cinfo:
                    line += (
                        f", nonzero_solver_status={cinfo['solver_nonzero_status_count']}, "
                        f"last_status={cinfo.get('solver_last_status', 0)}"
                    )
                f.write(line + "\n")
            f.write("\n")

    print(f"Saved run log to: {log_path}")


if __name__ == "__main__":
    main()
