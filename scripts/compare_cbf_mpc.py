#!/usr/bin/env python3
"""
Back-to-back comparison: pure RL vs RL+CBF vs RL+CBF2(acados) vs MPC on the same N episodes.
Each episode uses the same seed so all three controllers face identical
initial conditions. Trajectories are overlaid on the same plots.

All scenario parameters (spawn, goal, world box, episode count, etc.)
are read from a single comparison config YAML (default:
configs/compare_config.yaml).  Controller-specific tuning lives in
mpc_config.yaml and cbf_config.yaml.

Usage:
  python scripts/compare_cbf_mpc.py \\
    --compare_config configs/compare_config.yaml \\
    --checkpoint models/.../best_model \\
    [--mpc_config configs/mpc_config.yaml] \\
    [--cbf_config configs/cbf_config.yaml] \\
    [--env_config configs/drone_ppo_default.yaml]

Note: acados may print "ACADOS_MINSTEP" messages to stderr. These are
benign (solver converged). Suppress with: python ... 2>/dev/null
"""
import argparse
import copy
import glob
import os
import sys
import time

import numpy as np
import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config_loader import load_config, prepare_env_run_dir, get_vec_env_config_string
from scripts.context import flightmare_context
from scripts.custom_reward_wrapper import CustomRewardWrapper
from scripts.env_wrapper import FlightlibVecEnv, ActionHistoryWrapper, ObservationNoiseWrapper
from scripts.cbf_filter import CBFFilter
from scripts.quadrotor_model import POS, ATT, VEL, OME, QuadrotorModel, STATE_DIM
from scripts.mpc_controller import MPCController

STATE_OBS_DIM = 13


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_compare_config(path=None):
    if path is None:
        path = os.path.join(_REPO_ROOT, "configs", "compare_config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_env_cfg(ccfg, base_env_cfg_path=None):
    """
    Build a drone_ppo_default–style env config dict from the comparison
    config, optionally merging on top of an existing base env config.
    """
    if base_env_cfg_path is not None:
        base = load_config(base_env_cfg_path)
    else:
        base = {}

    base.setdefault("env", {})
    env = base["env"]

    scenario = ccfg.get("scenario", {})
    spawn = ccfg.get("spawn", {})
    goal_cfg = ccfg.get("goal", {})
    quad = ccfg.get("quadrotor", {})

    env["max_episode_steps"] = scenario.get("max_episode_steps", 500)

    env.setdefault("vec_env", {})
    env["vec_env"]["num_envs"] = 1
    env["vec_env"]["num_threads"] = 1

    env.setdefault("quadrotor_env", {})
    env["quadrotor_env"]["sim_dt"] = scenario.get("sim_dt", 0.02)

    if quad:
        env["quadrotor_dynamics"] = quad

    env["motor_init"] = spawn.get("motor_init", "zero")

    # Goal: fixed or per-episode random (handled in the loop)
    fixed_goal = goal_cfg.get("position", [0.0, 0.0, 5.0])
    env["goal_position"] = list(fixed_goal)

    # Spawn ranges
    env["spawn_ranges"] = {
        k: v for k, v in spawn.items() if k != "motor_init"
    }

    world_box = ccfg.get("world_box")
    if world_box is not None:
        env["world_box"] = list(world_box)

    # Custom reward (needed so the RL policy gets the same reward signal)
    rl_cfg = ccfg.get("rl_policy", {})
    cr = rl_cfg.get("custom_reward")
    if cr is not None:
        env["custom_reward"] = cr

    env["action_history_len"] = rl_cfg.get("action_history_len", 0)

    # Training section (normalize_obs flag lives here)
    base.setdefault("training", {})
    base["training"]["normalize_obs"] = rl_cfg.get("normalize_obs", True)
    base["training"]["seed"] = scenario.get("seed", 0)

    # Evaluation
    base.setdefault("evaluation", {})
    base["evaluation"]["deterministic"] = rl_cfg.get("deterministic", True)
    base["evaluation"]["max_episode_steps"] = scenario.get("max_episode_steps", 500)

    return base


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

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


def _set_env_seed(env, seed):
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


def _sample_random_goal(goal_cfg, rng):
    """Sample a goal from the configured ranges."""
    r = goal_cfg.get("goal_pos_range", {})
    return np.array([
        rng.uniform(*r.get("x", [-3.0, 3.0])),
        rng.uniform(*r.get("y", [-3.0, 3.0])),
        rng.uniform(*r.get("z", [2.0, 8.0])),
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-episode rollout helpers
# ---------------------------------------------------------------------------

def _run_cbf_episode(env, model_policy, cbf, model_quad, goal_pos, act_mean,
                     act_std, max_steps, ep_seed, deterministic):
    if ep_seed is not None:
        np.random.seed(ep_seed)
    _set_env_seed(env, ep_seed)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.asarray(obs[0]).ravel()
    obs_raw = _get_raw_obs(env, obs)

    state = model_quad.state_from_observation(obs_raw[:STATE_OBS_DIM], goal_pos=goal_pos)
    print(f"  [CBF] Init: obs_raw[:13]={obs_raw[:STATE_OBS_DIM]}")
    print(f"  [CBF] Init: goal={goal_pos}, reconstructed pos={state[POS]}, vel={state[VEL]}")
    for bar in cbf.barriers:
        h_val = bar.h(state[POS], state[VEL])
        print(f"  [CBF] Init barrier {bar.name}: h={h_val:.4f}")
    positions = [state[POS].copy()]
    obs_list = [obs_raw[:STATE_OBS_DIM].copy()]
    actions = []
    rewards = []
    act_dim = env.action_space.shape[0]

    done, steps = False, 0
    qp_fail_count = 0
    barrier_violations = {b.name: 0 for b in cbf.barriers}
    while not done and steps < max_steps:
        action, _ = model_policy.predict(
            obs.reshape(1, -1) if obs.ndim == 1 else obs,
            deterministic=deterministic,
        )
        action = action.ravel()
        u_raw = action[:4].astype(np.float64) * act_std + act_mean
        u_safe = cbf.filter(state, u_raw)

        if cbf.last_qp_failed:
            qp_fail_count += 1
            if qp_fail_count <= 5:
                print(f"  [CBF] QP FAIL step={steps} pos={state[POS]} "
                      f"reason={cbf.last_qp_failure_reason}")
        for bar in cbf.barriers:
            h_val = bar.h(state[POS], state[VEL])
            if h_val < 0:
                barrier_violations[bar.name] += 1
                if barrier_violations[bar.name] <= 3:
                    print(f"  [CBF] BARRIER VIOLATED: {bar.name} h={h_val:.4f} "
                          f"pos={state[POS]} vel={state[VEL]}")

        safe_norm = np.clip(
            (u_safe.astype(np.float64) - act_mean) / (act_std + 1e-8),
            -1.0, 1.0,
        ).astype(np.float32)
        safe_action = np.zeros(act_dim, dtype=np.float32)
        safe_action[:4] = safe_norm
        if act_dim > 4:
            safe_action[4:] = action[4:]

        actions.append(u_safe.copy())
        obs, reward, dones, infos = env.step(safe_action.reshape(1, -1))
        obs = np.asarray(obs[0]).ravel()
        obs_raw = _get_raw_obs(env, obs)
        state = model_quad.state_from_observation(obs_raw[:STATE_OBS_DIM], goal_pos=goal_pos)
        positions.append(state[POS].copy())
        obs_list.append(obs_raw[:STATE_OBS_DIM].copy())
        rewards.append(float(reward[0]))
        done = bool(dones[0])
        steps += 1
        if infos and getattr(infos[0], "__contains__", None) and "episode" in infos[0]:
            break

    print(f"  [CBF] Episode summary: steps={steps}, QP failures={qp_fail_count}/{steps}, "
          f"barrier violations={barrier_violations}")

    return {
        "positions": np.array(positions),
        "obs": np.array(obs_list),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "steps": steps,
    }


def _run_rl_episode(env, model_policy, model_quad, goal_pos, act_mean,
                    act_std, max_steps, ep_seed, deterministic):
    """Run one episode with pure RL (no CBF filter)."""
    if ep_seed is not None:
        np.random.seed(ep_seed)
    _set_env_seed(env, ep_seed)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.asarray(obs[0]).ravel()
    obs_raw = _get_raw_obs(env, obs)

    state = model_quad.state_from_observation(obs_raw[:STATE_OBS_DIM], goal_pos=goal_pos)
    positions = [state[POS].copy()]
    obs_list = [obs_raw[:STATE_OBS_DIM].copy()]
    actions = []
    rewards = []

    done, steps = False, 0
    while not done and steps < max_steps:
        action, _ = model_policy.predict(
            obs.reshape(1, -1) if obs.ndim == 1 else obs,
            deterministic=deterministic,
        )
        action = action.ravel()
        u_raw = action[:4].astype(np.float64) * act_std + act_mean

        actions.append(u_raw.copy())
        obs, reward, dones, infos = env.step(action.reshape(1, -1))
        obs = np.asarray(obs[0]).ravel()
        obs_raw = _get_raw_obs(env, obs)
        state = model_quad.state_from_observation(obs_raw[:STATE_OBS_DIM], goal_pos=goal_pos)
        positions.append(state[POS].copy())
        obs_list.append(obs_raw[:STATE_OBS_DIM].copy())
        rewards.append(float(reward[0]))
        done = bool(dones[0])
        steps += 1
        if infos and getattr(infos[0], "__contains__", None) and "episode" in infos[0]:
            break

    return {
        "positions": np.array(positions),
        "obs": np.array(obs_list),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "steps": steps,
    }


def _run_mpc_episode(env, mpc, model_quad, goal_pos, act_mean, act_std,
                     max_steps, ep_seed):
    if ep_seed is not None:
        np.random.seed(ep_seed)
    _set_env_seed(env, ep_seed)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    obs = np.asarray(obs[0]).ravel()
    obs_raw = _get_raw_obs(env, obs)

    obs_phys = obs_raw[:STATE_OBS_DIM].astype(np.float64)
    state = model_quad.state_from_observation(obs_phys, goal_pos=goal_pos)
    mpc.reset(state)

    positions = [state[POS].copy()]
    obs_list = [obs_phys.copy()]
    actions = []
    rewards = []
    solve_times = []

    done, steps = False, 0
    while not done and steps < max_steps:
        u_mpc = mpc.solve(state, goal_pos)
        solve_times.append(mpc.last_solve_time_ms)
        u_norm = np.clip(
            (u_mpc - act_mean) / (act_std + 1e-8), -1.0, 1.0,
        ).astype(np.float32)

        actions.append(u_mpc.copy())
        obs, reward, dones, infos = env.step(u_norm.reshape(1, -1))
        obs = np.asarray(obs[0]).ravel()
        obs_raw = _get_raw_obs(env, obs)
        obs_phys = obs_raw[:STATE_OBS_DIM].astype(np.float64)
        state = model_quad.state_from_observation(obs_phys, goal_pos=goal_pos)
        positions.append(state[POS].copy())
        obs_list.append(obs_phys.copy())
        rewards.append(float(reward[0]))
        done = bool(dones[0])
        steps += 1
        if infos and getattr(infos[0], "__contains__", None) and "episode" in infos[0]:
            break

    return {
        "positions": np.array(positions),
        "obs": np.array(obs_list),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "steps": steps,
        "solve_times": np.array(solve_times),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_barriers_2d(ax, barriers, dims_xy, z_fix=None):
    """Draw position barrier boundaries on a 2D projection.

    Each barrier is a dict with 'n' (normal, 3-vector) and 'q' (offset scalar).
    The boundary is n'p + q = 0.  For axis-aligned barriers this produces
    horizontal/vertical lines; for oblique barriers a diagonal line.
    """
    if not barriers:
        return
    dim_idx = {"x": 0, "y": 1, "z": 2}
    ia = dim_idx[dims_xy[0]]
    ib = dim_idx[dims_xy[1]]
    ic = 3 - ia - ib  # the "missing" dimension
    drawn_labels = False
    for b in barriers:
        n = np.asarray(b["n"], dtype=np.float64).ravel()[:3]
        q = float(b["q"])
        lbl = b.get("name", "") if not drawn_labels else None
        kw = dict(color="red", linewidth=1, alpha=0.8)
        if abs(n[ic]) < 1e-10:
            # barrier lives entirely in the plotted plane
            if abs(n[ib]) < 1e-10:
                if abs(n[ia]) > 1e-10:
                    ax.axvline(-q / n[ia], label=lbl, **kw)
            else:
                p_a = np.linspace(-50, 50, 200)
                p_b = (-q - n[ia] * p_a) / (n[ib] + 1e-12)
                ax.plot(p_a, p_b, label=lbl, **kw)
        else:
            # barrier involves the missing dimension; fix it to z_fix
            p_fix = z_fix if z_fix is not None else 5.0
            rhs = -q - n[ic] * p_fix
            kw["alpha"] = 0.6
            kw["linewidth"] = 0.8
            if abs(n[ib]) < 1e-10:
                if abs(n[ia]) > 1e-10:
                    ax.axvline(rhs / n[ia], label=lbl, **kw)
            else:
                p_a = np.linspace(-50, 50, 200)
                p_b = (rhs - n[ia] * p_a) / (n[ib] + 1e-12)
                ax.plot(p_a, p_b, label=lbl, **kw)
        if lbl:
            drawn_labels = True


def _barriers_to_axis_bounds(barriers):
    """Extract axis-aligned min/max from barriers for setting plot limits.

    Returns dict like {'x': (-4, 4), 'y': (-4, 4), 'z': (-1, 4)} or empty.
    """
    bounds = {}
    axis_names = {0: "x", 1: "y", 2: "z"}
    for b in barriers:
        n = np.asarray(b["n"], dtype=np.float64).ravel()[:3]
        q = float(b["q"])
        nonzero = [i for i in range(3) if abs(n[i]) > 1e-10]
        if len(nonzero) != 1:
            continue
        idx = nonzero[0]
        boundary = -q / n[idx]  # n[idx]*p + q = 0  =>  p = -q/n[idx]
        name = axis_names[idx]
        if n[idx] > 0:
            # n'p + q >= 0 with positive n => p >= -q/n = boundary (lower)
            bounds.setdefault(name, [None, None])
            lo = bounds[name][0]
            bounds[name][0] = boundary if lo is None else min(lo, boundary)
        else:
            # n'p + q >= 0 with negative n => p <= -q/n = boundary (upper)
            bounds.setdefault(name, [None, None])
            hi = bounds[name][1]
            bounds[name][1] = boundary if hi is None else max(hi, boundary)
    return {k: tuple(v) for k, v in bounds.items()}


def _quat_to_tilt_deg(qw, qx, qy, qz):
    body_z = np.clip(1.0 - 2.0 * (qx**2 + qy**2), -1.0, 1.0)
    return np.degrees(np.arccos(body_z))


def plot_comparison(datasets, sim_dt, barriers, goal_pos,
                    save_path=None, episode_idx=0):
    """Overlay controller trajectories on the same figure.

    Parameters
    ----------
    datasets : list of (label, data_dict, color, linestyle)
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 20))
    fig.suptitle(f"Episode {episode_idx}  |  "
                 f"goal={np.array2string(goal_pos, precision=1)}",
                 fontsize=14, fontweight="bold", y=0.99)

    legend_handles = [Line2D([0], [0], color=c, ls=ls, lw=2, label=lbl)
                      for lbl, _, c, ls in datasets]
    legend_handles.append(Line2D([0], [0], color="green", marker="*", ls="None",
                                 markersize=10, label="Goal"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=len(datasets) + 1,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.97))

    # Row 0: trajectory projections
    for col, (dims, title) in enumerate([
        (("x", "y"), "x vs y"), (("y", "z"), "y vs z"), (("z", "x"), "z vs x"),
    ]):
        ax = axes[0, col]
        ia = {"x": 0, "y": 1, "z": 2}[dims[0]]
        ib = {"x": 0, "y": 1, "z": 2}[dims[1]]
        for lbl, d, c, ls in datasets:
            pos = d["positions"]
            ax.plot(pos[:, ia], pos[:, ib], color=c, linewidth=2, linestyle=ls, label=lbl)
        if goal_pos is not None:
            ax.scatter([goal_pos[ia]], [goal_pos[ib]], color="green", s=100, marker="*", zorder=5, label="Goal")
        _draw_barriers_2d(ax, barriers, dims, z_fix=goal_pos[2] if goal_pos is not None else None)
        bnd = _barriers_to_axis_bounds(barriers) if barriers else {}
        for dim_key, set_lim in [(dims[0], ax.set_xlim), (dims[1], ax.set_ylim)]:
            if dim_key in bnd:
                lo, hi = bnd[dim_key]
                if lo is not None and hi is not None:
                    margin = 0.15 * (hi - lo)
                    set_lim(lo - margin, hi + margin)
        ax.set_xlabel(dims[0])
        ax.set_ylabel(dims[1])
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    # Row 1: position vs time, tilt, speed
    ax = axes[1, 0]
    for i, coord in enumerate(["x", "y", "z"]):
        for lbl, d, c, ls in datasets:
            pos = d["positions"]
            t = np.arange(len(pos)) * sim_dt
            ax.plot(t, pos[:, i], color=f"C{i}", linewidth=1.5, linestyle=ls,
                    label=f"{coord} {lbl}" if i == 0 else None)
    if goal_pos is not None:
        for i, c in enumerate(["C0", "C1", "C2"]):
            ax.axhline(goal_pos[i], color=c, linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_ylabel("Position (m)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Position vs time")

    ax = axes[1, 1]
    for lbl, d, c, ls in datasets:
        obs = d["obs"]
        t = np.arange(len(obs)) * sim_dt
        tilt = _quat_to_tilt_deg(obs[:, 3], obs[:, 4], obs[:, 5], obs[:, 6])
        ax.plot(t, tilt, color=c, linewidth=1.5, linestyle=ls, label=lbl)
    ax.set_ylabel("Tilt angle (deg)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Tilt angle")

    ax = axes[1, 2]
    for lbl, d, c, ls in datasets:
        obs = d["obs"]
        t = np.arange(len(obs)) * sim_dt
        speed = np.linalg.norm(obs[:, 7:10], axis=1)
        ax.plot(t, speed, color=c, linewidth=1.5, linestyle=ls, label=lbl)
    ax.set_ylabel("Speed |v| (m/s)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Linear speed")

    # Row 2: total thrust, angular velocity, cumulative reward
    ax = axes[2, 0]
    for lbl, d, c, ls in datasets:
        act = d["actions"]
        if len(act) > 0:
            t = np.arange(len(act)) * sim_dt
            ax.plot(t, np.sum(act, axis=1), color=c, linewidth=1, linestyle=ls, label=lbl)
    ax.set_ylabel("Total thrust (N)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Total thrust")

    ax = axes[2, 1]
    for lbl, d, c, ls in datasets:
        obs = d["obs"]
        t = np.arange(len(obs)) * sim_dt
        omega_mag = np.linalg.norm(obs[:, 10:13], axis=1)
        ax.plot(t, omega_mag, color=c, linewidth=1, linestyle=ls, label=lbl)
    ax.set_ylabel("|omega| (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Angular velocity magnitude")

    ax = axes[2, 2]
    for lbl, d, c, ls in datasets:
        rew = d["rewards"]
        if len(rew) > 0:
            t = np.arange(len(rew)) * sim_dt
            ax.plot(t, np.cumsum(rew), color=c, linewidth=1.5, linestyle=ls, label=lbl)
    ax.set_ylabel("Cumulative reward")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Cumulative reward")

    # Row 3: position error, per-motor thrust (total only for clarity), 3D trajectory
    ax = axes[3, 0]
    if goal_pos is not None:
        title_parts = []
        for lbl, d, c, ls in datasets:
            pos = d["positions"]
            t = np.arange(len(pos)) * sim_dt
            err = np.linalg.norm(pos - goal_pos, axis=1)
            ax.plot(t, err, color=c, linewidth=1.5, linestyle=ls, label=lbl)
            title_parts.append(f"{lbl}: {err[-1]:.3f}m")
        ax.set_ylabel("Position error (m)")
        ax.set_xlabel("Time (s)")
        ax.set_title("Final err — " + ", ".join(title_parts))
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    for lbl, d, c, ls in datasets:
        act = d["actions"]
        if len(act) > 0:
            t = np.arange(len(act)) * sim_dt
            total = np.sum(act, axis=1)
            ax.plot(t, total, linewidth=1, color=c, linestyle=ls, label=lbl)
    ax.set_ylabel("Total thrust (N)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Total thrust (detail)")

    axes[3, 2].remove()
    ax3d = fig.add_subplot(n_rows, n_cols, 12, projection="3d")
    for lbl, d, c, ls in datasets:
        pos = d["positions"]
        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=c, linewidth=2, linestyle=ls, label=lbl)
    if goal_pos is not None:
        ax3d.scatter([goal_pos[0]], [goal_pos[1]], [goal_pos[2]],
                     color="green", s=120, marker="*", zorder=5)
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
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
        description="Back-to-back comparison: RL+CBF vs MPC",
    )
    parser.add_argument("--compare_config", type=str, default=None,
                        help="Comparison scenario YAML (default: configs/compare_config.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="RL policy checkpoint (overrides compare_config)")
    parser.add_argument("--mpc_config", type=str, default=None,
                        help="MPC config YAML (default: configs/mpc_config.yaml)")
    parser.add_argument("--cbf_config", type=str, default=None,
                        help="CBF config YAML (default: configs/cbf_config.yaml)")
    parser.add_argument("--env_config", type=str, default=None,
                        help="Optional base env config to merge with (e.g. drone_ppo_default.yaml)")
    # CLI overrides (take precedence over compare_config)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--goal", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--save_plots", action="store_true", default=None)
    parser.add_argument("--plot_dir", type=str, default=None)
    args = parser.parse_args()

    # ---- Load comparison config ----
    ccfg = _load_compare_config(args.compare_config)
    scenario = ccfg.get("scenario", {})
    goal_cfg = ccfg.get("goal", {})
    plot_cfg = ccfg.get("plotting", {})

    # CLI overrides
    n_episodes = args.episodes or scenario.get("episodes", 5)
    max_steps = args.max_steps or scenario.get("max_episode_steps", 500)
    seed = args.seed if args.seed is not None else scenario.get("seed", 0)
    sim_dt = scenario.get("sim_dt", 0.02)
    save_plots = args.save_plots if args.save_plots is not None else plot_cfg.get("save_plots", False)
    plot_dir = args.plot_dir or plot_cfg.get("plot_dir", "comparison_plots")
    randomize_goal = goal_cfg.get("randomize_goal", False)

    if args.goal is not None:
        fixed_goal = np.array(args.goal, dtype=np.float64)
        randomize_goal = False
    else:
        fixed_goal = np.array(goal_cfg.get("position", [0.0, 0.0, 5.0]), dtype=np.float64)

    checkpoint = args.checkpoint or ccfg.get("rl_policy", {}).get("checkpoint")
    if checkpoint is None:
        parser.error("--checkpoint is required (or set rl_policy.checkpoint in compare_config)")

    # ---- Build env config from compare_config ----
    if args.goal is not None:
        ccfg.setdefault("goal", {})["position"] = list(args.goal)
    cfg = _build_env_cfg(ccfg, args.env_config)
    barriers = ccfg.get("position_barriers", [])

    qd = ccfg.get("quadrotor", {})
    mass = float(qd.get("mass", 0.774))
    g = 9.81
    act_mean = np.full(4, (mass * g) / 4.0, dtype=np.float64)
    act_std = np.full(4, (mass * 2 * g) / 4.0, dtype=np.float64)

    # ---- Build environment ----
    print("Creating environment ...")
    env = _make_env(cfg)

    # Wrappers for RL policy
    rl_cfg = ccfg.get("rl_policy", {})

    obs_noise_cfg = ccfg.get("observation_noise")
    if isinstance(obs_noise_cfg, dict) and (
        obs_noise_cfg.get("position", 0) > 0 or obs_noise_cfg.get("velocity", 0) > 0
    ):
        env = ObservationNoiseWrapper(env, obs_noise_cfg)
        print(f"Observation noise: position={obs_noise_cfg.get('position', 0)}, "
              f"velocity={obs_noise_cfg.get('velocity', 0)}")

    cr = rl_cfg.get("custom_reward")
    if cr is not None and cr.get("enabled", False):
        env = CustomRewardWrapper(env, cr)
    action_history_len = rl_cfg.get("action_history_len", 0)
    if action_history_len > 0:
        env = ActionHistoryWrapper(env, action_history_len)

    # ---- CBF filter ----
    print("Building CBF filter ...")
    cbf_kwargs = {}
    if args.cbf_config is not None:
        cbf_kwargs["config_path"] = args.cbf_config
    cbf = CBFFilter(**cbf_kwargs)
    model_quad = cbf.model

    # ---- RL policy ----
    print("Loading RL policy ...")
    if rl_cfg.get("normalize_obs", True):
        vecnorm_path = os.path.join(os.path.dirname(checkpoint), "vecnormalize.pkl")
        if os.path.isfile(vecnorm_path):
            from stable_baselines3.common.vec_env import VecNormalize
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False
    from stable_baselines3 import PPO
    model_policy = PPO.load(checkpoint, env=env)
    deterministic = rl_cfg.get("deterministic", True)

    # ---- MPC controllers (free + constrained) ----
    # Derive position bounds from barriers
    bnd = _barriers_to_axis_bounds(barriers)
    if bnd:
        barrier_pos_min = np.array([
            bnd.get("x", (-20, 20))[0] or -20,
            bnd.get("y", (-20, 20))[0] or -20,
            bnd.get("z", (0, 20))[0] or 0,
        ])
        barrier_pos_max = np.array([
            bnd.get("x", (-20, 20))[1] or 20,
            bnd.get("y", (-20, 20))[1] or 20,
            bnd.get("z", (0, 20))[1] or 20,
        ])
    else:
        barrier_pos_min = np.array([-20., -20., 0.])
        barrier_pos_max = np.array([20., 20., 20.])

    print("Building MPC solver (free) ...")
    mpc_free = MPCController(
        mpc_config_path=args.mpc_config,
        constrained=False,
        solver_label="free",
    )
    print(f"  MPC free: N={mpc_free.N}, dt={mpc_free.dt}, thrust_limits={mpc_free.thrust_limits}")

    print("Building MPC solver (constrained) ...")
    mpc_con = MPCController(
        mpc_config_path=args.mpc_config,
        pos_min=barrier_pos_min,
        pos_max=barrier_pos_max,
        constrained=True,
        solver_label="con",
    )
    print(f"  MPC constrained: pos_min={barrier_pos_min}, pos_max={barrier_pos_max}")

    model_quad_mpc = QuadrotorModel()

    # ---- Run ----
    print(f"\nScenario: {n_episodes} episodes, max_steps={max_steps}, "
          f"seed={seed}, randomize_goal={randomize_goal}")
    if not randomize_goal:
        print(f"  fixed goal: {fixed_goal}")

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    if seed is not None:
        np.random.seed(seed)

    goal_rng = np.random.RandomState(seed)

    # Controller registry: (label, color, linestyle)
    CTRL_META = [
        ("RL",       "C1", "-"),
        ("RL+CBF",   "C0", "-"),
        ("MPC",      "C3", "--"),
        ("MPC+Con",  "C2", "--"),
    ]
    ctrl_names = [m[0] for m in CTRL_META]
    stats = {name: {"rewards": [], "errors": [], "steps": []} for name in ctrl_names}

    for ep in range(n_episodes):
        ep_seed = (seed + ep) if seed is not None else None

        if randomize_goal:
            goal_pos = _sample_random_goal(goal_cfg, goal_rng)
        else:
            goal_pos = fixed_goal.copy()

        print(f"\n{'='*60}")
        print(f"Episode {ep}  seed={ep_seed}  goal={np.array2string(goal_pos, precision=2)}")
        print(f"{'='*60}")

        ep_data = {}

        # ---- Pure RL ----
        print("  Running pure RL ...")
        t0 = time.perf_counter()
        ep_data["RL"] = _run_rl_episode(
            env, model_policy, model_quad, goal_pos,
            act_mean, act_std, max_steps, ep_seed, deterministic,
        )
        rl_wall = time.perf_counter() - t0
        d = ep_data["RL"]
        rl_rew = float(np.sum(d["rewards"]))
        rl_err = float(np.linalg.norm(d["positions"][-1] - goal_pos))
        print(f"    steps={d['steps']}, reward={rl_rew:.2f}, "
              f"final_err={rl_err:.4f}m, wall={rl_wall:.1f}s")

        # ---- RL+CBF ----
        print("  Running RL+CBF ...")
        t0 = time.perf_counter()
        ep_data["RL+CBF"] = _run_cbf_episode(
            env, model_policy, cbf, model_quad, goal_pos,
            act_mean, act_std, max_steps, ep_seed, deterministic,
        )
        cbf_wall = time.perf_counter() - t0
        d = ep_data["RL+CBF"]
        cbf_rew = float(np.sum(d["rewards"]))
        cbf_err = float(np.linalg.norm(d["positions"][-1] - goal_pos))
        print(f"    steps={d['steps']}, reward={cbf_rew:.2f}, "
              f"final_err={cbf_err:.4f}m, wall={cbf_wall:.1f}s")

        # ---- MPC (free) ----
        print("  Running MPC (free) ...")
        t0 = time.perf_counter()
        ep_data["MPC"] = _run_mpc_episode(
            env, mpc_free, model_quad_mpc, goal_pos,
            act_mean, act_std, max_steps, ep_seed,
        )
        mpc_wall = time.perf_counter() - t0
        d = ep_data["MPC"]
        mpc_rew = float(np.sum(d["rewards"]))
        mpc_err = float(np.linalg.norm(d["positions"][-1] - goal_pos))
        avg_solve = float(np.mean(d["solve_times"])) if len(d["solve_times"]) else 0
        print(f"    steps={d['steps']}, reward={mpc_rew:.2f}, "
              f"final_err={mpc_err:.4f}m, wall={mpc_wall:.1f}s, "
              f"solve_avg={avg_solve:.2f}ms")

        # ---- MPC (constrained) ----
        print("  Running MPC+Con (constrained) ...")
        t0 = time.perf_counter()
        ep_data["MPC+Con"] = _run_mpc_episode(
            env, mpc_con, model_quad_mpc, goal_pos,
            act_mean, act_std, max_steps, ep_seed,
        )
        con_wall = time.perf_counter() - t0
        d = ep_data["MPC+Con"]
        con_rew = float(np.sum(d["rewards"]))
        con_err = float(np.linalg.norm(d["positions"][-1] - goal_pos))
        avg_solve_c = float(np.mean(d["solve_times"])) if len(d["solve_times"]) else 0
        print(f"    steps={d['steps']}, reward={con_rew:.2f}, "
              f"final_err={con_err:.4f}m, wall={con_wall:.1f}s, "
              f"solve_avg={avg_solve_c:.2f}ms")

        for name in ctrl_names:
            dd = ep_data[name]
            stats[name]["rewards"].append(float(np.sum(dd["rewards"])))
            stats[name]["errors"].append(float(np.linalg.norm(dd["positions"][-1] - goal_pos)))
            stats[name]["steps"].append(dd["steps"])

        save_path = os.path.join(plot_dir, f"compare_ep{ep}.png") if save_plots else None
        plot_datasets = [
            (lbl, ep_data[lbl], c, ls) for lbl, c, ls in CTRL_META
        ]
        plot_comparison(
            plot_datasets, sim_dt, barriers, goal_pos,
            save_path=save_path, episode_idx=ep,
        )

    env.close()

    # ---- Summary ----
    col_w = 20
    W = 25 + col_w * len(ctrl_names)
    print(f"\n{'='*W}")
    print("COMPARISON SUMMARY")
    print(f"{'='*W}")
    header = f"{'Metric':<25}" + "".join(f"{n:>{col_w}}" for n in ctrl_names)
    print(header)
    print("-" * W)

    def _fmt(vals):
        return f"{np.mean(vals):>8.2f} +/- {np.std(vals):<8.2f}"

    for metric, key in [("Avg reward", "rewards"), ("Avg final error (m)", "errors"), ("Avg episode steps", "steps")]:
        row = f"{metric:<25}" + "".join(f"{_fmt(stats[n][key]):>{col_w}}" for n in ctrl_names)
        print(row)
    print(f"{'='*W}")
    print(f"  Episodes: {n_episodes}, max_steps: {max_steps}")
    print(f"  Goal: {'random' if randomize_goal else str(fixed_goal)}")
    print(f"  Barriers: pos_min={barrier_pos_min}, pos_max={barrier_pos_max}")
    print(f"{'='*W}")


if __name__ == "__main__":
    main()
