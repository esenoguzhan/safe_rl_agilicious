#!/usr/bin/env python3
"""
MPC evaluation: run the nonlinear MPC controller in the Flightmare simulator
and plot trajectories, states, thrusts, and solver timing. Designed for
direct comparison with RL+CBF evaluation (scripts/eval_cbf.py).

Usage:
  python scripts/eval_mpc.py --config configs/drone_ppo_default.yaml \\
    --mpc_config configs/mpc_config.yaml \\
    [--episodes 5] [--save_plots] [--plot_dir mpc_plots] \\
    [--goal 0 0 5] [--seed 42]

Note: acados may print "ACADOS_MINSTEP" messages to stderr. These are benign
(solver essentially converged). Suppress with: python ... 2>/dev/null
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
from scripts.env_wrapper import FlightlibVecEnv
from scripts.quadrotor_model import POS, ATT, VEL, OME, QuadrotorModel, STATE_DIM
from scripts.mpc_controller import MPCController

STATE_OBS_DIM = 13


# ---------------------------------------------------------------------------
# Environment helpers (same as eval_cbf.py)
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_world_box_2d(ax, world_box, dims_xy):
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


def _quat_to_tilt_deg(qw, qx, qy, qz):
    body_z = np.clip(1.0 - 2.0 * (qx**2 + qy**2), -1.0, 1.0)
    return np.degrees(np.arccos(body_z))


def plot_mpc_episode(
    positions,
    obs_list,
    actions,
    solve_times,
    t_axis,
    world_box,
    goal_pos,
    rewards,
    solver_status,
    save_path=None,
    episode_idx=0,
):
    """Plot MPC episode: trajectories, states, thrusts, solve times."""
    import matplotlib.pyplot as plt

    pos = np.array(positions)
    obs = np.array(obs_list)
    act = np.array(actions)
    stimes = np.array(solve_times)
    status = np.array(solver_status)
    n_steps = len(act)
    t_act = t_axis[:n_steps] if len(t_axis) > n_steps else t_axis
    dt = (t_axis[1] - t_axis[0]) if len(t_axis) > 1 else 0.02

    n_rows = 4
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 14))
    fig.suptitle(f"MPC Eval — Episode {episode_idx}")

    # Row 0: Trajectory x-y, y-z, z-x
    for col, (dims, title) in enumerate([
        (("x", "y"), "x vs y"),
        (("y", "z"), "y vs z"),
        (("z", "x"), "z vs x"),
    ]):
        ax = axes[0, col]
        ia = {"x": 0, "y": 1, "z": 2}[dims[0]]
        ib = {"x": 0, "y": 1, "z": 2}[dims[1]]
        ax.plot(pos[:, ia], pos[:, ib], color="C0", linewidth=2, label="MPC")
        if goal_pos is not None:
            ax.scatter([goal_pos[ia]], [goal_pos[ib]], color="green", s=80, marker="*", zorder=5, label="Goal")
        _draw_world_box_2d(ax, world_box, dims)
        ax.set_xlabel(dims[0])
        ax.set_ylabel(dims[1])
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    # Row 1: Position vs time, orientation, linear velocity
    ax = axes[1, 0]
    t_pos = t_axis if len(t_axis) == len(pos) else np.arange(len(pos)) * dt
    for i, lbl in enumerate(["x", "y", "z"]):
        ax.plot(t_pos, pos[:, i], label=lbl)
    if goal_pos is not None:
        for i, (lbl, c) in enumerate(zip(["x_goal", "y_goal", "z_goal"], ["C0", "C1", "C2"])):
            ax.axhline(goal_pos[i], color=c, linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_ylabel("Position (m)")
    ax.set_xlabel("Time (s)")
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

    # Row 2: Motor thrusts, angular velocity, rewards
    ax = axes[2, 0]
    for i in range(4):
        ax.plot(t_act, act[:, i], label=f"motor_{i}")
    ax.set_ylabel("Motor thrust (N)")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t_axis, obs[:, 10], label="wx")
    ax.plot(t_axis, obs[:, 11], label="wy")
    ax.plot(t_axis, obs[:, 12], label="wz")
    ax.set_ylabel("Angular vel (rad/s)")
    ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 2]
    if len(rewards) > 0:
        cum_r = np.cumsum(rewards)
        ax.plot(t_act[:len(cum_r)], cum_r, color="C0", label="cumulative reward")
        ax.set_ylabel("Cumulative reward")
        ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)

    # Row 3: Solve times, position error, 3D trajectory
    ax = axes[3, 0]
    ax.plot(t_act, stimes, color="C0", linewidth=0.8)
    ax.set_ylabel("MPC solve time (ms)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    failed = np.where(np.isin(status, [1, 4]))[0]
    if len(failed) > 0:
        ax.scatter(t_act[failed], stimes[failed], color="red", s=15, zorder=5, label="solve failed")
        ax.legend(fontsize=6)
    ax.set_title(f"Avg solve: {np.mean(stimes):.2f} ms, max: {np.max(stimes):.2f} ms")

    ax = axes[3, 1]
    if goal_pos is not None:
        pos_err = np.linalg.norm(pos - goal_pos, axis=1)
        ax.plot(t_pos, pos_err, color="C0", label="|pos - goal| (m)")
        ax.set_ylabel("Position error (m)")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right", fontsize=6)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Final pos error: {pos_err[-1]:.4f} m" if goal_pos is not None else "")

    axes[3, 2].remove()
    ax3d = fig.add_subplot(n_rows, n_cols, 12, projection="3d")
    ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color="C0", linewidth=2, label="MPC")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MPC evaluation in Flightmare")
    parser.add_argument("--config", type=str, required=True, help="Path to env YAML config (e.g. configs/drone_ppo_default.yaml)")
    parser.add_argument("--mpc_config", type=str, default=None, help="Path to MPC YAML config (default: configs/mpc_config.yaml)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=None, help="Override max episode steps")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--plot_dir", type=str, default="mpc_plots")
    parser.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--debug", action="store_true", help="Print per-step MPC debug info")
    parser.add_argument("--debug_steps", type=int, default=None)
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

    env_cfg = cfg.get("env", {})
    sim_dt = env_cfg.get("quadrotor_env", {}).get("sim_dt", 0.02)
    max_episode_steps = args.max_steps or cfg_val.get("evaluation", {}).get("max_episode_steps", 1000)
    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 0)
    goal_pos = np.array(cfg_val.get("env", {}).get("goal_position", [0.0, 0.0, 5.0]), dtype=np.float64)
    world_box = env_cfg.get("world_box")

    qd = env_cfg.get("quadrotor_dynamics", {})
    mass = float(qd.get("mass", 0.774))
    g = 9.81
    act_mean = np.full(4, (mass * g) / 4.0, dtype=np.float64)
    act_std = np.full(4, (mass * 2 * g) / 4.0, dtype=np.float64)

    # Build environment
    print("Creating environment ...")
    env = _make_env(cfg_val)

    print("Building MPC solver ...")
    mpc = MPCController(mpc_config_path=args.mpc_config)
    model_quad = QuadrotorModel()
    print(f"  N={mpc.N}, dt={mpc.dt}, thrust_limits={mpc.thrust_limits}")

    if args.save_plots:
        os.makedirs(args.plot_dir, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    all_ep_rewards = []
    all_final_errors = []
    all_solve_times = []

    for ep in range(args.episodes):
        ep_seed = (seed + ep) if seed is not None else None
        if ep_seed is not None:
            np.random.seed(ep_seed)
        _set_env_seed(env, ep_seed)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = np.asarray(obs[0]).ravel()

        # Extract physical state from observation
        obs_phys = obs[:STATE_OBS_DIM].astype(np.float64)
        state = model_quad.state_from_observation(obs_phys, goal_pos=goal_pos)
        mpc.reset(state)

        positions = [state[POS].copy()]
        obs_list = [obs_phys.copy()]
        actions_list = []
        rewards_list = []
        solve_times = []
        solver_status_list = []
        done = False
        steps = 0

        print(f"\n--- Episode {ep} ---")
        print(f"  initial pos: [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}]")

        while not done and steps < max_episode_steps:
            # Solve MPC
            u_mpc = mpc.solve(state, goal_pos)
            solve_times.append(mpc.last_solve_time_ms)
            solver_status_list.append(mpc.last_status)

            if args.debug and (args.debug_steps is None or steps < args.debug_steps):
                t = steps * sim_dt
                p = state[POS]
                print(f"  step={steps} t={t:.3f}s  pos=[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]"
                      f"  u=[{u_mpc[0]:.3f}, {u_mpc[1]:.3f}, {u_mpc[2]:.3f}, {u_mpc[3]:.3f}]"
                      f"  solve={mpc.last_solve_time_ms:.2f}ms  status={mpc.last_status}")

            # Convert motor thrusts [N] -> normalized [-1, 1] for flightlib
            u_norm = np.clip((u_mpc - act_mean) / (act_std + 1e-8), -1.0, 1.0).astype(np.float32)

            actions_list.append(u_mpc.copy())

            obs, reward, dones, infos = env.step(u_norm.reshape(1, -1))
            obs = np.asarray(obs[0]).ravel()
            obs_phys = obs[:STATE_OBS_DIM].astype(np.float64)
            state = model_quad.state_from_observation(obs_phys, goal_pos=goal_pos)

            positions.append(state[POS].copy())
            obs_list.append(obs_phys.copy())
            rewards_list.append(float(reward[0]))
            done = bool(dones[0])
            steps += 1

            if infos and getattr(infos[0], "__contains__", None) and "episode" in infos[0]:
                break

        ep_reward = sum(rewards_list)
        final_err = float(np.linalg.norm(state[POS] - goal_pos))
        avg_solve = float(np.mean(solve_times)) if solve_times else 0.0
        # acados: 0=SUCCESS, 2=MAX_ITER (partial convergence, still usable),
        #         3=MINSTEP (essentially converged), 1/4=real failures
        from collections import Counter
        status_counts = Counter(solver_status_list)
        n_hard_fail = sum(v for k, v in status_counts.items() if k not in {0, 2, 3})

        print(f"  steps: {steps}, reward: {ep_reward:.2f}, final_pos_err: {final_err:.4f} m")
        print(f"  solve time — avg: {avg_solve:.2f} ms, max: {max(solve_times):.2f} ms")
        status_str = ", ".join(f"{k}:{v}" for k, v in sorted(status_counts.items()))
        print(f"  solver status — {status_str}")
        if n_hard_fail > 0:
            print(f"  WARNING: {n_hard_fail}/{steps} hard solver failures (status 1 or 4)")

        all_ep_rewards.append(ep_reward)
        all_final_errors.append(final_err)
        all_solve_times.extend(solve_times)

        t_axis = np.arange(len(positions)) * sim_dt
        save_path = os.path.join(args.plot_dir, f"mpc_eval_episode_{ep}.png") if args.save_plots else None
        plot_mpc_episode(
            positions,
            obs_list,
            actions_list,
            solve_times,
            t_axis,
            world_box,
            goal_pos,
            rewards_list,
            solver_status_list,
            save_path=save_path,
            episode_idx=ep,
        )

    env.close()

    print("\n" + "=" * 60)
    print("MPC Evaluation Summary")
    print("=" * 60)
    print(f"  Episodes:         {args.episodes}")
    print(f"  Avg reward:       {np.mean(all_ep_rewards):.2f} +/- {np.std(all_ep_rewards):.2f}")
    print(f"  Avg final error:  {np.mean(all_final_errors):.4f} +/- {np.std(all_final_errors):.4f} m")
    print(f"  Avg solve time:   {np.mean(all_solve_times):.2f} ms")
    print(f"  Max solve time:   {np.max(all_solve_times):.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
