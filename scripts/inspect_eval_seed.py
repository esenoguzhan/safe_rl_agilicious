#!/usr/bin/env python3
"""
Inspect what initial scenarios a fixed eval seed produces.

Builds the same eval env as train_v2 (1 env, same wrappers), sets the given seed
(np.random + C++ env), then runs N resets and logs each scenario's initial state:
- position_error (obs 0:3) = goal - position  =>  |pos_error| = distance to goal
- linear_velocity (obs 7:10)
- angular_velocity (obs 10:13)
- tilt_deg: angle from vertical in degrees (0=upright, 90=flat), derived from quat (obs 3:7)

So you can see whether the eval seed yields "hard" scenarios (large distance, high velocity, high tilt).

Usage:
  python scripts/inspect_eval_seed.py --config configs/curriculum/phase1_baseline.yaml
  python scripts/inspect_eval_seed.py --config configs/curriculum/phase1_baseline.yaml --seed 7777 --n 20
"""
import argparse
import copy
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config_loader import load_config, prepare_env_run_dir, get_vec_env_config_string
from scripts.context import flightmare_context
from scripts.custom_reward_wrapper import CustomRewardWrapper
from scripts.env_wrapper import (
    FlightlibVecEnv,
    VecMaxEpisodeSteps,
    DomainRandomizationWrapper,
    ObservationNoiseWrapper,
    ActionHistoryWrapper,
)


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
    import glob
    flightlib_dir = os.path.join(_REPO_ROOT, "flightmare", "flightlib")
    build_dir = os.path.join(flightlib_dir, "build")
    for pattern in ["lib.*", "lib"]:
        for path in glob.glob(os.path.join(build_dir, pattern)):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
    if flightlib_dir not in sys.path:
        sys.path.insert(0, flightlib_dir)
    try:
        from flightgym import QuadrotorEnv_v1
        return QuadrotorEnv_v1
    except ModuleNotFoundError:
        from flightlib import QuadrotorEnv_v1
        return QuadrotorEnv_v1


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
    impl.setMotorInitMode(_MOTOR_INIT_MODES.get(motor_init, 0))

    goal_pos = cfg.get("env", {}).get("goal_position")
    if goal_pos is not None:
        n = impl.getNumOfEnvs()
        goals = np.array([[goal_pos[0], goal_pos[1], goal_pos[2]]] * n, dtype=np.float32)
        impl.setEnvGoalPositions(goals)

    spawn_cfg = cfg.get("env", {}).get("spawn_ranges")
    if spawn_cfg is not None:
        impl.setSpawnRanges(_pack_spawn_ranges(spawn_cfg))

    world_box = cfg.get("env", {}).get("world_box")
    if world_box is not None:
        impl.setWorldBox(np.array(world_box, dtype=np.float32))

    return FlightlibVecEnv(impl)


def _wrap_env(env, cfg, add_obs_noise=True):
    env_cfg = cfg.get("env", {})
    dr_cfg = env_cfg.get("domain_randomization", {})
    if dr_cfg.get("enabled", False):
        env = DomainRandomizationWrapper(env, dr_cfg)

    if add_obs_noise:
        obs_noise_cfg = env_cfg.get("observation_noise")
        if isinstance(obs_noise_cfg, dict) and (
            obs_noise_cfg.get("position", 0) > 0 or obs_noise_cfg.get("velocity", 0) > 0
        ):
            env = ObservationNoiseWrapper(env, obs_noise_cfg)
        else:
            env = ObservationNoiseWrapper(env, {"position": 0.01, "velocity": 0.05})

    if env_cfg.get("custom_reward") and env_cfg["custom_reward"].get("enabled", False):
        env = CustomRewardWrapper(env, env_cfg["custom_reward"])

    if env_cfg.get("max_episode_steps") is not None:
        env = VecMaxEpisodeSteps(env, env_cfg["max_episode_steps"])

    ahl = env_cfg.get("action_history_len", 0)
    if ahl > 0:
        env = ActionHistoryWrapper(env, ahl)

    return env


def _unwrap_to_base(env):
    cur = env
    while cur is not None:
        if hasattr(cur, "set_seed") and hasattr(cur, "_impl"):
            return cur
        cur = getattr(cur, "venv", None)
    return None


def main():
    parser = argparse.ArgumentParser(description="Inspect initial scenarios produced by a fixed eval seed.")
    parser.add_argument("--config", type=str, default="configs/curriculum/phase1_baseline.yaml", help="Config YAML (same as training)")
    parser.add_argument("--seed", type=int, default=7777, help="Eval seed to inspect (default: 7777)")
    parser.add_argument("--n", type=int, default=20, help="Number of resets (scenarios) to log (default: 20)")
    parser.add_argument("--no-obs-noise", action="store_true", help="Disable observation noise to see raw C++ scenario only")
    parser.add_argument("--out", type=str, default=None, help="Optional: save table to this path (e.g. eval_scenarios.txt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_eval = copy.deepcopy(cfg)
    if "env" not in cfg_eval:
        cfg_eval["env"] = {}
    if "vec_env" not in cfg_eval["env"]:
        cfg_eval["env"]["vec_env"] = {}
    cfg_eval["env"]["vec_env"]["num_envs"] = 1
    cfg_eval["env"]["vec_env"]["num_threads"] = 1

    env = _make_env(cfg_eval)
    env = _wrap_env(env, cfg_eval, add_obs_noise=not args.no_obs_noise)

    base = _unwrap_to_base(env)
    if base is None:
        print("Warning: could not find base FlightlibVecEnv for set_seed")

    np.random.seed(args.seed)
    if base is not None:
        base.set_seed(args.seed)

    # Observation layout: pos_error(3), quat(4), lin_vel(3), omega(3) [+ action history if present]
    # Quat (w,x,y,z): tilt_deg = angle from vertical (0=upright, 90=flat). Body z in world has z-component 1-2*(x^2+y^2).
    obs_dim_base = 13
    scenarios = []

    for i in range(args.n):
        obs = env.reset()
        obs = np.asarray(obs).ravel()
        if obs.size < obs_dim_base:
            obs = np.pad(obs, (0, obs_dim_base - obs.size))
        pos_err = obs[0:3]
        quat = obs[3:7]   # (w, x, y, z)
        lin_vel = obs[7:10]
        ang_vel = obs[10:13]

        dist_to_goal = float(np.linalg.norm(pos_err))
        lin_vel_norm = float(np.linalg.norm(lin_vel))
        ang_vel_norm = float(np.linalg.norm(ang_vel))

        # Tilt from vertical (degrees): body z in world = 1 - 2*(qx^2 + qy^2) for unit quat (qw,qx,qy,qz)
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        body_z_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
        body_z_world_z = np.clip(body_z_world_z, -1.0, 1.0)
        tilt_deg = float(np.degrees(np.arccos(body_z_world_z)))

        scenarios.append({
            "ep": i + 1,
            "pos_err": pos_err,
            "dist_to_goal": dist_to_goal,
            "lin_vel": lin_vel,
            "lin_vel_norm": lin_vel_norm,
            "ang_vel": ang_vel,
            "ang_vel_norm": ang_vel_norm,
            "quat": quat,
            "tilt_deg": tilt_deg,
        })

    env.close()

    # Print report
    lines = []
    lines.append("=" * 100)
    lines.append(f"Eval seed: {args.seed}  |  Config: {args.config}  |  N = {args.n} scenarios")
    lines.append("Observation: pos_error = goal - position; tilt_deg = angle from vertical (0=upright, 90=flat)")
    lines.append("=" * 100)
    lines.append(
        f"{'Ep':>4}  {'dist':>8}  {'lin_vel':>8}  {'ang_vel':>8}  {'tilt_deg':>8}  |  "
        "pos_err [x,y,z]        lin_vel [x,y,z]       ang_vel [x,y,z]"
    )
    lines.append("-" * 100)

    for s in scenarios:
        pe = s["pos_err"]
        lv = s["lin_vel"]
        av = s["ang_vel"]
        lines.append(
            f"{s['ep']:>4}  {s['dist_to_goal']:>8.3f}  {s['lin_vel_norm']:>8.3f}  {s['ang_vel_norm']:>8.3f}  {s['tilt_deg']:>8.2f}  |  "
            f"[{pe[0]:>5.2f},{pe[1]:>5.2f},{pe[2]:>5.2f}]  "
            f"[{lv[0]:>5.2f},{lv[1]:>5.2f},{lv[2]:>5.2f}]  "
            f"[{av[0]:>5.2f},{av[1]:>5.2f},{av[2]:>5.2f}]"
        )

    dists = [s["dist_to_goal"] for s in scenarios]
    vels = [s["lin_vel_norm"] for s in scenarios]
    angvels = [s["ang_vel_norm"] for s in scenarios]
    tilts = [s["tilt_deg"] for s in scenarios]
    lines.append("-" * 100)
    lines.append(f"Mean dist_to_goal: {np.mean(dists):.4f}  (min={np.min(dists):.4f}, max={np.max(dists):.4f})")
    lines.append(f"Mean lin_vel_norm: {np.mean(vels):.4f}  (min={np.min(vels):.4f}, max={np.max(vels):.4f})")
    lines.append(f"Mean ang_vel_norm: {np.mean(angvels):.4f}  (min={np.min(angvels):.4f}, max={np.max(angvels):.4f})")
    lines.append(f"Mean tilt_deg:     {np.mean(tilts):.2f}  (min={np.min(tilts):.2f}, max={np.max(tilts):.2f})")
    lines.append("=" * 100)

    report = "\n".join(lines)
    print(report)

    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
