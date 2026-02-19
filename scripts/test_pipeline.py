#!/usr/bin/env python3
"""
Debug script to verify the training pipeline: env, wrappers, obs, reward, done, info.
Runs the same env stack as train (without VecNormalize by default) and prints
observations, rewards, dones, and infos step-by-step.
Usage:
  python scripts/test_pipeline.py --config configs/drone_ppo_default.yaml [--steps 50] [--seed 0]
  python scripts/test_pipeline.py --config configs/drone_ppo_default.yaml --episodes 2
  python scripts/test_pipeline.py --config configs/drone_ppo_default.yaml --action zero --steps 20
  python scripts/test_pipeline.py --config configs/drone_ppo_default.yaml --diagnose --steps 5
"""
import argparse
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.config_loader import load_config
from scripts.custom_reward_wrapper import CustomRewardWrapper
from scripts.env_wrapper import FlightlibVecEnv, VecMaxEpisodeSteps
from scripts.record_episode_statistics import VecRecordEpisodeStatistics

# Reuse train's env builder
from scripts import train as train_module

OBS_LABELS = (
    ["pos_x", "pos_y", "pos_z", "roll", "pitch", "yaw"]
    + ["vx", "vy", "vz", "wx", "wy", "wz"]
)


def _fmt_obs(obs, env_index=0):
    """Format obs for one env as labeled dict (assumes 12D: pos, ori, lin_vel, ang_vel)."""
    if obs.ndim == 1:
        arr = obs
    else:
        arr = obs[env_index]
    n = min(len(OBS_LABELS), len(arr))
    return dict(zip(OBS_LABELS[:n], np.round(arr[:n], 4).tolist()))


def _run_diagnose(cfg, args):
    """Low-level diagnostic: bypass wrappers and test C++ impl directly."""
    QuadrotorEnv_v1 = train_module._get_QuadrotorEnv_v1()
    from scripts.config_loader import get_vec_env_config_string
    vec_config_str = get_vec_env_config_string(cfg)
    impl = QuadrotorEnv_v1(vec_config_str, False)

    num_envs = impl.getNumOfEnvs()
    obs_dim = impl.getObsDim()
    act_dim = impl.getActDim()
    extra_info_names = impl.getExtraInfoNames()
    has_get_obs = hasattr(impl, "getObs")

    print("=" * 60)
    print("DIAGNOSE MODE — raw C++ impl (no Python wrappers)")
    print("=" * 60)
    print(f"num_envs={num_envs}, obs_dim={obs_dim}, act_dim={act_dim}")
    print(f"extra_info_names={extra_info_names} (len={len(extra_info_names)})")
    print(f"impl.getObs exposed: {has_get_obs}")
    print()

    obs = np.zeros((num_envs, obs_dim), dtype=np.float32)
    reward = np.zeros(num_envs, dtype=np.float32)
    done = np.zeros(num_envs, dtype=bool)
    extra_info = np.zeros(
        (num_envs, len(extra_info_names)), dtype=np.float32
    )

    print(f"Buffer layouts:")
    print(f"  obs:        shape={obs.shape}, dtype={obs.dtype}, strides={obs.strides}, C_CONTIGUOUS={obs.flags['C_CONTIGUOUS']}")
    print(f"  reward:     shape={reward.shape}, dtype={reward.dtype}, strides={reward.strides}")
    print(f"  done:       shape={done.shape}, dtype={done.dtype}, strides={done.strides}")
    print(f"  extra_info: shape={extra_info.shape}, dtype={extra_info.dtype}, strides={extra_info.strides}")
    print()

    # --- Test reset ---
    print("--- Testing reset() ---")
    obs_pre_reset = obs.copy()
    ok_reset = impl.reset(obs)
    obs_changed_reset = not np.array_equal(obs, obs_pre_reset)
    print(f"  reset() returned: {ok_reset}")
    print(f"  obs changed after reset: {obs_changed_reset}")
    print(f"  obs[0] = {obs[0].round(4).tolist()}")
    print()

    # --- Test step ---
    steps = args.steps if args.steps else 5
    print(f"--- Testing step() for {steps} steps ---")
    prev_obs = obs.copy()
    for i in range(steps):
        if args.action == "zero":
            act = np.zeros((num_envs, act_dim), dtype=np.float32)
        else:
            act = np.random.uniform(-1.0, 1.0, (num_envs, act_dim)).astype(np.float32)

        obs_before = obs.copy()
        rew_before = reward.copy()
        done_before = done.copy()

        ok = impl.step(act, obs, reward, done, extra_info)

        obs_changed = not np.array_equal(obs, obs_before)
        rew_changed = not np.array_equal(reward, rew_before)
        done_changed = not np.array_equal(done, done_before)

        print(f"  step {i}: ok={ok} | obs_changed={obs_changed} rew_changed={rew_changed} done_changed={done_changed}")
        print(f"    act[0]={act[0].round(3).tolist()}")
        print(f"    obs[0]={obs[0].round(4).tolist()}")
        print(f"    rew[0]={float(reward[0]):.6f}  done[0]={bool(done[0])}")

        if obs_changed:
            delta = np.abs(obs[0] - prev_obs[0])
            print(f"    obs delta[0]={delta.round(6).tolist()}")
        prev_obs = obs.copy()

    # --- Test getObs (if available) ---
    if has_get_obs:
        print()
        print("--- Testing getObs() separately ---")
        obs2 = np.zeros_like(obs)
        impl.getObs(obs2)
        matches_step_obs = np.allclose(obs, obs2)
        print(f"  getObs into fresh buffer:")
        print(f"    obs2[0]={obs2[0].round(4).tolist()}")
        print(f"    matches last step obs: {matches_step_obs}")
        if not matches_step_obs:
            print(f"    DIFF: {(obs2[0] - obs[0]).round(6).tolist()}")
            print(f"    => getObs retrieves different (likely correct) state from C++")
    else:
        print()
        print("--- getObs NOT exposed in binding (rebuild flightgym with updated pybind_wrapper.cpp) ---")

    # --- Test testStep (already bound, has no dim check) ---
    print()
    print("--- Testing testStep() (no dim check, calls getObs internally) ---")
    impl.reset(obs)
    obs_pre = obs.copy()
    if args.action == "zero":
        act = np.zeros((num_envs, act_dim), dtype=np.float32)
    else:
        act = np.random.uniform(-1.0, 1.0, (num_envs, act_dim)).astype(np.float32)
    impl.testStep(act, obs, reward, done, extra_info)
    obs_changed_ts = not np.array_equal(obs, obs_pre)
    print(f"  obs changed after testStep: {obs_changed_ts}")
    print(f"  obs[0]={obs[0].round(4).tolist()}")
    print(f"  rew[0]={float(reward[0]):.6f}  done[0]={bool(done[0])}")
    if obs_changed_ts and not obs_changed:
        print("  => testStep updates obs but step() does not — confirms Ref propagation issue")

    print()
    print("DIAGNOSE COMPLETE")


def main():
    parser = argparse.ArgumentParser(description="Debug pipeline: run env and print obs, reward, done, info")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--steps", type=int, default=None, help="Max steps to run (default: 50)")
    parser.add_argument("--episodes", type=int, default=None, help="Run until this many episode ends (overrides --steps if set)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action", type=str, default="random", choices=("random", "zero"), help="Action: random in [-1,1] or zero")
    parser.add_argument("--normalize", action="store_true", help="Wrap with VecNormalize (like train)")
    parser.add_argument("--verbose", action="store_true", help="Print full obs/action arrays every step")
    parser.add_argument("--diagnose", action="store_true", help="Run low-level C++ binding diagnostics (bypasses wrappers)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(args.seed)

    if args.diagnose:
        _run_diagnose(cfg, args)
        return

    # Build env same as train (no run_dir override; use config as-is for debug)
    env = train_module._make_env(cfg)
    env_cfg = cfg.get("env", {})
    custom_reward_cfg = env_cfg.get("custom_reward")
    if custom_reward_cfg and custom_reward_cfg.get("enabled", False):
        env = CustomRewardWrapper(env, custom_reward_cfg)
        print("[Pipeline] CustomRewardWrapper enabled")
    max_episode_steps = env_cfg.get("max_episode_steps")
    if max_episode_steps is not None:
        env = VecMaxEpisodeSteps(env, max_episode_steps)
        print(f"[Pipeline] VecMaxEpisodeSteps(max_steps={max_episode_steps})")
    env = VecRecordEpisodeStatistics(env, deque_size=10)
    if args.normalize:
        from stable_baselines3.common.vec_env import VecNormalize
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        print("[Pipeline] VecNormalize enabled")

    num_envs = env.num_envs
    obs_space = env.observation_space
    act_space = env.action_space
    print(f"[Pipeline] num_envs={num_envs}, obs_shape={obs_space.shape}, act_shape={act_space.shape}")
    print()

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    run_until_episodes = args.episodes is not None
    max_steps = args.steps if args.steps is not None else (10000 if run_until_episodes else 50)
    episode_ends = 0
    step = 0

    while step < max_steps:
        if args.action == "zero":
            action = np.zeros((num_envs, act_space.shape[0]), dtype=np.float32)
        else:
            action = np.random.uniform(-1.0, 1.0, (num_envs, act_space.shape[0])).astype(np.float32)

        obs, rewards, dones, infos = env.step(action)

        if args.verbose:
            print(f"--- step {step} ---")
            print(f"  action[0] = {action[0].round(4).tolist()}")
            print(f"  obs[0]    = {obs[0].round(4).tolist()}")
            print(f"  reward[0] = {float(rewards[0]):.6f}, done[0] = {bool(dones[0])}")
            print(f"  info[0] keys = {list(infos[0].keys())}")
        else:
            print(f"step {step:4d} | action[0] {action[0].round(3).tolist()} | r[0]={float(rewards[0]):.4f} done[0]={dones[0]}")
            print(f"         obs[0] pos(x,y,z)=({obs[0][0]:.3f},{obs[0][1]:.3f},{obs[0][2]:.3f}) ori(r,p,y)=({obs[0][3]:.3f},{obs[0][4]:.3f},{obs[0][5]:.3f}) vel(vx,vy,vz)=({obs[0][6]:.3f},{obs[0][7]:.3f},{obs[0][8]:.3f})")

        for i in range(num_envs):
            if dones[i]:
                episode_ends += 1
                ep = infos[i].get("episode", {})
                print(f"         >>> env[{i}] DONE episode r={ep.get('r', '?')} l={ep.get('l', '?')} t={ep.get('t', '?')}")

        if run_until_episodes and episode_ends >= args.episodes:
            print(f"\nReached {episode_ends} episode end(s). Stopping.")
            break

        step += 1
        if not run_until_episodes and step >= max_steps:
            break

    # Summary
    print()
    print("============ Summary ============")
    print(f"Total steps: {step}")
    print(f"Episode ends (dones): {episode_ends}")
    if hasattr(env, "return_queue") and len(env.return_queue) > 0:
        print(f"Return queue (last {len(env.return_queue)}): {[round(r, 4) for r in list(env.return_queue)]}")
    if hasattr(env, "length_queue") and len(env.length_queue) > 0:
        print(f"Length queue (last {len(env.length_queue)}): {list(env.length_queue)}")
    print("Final obs[0] (labeled):", _fmt_obs(obs, 0))
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
