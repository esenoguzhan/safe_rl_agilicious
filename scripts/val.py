#!/usr/bin/env python3
"""
Validation/evaluation for PPO drone control with matplotlib visualization of all states.
Eval results (plots and eval_results.txt) are saved in the checkpoint's folder by default
(e.g. models/PPO_steps/ when checkpoint is models/PPO_steps/best_model.zip).
Usage: python val.py --config configs/drone_ppo_default.yaml --checkpoint models/PPO_steps/best_model.zip [--episodes 5] [--save_plots] [--plot_dir ...]
"""
import argparse
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
from scripts.env_wrapper import FlightlibVecEnv


def _ensure_flightgym_path():
    """If flightgym not found, add flightlib source and build dirs to sys.path (editable install)."""
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
    """Import QuadrotorEnv_v1 from flightgym or flightlib (build-dependent)."""
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
        pass
    try:
        from flightlib import QuadrotorEnv_v1
        return QuadrotorEnv_v1
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Could not import 'flightgym' or 'flightlib'. Install the flightlib Python binding:\n"
            "  export FLIGHTMARE_PATH=/path/to/safe_rl_agilicious/flightmare\n"
            "  cd $FLIGHTMARE_PATH/flightlib && pip install .\n"
            "Use the same Python for pip and for running (e.g. python -m pip install .).\n"
            "See scripts/README.md for full setup."
        ) from None


def _make_env(cfg):
    """Create FlightlibVecEnv, optionally inside flightmare_context."""
    QuadrotorEnv_v1 = _get_QuadrotorEnv_v1()

    run_dir = prepare_env_run_dir(cfg)
    if run_dir:
        with flightmare_context(run_dir):
            impl = QuadrotorEnv_v1()
    else:
        vec_config_str = get_vec_env_config_string(cfg)
        impl = QuadrotorEnv_v1(vec_config_str, False)

    return FlightlibVecEnv(impl)


def _plot_episode(obs_list, act_list, reward_list, t_axis, save_path=None, episode_idx=0):
    """Plot position, orientation, velocities, rewards, actions vs time for one episode."""
    import matplotlib.pyplot as plt

    obs = np.array(obs_list)
    act = np.array(act_list)
    rewards = np.array(reward_list)

    n_rows = 4
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    fig.suptitle(f"Episode {episode_idx} — state and action traces")

    # Position (0:3)
    ax = axes[0, 0]
    ax.plot(t_axis, obs[:, 0], label="x")
    ax.plot(t_axis, obs[:, 1], label="y")
    ax.plot(t_axis, obs[:, 2], label="z")
    ax.set_ylabel("Position (m)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Orientation euler (3:6)
    ax = axes[0, 1]
    ax.plot(t_axis, obs[:, 3], label="roll")
    ax.plot(t_axis, obs[:, 4], label="pitch")
    ax.plot(t_axis, obs[:, 5], label="yaw")
    ax.set_ylabel("Orientation (rad)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Linear velocity (6:9)
    ax = axes[1, 0]
    ax.plot(t_axis, obs[:, 6], label="vx")
    ax.plot(t_axis, obs[:, 7], label="vy")
    ax.plot(t_axis, obs[:, 8], label="vz")
    ax.set_ylabel("Linear vel (m/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Angular velocity (9:12)
    ax = axes[1, 1]
    ax.plot(t_axis, obs[:, 9], label="wx")
    ax.plot(t_axis, obs[:, 10], label="wy")
    ax.plot(t_axis, obs[:, 11], label="wz")
    ax.set_ylabel("Angular vel (rad/s)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Reward and cumulative return
    ax = axes[2, 0]
    ax.plot(t_axis, rewards, label="reward", color="C0")
    ax.set_ylabel("Reward")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    cum = np.cumsum(rewards)
    ax2.plot(t_axis, cum, label="cumulative", color="C1", alpha=0.8)
    ax2.set_ylabel("Cumulative return", color="C1")

    # Actions (4-dim)
    ax = axes[2, 1]
    for i in range(act.shape[1]):
        ax.plot(t_axis, act[:, i], label=f"a{i}")
    ax.set_ylabel("Actions")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3D trajectory
    axes[3, 0].remove()
    ax3d = fig.add_subplot(n_rows, n_cols, 7, projection="3d")
    ax3d.plot(obs[:, 0], obs[:, 1], obs[:, 2], color="C0")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.set_title("3D trajectory")
    axes[3, 1].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Validate PPO drone policy with state plots")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model zip or folder")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None, help="Override evaluation.n_episodes")
    parser.add_argument("--save_plots", action="store_true", help="Save figures to disk")
    parser.add_argument("--plot_dir", type=str, default=None, help="Directory to save plots")
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Use single env for validation so we get one trajectory per episode
    import copy
    cfg_val = copy.deepcopy(cfg)
    if "env" not in cfg_val:
        cfg_val["env"] = {}
    if "vec_env" not in cfg_val["env"]:
        cfg_val["env"]["vec_env"] = {}
    cfg_val["env"]["vec_env"]["num_envs"] = 1
    cfg_val["env"]["vec_env"]["num_threads"] = 1

    eval_cfg = cfg_val.get("evaluation", {})
    n_episodes = args.episodes if args.episodes is not None else eval_cfg.get("n_episodes", 5)
    deterministic = eval_cfg.get("deterministic", True)
    # Flightlib only sets done on ground contact; cap steps so episodes always end (e.g. max_t=5s @ 0.02 => 250)
    max_episode_steps = eval_cfg.get("max_episode_steps", 250)
    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 0)

    if seed is not None:
        np.random.seed(seed)

    env = _make_env(cfg_val)
    custom_reward_cfg = cfg.get("env", {}).get("custom_reward")
    if custom_reward_cfg and custom_reward_cfg.get("enabled", False):
        env = CustomRewardWrapper(env, custom_reward_cfg)
    training_cfg = cfg.get("training", {})
    normalize_obs = training_cfg.get("normalize_obs", True)

    if normalize_obs:
        vecnorm_path = os.path.join(os.path.dirname(args.checkpoint), "vecnormalize.pkl")
        if os.path.isfile(vecnorm_path):
            from stable_baselines3.common.vec_env import VecNormalize
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False

    from stable_baselines3 import PPO

    model = PPO.load(args.checkpoint, env=env)

    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    plot_dir = args.plot_dir if args.plot_dir is not None else run_dir
    save_plots = args.save_plots or (args.plot_dir is None)
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    returns = []
    lengths = []
    for ep in range(n_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        obs_list, act_list, reward_list = [], [], []
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            if env.num_envs == 1:
                action = action.reshape(1, -1)
            obs_list.append(obs[0].copy())
            n_obs, rewards, dones, infos = env.step(action)
            act_list.append(action[0].copy())
            reward_list.append(float(rewards[0]))
            obs = n_obs
            done = dones[0]
            steps += 1
            if infos and "episode" in infos[0]:
                break
            if steps >= max_episode_steps:
                break

        ep_return = sum(reward_list)
        ep_len = len(reward_list)
        returns.append(ep_return)
        lengths.append(ep_len)

        # Unnormalize obs to physical units for plotting (if using VecNormalize)
        if hasattr(env, "obs_rms") and env.obs_rms is not None:
            eps = getattr(env, "epsilon", 1e-8)
            std = np.sqrt(env.obs_rms.var + eps)
            obs_list = [np.asarray(obs) * std + env.obs_rms.mean for obs in obs_list]

        t_axis = np.arange(len(obs_list)) * 0.02

        save_path = os.path.join(plot_dir, f"episode_{ep}.png") if save_plots else None
        _plot_episode(obs_list, act_list, reward_list, t_axis, save_path=save_path, episode_idx=ep)

    env.close()

    mean_return = float(np.mean(returns))
    mean_length = float(np.mean(lengths))
    print(f"Episodes: {n_episodes}, mean return: {mean_return:.2f}, mean length: {mean_length:.1f}")

    eval_results_path = os.path.join(run_dir, "eval_results.txt")
    with open(eval_results_path, "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"n_episodes: {n_episodes}\n")
        f.write(f"mean_return: {mean_return:.4f}\n")
        f.write(f"mean_length: {mean_length:.1f}\n")
        f.write("returns: " + ", ".join(f"{r:.4f}" for r in returns) + "\n")
        f.write("lengths: " + ", ".join(str(L) for L in lengths) + "\n")
    print("Eval results saved to", eval_results_path)


if __name__ == "__main__":
    main()
