# PPO Drone Control (flightlib + Stable-Baselines3)

Training and validation scripts for PPO-based drone control using **flightlib** (`QuadrotorEnv_v1`) as the environment. All parameters are configured via a single YAML file. This pipeline does **not** use the legacy flightrl stack.

## Requirements

- **flightlib** (flightgym): build and install from `flightmare/flightlib` with `FLIGHTMARE_PATH` set.
- **Python**: `gymnasium`, `numpy`, `pyyaml`, `stable-baselines3`, `matplotlib` (for `val.py`).
- Set `FLIGHTMARE_PATH` to the root of the flightmare repo (e.g. `export FLIGHTMARE_PATH=/path/to/safe_rl_agilicious/flightmare`).

## Install

1. Set the flightmare path and build flightlib:
   ```bash
   export FLIGHTMARE_PATH=/path/to/safe_rl_agilicious/flightmare
   cd $FLIGHTMARE_PATH/flightlib && pip install .
   ```
2. Install Python deps (from repo root):
   ```bash
   pip install gymnasium numpy pyyaml stable-baselines3 matplotlib
   ```

## Config

Edit `configs/drone_ppo_default.yaml` to set:

- **env.vec_env**: `num_envs`, `num_threads`, `seed`, `scene_id`, `render`
- **env.quadrotor_env** (optional): overrides for sim_dt, max_t, dynamics, rl reward coefficients
- **ppo**: SB3 PPO hyperparameters (learning_rate, n_steps, batch_size, etc.)
- **training**: total_timesteps, save_interval, eval_freq, normalize_obs
- **evaluation**: n_episodes, deterministic (for val.py)
- **paths**: log_dir, save_dir, checkpoint_path, plot_dir

## Train

From the **repository root**:

```bash
python scripts/train.py --config configs/drone_ppo_default.yaml [--seed 0]
```

Checkpoints and the final model are saved under `paths.save_dir` (default: `saved/`). If `training.normalize_obs` is true, `vecnormalize.pkl` is saved there too and must be used when evaluating.

## Validate (with state plots)

From the **repository root**:

```bash
python scripts/val.py --config configs/drone_ppo_default.yaml --checkpoint saved/ppo_drone_final.zip [--episodes 5] [--save_plots] [--plot_dir val_plots]
```

- Runs N episodes with the loaded policy and plots **position**, **orientation**, **linear/angular velocity**, **reward**, **actions**, and **3D trajectory** per episode (matplotlib).
- Use `--save_plots` to write figures to `--plot_dir` (or `paths.plot_dir` from config).

## File layout

- `context.py` — `flightmare_context(run_dir)` context manager for `FLIGHTMARE_PATH`
- `config_loader.py` — load YAML, write vec_env/quadrotor_env under run_dir, build config strings
- `env_wrapper.py` — Gymnasium VecEnv wrapping `QuadrotorEnv_v1` (action clipping, terminal_observation, episode in infos)
- `train.py` — PPO training only (optional VecNormalize; saves policy + vecnormalize)
- `val.py` — load checkpoint, run episodes, pyplot visualization of all states
