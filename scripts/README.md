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
- Use `--use_cbf` to run evaluation through the CBF safety filter (position barriers). **Do not use CBF during training.**

## CBF safety filter (deployment only)

A separate CBF (Control Barrier Function) filter and wrapper ensure position barriers are satisfied when deploying a trained policy. Formulation follows the RL-CBF paper (Cheng et al.): minimum-intervention QP that keeps the system in the safe set.

- **configs/cbf_config.yaml** — position barriers (half-spaces `n'p + q >= 0`), `alpha` (continuous-time), or **discrete CBF** (`discrete_cbf: true`) with `h_{k+1} >= (1-gamma)*h_k`, `gamma_min`/`gamma_max`/`sigma_param`; QP solver: `acados` (HPIPM), `osqp`, or `scipy`. For acados, install `casadi` and `acados_template`; generated code is cached under `build/cbf_acados/`.
- **scripts/cbf_filter.py** — `CBFFilter`: given state `x` and RL motor thrusts `u_RL`, returns `u_safe` via QP. Uses `scripts/quadrotor_model.py` for Lie derivatives.
- **scripts/cbf_wrapper.py** — `CBFWrapper(venv, ...)`: VecEnv wrapper that filters each action through the CBF before stepping. Use with `goal_position` and optional `act_mean`/`act_std` for action denormalization.

**Validation with CBF:**  
`python scripts/val.py --config configs/drone_ppo_default.yaml --checkpoint saved/ppo_drone_final.zip --use_cbf`

**CBF-specific evaluation (raw RL vs filtered trajectories):**  
`scripts/eval_cbf.py` runs the policy and CBF filter without the wrapper, records both raw RL and CBF-filtered trajectories (the latter from the env; the former by integrating the quad model with raw thrusts). It plots trajectory views (x–y, y–z, z–x), states, raw vs filtered motor thrusts, and draws **world limits** (from `env.world_box`) and **CBF barrier limits** (from `configs/cbf_config.yaml`) on the trajectory plots.

```bash
python scripts/eval_cbf.py --config configs/drone_ppo_default.yaml --checkpoint saved/ppo_drone_final.zip [--episodes 5] [--save_plots] [--plot_dir ...]
```

**QP solver:** Default is **OSQP** (`pip install osqp`). Set `cbf.solver: acados` in `configs/cbf_config.yaml` to try acados (falls back to OSQP if acados is not available).

## Component tests

Validate the quadrotor model, CBF filter, CBF wrapper, and pipeline (Flightmare / val) compatibility:

```bash
python scripts/test_components.py
```

Or with pytest: `pytest scripts/test_components.py -v`

**Pipeline tests** (section "Pipeline (Flightmare / val)"): build the same env as val/train and run a few steps, with and without the CBF wrapper. They are **skipped** if `flightlib`/`flightgym` is not installed; run with flightlib available to confirm full compatibility.

## File layout

- `context.py` — `flightmare_context(run_dir)` context manager for `FLIGHTMARE_PATH`
- `config_loader.py` — load YAML, write vec_env/quadrotor_env under run_dir, build config strings
- `env_wrapper.py` — Gymnasium VecEnv wrapping `QuadrotorEnv_v1` (action clipping, terminal_observation, episode in infos)
- `train.py` — PPO training only (optional VecNormalize; saves policy + vecnormalize)
- `val.py` — load checkpoint, run episodes, pyplot visualization of all states
- `eval_cbf.py` — CBF eval: raw RL vs CBF-filtered trajectories, world/CBF limits on plots
- `quadrotor_model.py` — standalone quadrotor dynamics (state 13, motor thrusts 4); used by CBF
- `cbf_filter.py` — CBF safety filter (position barriers, QP solved with OSQP or acados)
- `cbf_wrapper.py` — VecEnv wrapper that filters RL actions through the CBF
- `test_components.py` — tests for quadrotor model, CBF filter, CBF wrapper
