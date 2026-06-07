# Safe RL for Quadrotor Recovery and Stabilization (RL + CBF on Flightmare)

A practical framework for training **goal-conditioned quadrotor recovery policies** with PPO and
running them safely behind a **Control Barrier Function (CBF) safety filter**. It trains a policy that
flies a quadrotor back to a goal from *arbitrary* initial conditions — large velocities, high body rates,
even fully inverted — inside an extended [Flightmare](https://github.com/uzh-rpg/flightmare) simulator
with [Agilicious](https://agilicious.dev) dynamics, and ships tooling to evaluate it against nonlinear
MPC baselines and deploy it on real hardware over ROS.

Everything is **config-driven** (YAML): you define training settings, a curriculum, safety barriers, and
evaluation scenarios in files under `configs/`, and the scripts under `scripts/` do the rest. The CBF is
**deployment-only** — it never touches the training loop, so you can train once and add/remove safety
filtering freely at run time.

> This is the reference implementation for the paper *"Safe Reinforcement Learning with Control Barrier
> Functions for Quadrotor Recovery"* (Eşen, Yuan, Ryll — TU Munich). See [Citation](#citation).

---

## Contents

- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Training](#training)
- [Curriculum (YAML-driven)](#curriculum-yaml-driven)
- [Evaluation & comparison](#evaluation--comparison)
- [CBF safety filter](#cbf-safety-filter)
- [Configuration reference](#configuration-reference)
- [What we changed in Flightmare (and why)](#what-we-changed-in-flightmare-and-why)
- [Extending the framework](#extending-the-framework)
- [Real-drone deployment (ROS)](#real-drone-deployment-ros)
- [Citation](#citation)

## How it works

```
            ┌───────────────────────── deployment ─────────────────────────┐
 obs ──►  PPO policy  ──► u_RL ──►  CBF-QP filter  ──► u_safe ──►  plant
 (33-D)   (SB3, MLP)              (acados/OSQP, QP)            (Flightmare sim or real drone)
```

- **Training:** PPO (Stable-Baselines3) on `QuadrotorEnv_v1` from `flightlib`, with action-history
  observations, a reward that stays informative across the whole flight envelope, domain randomization
  (mass, motor time constant, command delay), and calibrated disturbances (wind, drag, noise).
- **Safety (optional, deployment-only):** a minimum-intervention CBF-QP wraps the environment and nudges
  the RL action each step to keep the drone inside a safe set (default: a vertical box `1 m ≤ z ≤ 8 m`).
- **Baselines:** nonlinear MPC (NMPC) and a soft-constrained variant (C-NMPC) share the same dynamics
  model and solver as the CBF, for fair comparison.

## Repository layout

```
safe_rl_agilicious/
├── scripts/                     # all Python entrypoints + libraries
│   ├── train.py                             # main trainer (high-rate physics, low-rate control)
│   ├── train_core.py                        # core PPO trainer internals (env factory, schedules, callbacks)
│   ├── run_curriculum.py                    # YAML-driven sequential curriculum runner
│   ├── compare.py                           # main evaluation/comparison entrypoint
│   ├── compare_core.py / compare_fair.py / compare_sim_plant.py  # comparison libraries (used by compare.py)
│   ├── cbf_filter.py / cbf_wrapper.py       # CBF-QP safety filter + VecEnv wrapper
│   ├── mpc_controller.py / mpc_barrier_bounds.py  # NMPC / C-NMPC baselines (acados)
│   ├── quadrotor_model.py                    # symbolic 13-state dynamics (shared by CBF + MPC)
│   ├── env_wrapper.py                         # SB3 VecEnv + history/noise/DR/decimation wrappers
│   ├── custom_reward_wrapper.py              # reward shaping
│   ├── config_loader.py / context.py         # YAML merge + FLIGHTMARE_PATH handling
│   └── requirements.txt
├── configs/                     # YAML config (training, curriculum, CBF, MPC, scenarios)
├── models/                      # trained checkpoints
├── flightmare/                  # vendored, extended Flightmare simulator (build flightlib here)
└── standalone_ros_rl_feedthrough/   # self-contained ROS package for on-drone deployment
```

## Installation

Linux, Python 3.8+, and a C++ toolchain (to build `flightlib`).

```bash
git clone https://github.com/esenoguzhan/safe_rl_agilicious.git
cd safe_rl_agilicious

# Create & activate a conda environment
conda create -n safe_rl python=3.10 -y
conda activate safe_rl

# Build & install flightlib (Flightmare's Python bindings)
export FLIGHTMARE_PATH=$PWD/flightmare
cd "$FLIGHTMARE_PATH/flightlib" && pip install .
cd -

# Python dependencies
pip install -r scripts/requirements.txt
```

- In every shell that runs training/eval, `conda activate safe_rl` and export `FLIGHTMARE_PATH`
  (or add the export to your shell profile).
- The CBF and MPC use **acados** (HPIPM) by default. Install [`acados`](https://docs.acados.org) with
  `acados_template` and `casadi` to use it; otherwise the CBF falls back to OSQP. Generated code is
  cached under `build/`.

## Quick start

```bash
# 1) Train one stage on the base config
python scripts/train.py \
    --config configs/ppo_config_base.yaml --physics-hz 500 --rl-hz 50

# 2) Or run the whole curriculum (each stage resumes from the previous)
python scripts/run_curriculum.py --dry-run --physics-hz 500 --rl-hz 50   # preview first
python scripts/run_curriculum.py --physics-hz 500 --rl-hz 50             # then train

# 3) Evaluate RL / RL+CBF / NMPC / C-NMPC on the scenario suite
python scripts/compare.py \
    --scenarios_config configs/scenarious.yaml \
    --checkpoint models/PPO_50000000_seq6dec_s6_tau_dr_ph500_rl50/best_model.zip \
    --cbf_config configs/cbf_config.yaml --mpc_config configs/mpc_config.yaml \
    --physics-hz 500 --rl-hz 50 --save_plots
```

## Training

`scripts/train.py` is the main trainer. It integrates physics at `--physics-hz`
while stepping the policy at `--rl-hz`, so simulation fidelity is decoupled from the control rate. It:

- merges your `--config` over the flightlib defaults and writes the per-run env config,
- applies action-history stacking, observation noise, domain randomization, and (if enabled) disturbances,
- trains PPO and writes checkpoints + `vecnormalize.pkl` + `config.yaml` to `models/PPO_<steps>_<run-name>/`,
- supports `--resume <run_dir>` to continue from a previous run's weights and normalizer.

Common flags: `--config`, `--physics-hz`, `--rl-hz`, `--reward-aggregate {sum,mean}`, `--run-name`,
`--seed`, `--resume`. (Use `scripts/train_core.py` for plain single-rate training.)

## Curriculum (YAML-driven)

`scripts/run_curriculum.py` runs the trainer in **N sequential stages**, each resuming from the previous
one. Stages live entirely in a YAML file — **no code changes needed to add or reorder them**. Each stage's
`overrides` are deep-merged **cumulatively** on top of the base config (stage *k* = base + overrides of
stages 1..k). Adding a stage is as simple as appending a block with only the parameters you want to change:

```yaml
# configs/curriculum.yaml
run_name_prefix: seq7dec
base_config: configs/ppo_config_base.yaml
stages:
  - name: base
    description: "Base config as-is (no parameter change)."
  - name: extended_position          # only provide the new bounds; everything else is inherited
    description: "Widen position spawn ranges."
    overrides:
      env:
        spawn_ranges:
          pos_x: [-8.0, 8.0]
          pos_y: [-8.0, 8.0]
          pos_z: [0.5, 8.0]
  # ... add as many stages as you like ...
```

```bash
python scripts/run_curriculum.py --dry-run                    # print stages + commands only
python scripts/run_curriculum.py --physics-hz 500 --rl-hz 50  # change decimation rates
python scripts/run_curriculum.py --start-from 3 \             # resume mid-curriculum after a failure
    --resume-from-run-dir models/PPO_..._s2_extended_position_ph500_rl50
```

Base-config precedence: `--base-config` (CLI) > curriculum `base_config` key > `configs/ppo_config_base.yaml`.

## Evaluation & comparison

`scripts/compare.py` runs RL, RL+CBF, NMPC, and C-NMPC across a
scenario suite and produces CSV/XLSX summaries plus per-scenario plots (`--save_plots`). It supports:

- **per-run sim-plant overrides** — apply mass / motor-τ mismatch to the *simulator only* while
  controllers keep nominal parameters (great for robustness studies),
- **decimated timing** (`--physics-hz`, `--rl-hz`), and matched seeds across controllers.

Key flags: `--scenarios_config`, `--checkpoint`, `--cbf_config`, `--mpc_config`,
`--controllers RL RL+CBF MPC MPC+Con`, `--skip_controllers`, `--save_plots`, `--plot_dir`.

## CBF safety filter

The CBF (`scripts/cbf_filter.py`, configured by `configs/cbf_config.yaml`) is a minimum-intervention QP
that keeps the system in a safe set, applied only at deployment:

- **Velocity-aware barriers** `h(p, v) = nᵀp + q + k_v (nᵀv) ≥ 0` (default: ground + ceiling). The
  velocity term makes the filter intervene *before* the geometric boundary.
- **Continuous-time condition** `L_f h + L_g h·u ≥ -α h`, with optional slack for feasibility at actuator
  limits; falls back to the clamped RL action on rare QP failure.
- **Solver:** acados (HPIPM) or OSQP/SciPy.

Add lateral walls or change `α`, `k_v`, `r_uav`, or the solver by editing `configs/cbf_config.yaml`
(commented `x_min`/`x_max`/`y_min`/`y_max` examples are included). To apply the filter programmatically,
wrap a VecEnv with `scripts/cbf_wrapper.py`.

## Configuration reference

| File | Purpose |
| --- | --- |
| `configs/ppo_config_base.yaml` | Base training config: vec_env, dynamics, spawn ranges, DR ranges, disturbances, PPO/training settings. |
| `configs/ppo_config_base_latency.yaml` | Same, plus a nonzero command delay. |
| `configs/curriculum.yaml` | Curriculum stages for `run_curriculum.py`. |
| `configs/cbf_config.yaml` | CBF barriers, `alpha`, `k_v`, `r_uav`, slack, QP solver. |
| `configs/mpc_config.yaml` | NMPC / C-NMPC horizon, weights, solver options. |
| `configs/quadrotor_model.yaml` | Dynamics parameters for the CBF/MPC internal models. |
| `configs/scenarious.yaml` | Evaluation scenarios + per-run sim-plant overrides. |

`env.*` keys in a training config merge into the flightlib `quadrotor_env.yaml` (`quadrotor_env`,
`quadrotor_dynamics`, `rl`, `disturbances`, plus `spawn_ranges`, `domain_randomization`, `vec_env`) —
see `scripts/config_loader.py`.

## What we changed in Flightmare (and why)

[Flightmare](https://github.com/uzh-rpg/flightmare) is an excellent quadrotor simulator — fast,
photorealistic, and built on solid rigid-body dynamics. But the upstream release predates the modern
RL tooling we wanted to use: it ships an old Gym-style interface that does **not** plug into the
current **Stable-Baselines3 / Gymnasium** stack, and it exposes **no hooks for domain randomization or
disturbances**, which are exactly what you need for **sim-to-real transfer** and **generalization**.
Rather than fork to a different simulator, we extended the vendored Flightmare in `flightmare/` so it
keeps everything that made it great while supporting our training pipeline. The main changes:

- **Up-to-date SB3 / Gymnasium interface.** A thin Gymnasium `VecEnv` adapter
  (`scripts/env_wrapper.py: FlightlibVecEnv`) wraps the C++ `QuadrotorEnv_v1` with modern SB3 semantics:
  `gymnasium.spaces`, `step_async`/`step_wait`, automatic action clipping to `[-1, 1]`,
  `terminal_observation` on episode end, and `episode {"r", "l"}` statistics. This lets us train with
  the current SB3 PPO out of the box instead of being pinned to a legacy Gym version.
- **Domain randomization hooks (new C++ API).** The env now exposes per-env setters
  (`setEnvMasses`, `setEnvMotorTauInvs`, `setEnvGoalPositions`, `reinitHoverMotor`) so we can randomize
  **mass, motor time constant, and goal position** at every episode boundary
  (`DomainRandomizationWrapper`). Upstream Flightmare has fixed dynamics per run; this is what makes the
  policy robust to model mismatch and transferable to real hardware.
- **Calibrated disturbance models in the dynamics.** The C++ environment gained an optional disturbance
  block (`flightmare/flightlib/configs/quadrotor_env.yaml → disturbances`): Ornstein–Uhlenbeck wind
  gusts, world-frame mean wind, body-frame quadratic drag, and additive force/torque noise, plus a
  reseedable RNG (`seedDisturbance`) for reproducible evaluation. These close part of the sim-to-real
  gap that the stock simulator cannot model.
- **Full-envelope state injection.** `setQuadState` lets us place the drone at any pose/velocity/rate
  (including inverted, high-speed, high-rate starts) for recovery training and matched-seed evaluation.
- **Decimated physics.** High-rate integration with a lower-rate control/observation step
  (`VecPhysicsDecimationWrapper`) decouples simulation fidelity from the policy rate.

## Extending the framework

The Flightmare simulator in `flightmare/` has been extended so it can be reused for your own experiments:

- **Stable-Baselines3 compatibility** — Gymnasium `VecEnv` adapter around `QuadrotorEnv_v1`
  (`env_wrapper.py: FlightlibVecEnv`).
- **Disturbance models** — Ornstein–Uhlenbeck wind, body-frame quadratic drag, and force/torque noise
  (`flightmare/flightlib/configs/quadrotor_env.yaml → disturbances`; toggle via `env.disturbances.enable`).
- **Domain randomization** — mass, motor time constant, command delay, goal position.
- **Full-envelope spawn ranges** — wide position/velocity/rate initialisation, including inverted starts.
- **Decimated physics** — high-rate integration with a lower-rate control step (`VecPhysicsDecimationWrapper`).

Typical customizations:

- **New training regime:** copy `configs/ppo_config_base.yaml`, change the parameters, point
  `--config` (or the curriculum `base_config`) at it.
- **New curriculum:** edit/clone `configs/curriculum.yaml` — add stages with just the overrides you want.
- **New safety constraints:** add barriers in `configs/cbf_config.yaml`.
- **New evaluation scenarios:** add entries to a scenarios YAML and pass it via `--scenarios_config`.

## Real-drone deployment (ROS)

`standalone_ros_rl_feedthrough/` is a self-contained ROS 1 package that runs a trained policy on hardware
(Agilicious / RosPilot) by publishing `agiros_msgs/Command` on `feedthrough_command`. It includes a native
`rospy` node, a `rosbridge` + `roslibpy` client (run the policy on a laptop while ROS stays in a
container), and a **CBF-enabled** variant that filters every step on-board and logs barrier values /
interventions. See `standalone_ros_rl_feedthrough/README.md`.

## Citation

```bibtex
@inproceedings{esen2026saferl,
  title     = {Safe Reinforcement Learning with Control Barrier Functions for Quadrotor Recovery},
  author    = {E{\c{s}}en, O{\u{g}}uzhan and Yuan, Yuxia and Ryll, Markus},
  year      = {2026},
  organization = {Chair of Autonomous Aerial Systems, Technical University of Munich}
}
```

Built on [Flightmare](https://github.com/uzh-rpg/flightmare),
[Agilicious](https://agilicious.dev),
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), and
[acados](https://docs.acados.org).
```
