# standalone_ros_rl_feedthrough

Self-contained ROS package (not wired into Agilicious sources). Publishes `agiros_msgs/Command` on `feedthrough_command` so **RosPilot** applies your RL command when feedthrough is active.

## Dependencies

- ROS 1 (`rospy`), `geometry_msgs`
- `agiros_msgs` from your Agilicious workspace (`/home/oguz/Desktop/agilicious_repo/agilicious` → build `agiros_msgs`)

## Build

Place this folder under your catkin workspace `src/`, then:

```bash
cd <catkin_ws>
catkin_make   # or catkin build
source devel/setup.bash
chmod +x src/standalone_ros_rl_feedthrough/scripts/rl_feedthrough_node.py
```

## Run (same namespace as pilot)

Agiros subscribes to **private** topics on the pilot node, e.g. `state` and `feedthrough_command`. Run this node **in the same ROS namespace** as `RosPilot`:

```bash
rosrun standalone_ros_rl_feedthrough rl_feedthrough_node.py \
  _quad_state_topic:=state \
  _feedthrough_topic:=feedthrough_command \
  _rate_hz:=100
```

If your pilot uses a different prefix, use remaps or `rosrun ... __ns:=/your_pilot_ns`.

## What to edit

1. `_load_policy()` — `PPO.load`, optional `VecNormalize.load`
2. `build_observation()` — match your training observation (history, noise off, etc.)
3. `_action_to_command()` — map SB3 `[-1,1]` actions to Newton thrusts using your `act_mean` / `act_std`
4. Parameters `~fixed_goal_xyz` vs telemetry reference for position error

With `~model_path` empty, the node runs and publishes **zero thrust** commands (for wiring tests only).

## Host without ROS: rosbridge + roslibpy

Run the policy on your laptop with **Python 3 + pip** only (`pip install -r requirements.txt`), while ROS + pilot stay in Docker.

**In the container** (same ROS master as the pilot), start WebSocket rosbridge, e.g.:

```bash
sudo apt-get install ros-${ROS_DISTRO}-rosbridge-suite   # if needed
roslaunch rosbridge_server rosbridge_websocket.launch port:=9090
```

Expose port **9090** from Docker (`-p 9090:9090`) or use host networking.

**On the host**, from this package’s `scripts/` (or after `catkin build` + `source devel/setup.bash`):

```bash
python3 rl_feedthrough_rosbridge_client.py \
  --model-path /path/to/run_dir_or_best_model.zip \
  --fixed-goal 1 2 0.5
```

Everything else uses sensible defaults (host `127.0.0.1:9090`, topics under
`/angrybird/agiros_pilot/...`, motor permutation `1,3,2,0`, telemetry off,
zero state-estimator noise, 50-step warm-up, log-every 25). Override any of
them with the matching CLI flag if your setup differs.

`--model-path` accepts either the run directory or the `best_model.zip` file
itself; `vecnormalize.pkl` is auto-discovered alongside it (override with
`--vecnormalize-path` if needed).

The script subscribes to `agiros_msgs/QuadState` and publishes
`agiros_msgs/Command` through rosbridge; message definitions must be
available on the ROS side (standard for custom msgs when using rosbridge
with that workspace sourced).

Shared logic lives in `rl_feedthrough_core.py` (used by both `rl_feedthrough_node.py` and the rosbridge client).

## Observing the CBF safety filter

The same package ships a CBF-enabled variant of the rosbridge client so you can watch a Control Barrier Function (Cheng et al., AAAI 2019) sit between the RL policy and the published `feedthrough_command`, without touching the no-CBF entrypoints.

Files (all under `scripts/`):

| File | Purpose |
| --- | --- |
| `cbf_core_lib.py` | Self-contained CBF + minimal `QuadrotorModel`. Continuous-time HOCBF QP, OSQP / SciPy backends, optional slack. Trimmed subset of `source_scripts/cbf_filter.py` (no acados, no discrete CBF). |
| `cbf_config.yaml` | Default CBF config (ground + ceiling velocity-aware barriers, OSQP, slack on). Edit to add x/y walls or change `alpha`, `kv`, `r_uav`, etc. |
| `quadrotor_model.yaml` | Dynamics parameters for the CBF Lie derivatives. Must match the model the policy was trained against. |
| `rl_cbf_feedthrough_core.py` | Composes `RlFeedthroughCore` with `CBFFilter`. Adds `predict_and_filter(obs, state_dict)` returning `(raw_action, safe_action, info)` plus barrier-value / QP-status telemetry. |
| `rl_cbf_feedthrough_rosbridge_client.py` | Drop-in replacement for `rl_feedthrough_rosbridge_client.py` that routes every step through the CBF, periodically logs barrier values + interventions, and (optionally) appends a CSV trace for offline analysis. |

Extra dependencies (already added to `requirements.txt`): `pyyaml`, `scipy`, `osqp`.

### Run

```bash
python3 rl_cbf_feedthrough_rosbridge_client.py \
  --host 127.0.0.1 --port 9090 \
  --state-topic /your_pilot_ns/state \
  --cmd-topic   /your_pilot_ns/feedthrough_command \
  --telemetry-topic /your_pilot_ns/telemetry \
  --model-path /path/to/best_model.zip \
  --vecnormalize-path /path/to/vecnormalize.pkl \
  --motor-perm 1,3,2,0 \
  --fixed-goal 0 0 5 \
  --cbf-log-every 25 \
  --cbf-csv /tmp/cbf_trace.csv
```

Every flag from `rl_feedthrough_rosbridge_client.py` (state-estimator noise, action LPF, torque scale, ...) is preserved with the same semantics. Note that the safety watchdog flags previously bundled with this entrypoint have been removed; the CBF entrypoint still ships its own `--safety-*` watchdog for now. The only additions over the no-CBF script are:

- `--cbf-config PATH` / `--quadrotor-model-config PATH` – override the bundled YAMLs.
- `--no-cbf` – bypass the filter to A/B against the raw-policy deployment with every other knob byte-identical.
- `--push-raw-to-history` – feed the policy's action-history buffer with the raw RL action instead of the CBF-filtered one (default mirrors `source_scripts/cbf_wrapper.py`).
- `--no-align-thrust-limits` – keep the CBF QP bounds at the physical motor envelope. Default clamps to `[max(0, mean-std), mean+std]`, matching what Flightmare actually applies through the policy's normalised-action interface.
- `--cbf-log-every N` – one-line CBF summary every N steps: `|u_cbf|`, h(p,v) per barrier, slack, `INTERVENED`, `QP_FAILED(reason)`.
- `--cbf-csv PATH` – append per-step `(t, p, v, ω, q, u_rl, u_safe, |u_cbf|, intervened, qp_failed, h_*, n·v_*, slack_*)` to a CSV for offline plots.

The script also prints a final summary on exit (`steps`, `interventions (%)`, `qp_failures (%)`) so you can quickly tell whether the filter actually fired during the run.
