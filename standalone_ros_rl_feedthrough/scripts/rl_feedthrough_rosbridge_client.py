#!/usr/bin/env python3
"""
Run the RL feedthrough policy on the host without rospy — talk to rosbridge_server (WebSocket).

Requires in the ROS side (e.g. Docker): rosbridge_server running and reachable, e.g.
  roslaunch rosbridge_server rosbridge_websocket.launch port:=9090

Install on host: pip install roslibpy  (+ requirements.txt for SB3/torch)

Minimal invocation (everything else uses sensible defaults; vecnormalize.pkl is
auto-discovered next to best_model.zip):

  python3 rl_feedthrough_rosbridge_client.py \\
    --model-path /path/to/run_dir_or_best_model.zip \\
    --fixed-goal 1 2 0.5

Re-target the policy while the script is running (stdin reader, on by
default when attached to a terminal):

    type  ``X Y Z<Enter>``       — set a new goal
    type  ``snap<Enter>``        — snap goal to current drone pose
    type  ``show<Enter>``        — print the current goal
"""
import argparse
import logging
import os
import sys
import threading
import time

# Allow importing sibling core when run as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rl_feedthrough_core import (
    GoalState,
    RlFeedthroughCore,
    StateCsvTracer,
    default_paths_under_repo,
    default_trace_csv_path,
    resolve_policy_paths,
    start_stdin_goal_reader,
)

try:
    from roslibpy import Message, Ros, Topic
except ImportError as e:
    print("Install roslibpy: pip install roslibpy", file=sys.stderr)
    raise


def _parse_args():
    p = argparse.ArgumentParser(description="RL feedthrough via rosbridge (roslibpy)")
    p.add_argument("--host", default="127.0.0.1", help="rosbridge WebSocket host")
    p.add_argument("--port", type=int, default=9090, help="rosbridge WebSocket port")
    p.add_argument(
        "--state-topic",
        default="/angrybird2_/agiros_pilot/state",
        help="agiros_msgs/QuadState subscription (full ROS name)",
    )
    p.add_argument(
        "--cmd-topic",
        default="/angrybird2_/agiros_pilot/feedthrough_command",
        help="agiros_msgs/Command publication (full ROS name)",
    )
    p.add_argument(
        "--telemetry-topic",
        default="/angrybird2_/agiros_pilot/telemetry",
        help="agiros_msgs/Telemetry subscription (optional reference)",
    )
    p.add_argument(
        "--telemetry",
        action="store_true",
        default=False,
        help="Subscribe to telemetry topic and use its reference for the goal. "
             "By default telemetry is OFF (goal comes from --fixed-goal or zeros).",
    )
    # Kept for backward-compat: --no-telemetry is the default behaviour now.
    p.add_argument(
        "--no-telemetry",
        action="store_false",
        dest="telemetry",
        help="(default; kept for back-compat with prior invocations).",
    )
    p.add_argument("--rate-hz", type=float, default=50.0)
    p.add_argument("--model-path", default="", help="Path to SB3 best_model.zip")
    p.add_argument("--vecnormalize-path", default="", help="Path to vecnormalize.pkl")
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--fixed-goal",
        "--fixed_goal",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="World-frame goal (m); overrides telemetry reference for position error",
    )
    p.add_argument("--quad-mass-kg", type=float, default=0.774)
    p.add_argument("--gravity-z", type=float, default=-9.81)
    p.add_argument(
        "--enforce-quaternion-hemisphere",
        action="store_true",
        default=True,
    )
    p.add_argument(
        "--no-enforce-quaternion-hemisphere",
        dest="enforce_quaternion_hemisphere",
        action="store_false",
    )
    p.add_argument(
        "--pos-err-convention",
        choices=("goal_minus_pos", "pos_minus_goal"),
        default="goal_minus_pos",
        help="Sign of pos_err in the observation (must match training).",
    )
    p.add_argument(
        "--motor-perm",
        default="1,3,2,0",
        help=(
            "Comma-separated permutation P (length 4) applied as "
            "agi_thrusts[i] = policy_thrusts[P[i]]. "
            "Default '1,3,2,0' matches the stock flightmare order "
            "[FL, FR, BR, BL] -> agilicious [FR, BL, BR, FL]. "
            "Use '0,1,2,3' if your policy was already trained in agilicious order."
        ),
    )
    p.add_argument(
        "--body-velocity",
        action="store_true",
        help="Rotate the linear velocity from world to body frame before "
             "feeding the policy (use only if your training did this).",
    )
    p.add_argument(
        "--action-lpf-alpha",
        type=float,
        default=1.0,
        help="One-pole low-pass on the action: a_t = alpha*raw + (1-alpha)*a_{t-1}. "
             "1.0 disables (default). Try 0.3-0.5 to dampen bang-bang behavior.",
    )
    p.add_argument(
        "--action-clip",
        type=float,
        default=1.0,
        help="Clip raw action magnitude before LPF/affine. Default 1.0; try 0.5 "
             "to halve control authority while debugging.",
    )
    p.add_argument(
        "--action-history-init",
        type=float,
        default=0.0,
        help="Constant value to fill the action history buffer with at startup. "
             "Defaults to 0.0; the policy's running stats suggest ~0.05.",
    )
    p.add_argument(
        "--torque-scale",
        type=float,
        default=1.0,
        help="Shrink the differential (roll/pitch/yaw) component of the action "
             "vector by this factor while leaving the collective component "
             "untouched. 1.0 disables (default). Try 0.2-0.4 to stop the "
             "policy from going bang-bang on attitude when training-vs-deploy "
             "rotational damping is mismatched.",
    )
    p.add_argument(
        "--thrust-bias",
        type=float,
        default=0.0,
        help="Additive offset on the collective component of the action "
             "(units = action space, ~3.8 N per motor per unit). Use a small "
             "positive value (e.g. 0.02-0.05) if probe shows the policy "
             "outputs slightly less than m*g at perfect hover.",
    )
    p.add_argument(
        "--noise-pos",
        type=float,
        default=0.0,
        help="Std-dev (m) of i.i.d. Gaussian noise added to the position "
             "fed to the policy. Simulates state-estimator / Vicon noise. "
             "Try 0.005-0.02 m for indoor mocap, 0.05-0.10 m for VIO.",
    )
    p.add_argument(
        "--noise-vel",
        type=float,
        default=0.0,
        help="Std-dev (m/s) of i.i.d. Gaussian noise on the linear velocity. "
             "Try 0.02-0.10 m/s for typical state-estimator output.",
    )
    p.add_argument(
        "--noise-omega",
        type=float,
        default=0.0,
        help="Std-dev (rad/s) of i.i.d. Gaussian noise on the body angular "
             "velocity (gyro-equivalent). Try 0.02-0.10 rad/s.",
    )
    p.add_argument(
        "--noise-quat",
        type=float,
        default=0.0,
        help="Std-dev (rad) of small-angle attitude perturbation applied as "
             "a random rotation. Try 0.005-0.02 rad (≈0.3-1.1 deg).",
    )
    p.add_argument(
        "--noise-seed",
        type=int,
        default=None,
        help="Optional RNG seed for the observation-noise generator. "
             "Defaults to None (non-reproducible).",
    )
    p.add_argument(
        "--log-step-every",
        type=int,
        default=0,
        help="If >0, log per-step obs summary + action every N steps to stderr. "
             "Default 0 (silent); a CSV trace is recorded for offline plotting.",
    )
    p.add_argument(
        "--snap-goal-to-current",
        action="store_true",
        help="Override --fixed-goal with the drone's pose at the first received "
             "state message, so the policy starts with pos_err == 0.",
    )
    p.add_argument(
        "--engage-after-steps",
        type=int,
        default=50,
        help="Wait this many control periods after connect before publishing "
             "commands (lets the state subscriber settle).",
    )
    p.add_argument(
        "--freefall-steps",
        type=int,
        default=0,
        help="If >0, publish N control periods of zero-thrust [0,0,0,0] "
             "commands AFTER --engage-after-steps and BEFORE the RL policy "
             "engages. Mimics a brief freefall so the policy has to recover. "
             "At --rate-hz 50, N=10 ~ 0.2 s of freefall. Default 0 (disabled).",
    )
    p.add_argument(
        "--stdin-goals",
        action="store_true",
        default=True,
        help="Read 'x y z' lines from stdin to retarget the goal at runtime "
             "(also 'snap' to snap to current pose, 'show' to print current "
             "goal). Auto-disabled when stdin is not a terminal. Default ON.",
    )
    p.add_argument(
        "--no-stdin-goals",
        action="store_false",
        dest="stdin_goals",
        help="Disable the stdin goal reader.",
    )
    p.add_argument(
        "--data-csv",
        default="",
        help="Path to append per-step state/action/thrust CSV trace. Empty "
             "(default) -> scripts/recordings/rl_feedthrough_trace_"
             "<timestamp>.csv. Use --no-data-csv to disable recording entirely.",
    )
    p.add_argument(
        "--no-data-csv",
        action="store_true",
        help="Disable per-step CSV trace recording.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    dm, dv = default_paths_under_repo(_SCRIPT_DIR)
    model_path, vnorm_path = resolve_policy_paths(
        args.model_path, args.vecnormalize_path, dm, dv
    )
    logging.info("model_path     = %s", model_path or "(none)")
    logging.info("vecnormalize   = %s", vnorm_path or "(none)")

    try:
        motor_perm = tuple(int(x) for x in args.motor_perm.split(","))
    except ValueError:
        logging.error("Invalid --motor-perm %r; expected e.g. '0,1,2,3'", args.motor_perm)
        return 2
    if len(motor_perm) != 4 or sorted(motor_perm) != [0, 1, 2, 3]:
        logging.error("--motor-perm must be a permutation of (0,1,2,3); got %s",
                      motor_perm)
        return 2

    pos_err_sign = +1 if args.pos_err_convention == "goal_minus_pos" else -1

    core = RlFeedthroughCore(
        model_path=model_path,
        vecnormalize_path=vnorm_path,
        device=args.device,
        action_dim=4,
        action_history_len=None,
        use_single_rotor_thrust=True,
        quad_mass_kg=args.quad_mass_kg,
        gravity_z=args.gravity_z,
        enforce_quat_hemisphere=args.enforce_quaternion_hemisphere,
        fixed_goal_xyz=list(args.fixed_goal) if args.fixed_goal else None,
        pos_err_sign=pos_err_sign,
        motor_perm=motor_perm,
        use_body_velocity=args.body_velocity,
        action_lpf_alpha=args.action_lpf_alpha,
        action_clip=args.action_clip,
        action_history_init=args.action_history_init,
        torque_scale=args.torque_scale,
        thrust_bias=args.thrust_bias,
        noise_pos=args.noise_pos,
        noise_vel=args.noise_vel,
        noise_omega=args.noise_omega,
        noise_quat=args.noise_quat,
        noise_seed=args.noise_seed,
    )
    logging.info(
        "core: pos_err_sign=%+d motor_perm=%s body_velocity=%s "
        "action_lpf_alpha=%.3f action_clip=%.3f hist_init=%.3f "
        "torque_scale=%.3f thrust_bias=%+.3f "
        "noise_pos=%.4f noise_vel=%.4f noise_omega=%.4f noise_quat=%.4f "
        "noise_seed=%s",
        pos_err_sign, motor_perm, args.body_velocity,
        args.action_lpf_alpha, args.action_clip, args.action_history_init,
        args.torque_scale, args.thrust_bias,
        args.noise_pos, args.noise_vel, args.noise_omega, args.noise_quat,
        args.noise_seed,
    )

    if not core.load_policy():
        logging.error("Policy not loaded; exiting.")
        return 1

    ros = Ros(args.host, args.port)

    lock = threading.Lock()
    last_state = {"msg": None}  # type: ignore

    def on_state(message):
        with lock:
            last_state["msg"] = message

    def on_telemetry(message):
        core.set_telemetry_dict(message)

    logging.info("Connecting to rosbridge at ws://%s:%s ...", args.host, args.port)
    try:
        ros.run()
    except Exception as e:
        logging.error("ros.run() failed: %s", e)
        return 1

    if not ros.is_connected:
        logging.error("Could not connect to rosbridge.")
        return 1

    listener = Topic(ros, args.state_topic, "agiros_msgs/QuadState", queue_size=1)
    listener.subscribe(on_state)

    tele = None
    if args.telemetry:
        tele = Topic(ros, args.telemetry_topic, "agiros_msgs/Telemetry", queue_size=1)
        tele.subscribe(on_telemetry)

    cmd_pub = Topic(ros, args.cmd_topic, "agiros_msgs/Command", queue_size=10)
    cmd_pub.advertise()

    period = 1.0 / max(args.rate_hz, 1.0)

    def _on_goal_change(xyz, source):
        core.fixed_goal_xyz = list(xyz)
        logging.info(
            "goal -> [%+.3f, %+.3f, %+.3f]  (%s)",
            xyz[0], xyz[1], xyz[2], source,
        )

    goal = GoalState(
        initial_xyz=list(args.fixed_goal) if args.fixed_goal else None,
        on_change=_on_goal_change,
    )

    def _latest_state_for_snap():
        with lock:
            return last_state["msg"]

    start_stdin_goal_reader(goal, _latest_state_for_snap, enabled=args.stdin_goals)

    if args.no_data_csv:
        tracer = None
        logging.info("data-csv recording: DISABLED (--no-data-csv).")
    else:
        csv_path = args.data_csv.strip() or default_trace_csv_path("rl_feedthrough")
        tracer = StateCsvTracer(csv_path)
        logging.info("data-csv recording: %s", csv_path)

    logging.info(
        "Running controller loop @ %.1f Hz | state=%s cmd=%s | start_goal=%s",
        args.rate_hz,
        args.state_topic,
        args.cmd_topic,
        goal.get(),
    )

    log_every = int(max(0, args.log_step_every))
    step_idx = 0
    pre_engage_idx = 0
    warmup_left = int(max(0, args.engage_after_steps))
    freefall_left = int(max(0, args.freefall_steps))
    freefall_total = freefall_left
    freefall_announced = False
    zero_action = [0.0, 0.0, 0.0, 0.0]
    # Reused, never-mutated placeholder command for phases where we publish
    # nothing (so no thrust is overwritten but the state is still recorded).
    no_publish_cmd = {"thrusts": [0.0, 0.0, 0.0, 0.0]}
    try:
        while ros.is_connected:
            time.sleep(period)
            with lock:
                st = last_state["msg"]
            if st is None:
                continue
            if warmup_left > 0:
                warmup_left -= 1
                # Record the baseline trajectory *before* we take over the
                # command topic. We do NOT publish here, so whatever controller
                # currently owns the drone stays in charge; we only observe.
                if tracer is not None:
                    try:
                        t_sec = float(st.get("t", time.time()))
                        current_goal = goal.get()
                        obs = core.build_observation_from_state_dict(
                            st, fixed_goal_xyz=current_goal
                        )
                        tracer.write(
                            pre_engage_idx, t_sec, current_goal, st, obs,
                            zero_action, no_publish_cmd, phase="pre_engage",
                        )
                        pre_engage_idx += 1
                    except Exception as e:
                        logging.warning("pre-engage tracer.write failed: %s", e)
                continue

            if freefall_left > 0:
                if not freefall_announced:
                    logging.info(
                        "freefall: publishing [0,0,0,0] thrust for %d steps "
                        "(~%.2f s @ %.1f Hz) before RL engages.",
                        freefall_total, freefall_total * period, args.rate_hz,
                    )
                    freefall_announced = True
                try:
                    t_sec = float(st.get("t", time.time()))
                    current_goal = goal.get()
                    obs = core.build_observation_from_state_dict(
                        st, fixed_goal_xyz=current_goal
                    )
                    cmd = {
                        "header": {
                            "stamp": {
                                "secs": int(time.time()),
                                "nsecs": int((time.time() % 1) * 1e9),
                            },
                            "frame_id": "",
                        },
                        "t": t_sec,
                        "is_single_rotor_thrust": True,
                        "collective_thrust": 0.0,
                        "bodyrates": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "thrusts": [0.0, 0.0, 0.0, 0.0],
                    }
                    cmd_pub.publish(Message(cmd))
                    step_idx += 1
                    freefall_left -= 1
                    if tracer is not None:
                        try:
                            tracer.write(
                                step_idx, t_sec, current_goal, st, obs,
                                zero_action, cmd, phase="freefall",
                            )
                        except Exception as e:
                            logging.warning("tracer.write failed: %s", e)
                    if freefall_left == 0:
                        logging.info("freefall: done; RL policy engaging next step.")
                except Exception as e:
                    logging.exception("freefall step error: %s", e)
                continue

            if args.snap_goal_to_current and step_idx == 0:
                try:
                    p = st["pose"]["position"]
                    goal.set(
                        [float(p["x"]), float(p["y"]), float(p["z"])],
                        "snap_goal_to_current",
                    )
                except (KeyError, TypeError) as e:
                    logging.warning("could not snap goal: %s", e)
            try:
                t_sec = float(st.get("t", time.time()))
                current_goal = goal.get()
                obs = core.build_observation_from_state_dict(
                    st, fixed_goal_xyz=current_goal
                )

                act = core.predict_action(obs)
                cmd = core.action_to_command_dict(t_sec, act)
                core.push_action_history(act)
                cmd_pub.publish(Message(cmd))
                step_idx += 1

                if tracer is not None:
                    try:
                        tracer.write(step_idx, t_sec, current_goal, st, obs, act, cmd)
                    except Exception as e:
                        logging.warning("tracer.write failed: %s", e)

                if log_every > 0 and (step_idx % log_every == 0 or step_idx <= 5):
                    p = obs[0:3]
                    q = obs[3:7]
                    v = obs[7:10]
                    w = obs[10:13]
                    th = cmd.get("thrusts", [0, 0, 0, 0])
                    logging.info(
                        "step=%d t=%.3f pos_err=[%+.2f,%+.2f,%+.2f] "
                        "q=[%+.2f,%+.2f,%+.2f,%+.2f] v=[%+.2f,%+.2f,%+.2f] "
                        "w=[%+.2f,%+.2f,%+.2f] act=[%+.2f,%+.2f,%+.2f,%+.2f] "
                        "thr(N)=[%+.2f,%+.2f,%+.2f,%+.2f] sum=%.2f",
                        step_idx, t_sec, p[0], p[1], p[2],
                        q[0], q[1], q[2], q[3], v[0], v[1], v[2], w[0], w[1], w[2],
                        act[0], act[1], act[2], act[3],
                        th[0], th[1], th[2], th[3], sum(th),
                    )
            except Exception as e:
                logging.exception("step error: %s", e)
    except KeyboardInterrupt:
        pass
    finally:
        if tracer is not None:
            logging.info(
                "data-csv: wrote %d rows -> %s",
                tracer.row_count, tracer.path,
            )
            tracer.close()
        try:
            listener.unsubscribe()
            if tele is not None:
                tele.unsubscribe()
            cmd_pub.unadvertise()
        except Exception:
            pass
        try:
            ros.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
