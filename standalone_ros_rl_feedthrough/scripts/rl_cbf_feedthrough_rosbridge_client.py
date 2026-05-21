#!/usr/bin/env python3
"""
RL + CBF feedthrough policy via rosbridge (roslibpy).

Mirrors ``rl_feedthrough_rosbridge_client.py`` but inserts a continuous-time
CBF safety filter (see :mod:`cbf_core_lib`, :mod:`rl_cbf_feedthrough_core`)
between the RL policy and the published ``agiros_msgs/Command``. Use this to
observe the CBF's behaviour in agisim while keeping the rest of the
deployment pipeline (transport, observation pre-processing, action affine
+ motor permutation, runtime goal updates) byte-identical to the no-CBF
script.

Minimal invocation (everything else uses sensible defaults; ``vecnormalize.pkl``
is auto-discovered next to ``best_model.zip``; a CSV trace is written to
``/tmp/rl_cbf_feedthrough_trace_<timestamp>.csv`` by default for offline
plotting):

  python3 rl_cbf_feedthrough_rosbridge_client.py \\
    --model-path /path/to/run_dir_or_best_model.zip \\
    --fixed-goal 1 2 0.5

Re-target the policy while running (stdin reader, on by default when
attached to a terminal):

    type  ``X Y Z<Enter>``       — set a new goal
    type  ``snap<Enter>``        — snap goal to current drone pose
    type  ``show<Enter>``        — print the current goal

CBF-specific flags vs the no-CBF script:
  - ``--cbf-config``             path to a non-default CBF YAML
  - ``--quadrotor-model-config`` path to a non-default model YAML
  - ``--no-cbf``                 bypass the filter (A/B against the no-CBF
                                 deployment using all other knobs)
  - ``--push-raw-to-history``    feed the policy's history with the raw RL
                                 action instead of the CBF-filtered one
                                 (default: push the safe action, as in
                                 ``source_scripts/cbf_wrapper.py``)
  - ``--no-align-thrust-limits`` keep CBF QP bounds at the physical motor
                                 envelope (default: clamp to the deployment's
                                 action affine band)
  - ``--cbf-log-every N``        periodic CBF summary log (default 0 = silent)
"""
import argparse
import logging
import os
import sys
import threading
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np

from rl_feedthrough_core import (
    GoalState,
    RlFeedthroughCore,
    StateCsvTracer,
    default_paths_under_repo,
    default_trace_csv_path,
    resolve_policy_paths,
    start_stdin_goal_reader,
)
from rl_cbf_feedthrough_core import RlCbfFeedthroughCore

try:
    from roslibpy import Message, Ros, Topic
except ImportError:
    print("Install roslibpy: pip install roslibpy", file=sys.stderr)
    raise


def _parse_args():
    p = argparse.ArgumentParser(
        description="RL + CBF feedthrough via rosbridge (roslibpy)"
    )

    # ----------------------------------------------------------- transport
    p.add_argument("--host", default="127.0.0.1", help="rosbridge WebSocket host")
    p.add_argument("--port", type=int, default=9090, help="rosbridge WebSocket port")
    p.add_argument(
        "--state-topic",
        default="/angrybird/agiros_pilot/state",
        help="agiros_msgs/QuadState subscription (full ROS name)",
    )
    p.add_argument(
        "--cmd-topic",
        default="/angrybird/agiros_pilot/feedthrough_command",
        help="agiros_msgs/Command publication (full ROS name)",
    )
    p.add_argument(
        "--telemetry-topic",
        default="/angrybird/agiros_pilot/telemetry",
        help="agiros_msgs/Telemetry subscription (optional reference)",
    )
    p.add_argument(
        "--telemetry",
        action="store_true",
        default=False,
        help="Subscribe to telemetry topic and use its reference for the goal. "
             "By default telemetry is OFF (goal comes from --fixed-goal or zeros).",
    )
    p.add_argument(
        "--no-telemetry",
        action="store_false",
        dest="telemetry",
        help="(default; kept for back-compat with prior invocations).",
    )
    p.add_argument("--rate-hz", type=float, default=50.0)

    # --------------------------------------------------------------- policy
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
    p.add_argument("--action-lpf-alpha", type=float, default=1.0)
    p.add_argument("--action-clip", type=float, default=1.0)
    p.add_argument("--action-history-init", type=float, default=0.0)
    p.add_argument("--torque-scale", type=float, default=1.0)
    p.add_argument("--thrust-bias", type=float, default=0.0)
    p.add_argument("--noise-pos", type=float, default=0.0)
    p.add_argument("--noise-vel", type=float, default=0.0)
    p.add_argument("--noise-omega", type=float, default=0.0)
    p.add_argument("--noise-quat", type=float, default=0.0)
    p.add_argument("--noise-seed", type=int, default=None)

    # ------------------------------------------------------------------ CBF
    p.add_argument(
        "--cbf-config",
        default="",
        help="Path to a CBF YAML. Empty -> bundled cbf_config.yaml next to "
             "cbf_core_lib.py.",
    )
    p.add_argument(
        "--quadrotor-model-config",
        default="",
        help="Path to a quadrotor model YAML for the CBF. Empty -> bundled "
             "quadrotor_model.yaml.",
    )
    p.add_argument(
        "--no-cbf",
        action="store_true",
        help="Disable the CBF filter (publish the raw RL action). Use to A/B "
             "with the no-CBF deployment while keeping every other knob "
             "byte-identical.",
    )
    p.add_argument(
        "--push-raw-to-history",
        action="store_true",
        help="Push the raw RL action into the policy's history buffer instead "
             "of the CBF-filtered action. Matches the no-CBF training "
             "distribution. Default: push the safe action (as in "
             "source_scripts/cbf_wrapper.py).",
    )
    p.add_argument(
        "--no-align-thrust-limits",
        action="store_true",
        help="Keep the CBF QP's per-motor thrust bounds at the physical "
             "motor envelope (from quadrotor_model.yaml). Default: clamp to "
             "[max(0, mean - std), mean + std] from the deployment's action "
             "affine, matching what Flightmare actually applies.",
    )
    p.add_argument(
        "--cbf-log-every",
        type=int,
        default=0,
        help="If >0, log a one-line CBF summary every N steps (h values, "
             "|u_cbf|, slack, QP status). 0 disables.",
    )
    p.add_argument(
        "--cbf-csv",
        default="",
        help="Path to append per-step state + CBF telemetry CSV. Empty "
             "(default) -> /tmp/rl_cbf_feedthrough_trace_<timestamp>.csv. "
             "Use --no-cbf-csv to disable recording entirely.",
    )
    p.add_argument(
        "--no-cbf-csv",
        action="store_true",
        help="Disable per-step CSV trace recording.",
    )

    # ------------------------------------------------------------- runtime
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
             "commands AFTER --engage-after-steps and BEFORE the RL+CBF "
             "pipeline engages. Mimics a brief freefall so the policy has "
             "to recover. At --rate-hz 50, N=10 ~ 0.2 s of freefall. "
             "Default 0 (disabled).",
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
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _cbf_extra_columns(barrier_names):
    """Column names appended to StateCsvTracer.BASE_COLUMNS for CBF telemetry."""
    cols = [
        "u_rl_0", "u_rl_1", "u_rl_2", "u_rl_3",
        "u_safe_0", "u_safe_1", "u_safe_2", "u_safe_3",
        "u_cbf_norm",
        "intervened", "qp_failed", "qp_failure_reason",
    ]
    for name in barrier_names:
        cols.append("h_{}".format(name))
    for name in barrier_names:
        cols.append("ndotv_{}".format(name))
    for name in barrier_names:
        cols.append("slack_{}".format(name))
    return cols


def _cbf_extra_values(info, cmd, barrier_names):
    """Row values matching :func:`_cbf_extra_columns`."""
    u_rl = info.get("u_rl_n", np.zeros(4))
    u_safe = info.get("u_safe_n", np.array(cmd.get("thrusts", [0, 0, 0, 0])))
    u_cbf_norm = float(info.get("u_cbf_norm", 0.0))
    h_values = info.get("h_values") or {}
    ndotv = info.get("n_dot_v") or {}
    slack = info.get("slack") or {}

    vals = [
        "%.6f" % float(u_rl[0]), "%.6f" % float(u_rl[1]),
        "%.6f" % float(u_rl[2]), "%.6f" % float(u_rl[3]),
        "%.6f" % float(u_safe[0]), "%.6f" % float(u_safe[1]),
        "%.6f" % float(u_safe[2]), "%.6f" % float(u_safe[3]),
        "%.6f" % u_cbf_norm,
        int(bool(info.get("intervened", False))),
        int(bool(info.get("qp_failed", False))),
        info.get("qp_failure_reason") or "",
    ]
    for name in barrier_names:
        vals.append("%.6f" % float(h_values.get(name, float("nan"))))
    for name in barrier_names:
        vals.append("%.6f" % float(ndotv.get(name, float("nan"))))
    for name in barrier_names:
        vals.append("%.6f" % float(slack.get(name, 0.0)))
    return vals


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

    rl_core = RlFeedthroughCore(
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

    if not rl_core.load_policy():
        logging.error("Policy not loaded; exiting.")
        return 1

    try:
        cbf_core = RlCbfFeedthroughCore(
            rl_core=rl_core,
            cbf_config_path=args.cbf_config or None,
            quadrotor_model_config_path=args.quadrotor_model_config or None,
            enable_cbf=not args.no_cbf,
            push_safe_to_history=not args.push_raw_to_history,
            align_thrust_limits_to_action_affine=not args.no_align_thrust_limits,
        )
    except Exception as e:
        logging.exception("Failed to construct RlCbfFeedthroughCore: %s", e)
        return 1

    if cbf_core.enable_cbf:
        logging.info(
            "CBF: ENABLED  barriers=%s  thrust_limits=%s  "
            "push_safe_to_history=%s  align_thrust_limits=%s",
            list(cbf_core.barrier_names),
            cbf_core.effective_thrust_limits,
            cbf_core.push_safe_to_history,
            not args.no_align_thrust_limits,
        )
    else:
        logging.info("CBF: DISABLED (--no-cbf) — raw RL action will be published.")

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

    if args.no_cbf_csv:
        csv_tracer = None
        logging.info("CBF telemetry CSV: DISABLED (--no-cbf-csv).")
    else:
        csv_path = (args.cbf_csv or "").strip() or default_trace_csv_path(
            "rl_cbf_feedthrough"
        )
        csv_tracer = StateCsvTracer(
            csv_path,
            extra_columns=_cbf_extra_columns(cbf_core.barrier_names),
        )
        logging.info("CBF telemetry CSV: %s", csv_path)

    ros = Ros(args.host, args.port)

    lock = threading.Lock()
    last_state = {"msg": None}

    def on_state(message):
        with lock:
            last_state["msg"] = message

    def on_telemetry(message):
        rl_core.set_telemetry_dict(message)

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
        rl_core.fixed_goal_xyz = list(xyz)
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

    logging.info(
        "Running CBF controller loop @ %.1f Hz | state=%s cmd=%s | start_goal=%s",
        args.rate_hz,
        args.state_topic,
        args.cmd_topic,
        goal.get(),
    )

    log_every = int(max(0, args.log_step_every))
    cbf_log_every = int(max(0, args.cbf_log_every))
    step_idx = 0
    warmup_left = int(max(0, args.engage_after_steps))
    freefall_left = int(max(0, args.freefall_steps))
    freefall_total = freefall_left
    freefall_announced = False
    zero_action = np.zeros(4, dtype=np.float32)
    zero_cbf_extras = _cbf_extra_values(
        {
            "u_rl_n": np.zeros(4),
            "u_safe_n": np.zeros(4),
            "u_cbf_norm": 0.0,
            "intervened": False,
            "qp_failed": False,
            "qp_failure_reason": "",
            "h_values": {},
            "n_dot_v": {},
            "slack": {},
        },
        {"thrusts": [0.0, 0.0, 0.0, 0.0]},
        cbf_core.barrier_names,
    )

    cbf_intervened_count = 0
    cbf_qp_fail_count = 0
    try:
        while ros.is_connected:
            time.sleep(period)
            with lock:
                st = last_state["msg"]
            if st is None:
                continue
            if warmup_left > 0:
                warmup_left -= 1
                continue

            if freefall_left > 0:
                if not freefall_announced:
                    logging.info(
                        "freefall: publishing [0,0,0,0] thrust for %d steps "
                        "(~%.2f s @ %.1f Hz) before RL+CBF engages.",
                        freefall_total, freefall_total * period, args.rate_hz,
                    )
                    freefall_announced = True
                try:
                    t_sec = float(st.get("t", time.time()))
                    current_goal = goal.get()
                    obs = rl_core.build_observation_from_state_dict(
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
                    if csv_tracer is not None:
                        try:
                            csv_tracer.write(
                                step_idx, t_sec, current_goal, st, obs,
                                zero_action, cmd, extra_values=zero_cbf_extras,
                            )
                        except Exception as e:
                            logging.warning("tracer.write failed: %s", e)
                    if freefall_left == 0:
                        logging.info("freefall: done; RL+CBF engaging next step.")
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
                obs = rl_core.build_observation_from_state_dict(
                    st, fixed_goal_xyz=current_goal
                )

                raw_act, safe_act, info = cbf_core.predict_and_filter(
                    obs, state_dict=st
                )
                cmd = cbf_core.action_to_command_dict(t_sec, safe_act)
                cbf_core.push_action_history(raw_act, safe_act)
                cmd_pub.publish(Message(cmd))
                step_idx += 1

                if info.get("intervened"):
                    cbf_intervened_count += 1
                if info.get("qp_failed"):
                    cbf_qp_fail_count += 1

                if csv_tracer is not None:
                    try:
                        csv_tracer.write(
                            step_idx, t_sec, current_goal, st, obs, safe_act, cmd,
                            extra_values=_cbf_extra_values(
                                info, cmd, cbf_core.barrier_names
                            ),
                        )
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
                        "w=[%+.2f,%+.2f,%+.2f] "
                        "raw=[%+.2f,%+.2f,%+.2f,%+.2f] "
                        "safe=[%+.2f,%+.2f,%+.2f,%+.2f] "
                        "thr(N)=[%+.2f,%+.2f,%+.2f,%+.2f] sum=%.2f",
                        step_idx, t_sec, p[0], p[1], p[2],
                        q[0], q[1], q[2], q[3], v[0], v[1], v[2], w[0], w[1], w[2],
                        raw_act[0], raw_act[1], raw_act[2], raw_act[3],
                        safe_act[0], safe_act[1], safe_act[2], safe_act[3],
                        th[0], th[1], th[2], th[3], sum(th),
                    )

                if cbf_log_every > 0 and (
                    step_idx % cbf_log_every == 0 or step_idx <= 3
                ):
                    summary = cbf_core.format_cbf_log_line(info)
                    sp = info.get("state_p", np.zeros(3))
                    sv = info.get("state_v", np.zeros(3))
                    logging.info(
                        "CBF step=%d p=[%+.2f,%+.2f,%+.2f] v=[%+.2f,%+.2f,%+.2f] %s",
                        step_idx, sp[0], sp[1], sp[2], sv[0], sv[1], sv[2], summary,
                    )
            except Exception as e:
                logging.exception("step error: %s", e)
    except KeyboardInterrupt:
        pass
    finally:
        if cbf_core.enable_cbf:
            logging.info(
                "CBF summary: steps=%d  interventions=%d (%.1f%%)  "
                "qp_failures=%d (%.1f%%)",
                step_idx,
                cbf_intervened_count,
                100.0 * cbf_intervened_count / max(step_idx, 1),
                cbf_qp_fail_count,
                100.0 * cbf_qp_fail_count / max(step_idx, 1),
            )
        if csv_tracer is not None:
            logging.info(
                "data-csv: wrote %d rows -> %s",
                csv_tracer.row_count, csv_tracer.path,
            )
            csv_tracer.close()
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
