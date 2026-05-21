#!/usr/bin/env python3
"""
Measure the end-to-end input -> response delay of the agisim feedthrough chain.

It publishes the SAME message type the RL policy uses
(``agiros_msgs/Command`` on ``feedthrough_command``) and times how long until
the body-rate response shows up in ``agiros_msgs/QuadState``.

The protocol per trial is:
  1. Send a uniform 'hold' thrust ([h,h,h,h]) for ``--hold-seconds`` so the
     drone settles to |w| ~ 0 and the pilot stays in feedthrough mode.
  2. Drain the state ring.
  3. At a known wall-clock + sim-time t0, publish a 'step' thrust that produces
     a clear single-axis torque (default = roll torque).
  4. Keep streaming the step command at ``--cmd-rate-hz`` so the pilot does
     not time out, while watching |w| in incoming QuadState messages.
  5. Report the time between t0 and the first QuadState whose |w| crosses
     ``--threshold-rad-s`` -- computed both in wall-clock (what the policy
     actually experiences) and in sim-time (what agisim physics adds, isolated
     from rosbridge / WebSocket jitter).

Run it WITHOUT the RL client (the script holds the pilot in feedthrough mode
by itself). Example:

    python3 measure_end_to_end_delay.py \\
        --host 127.0.0.1 --port 9090 \\
        --state-topic /angrybird/agiros_pilot/state \\
        --cmd-topic   /angrybird/agiros_pilot/feedthrough_command \\
        --n-trials 20 \\
        --csv /tmp/agisim_delay.csv

Two delays are reported. The difference of (wall - sim) is the rosbridge /
WebSocket round-trip overhead; the absolute ``sim`` value is what should be
compared against the training-time delay range.
"""
import argparse
import collections
import csv
import logging
import os
import statistics
import sys
import threading
import time

try:
    from roslibpy import Message, Ros, Topic
except ImportError:
    print("Install roslibpy: pip install roslibpy", file=sys.stderr)
    raise


def _parse_args():
    p = argparse.ArgumentParser(
        description="Measure end-to-end input delay of the agisim feedthrough chain."
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument(
        "--state-topic",
        default="/angrybird/agiros_pilot/state",
        help="agiros_msgs/QuadState subscription (full ROS name).",
    )
    p.add_argument(
        "--cmd-topic",
        default="/angrybird/agiros_pilot/feedthrough_command",
        help="agiros_msgs/Command publication (full ROS name).",
    )
    p.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of step trials. More trials -> tighter statistics.",
    )
    p.add_argument(
        "--hold-thrust",
        type=float,
        default=1.9,
        help="Per-motor thrust (N) in the baseline 'hold' phase. "
             "Default 1.9 N approximates m*g/4 for the angrybird (~0.774 kg).",
    )
    p.add_argument(
        "--step-bias",
        type=float,
        default=1.6,
        help="Per-motor differential thrust (N) for the step. Two motors get "
             "(hold+bias) and the other two (hold-bias), producing a clear "
             "single-axis torque. Default 1.6 N gives 0.3..3.5 N spread, well "
             "within the 5.7 N training cap.",
    )
    p.add_argument(
        "--step-axis",
        choices=("roll", "pitch", "collective"),
        default="roll",
        help="Which torque axis the step excites. 'collective' is a pure "
             "vertical thrust step (no torque) -- useful as a sanity check; "
             "the response then has to be detected via |a_z|.",
    )
    p.add_argument(
        "--hold-seconds",
        type=float,
        default=1.0,
        help="Baseline command duration before each step. Increase if the "
             "drone has not settled by the time the next trial starts.",
    )
    p.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="Baseline command duration after each step (lets w bleed off).",
    )
    p.add_argument(
        "--max-wait-ms",
        type=float,
        default=300.0,
        help="Per-trial timeout while waiting for the body-rate response.",
    )
    p.add_argument(
        "--threshold-rad-s",
        type=float,
        default=0.3,
        help="Threshold on max(|w_x|,|w_y|,|w_z|) (rad/s) that defines "
             "'response detected'. Lower = more sensitive but more noise-prone.",
    )
    p.add_argument(
        "--accel-threshold",
        type=float,
        default=3.0,
        help="For --step-axis=collective only: threshold on |a_z - a_z_baseline| "
             "(m/s^2) used to detect the response.",
    )
    p.add_argument(
        "--cmd-rate-hz",
        type=float,
        default=200.0,
        help="Rate at which hold/step commands are republished (keeps the "
             "pilot in feedthrough mode and matches typical agisim listening).",
    )
    p.add_argument(
        "--csv",
        default="",
        help="Optional path; appends per-trial measurements (with header on "
             "first write).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def _make_command(t_sec, thrusts):
    """Build an ``agiros_msgs/Command`` dict matching ``RlFeedthroughCore``."""
    now = time.time()
    secs = int(now)
    nsecs = int((now - secs) * 1e9)
    return {
        "header": {"stamp": {"secs": secs, "nsecs": nsecs}, "frame_id": ""},
        "t": float(t_sec),
        "is_single_rotor_thrust": True,
        "collective_thrust": 0.0,
        "bodyrates": {"x": 0.0, "y": 0.0, "z": 0.0},
        "thrusts": [float(thrusts[i]) for i in range(4)],
    }


def _build_step_thrusts(hold, bias, axis):
    """Return four per-motor thrusts (N) in the same order as ``Command.thrusts``.

    The agilicious motor allocation is

        F   = u0 + u1 + u2 + u3
        tx ~  u0 - u1 - u2 + u3   (roll)
        ty ~ -u0 - u1 + u2 + u3   (pitch)
        tz ~  u0 - u1 + u2 - u3   (yaw)

    so:
      * ``roll`` excites tx by raising motors {0,3} and lowering {1,2}.
      * ``pitch`` excites ty by raising motors {2,3} and lowering {0,1}.
      * ``collective`` raises all four by ``bias`` (zero torque).

    Whichever axis the agisim allocation actually maps these to, the
    response shows up on one body-rate axis and the maximum-magnitude
    component captures the timing. ``roll`` is the default because the
    user has been observing roll/pitch issues with the lowest inertia
    (and the highest expected sensitivity to delay).
    """
    if axis == "collective":
        u = [hold + bias] * 4
    elif axis == "pitch":
        u = [hold - bias, hold - bias, hold + bias, hold + bias]
    else:  # roll
        u = [hold + bias, hold - bias, hold - bias, hold + bias]
    return [max(0.0, float(t)) for t in u]


def _percentile_sorted(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _summary(values, label, logger_):
    if not values:
        logger_.info("%s: NO COMPLETED TRIALS.", label)
        return
    s = sorted(values)
    logger_.info(
        "%s n=%d  mean=%.2f  median=%.2f  min=%.2f  max=%.2f  "
        "p90=%.2f  p99=%.2f  stdev=%.2f  (ms)",
        label,
        len(values),
        statistics.mean(s),
        statistics.median(s),
        s[0],
        s[-1],
        _percentile_sorted(s, 90.0),
        _percentile_sorted(s, 99.0),
        statistics.stdev(s) if len(s) > 1 else 0.0,
    )


def main():
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("delay")

    hold_thrusts = [args.hold_thrust] * 4
    step_thrusts = _build_step_thrusts(args.hold_thrust, args.step_bias, args.step_axis)
    logger.info(
        "hold=%s  step=%s  axis=%s  threshold_rad_s=%.3f  max_wait_ms=%.1f",
        ["%.2f" % t for t in hold_thrusts],
        ["%.2f" % t for t in step_thrusts],
        args.step_axis,
        args.threshold_rad_s,
        args.max_wait_ms,
    )

    use_accel = (args.step_axis == "collective")

    ros = Ros(args.host, args.port)
    state_lock = threading.Lock()
    state_ring = collections.deque(maxlen=2048)

    def on_state(message):
        wall_recv = time.time()
        try:
            t = float(message.get("t", wall_recv))
            wx = float(message["velocity"]["angular"]["x"])
            wy = float(message["velocity"]["angular"]["y"])
            wz = float(message["velocity"]["angular"]["z"])
            az = float(message["acceleration"]["linear"]["z"]) if use_accel else 0.0
        except (KeyError, TypeError):
            return
        with state_lock:
            state_ring.append((wall_recv, t, wx, wy, wz, az))

    logger.info("Connecting to rosbridge at ws://%s:%s ...", args.host, args.port)
    try:
        ros.run()
    except Exception as e:
        logger.error("ros.run() failed: %s", e)
        return 1
    if not ros.is_connected:
        logger.error("Could not connect to rosbridge.")
        return 1

    listener = Topic(ros, args.state_topic, "agiros_msgs/QuadState", queue_size=1)
    listener.subscribe(on_state)
    cmd_pub = Topic(ros, args.cmd_topic, "agiros_msgs/Command", queue_size=10)
    cmd_pub.advertise()

    # Wait for the first QuadState so we know the topic is alive.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with state_lock:
            if state_ring:
                break
        time.sleep(0.05)
    else:
        logger.error("No QuadState received from %s within 5 s.", args.state_topic)
        ros.close()
        return 1

    def latest_sim_t():
        with state_lock:
            return state_ring[-1][1] if state_ring else time.time()

    def stream_for(duration_s, thrusts):
        period = 1.0 / max(args.cmd_rate_hz, 1.0)
        t0 = time.time()
        while time.time() - t0 < duration_s:
            cmd_pub.publish(Message(_make_command(latest_sim_t(), thrusts)))
            time.sleep(period)

    csv_file = None
    csv_writer = None
    if args.csv:
        new_file = (not os.path.isfile(args.csv)) or (os.path.getsize(args.csv) == 0)
        csv_file = open(args.csv, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if new_file:
            csv_writer.writerow([
                "trial_idx",
                "wall_delay_ms",
                "sim_delay_ms",
                "w_at_response_x",
                "w_at_response_y",
                "w_at_response_z",
                "peak_w_mag",
                "a_z_baseline",
                "a_z_at_response",
                "axis",
                "host_publish_unix_s",
                "sim_t_at_publish",
            ])

    wall_delays_ms = []
    sim_delays_ms = []
    threshold = float(args.threshold_rad_s)
    accel_threshold = float(args.accel_threshold)
    max_wait_s = float(args.max_wait_ms) * 1e-3

    try:
        for trial in range(args.n_trials):
            logger.info(
                "Trial %2d/%d: hold %.2f s ...", trial + 1, args.n_trials, args.hold_seconds
            )
            stream_for(args.hold_seconds, hold_thrusts)

            # Compute the baseline a_z just before stepping (collective mode only).
            with state_lock:
                last_snap = list(state_ring)[-10:]
                state_ring.clear()
            az_base = (
                statistics.fmean(s[5] for s in last_snap) if (use_accel and last_snap) else 0.0
            )

            sim_t_at_publish = latest_sim_t()
            step_cmd = _make_command(sim_t_at_publish, step_thrusts)
            host_publish_t = time.time()
            cmd_pub.publish(Message(step_cmd))

            wall_resp_t = None
            sim_resp_t = None
            resp_w = (0.0, 0.0, 0.0)
            resp_az = az_base
            peak_w = 0.0
            prev_signal = 0.0
            prev_wall = host_publish_t
            prev_sim = sim_t_at_publish
            t_deadline = host_publish_t + max_wait_s

            while time.time() < t_deadline:
                # Keep streaming so the pilot does not time out.
                cmd_pub.publish(Message(_make_command(latest_sim_t(), step_thrusts)))
                time.sleep(1.0 / max(args.cmd_rate_hz, 1.0))

                with state_lock:
                    snap = list(state_ring)
                    state_ring.clear()

                for wall_recv, sim_t, wx, wy, wz, az in snap:
                    if wall_recv < host_publish_t:
                        continue
                    if use_accel:
                        signal = abs(az - az_base)
                        thresh = accel_threshold
                    else:
                        signal = max(abs(wx), abs(wy), abs(wz))
                        thresh = threshold
                    if signal > peak_w:
                        peak_w = signal
                    if wall_resp_t is None and signal >= thresh:
                        # Linearly interpolate to sub-sample precision when
                        # the previous sample was strictly below the threshold;
                        # otherwise fall back to the raw receive time (slightly
                        # conservative on the high side, which is safer).
                        if prev_signal < thresh and signal > prev_signal:
                            alpha = (thresh - prev_signal) / (signal - prev_signal)
                            alpha = min(max(alpha, 0.0), 1.0)
                            wall_resp_t = prev_wall + alpha * (wall_recv - prev_wall)
                            sim_resp_t = prev_sim + alpha * (sim_t - prev_sim)
                        else:
                            wall_resp_t = wall_recv
                            sim_resp_t = sim_t
                        resp_w = (wx, wy, wz)
                        resp_az = az
                        break
                    prev_signal = signal
                    prev_wall = wall_recv
                    prev_sim = sim_t

                if wall_resp_t is not None:
                    break

            if wall_resp_t is None:
                logger.warning(
                    "Trial %2d: no response (peak %s=%.3f, threshold=%.3f) within %.0f ms",
                    trial + 1,
                    "|dAz|" if use_accel else "|w|",
                    peak_w,
                    accel_threshold if use_accel else threshold,
                    max_wait_s * 1e3,
                )
                if csv_writer is not None:
                    csv_writer.writerow([
                        trial + 1, "", "", "", "", "",
                        "%.4f" % peak_w,
                        "%.4f" % az_base, "",
                        args.step_axis,
                        "%.6f" % host_publish_t,
                        "%.6f" % sim_t_at_publish,
                    ])
                    csv_file.flush()
            else:
                wall_delay_ms = (wall_resp_t - host_publish_t) * 1e3
                sim_delay_ms = (sim_resp_t - sim_t_at_publish) * 1e3
                wall_delays_ms.append(wall_delay_ms)
                sim_delays_ms.append(sim_delay_ms)
                logger.info(
                    "Trial %2d: wall=%6.2f ms  sim=%6.2f ms  "
                    "w_resp=[%+.3f,%+.3f,%+.3f]  peak=%.3f%s",
                    trial + 1,
                    wall_delay_ms,
                    sim_delay_ms,
                    resp_w[0],
                    resp_w[1],
                    resp_w[2],
                    peak_w,
                    "  az=%+.2f (base %+.2f)" % (resp_az, az_base) if use_accel else "",
                )
                if csv_writer is not None:
                    csv_writer.writerow([
                        trial + 1,
                        "%.3f" % wall_delay_ms,
                        "%.3f" % sim_delay_ms,
                        "%.4f" % resp_w[0],
                        "%.4f" % resp_w[1],
                        "%.4f" % resp_w[2],
                        "%.4f" % peak_w,
                        "%.4f" % az_base,
                        "%.4f" % resp_az,
                        args.step_axis,
                        "%.6f" % host_publish_t,
                        "%.6f" % sim_t_at_publish,
                    ])
                    csv_file.flush()

            stream_for(args.settle_seconds, hold_thrusts)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        try:
            listener.unsubscribe()
            cmd_pub.unadvertise()
        except Exception:
            pass
        try:
            ros.close()
        except Exception:
            pass
        if csv_file is not None:
            csv_file.close()

    _summary(wall_delays_ms, "WALL-CLOCK DELAY (host publish -> host recv): ", logger)
    _summary(sim_delays_ms,  "SIM-TIME    DELAY (sim t_publish -> sim t_resp):", logger)

    if wall_delays_ms and sim_delays_ms:
        diff_ms = [w - s for w, s in zip(wall_delays_ms, sim_delays_ms)]
        _summary(diff_ms, "ROSBRIDGE OVERHEAD (wall - sim) approx.:        ", logger)
        median_wall = statistics.median(wall_delays_ms)
        median_sim = statistics.median(sim_delays_ms)
        logger.info(
            "Compare:  median end-to-end (policy view) = %.1f ms  vs.  "
            "training delay window = e.g. 15-25 ms (sim-side).",
            median_wall,
        )
        logger.info(
            "If sim-time median (%.1f ms) > simulation.yaml `delay` "
            "(plus a few ms for motor lag), the extra is from the motor "
            "lag / BEM transients; if wall-clock median is much larger "
            "than sim, the rosbridge round-trip is the dominant unmodelled "
            "factor.",
            median_sim,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
