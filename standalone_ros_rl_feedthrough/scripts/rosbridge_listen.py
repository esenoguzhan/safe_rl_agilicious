#!/usr/bin/env python3
"""
Listen to ROS topics from the host PC via rosbridge (roslibpy).

Requires on the ROS side (e.g. Docker, same master as the pilot):
  roslaunch rosbridge_server rosbridge_websocket.launch port:=9090

Install on host: pip install roslibpy

Examples:
  # State + feedthrough commands (defaults match rl_feedthrough_rosbridge_client.py)
  python3 rosbridge_listen.py

  # Only commands
  python3 rosbridge_listen.py --no-state --command

  # Custom topic
  python3 rosbridge_listen.py --topic /angrybird/agiros_pilot/mpc_command \\
      --msg-type agiros_msgs/Command

  # Remote rosbridge
  python3 rosbridge_listen.py --host 192.168.1.10 --port 9090
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

try:
    from roslibpy import Ros, Topic
except ImportError:
    print("Install roslibpy: pip install roslibpy", file=sys.stderr)
    raise


def _parse_args():
    p = argparse.ArgumentParser(
        description="Subscribe to ROS topics over rosbridge and print messages."
    )
    p.add_argument("--host", default="127.0.0.1", help="rosbridge WebSocket host")
    p.add_argument("--port", type=int, default=9090, help="rosbridge WebSocket port")
    p.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Seconds to wait for WebSocket connection",
    )

    p.add_argument(
        "--state",
        action="store_true",
        default=True,
        help="Subscribe to QuadState (default: on)",
    )
    p.add_argument("--no-state", action="store_false", dest="state")
    p.add_argument(
        "--state-topic",
        default="/angrybird2_/agiros_pilot/state",
        help="agiros_msgs/QuadState",
    )

    p.add_argument(
        "--command",
        action="store_true",
        default=True,
        help="Subscribe to feedthrough Command (default: on)",
    )
    p.add_argument("--no-command", action="store_false", dest="command")
    p.add_argument(
        "--cmd-topic",
        default="/angrybird2_/agiros_pilot/feedthrough_command",
        help="agiros_msgs/Command",
    )

    p.add_argument(
        "--telemetry",
        action="store_true",
        default=False,
        help="Also subscribe to Telemetry",
    )
    p.add_argument(
        "--telemetry-topic",
        default="/angrybird2_/agiros_pilot/telemetry",
        help="agiros_msgs/Telemetry",
    )

    p.add_argument(
        "--topic",
        action="append",
        default=[],
        metavar="ROS_TOPIC",
        help="Extra topic to subscribe (repeatable); requires --msg-type",
    )
    p.add_argument(
        "--msg-type",
        action="append",
        default=[],
        metavar="TYPE",
        help="Message type for each --topic (e.g. agiros_msgs/Command)",
    )

    p.add_argument(
        "--rate-hz",
        type=float,
        default=0.0,
        help="Max print rate per topic (0 = no limit)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print full message JSON instead of a one-line summary",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logging",
    )
    return p.parse_args()


def _summarize_state(msg: Dict[str, Any]) -> str:
    try:
        p = msg["pose"]["position"]
        v = msg["velocity"]["linear"]
        t = msg.get("t", "?")
        return (
            f"t={t} pos=[{p['x']:.3f},{p['y']:.3f},{p['z']:.3f}] "
            f"vel=[{v['x']:.3f},{v['y']:.3f},{v['z']:.3f}]"
        )
    except (KeyError, TypeError):
        return json.dumps(msg, separators=(",", ":"))


def _summarize_command(msg: Dict[str, Any]) -> str:
    try:
        t = msg.get("t", "?")
        thrusts = msg.get("thrusts", [])
        if thrusts:
            ts = ", ".join(f"{float(x):.3f}" for x in thrusts[:4])
            return f"t={t} thrusts=[{ts}]"
        ct = msg.get("collective_thrust", 0.0)
        br = msg.get("bodyrates", {})
        return (
            f"t={t} collective={float(ct):.3f} "
            f"bodyrates=[{br.get('x',0):.3f},{br.get('y',0):.3f},{br.get('z',0):.3f}]"
        )
    except (KeyError, TypeError, ValueError):
        return json.dumps(msg, separators=(",", ":"))


def _summarize_telemetry(msg: Dict[str, Any]) -> str:
    try:
        ref = msg.get("reference", msg)
        p = ref["pose"]["position"]
        return f"ref=[{p['x']:.3f},{p['y']:.3f},{p['z']:.3f}]"
    except (KeyError, TypeError):
        return json.dumps(msg, separators=(",", ":"))


def _make_handler(
    label: str,
    summarize: Optional[Callable[[Dict[str, Any]], str]],
    rate_hz: float,
    as_json: bool,
) -> Callable[[Dict[str, Any]], None]:
    lock = threading.Lock()
    last_print = 0.0
    min_interval = (1.0 / rate_hz) if rate_hz > 0 else 0.0

    def on_message(msg: Dict[str, Any]) -> None:
        nonlocal last_print
        now = time.time()
        with lock:
            if min_interval > 0 and (now - last_print) < min_interval:
                return
            last_print = now
        if as_json:
            line = json.dumps(msg, separators=(",", ":"))
        elif summarize is not None:
            line = summarize(msg)
        else:
            line = json.dumps(msg, separators=(",", ":"))
        logging.info("[%s] %s", label, line)

    return on_message


def _subscribe(
    ros: Ros,
    topic_name: str,
    msg_type: str,
    label: str,
    summarize: Optional[Callable[[Dict[str, Any]], str]],
    rate_hz: float,
    as_json: bool,
) -> Topic:
    t = Topic(ros, topic_name, msg_type, queue_size=10)
    t.subscribe(_make_handler(label, summarize, rate_hz, as_json))
    logging.info("subscribed: %s (%s)", topic_name, msg_type)
    return t


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    extra_topics = list(args.topic)
    extra_types = list(args.msg_type)
    if len(extra_topics) != len(extra_types):
        logging.error(
            "Each --topic needs a matching --msg-type "
            "(got %d topics, %d types).",
            len(extra_topics),
            len(extra_types),
        )
        return 1

    if not (args.state or args.command or args.telemetry or extra_topics):
        logging.error("Nothing to subscribe; enable --state, --command, --telemetry, or --topic.")
        return 1

    ros = Ros(args.host, args.port)
    subscribers: List[Topic] = []
    stop = threading.Event()

    def _shutdown(*_):
        stop.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logging.info("Connecting to rosbridge at ws://%s:%s ...", args.host, args.port)
    try:
        ros.run(timeout=args.connect_timeout)
    except Exception as e:
        logging.error("ros.run() failed: %s", e)
        return 1

    if not ros.is_connected:
        logging.error("Could not connect to rosbridge.")
        return 1

    rate = args.rate_hz
    as_json = args.json

    if args.state:
        subscribers.append(
            _subscribe(
                ros,
                args.state_topic,
                "agiros_msgs/QuadState",
                "state",
                _summarize_state,
                rate,
                as_json,
            )
        )
    if args.command:
        subscribers.append(
            _subscribe(
                ros,
                args.cmd_topic,
                "agiros_msgs/Command",
                "command",
                _summarize_command,
                rate,
                as_json,
            )
        )
    if args.telemetry:
        subscribers.append(
            _subscribe(
                ros,
                args.telemetry_topic,
                "agiros_msgs/Telemetry",
                "telemetry",
                _summarize_telemetry,
                rate,
                as_json,
            )
        )

    for topic_name, msg_type in zip(extra_topics, extra_types):
        label = topic_name.rsplit("/", 1)[-1] or topic_name
        subscribers.append(
            _subscribe(ros, topic_name, msg_type, label, None, rate, as_json)
        )

    logging.info("Listening (Ctrl+C to stop).")
    try:
        while ros.is_connected and not stop.is_set():
            time.sleep(0.1)
    finally:
        for sub in subscribers:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        try:
            ros.close()
        except Exception:
            pass
        logging.info("Disconnected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
