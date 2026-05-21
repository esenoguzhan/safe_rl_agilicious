#!/usr/bin/env python3
"""
ROS 1 node: same RL policy as rl_feedthrough_core.RlFeedthroughCore (rospy + agiros_msgs).

For host without ROS, use: rl_feedthrough_rosbridge_client.py + rosbridge_server in Docker.
"""
import os
import sys
import threading

import numpy as np
import rospy
from agiros_msgs.msg import Command, QuadState, Telemetry
from geometry_msgs.msg import Vector3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rl_feedthrough_core import (  # noqa: E402
    IMPORT_ERROR,
    RlFeedthroughCore,
    default_paths_under_repo,
)


class RlFeedthroughNode(object):
    def __init__(self):
        rospy.init_node("rl_feedthrough_node", anonymous=False)

        if IMPORT_ERROR is not None:
            rospy.logfatal("Missing RL deps: %s", IMPORT_ERROR)
            raise IMPORT_ERROR

        default_model, default_vnorm = default_paths_under_repo(_SCRIPT_DIR)

        mp = rospy.get_param("~model_path", None)
        model_path = (mp or "").strip() or (
            default_model if os.path.isfile(default_model) else ""
        )
        vp = rospy.get_param("~vecnormalize_path", None)
        vecnorm_path = (vp or "").strip() or (
            default_vnorm if os.path.isfile(default_vnorm) else ""
        )

        fg = rospy.get_param("~fixed_goal_xyz", None)

        self._core = RlFeedthroughCore(
            model_path=model_path,
            vecnormalize_path=vecnorm_path,
            device=rospy.get_param("~device", "auto"),
            action_dim=int(rospy.get_param("~action_dim", 4)),
            action_history_len=rospy.get_param("~action_history_len", None),
            use_single_rotor_thrust=rospy.get_param("~use_single_rotor_thrust", True),
            collective_m_s2=float(rospy.get_param("~collective_thrust_m_s2", 9.81)),
            quad_mass_kg=float(rospy.get_param("~quad_mass_kg", 0.774)),
            gravity_z=float(rospy.get_param("~gravity_z", -9.81)),
            act_mean=rospy.get_param("~act_mean", None),
            act_std=rospy.get_param("~act_std", None),
            enforce_quat_hemisphere=rospy.get_param(
                "~enforce_quaternion_hemisphere", True
            ),
            fixed_goal_xyz=fg if fg is not None and len(fg) == 3 else None,
        )

        self._quad_state_topic = rospy.get_param("~quad_state_topic", "state")
        self._feedthrough_topic = rospy.get_param("~feedthrough_topic", "feedthrough_command")
        self._telemetry_topic = rospy.get_param("~telemetry_topic", "telemetry")
        self._use_telemetry_reference = rospy.get_param("~use_telemetry_reference", True)
        self._rate_hz = float(rospy.get_param("~rate_hz", 50.0))

        self._lock = threading.Lock()
        self._last_state = None
        self._last_telemetry = None

        if not self._core.load_policy():
            rospy.logwarn("Policy not loaded; publishing zero thrust.")

        self._state_sub = rospy.Subscriber(
            self._quad_state_topic,
            QuadState,
            self._on_state,
            queue_size=1,
            tcp_nodelay=True,
        )
        if self._use_telemetry_reference:
            self._tele_sub = rospy.Subscriber(
                self._telemetry_topic,
                Telemetry,
                self._on_telemetry,
                queue_size=1,
                tcp_nodelay=True,
            )
        self._cmd_pub = rospy.Publisher(
            self._feedthrough_topic, Command, queue_size=1, tcp_nodelay=True
        )

        self._timer = rospy.Timer(
            rospy.Duration(1.0 / max(self._rate_hz, 1.0)), self._on_timer
        )

        rospy.loginfo(
            "rl_feedthrough_node: sub=%s pub=%s @ %.1f Hz | model=%s | vecnorm=%s",
            self._quad_state_topic,
            self._feedthrough_topic,
            self._rate_hz,
            model_path or "(none)",
            vecnorm_path or "(none)",
        )
        rospy.loginfo(
            "action denorm: act_mean=%s act_std=%s",
            self._core._act_mean.tolist(),
            self._core._act_std.tolist(),
        )

    def _on_state(self, msg):
        with self._lock:
            self._last_state = msg

    def _on_telemetry(self, msg):
        with self._lock:
            self._last_telemetry = msg

    def _dict_to_command_msg(self, d):
        msg = Command()
        msg.header.stamp = rospy.Time.now()
        msg.t = float(d["t"])
        msg.is_single_rotor_thrust = bool(d["is_single_rotor_thrust"])
        msg.collective_thrust = float(d["collective_thrust"])
        br = d["bodyrates"]
        msg.bodyrates = Vector3(float(br["x"]), float(br["y"]), float(br["z"]))
        for i in range(4):
            msg.thrusts[i] = float(d["thrusts"][i])
        return msg

    def _on_timer(self, _evt):
        with self._lock:
            st = self._last_state
            if self._use_telemetry_reference and self._last_telemetry is not None:
                t = self._last_telemetry
                self._core.set_telemetry_dict(
                    {
                        "reference": {
                            "pose": {
                                "position": {
                                    "x": t.reference.pose.position.x,
                                    "y": t.reference.pose.position.y,
                                    "z": t.reference.pose.position.z,
                                }
                            }
                        }
                    }
                )
        if st is None:
            return
        t_sec = float(st.t) if st.t else rospy.Time.now().to_sec()
        fg = rospy.get_param("~fixed_goal_xyz", None)
        try:
            obs = self._core.build_observation_from_ros_quadstate(st, fg)
            act = self._core.predict_action(obs)
            d = self._core.action_to_command_dict(t_sec, act)
            self._core.push_action_history(act)
            self._cmd_pub.publish(self._dict_to_command_msg(d))
        except Exception as e:
            rospy.logerr_throttle(1.0, "build/predict error: %s", e)


def main():
    RlFeedthroughNode()
    rospy.spin()


if __name__ == "__main__":
    main()
