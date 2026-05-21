#!/usr/bin/env python3
"""
RL + CBF policy core for the standalone ROS RL feedthrough package.

Wraps :class:`rl_feedthrough_core.RlFeedthroughCore` with a continuous-time
CBF safety filter (see :mod:`cbf_core_lib`). The CBF runs in per-rotor thrust
[N] space, between the policy's normalised-action output and the affine
denormalisation that builds the ``agiros_msgs/Command``.

Pipeline per step (matches what :mod:`rl_feedthrough_rosbridge_client` does,
but routed through ``RlCbfFeedthroughCore.predict_and_filter`` instead of
``predict_action`` + ``action_to_command_dict``):

    raw_norm = RL.predict_action(obs)                           # in policy order, [-1, +1]
    u_rl    = max(clip(raw_norm, -1, 1) * act_std + act_mean, 0)  # [N], policy order
    state   = [p, q, v, omega]  from QuadState                  # raw, world / body conv.
    u_safe  = CBFFilter.filter(state, u_rl)                     # [N], policy order
    safe_norm = clip((u_safe - act_mean) / act_std, -1, +1)     # back to [-1, +1]
    cmd     = RL.action_to_command_dict(t, safe_norm)           # permutes to agi order
    publish(cmd)

The CBF's per-rotor thrust bounds are aligned with what the deployment can
actually command (``[max(0, mean - std), mean + std]``), so the QP plans only
in the feasible normalised-action band rather than the full physical motor
envelope.

State for the CBF: the raw world-frame position, quaternion, world-frame
linear velocity and body-frame angular velocity from ``QuadState``. There is
no ``goal - pos_err`` transform here (we have raw p directly), so barriers
defined in world frame work out of the box (e.g. ``z >= 1`` ground, ``z <= 8``
ceiling).

Motor ordering: the policy output and the CBF QP both live in *policy* /
Flightmare motor order. The conversion to agilicious order
``[FR, BL, BR, FL]`` happens inside
:py:meth:`RlFeedthroughCore.action_to_command_dict` via ``motor_perm`` (same
permutation as the non-CBF path).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rl_feedthrough_core import (  # noqa: E402
    RlFeedthroughCore,
    quat_wxyz_from_pose_dict,
    quat_wxyz_hemisphere,
    vec3_from_dict,
)

from cbf_core_lib import (  # noqa: E402
    ATT,
    CBFFilter,
    OME,
    POS,
    STATE_DIM,
    VEL,
)

logger = logging.getLogger(__name__)


class RlCbfFeedthroughCore:
    """Composes an :class:`RlFeedthroughCore` with a :class:`CBFFilter`.

    Holds a reference to the existing RL core (does not subclass it) so the
    non-CBF entrypoints in :mod:`rl_feedthrough_rosbridge_client` and
    :mod:`rl_feedthrough_node` keep working unchanged.

    Use :py:meth:`predict_and_filter` instead of ``RL.predict_action`` so the
    safe action and the CBF diagnostics come out atomically (the filter
    state in ``CBFFilter`` is updated on every call).
    """

    def __init__(
        self,
        rl_core: RlFeedthroughCore,
        cbf_config_path: Optional[str] = None,
        quadrotor_model_config_path: Optional[str] = None,
        enable_cbf: bool = True,
        push_safe_to_history: bool = True,
        align_thrust_limits_to_action_affine: bool = True,
    ):
        """
        Parameters
        ----------
        rl_core
            The RL feedthrough core that owns the SB3 policy and the
            action-affine ``(act_mean, act_std)`` for the deployment.
        cbf_config_path
            Path to a CBF YAML. ``None`` -> the bundled ``cbf_config.yaml``
            next to ``cbf_core_lib.py``.
        quadrotor_model_config_path
            Path to a quadrotor-model YAML. ``None`` -> the bundled
            ``quadrotor_model.yaml`` next to ``cbf_core_lib.py``.
        enable_cbf
            If ``False`` the core acts as a thin pass-through (useful for
            A/B comparisons in the deployment script: ``--no-cbf``).
        push_safe_to_history
            If ``True`` the policy's action-history input is fed the *safe*
            action (what the env actually applied). If ``False`` the raw RL
            action is pushed (matches the no-CBF training distribution more
            closely). Default matches the original ``CBFWrapper`` in
            ``source_scripts/cbf_wrapper.py``.
        align_thrust_limits_to_action_affine
            If ``True``, override the CBF's per-motor thrust bounds with the
            range the deployment can actually deliver through the policy's
            normalised-action interface: ``[max(0, mean - std), mean + std]``.
            Disable only if you intentionally want the CBF to plan within the
            physical motor envelope from ``quadrotor_model.yaml``.
        """
        self.rl = rl_core
        self.enable_cbf = bool(enable_cbf)
        self.push_safe_to_history = bool(push_safe_to_history)

        self._cbf: Optional[CBFFilter] = None
        self._effective_thrust_limits: Optional[Tuple[float, float]] = None

        if self.enable_cbf:
            self._cbf = CBFFilter(
                config_path=cbf_config_path,
                quadrotor_model_config_path=quadrotor_model_config_path,
            )

            if align_thrust_limits_to_action_affine:
                mean = np.asarray(self.rl._act_mean, dtype=np.float64).ravel()[:4]
                std = np.asarray(self.rl._act_std, dtype=np.float64).ravel()[:4]
                t_min = float(max(0.0, float(np.min(mean - std))))
                t_max = float(np.max(mean + std))
                if t_max > t_min:
                    self._cbf.set_thrust_limits(t_min, t_max)
                    self._effective_thrust_limits = (t_min, t_max)
                else:
                    logger.warning(
                        "RlCbfFeedthroughCore: degenerate action affine "
                        "(mean=%s, std=%s); leaving CBF thrust limits at physical "
                        "motor envelope.",
                        mean.tolist(), std.tolist(),
                    )
            else:
                self._effective_thrust_limits = self._cbf.model.get_thrust_limits()

    # ------------------------------------------------------------------ getters
    @property
    def cbf(self) -> Optional[CBFFilter]:
        return self._cbf

    @property
    def effective_thrust_limits(self) -> Optional[Tuple[float, float]]:
        """Per-motor [N] bounds the CBF QP is using (post-alignment)."""
        return self._effective_thrust_limits

    @property
    def barrier_names(self):
        if self._cbf is None:
            return ()
        return tuple(b.name for b in self._cbf.barriers)

    # --------------------------------------------------------------- state map
    def state_from_state_dict(self, state_dict: Dict[str, Any]) -> np.ndarray:
        """Build the 13D CBF state ``[p, q (wxyz), v, omega]`` directly from a
        rosbridge ``agiros_msgs/QuadState`` dict (no goal transform: the CBF
        works in world frame here).
        """
        p = vec3_from_dict(state_dict["pose"]["position"]).astype(np.float64)
        q = quat_wxyz_from_pose_dict(state_dict["pose"]).astype(np.float64)
        v = vec3_from_dict(state_dict["velocity"]["linear"]).astype(np.float64)
        w = vec3_from_dict(state_dict["velocity"]["angular"]).astype(np.float64)
        if self.rl.enforce_quat_hemisphere:
            q = quat_wxyz_hemisphere(q.astype(np.float32)).astype(np.float64)
        x = np.empty(STATE_DIM, dtype=np.float64)
        x[POS] = p
        x[ATT] = q
        x[VEL] = v
        x[OME] = w
        return x

    def state_from_ros_quadstate(self, state_msg) -> np.ndarray:
        """Same as :py:meth:`state_from_state_dict` but from a rospy QuadState."""
        p = np.array(
            [
                state_msg.pose.position.x,
                state_msg.pose.position.y,
                state_msg.pose.position.z,
            ],
            dtype=np.float64,
        )
        q = np.array(
            [
                state_msg.pose.orientation.w,
                state_msg.pose.orientation.x,
                state_msg.pose.orientation.y,
                state_msg.pose.orientation.z,
            ],
            dtype=np.float64,
        )
        v = np.array(
            [
                state_msg.velocity.linear.x,
                state_msg.velocity.linear.y,
                state_msg.velocity.linear.z,
            ],
            dtype=np.float64,
        )
        w = np.array(
            [
                state_msg.velocity.angular.x,
                state_msg.velocity.angular.y,
                state_msg.velocity.angular.z,
            ],
            dtype=np.float64,
        )
        if self.rl.enforce_quat_hemisphere:
            q = quat_wxyz_hemisphere(q.astype(np.float32)).astype(np.float64)
        x = np.empty(STATE_DIM, dtype=np.float64)
        x[POS] = p
        x[ATT] = q
        x[VEL] = v
        x[OME] = w
        return x

    # ------------------------------------------------------ predict + filter
    def _denormalize_to_thrusts(self, action_norm: np.ndarray) -> np.ndarray:
        """``[-1, +1]`` policy-order normalised action -> per-motor thrust [N]."""
        a = np.clip(np.asarray(action_norm, dtype=np.float64).reshape(-1), -1.0, 1.0)
        a4 = a[:4] if a.size >= 4 else np.pad(a, (0, 4 - a.size))
        thrusts = a4 * self.rl._act_std + self.rl._act_mean
        return np.maximum(thrusts, 0.0)

    def _normalize_from_thrusts(self, thrusts: np.ndarray) -> np.ndarray:
        """Inverse of :py:meth:`_denormalize_to_thrusts` (also ``[-1, +1]`` clamped)."""
        std_safe = np.where(np.abs(self.rl._act_std) < 1e-12, 1e-12, self.rl._act_std)
        anorm = (np.asarray(thrusts, dtype=np.float64) - self.rl._act_mean) / std_safe
        return np.clip(anorm, -1.0, 1.0).astype(np.float32)

    def predict_and_filter(
        self,
        obs: np.ndarray,
        state_dict: Optional[Dict[str, Any]] = None,
        state_msg: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Run the RL policy and the CBF in sequence.

        Either ``state_dict`` (rosbridge JSON) or ``state_msg`` (rospy
        QuadState) must be provided; both are accepted so the same core
        works for both transports.

        Returns
        -------
        raw_action : np.ndarray, shape (action_dim,), dtype float32
            Normalised action straight from :py:meth:`RlFeedthroughCore.predict_action`
            (post LPF / clip / torque-scale / thrust-bias). Policy-order.
        safe_action : np.ndarray, shape (action_dim,), dtype float32
            Normalised action after the CBF filter. Equal to ``raw_action`` when
            ``enable_cbf=False``. Policy-order. Feed this into
            :py:meth:`RlFeedthroughCore.action_to_command_dict` to publish.
        info : dict
            Per-step diagnostics. Always contains:
              - ``enable_cbf`` (bool)
              - ``state_p`` (3,), ``state_v`` (3,), ``state_omega`` (3,)
            When ``enable_cbf`` is True it additionally contains:
              - ``u_rl_n``    (4,)    raw RL thrust per motor [N], policy order
              - ``u_safe_n``  (4,)    CBF-filtered thrust per motor [N]
              - ``u_cbf_n``   (4,)    delta = u_safe_n - u_rl_n
              - ``u_cbf_norm``      L2 norm of u_cbf_n
              - ``h_values``  dict barrier_name -> h(p, v)
              - ``n_dot_v``   dict barrier_name -> n . v
              - ``qp_failed`` (bool)
              - ``slack``     None or dict barrier_name -> slack value
              - ``qp_failure_reason`` (str or None)
              - ``effective_thrust_limits`` (t_min, t_max) [N]
              - ``intervened`` (bool)  True if ``u_cbf_norm`` exceeded ``intervention_eps``
        """
        raw_action = self.rl.predict_action(obs)

        info: Dict[str, Any] = {
            "enable_cbf": bool(self.enable_cbf),
        }

        if state_dict is not None:
            state = self.state_from_state_dict(state_dict)
        elif state_msg is not None:
            state = self.state_from_ros_quadstate(state_msg)
        else:
            if self.enable_cbf:
                raise ValueError(
                    "predict_and_filter() requires state_dict or state_msg "
                    "when CBF is enabled."
                )
            state = None  # type: ignore

        if state is not None:
            info["state_p"] = state[POS].copy()
            info["state_v"] = state[VEL].copy()
            info["state_omega"] = state[OME].copy()

        if not self.enable_cbf or self._cbf is None:
            return raw_action, raw_action.copy(), info

        u_rl_n = self._denormalize_to_thrusts(raw_action)
        u_safe_n = self._cbf.filter(state, u_rl_n)
        u_safe_n = np.clip(u_safe_n, 0.0, None)

        safe_action = self._normalize_from_thrusts(u_safe_n)

        u_cbf_n = u_safe_n - u_rl_n
        info.update({
            "u_rl_n": u_rl_n,
            "u_safe_n": u_safe_n,
            "u_cbf_n": u_cbf_n,
            "u_cbf_norm": float(np.linalg.norm(u_cbf_n)),
            "h_values": self._cbf.barrier_values(state),
            "n_dot_v": self._cbf.barrier_normal_velocities(state),
            "qp_failed": bool(self._cbf.last_qp_failed),
            "slack": self._cbf.last_slack,
            "qp_failure_reason": self._cbf.last_qp_failure_reason,
            "effective_thrust_limits": self._effective_thrust_limits,
            "intervened": float(np.linalg.norm(u_cbf_n)) > 1e-4,
        })
        return raw_action, safe_action, info

    # --------------------------------------------------------------- helpers
    def push_action_history(
        self,
        raw_action: np.ndarray,
        safe_action: Optional[np.ndarray] = None,
    ) -> None:
        """Push the appropriate action into the policy's history buffer.

        Mirrors :py:meth:`RlFeedthroughCore.push_action_history`. The choice
        of which action enters the history is governed by
        ``push_safe_to_history`` (set at construction time):

          - True  (default, matches ``source_scripts/cbf_wrapper.py``): the
            CBF-filtered action enters the history -> the policy sees what
            the env actually applied. This is the more faithful sensor model
            once a CBF is in the loop.
          - False: the raw RL action enters the history -> matches the
            no-CBF training distribution exactly, but the history then
            diverges from physical reality whenever the CBF intervenes.
        """
        if safe_action is None or not self.push_safe_to_history:
            self.rl.push_action_history(raw_action)
        else:
            self.rl.push_action_history(safe_action)

    def action_to_command_dict(self, t_sec: float, action: np.ndarray) -> Dict[str, Any]:
        """Pass-through to :py:meth:`RlFeedthroughCore.action_to_command_dict`.

        The CBF returns a policy-order normalised action, so the same affine
        + permutation as the non-CBF path is reused.
        """
        return self.rl.action_to_command_dict(t_sec, action)

    def reset_runtime_state(self) -> None:
        """Reset the RL core's LPF + action history. (CBF is stateless.)"""
        self.rl.reset_runtime_state()

    # --------------------------------------------------------------- logging
    def format_cbf_log_line(self, info: Dict[str, Any]) -> str:
        """One-line summary of a per-step ``info`` dict for periodic logging."""
        if not info.get("enable_cbf", False):
            return "CBF=off"
        parts = []
        u_cbf_norm = info.get("u_cbf_norm")
        if u_cbf_norm is not None:
            parts.append(f"|u_cbf|={u_cbf_norm:+.3f}N")
        h_values = info.get("h_values") or {}
        if h_values:
            h_str = ", ".join(f"{k}={v:+.3f}" for k, v in h_values.items())
            parts.append(f"h=[{h_str}]")
        slack = info.get("slack")
        if slack:
            nz = {k: v for k, v in slack.items() if abs(v) > 1e-6}
            if nz:
                s_str = ", ".join(f"{k}={v:+.2e}" for k, v in nz.items())
                parts.append(f"slack=[{s_str}]")
        if info.get("intervened"):
            parts.append("INTERVENED")
        if info.get("qp_failed"):
            reason = info.get("qp_failure_reason") or "?"
            parts.append(f"QP_FAILED({reason})")
        return " ".join(parts) if parts else "CBF=idle"


__all__ = ["RlCbfFeedthroughCore"]
