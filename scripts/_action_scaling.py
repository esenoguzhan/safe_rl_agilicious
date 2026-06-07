"""Shared nominal action-scaling helpers for Flightmare RL/MPC evaluation.

The Flightmare env denormalizes policy actions via
    thrust = act_std * action + act_mean

Historically these constants were computed from the *current* quadrotor mass,
which meant mass mismatch (sim-plant overrides, domain randomization) silently
rescaled the policy output and hid the effect of the mismatch. We now freeze
them at the *nominal* (training-time) mass so mass overrides only affect
simulator dynamics.

Configs expose two related keys:

  quadrotor.mass            : original Flightmare plant mass (may be
                              overridden for sim-plant mismatch runs)
  _nominal_action_mass      : explicit nominal mass for action scaling
                              (set by the sim-plant runner before overrides
                              are applied so action denormalization is
                              independent of the overridden plant mass)

Callers pass either the scenarios/compare config dict (``scfg``) or a nested
training config dict (``cfg``) whose ``env`` subtree contains
``quadrotor_dynamics``.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

DEFAULT_NOMINAL_MASS = 0.774
GRAVITY = 9.81


def _coerce_mass(value: Any) -> float:
    if value is None:
        return float(DEFAULT_NOMINAL_MASS)
    try:
        m = float(value)
    except (TypeError, ValueError):
        return float(DEFAULT_NOMINAL_MASS)
    if not np.isfinite(m) or m <= 0.0:
        return float(DEFAULT_NOMINAL_MASS)
    return m


def nominal_action_mass(scfg: Dict[str, Any]) -> float:
    """Return the nominal mass used for action denormalization.

    Resolution order:
      1. ``scfg["_nominal_action_mass"]`` (explicit, set by sim-plant runner)
      2. ``scfg["quadrotor"]["mass"]``    (scenarios / compare configs)
      3. ``scfg["env"]["quadrotor_dynamics"]["mass"]`` (training configs)
      4. ``DEFAULT_NOMINAL_MASS`` (0.774)
    """
    if not isinstance(scfg, dict):
        return float(DEFAULT_NOMINAL_MASS)
    if "_nominal_action_mass" in scfg:
        return _coerce_mass(scfg.get("_nominal_action_mass"))
    q = scfg.get("quadrotor")
    if isinstance(q, dict) and "mass" in q:
        return _coerce_mass(q.get("mass"))
    env = scfg.get("env")
    if isinstance(env, dict):
        qd = env.get("quadrotor_dynamics")
        if isinstance(qd, dict) and "mass" in qd:
            return _coerce_mass(qd.get("mass"))
    return float(DEFAULT_NOMINAL_MASS)


def nominal_act_scaling(
    scfg: Dict[str, Any],
    dtype: np.dtype = np.float64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(act_mean, act_std)`` vectors of length 4 pinned to nominal mass.

    Matches the C++ Flightmare convention: per-motor denormalization with
    ``act_mean = mass*g/4`` and ``act_std = mass*2*g/4``.
    """
    m = nominal_action_mass(scfg)
    act_mean = np.full(4, (m * GRAVITY) / 4.0, dtype=dtype)
    act_std = np.full(4, (m * 2.0 * GRAVITY) / 4.0, dtype=dtype)
    return act_mean, act_std


def flightmare_applied_thrusts(
    action: np.ndarray,
    act_mean: np.ndarray,
    act_std: np.ndarray,
) -> np.ndarray:
    """Replicate the per-motor thrust actually applied by the Flightmare env.

    Flightmare's step interface:
      1. clip the normalised policy action to ``[-1, 1]``,
      2. denormalise via ``thrust = act_std * action + act_mean``,
      3. clamp each motor thrust to ``>= 0`` (motors cannot push backwards).

    The CBF / MPC filter must reason about the thrust that the plant will
    actually see, not the raw (unclipped, possibly-negative) RL sample.
    Feeding the raw value to the CBF causes the QP to correct against a
    ghost thrust that Flightmare never applies, which in turn pushes the
    filtered action into a regime with no authority left in the plant.

    Parameters
    ----------
    action : array-like, shape (..., >=4)
        Normalised policy output; only the first 4 entries are used.
    act_mean, act_std : array-like, shape (4,)
        Nominal denormalisation constants (see ``nominal_act_scaling``).

    Returns
    -------
    np.ndarray, shape (4,), dtype float64
        Per-motor thrust in Newtons as actually applied by Flightmare.
    """
    a = np.asarray(action, dtype=np.float64).ravel()[:4]
    a = np.clip(a, -1.0, 1.0)
    mean = np.asarray(act_mean, dtype=np.float64).ravel()[:4]
    std = np.asarray(act_std, dtype=np.float64).ravel()[:4]
    u = std * a + mean
    return np.maximum(u, 0.0)


def effective_thrust_limits(scfg: Dict[str, Any]) -> Tuple[float, float]:
    """Return ``(thrust_min, thrust_max)`` per motor matching the Flightmare step interface.

    The Flightmare env clips normalised actions to ``[-1, 1]`` before the
    denormalisation ``thrust = act_std * action + act_mean``.  The resulting
    per-motor thrust range is therefore:

        thrust_min_eff = max(0.0, act_mean - act_std)   # action = -1, then C++ clamps ≥ 0
        thrust_max_eff = act_mean + act_std              # action = +1

    Using these limits in MPC and CBF ensures the optimiser only plans thrusts
    that can actually be delivered through the interface — the wider physical
    bounds (from the thrust-map formula ≈ 12.25 N) are unreachable because the
    normaliser clips first.

    At nominal mass 0.774 kg:
        act_mean = 0.774*9.81/4  ≈ 1.898 N
        act_std  = 0.774*9.81/2  ≈ 3.797 N
        → thrust ∈ [0.0 N, 5.695 N] per motor
    """
    m = nominal_action_mass(scfg)
    act_mean_val = (m * GRAVITY) / 4.0
    act_std_val = (m * 2.0 * GRAVITY) / 4.0
    t_min = max(0.0, act_mean_val - act_std_val)
    t_max = act_mean_val + act_std_val
    return (t_min, t_max)
