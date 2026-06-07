"""
Derive axis-aligned z bounds for constrained MPC from CBF-style position barriers.

CBF uses h(p,v) = n'p + q + kv (n'v) >= 0 with q_eff = q - r_uav (Cheng et al. paper 3.1).
MPC only constrains z (ground / ceiling); x,y are not in the OCP state box.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _barriers_to_axis_bounds(barriers: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Same geometry as ``compare_scenarios._barriers_to_axis_bounds`` (single-axis n only)."""
    bounds: Dict[str, List[Optional[float]]] = {}
    axis_names = {0: "x", 1: "y", 2: "z"}
    for b in barriers:
        n = np.asarray(b["n"], dtype=np.float64).ravel()[:3]
        q = float(b["q"])
        nonzero = [i for i in range(3) if abs(n[i]) > 1e-10]
        if len(nonzero) != 1:
            continue
        idx = nonzero[0]
        boundary = -q / n[idx]
        name = axis_names[idx]
        bounds.setdefault(name, [None, None])
        if n[idx] > 0:
            lo = bounds[name][0]
            bounds[name][0] = boundary if lo is None else min(lo, boundary)
        else:
            hi = bounds[name][1]
            bounds[name][1] = boundary if hi is None else max(hi, boundary)
    return {k: tuple(v) for k, v in bounds.items()}


def barriers_to_mpc_z_interval(
    barriers: Optional[Sequence[Dict[str, Any]]],
    r_uav: float = 0.0,
    default_z: Tuple[float, float] = (0.0, 20.0),
) -> Tuple[float, float]:
    """
    (z_lo, z_hi) from axis-aligned *z* barriers, with CBF q_eff = q - r_uav.
    x/y barriers are ignored. Missing z from barriers uses default_z.
    """
    adj: List[Dict[str, Any]] = []
    for b in barriers or []:
        b2 = dict(b)
        b2["q"] = float(b["q"]) - float(r_uav)
        adj.append(b2)
    bnd = _barriers_to_axis_bounds(adj)
    zt = bnd.get("z", (None, None))
    z_lo, z_hi = zt[0], zt[1]
    if z_lo is None:
        z_lo = default_z[0]
    if z_hi is None:
        z_hi = default_z[1]
    return float(z_lo), float(z_hi)


def mpc_pos_vectors_from_z(z_lo: float, z_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    (3,) arrays for MPCController(pos_min, pos_max). Only index 2 is used as z bounds; x,y entries are 0
    and are not applied in the OCP.
    """
    pos_min = np.array([0.0, 0.0, z_lo], dtype=np.float64)
    pos_max = np.array([0.0, 0.0, z_hi], dtype=np.float64)
    return pos_min, pos_max
