"""
Minimal embedded CBF filter for the standalone ROS RL feedthrough package.

Self-contained subset of
``agiros/agiros/src/source_scripts/{quadrotor_model.py, cbf_filter.py}``.

Contains:
  - QuadrotorModel  (Flightmare-compatible 13D dynamics for CBF derivatives)
  - HOCBFBarrier    (velocity-aware half-plane: h(p, v) = n'p + q + kv*(n'v))
  - CBFFilter       (continuous-time CBF QP, OSQP / SciPy backends, slack)

Intentionally omitted from the source modules (to keep deps light and the
package self-contained):
  - Discrete CBF formulation and its numerical Jacobian helpers
  - acados solver backend (and the cached code-generation cache dirs)
  - Diagnostic / probe entrypoints

If you need any of the omitted features, copy the full
``cbf_filter.py`` + ``quadrotor_model.py`` here and swap the imports in
``rl_cbf_feedthrough_core.py``.

Math:
  - Continuous-time CBF (Cheng et al. AAAI 2019):
        L_f h + L_g h * u_safe >= -alpha * h
    For barrier h(p, v) = n'p + q + kv*(n'v) and dynamics with mass m,
    body-to-world rotation R(q), gravity g and per-rotor thrust u:
        L_f h = n'v + kv * (n'g)
        L_g h = (kv / m) * (n' R e_z) * [1, 1, 1, 1]
    (When kv = 0 we recover the position-only h(p) = n'p + q:
        L_f h = n'v, L_g h = (1/m) * (n' R e_z) * [1, 1, 1, 1].)
  - QP (no slack):
        min ||u_cbf||^2   s.t.   L_g h * u_cbf >= rhs   and   lb <= u_cbf <= ub
    with u_safe = u_rl + u_cbf, lb = u_min - u_rl, ub = u_max - u_rl,
    rhs = -alpha * h - L_f h - L_g h * u_rl.
  - QP (slack):
        min ||u_cbf||^2 + K_lin * sum(eps) + K_quad * sum(eps^2)
        s.t.  L_g h * u_cbf + eps >= rhs,  eps >= 0,  lb <= u_cbf <= ub
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import yaml
except ImportError as _e:  # pragma: no cover
    yaml = None  # type: ignore
    _YAML_IMPORT_ERROR: Optional[ImportError] = _e
else:
    _YAML_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

# State indices (Flightmare QuadState compatible)
POS = slice(0, 3)
ATT = slice(3, 7)  # quaternion (w, x, y, z)
VEL = slice(7, 10)
OME = slice(10, 13)
STATE_DIM = 13
INPUT_DIM = 4

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_QUADROTOR_MODEL_CONFIG = _SCRIPT_DIR / "quadrotor_model.yaml"
_DEFAULT_CBF_CONFIG = _SCRIPT_DIR / "cbf_config.yaml"


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (Flightmare convention: q = w, x, y, z)
# ---------------------------------------------------------------------------
def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Unit quaternion with the Flightmare convention w >= 0."""
    q = np.asarray(q, dtype=np.float64).ravel()
    n = float(np.linalg.norm(q))
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / n
    if q[0] < 0.0:
        q = -q
    return q


def R_from_q(q: np.ndarray) -> np.ndarray:
    """Rotation matrix body -> world from unit quaternion (w, x, y, z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array(
        [
            [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y)],
            [    2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x)],
            [    2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def _require_yaml() -> None:
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load CBF / quadrotor model configs. "
            f"Install with `pip install pyyaml`. Original error: {_YAML_IMPORT_ERROR}"
        )


def _load_yaml_section(
    config_path: Optional[Union[str, Path]],
    default_path: Path,
    section: str,
) -> dict:
    _require_yaml()
    path = Path(config_path) if config_path else default_path
    if not path.is_absolute():
        path = _SCRIPT_DIR / path
    if not path.is_file():
        raise FileNotFoundError(
            f"Could not find config file {path}. "
            f"Pass an explicit path or place '{default_path.name}' next to {__file__}."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if section not in cfg:
        raise KeyError(f"Expected '{section}' top-level key in {path}, got {list(cfg)}")
    return cfg[section]


# ---------------------------------------------------------------------------
# QuadrotorModel: 13D dynamics with motor lag option (used for CBF Lie derivs)
# ---------------------------------------------------------------------------
class QuadrotorModel:
    """
    Quadrotor dynamics model: state [p, q, v, omega] (13), control = 4 motor thrusts [N].
    The CBF only uses :py:meth:`get_thrust_limits`, mass, gravity and the body-to-world
    rotation, but the full dynamics are kept here so the same model can drive
    other diagnostics (forward simulation, allocation sanity checks, etc.).
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        params = _load_yaml_section(
            config_path, _DEFAULT_QUADROTOR_MODEL_CONFIG, "quadrotor_model"
        )

        self._mass = float(params["mass"])
        self._arm_l = float(params["arm_l"])
        inertia_scale = np.array(
            params.get("inertia_scale", [4.5, 4.5, 7.0]), dtype=np.float64
        )
        self._J = (self._mass / 12.0) * (self._arm_l ** 2) * np.diag(inertia_scale)
        self._J_inv = np.linalg.inv(self._J)

        self._motor_omega_min = float(params["motor_omega_min"])
        self._motor_omega_max = float(params["motor_omega_max"])
        self._motor_tau = float(params["motor_tau"])
        self._use_motor_lag = bool(params.get("use_motor_lag", False))
        self._thrust_map = np.array(params["thrust_map"], dtype=np.float64)
        self._kappa = float(params["kappa"])
        self._omega_max = np.array(params["omega_max"], dtype=np.float64)
        self._gravity = float(params.get("gravity", -9.81))
        self._gz = np.array([0.0, 0.0, self._gravity], dtype=np.float64)

        a, b, c = self._thrust_map[0], self._thrust_map[1], self._thrust_map[2]
        self._thrust_min = 0.0
        self._thrust_max = float(
            a * (self._motor_omega_max ** 2) + b * self._motor_omega_max + c
        )
        if self._thrust_max < self._thrust_min:
            self._thrust_max = self._thrust_min + 1e-6

        sqrt_half = float(np.sqrt(0.5))
        t_BM = self._arm_l * sqrt_half * np.array(
            [
                [1, -1, -1, 1],
                [-1, -1, 1, 1],
                [0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        k_row = self._kappa * np.array([[1, -1, 1, -1]], dtype=np.float64)
        self._B = np.vstack([np.ones((1, 4), dtype=np.float64), t_BM[:2, :], k_row])
        self._B_inv = np.linalg.inv(self._B)

        self._u_motor: Optional[np.ndarray] = None

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def gravity(self) -> float:
        return self._gravity

    def get_thrust_limits(self) -> Tuple[float, float]:
        return (self._thrust_min, self._thrust_max)

    def set_thrust_limits(self, thrust_min: float, thrust_max: float) -> None:
        if thrust_max <= thrust_min:
            raise ValueError(
                f"thrust_max ({thrust_max}) must be > thrust_min ({thrust_min})"
            )
        self._thrust_min = float(thrust_min)
        self._thrust_max = float(thrust_max)

    def clamp_motor_thrusts(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64).ravel()[:INPUT_DIM]
        return np.clip(u, self._thrust_min, self._thrust_max)

    def get_allocation_matrix(self) -> np.ndarray:
        return self._B.copy()


# ---------------------------------------------------------------------------
# Barrier + Lie derivatives
# ---------------------------------------------------------------------------
class HOCBFBarrier:
    """
    Velocity-aware half-plane barrier: h(p, v) = n'p + q + kv * (n'v).
    Safety set is {h >= 0}. As approach speed (n'v < 0) grows, h shrinks,
    so the filter intervenes earlier when racing toward the boundary.
    """

    def __init__(self, n: np.ndarray, q: float, kv: float = 0.0, name: str = ""):
        self._n = np.asarray(n, dtype=np.float64).ravel()[:3]
        self._q = float(q)
        self._kv = float(kv)
        self._name = name or "barrier"

    def h(self, p: np.ndarray, v: np.ndarray) -> float:
        p = np.asarray(p, dtype=np.float64).ravel()[:3]
        v = np.asarray(v, dtype=np.float64).ravel()[:3]
        dist = float(np.dot(self._n, p) + self._q)
        approach_speed = float(np.dot(self._n, v))
        return dist + self._kv * approach_speed

    @property
    def n(self) -> np.ndarray:
        return self._n.copy()

    @property
    def kv(self) -> float:
        return self._kv

    @property
    def name(self) -> str:
        return self._name


def compute_hocbf_derivatives(
    model: QuadrotorModel,
    x: np.ndarray,
    barrier: HOCBFBarrier,
) -> Tuple[float, np.ndarray]:
    """
    Lie derivatives of h(p, v) = n'p + q + kv*(n'v) under the per-rotor-thrust
    quadrotor dynamics. Returns (L_f h, L_g h) with L_g h shape (4,).
    """
    x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
    v = x[VEL]
    q = quaternion_normalize(x[ATT])
    R = R_from_q(q)
    n = barrier.n
    kv = barrier.kv
    g_vec = np.array([0.0, 0.0, model.gravity], dtype=np.float64)
    nRez = float(np.dot(n, R[:, 2]))
    if kv == 0.0:
        L_f_h = float(np.dot(n, v))
        L_g_h = (1.0 / model.mass) * nRez * np.ones(4, dtype=np.float64)
    else:
        L_f_h = float(np.dot(n, v)) + kv * float(np.dot(n, g_vec))
        L_g_h = (kv / model.mass) * nRez * np.ones(4, dtype=np.float64)
    return L_f_h, L_g_h


# ---------------------------------------------------------------------------
# QP solvers
# ---------------------------------------------------------------------------
_last_qp_failure_reason: Optional[str] = None


def _solve_qp_osqp(
    H: np.ndarray,
    g: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_iter: int = 4000,
) -> Optional[np.ndarray]:
    """min 0.5 x'Hx + g'x  s.t. A_ineq x <= b_ineq, lb <= x <= ub. Returns x or None."""
    global _last_qp_failure_reason
    try:
        import osqp
        from scipy import sparse as scipy_sparse
    except ImportError:
        _last_qp_failure_reason = "OSQP / scipy not installed"
        return None
    n = H.shape[0]
    P = scipy_sparse.csc_matrix(H)
    q_arr = np.asarray(g, dtype=np.float64)
    A_ineq = np.asarray(A_ineq, dtype=np.float64)
    if A_ineq.size == 0:
        A = np.eye(n, dtype=np.float64)
        l = np.asarray(lb, dtype=np.float64)
        u = np.asarray(ub, dtype=np.float64)
    else:
        A = np.vstack([A_ineq, np.eye(n)])
        l = np.concatenate(
            [np.full(A_ineq.shape[0], -1e30), np.asarray(lb, dtype=np.float64)]
        )
        u = np.concatenate(
            [np.asarray(b_ineq, dtype=np.float64).ravel(), np.asarray(ub, dtype=np.float64)]
        )
    A = scipy_sparse.csc_matrix(A)
    prob = osqp.OSQP()
    prob.setup(P, q_arr, A, l, u, verbose=False, max_iter=int(max_iter))
    res = prob.solve()
    if res.info.status in ("solved", "solved inaccurate"):
        return np.asarray(res.x, dtype=np.float64)
    _last_qp_failure_reason = (
        f"OSQP: status={res.info.status!r}, iter={getattr(res.info, 'iter', None)}"
    )
    return None


def _solve_qp_scipy(
    H: np.ndarray,
    g: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> Optional[np.ndarray]:
    """SLSQP fallback. Slow but always available."""
    global _last_qp_failure_reason
    try:
        from scipy.optimize import minimize
    except ImportError:
        _last_qp_failure_reason = "scipy not installed"
        return None
    n = H.shape[0]

    def obj(x):
        return 0.5 * float(x @ H @ x) + float(g @ x)

    constraints = []
    if A_ineq.size > 0:
        A_ineq = np.asarray(A_ineq, dtype=np.float64)
        b_ineq = np.asarray(b_ineq, dtype=np.float64).ravel()
        for i in range(A_ineq.shape[0]):
            a = A_ineq[i]
            b = b_ineq[i] if b_ineq.size > i else b_ineq.flat[0]
            constraints.append(
                {"type": "ineq", "fun": lambda x, ai=a, bi=b: bi - np.dot(ai, x)}
            )
    bounds = list(zip(np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)))
    x0 = np.clip(np.zeros(n), lb, ub)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        return np.asarray(res.x, dtype=np.float64)
    _last_qp_failure_reason = f"scipy SLSQP: {getattr(res, 'message', str(res))}"
    return None


def solve_cbf_qp(
    H: np.ndarray,
    g: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    solver: str = "osqp",
    max_iter: int = 4000,
) -> Optional[np.ndarray]:
    """Dispatch to OSQP / SciPy. Falls back to SciPy if OSQP fails."""
    global _last_qp_failure_reason
    _last_qp_failure_reason = None
    solver = solver.lower()
    if solver in ("acados", "osqp"):
        if solver == "acados":
            logger.warning(
                "cbf_core_lib: 'acados' backend is not bundled here. "
                "Falling back to OSQP. (Use the full cbf_filter.py for acados.)"
            )
        x = _solve_qp_osqp(H, g, A_ineq, b_ineq, lb, ub, max_iter=max_iter)
        if x is not None:
            _last_qp_failure_reason = None
            return x
        x = _solve_qp_scipy(H, g, A_ineq, b_ineq, lb, ub)
        if x is not None:
            _last_qp_failure_reason = None
        return x
    if solver == "scipy":
        x = _solve_qp_scipy(H, g, A_ineq, b_ineq, lb, ub)
        if x is not None:
            _last_qp_failure_reason = None
        return x
    raise ValueError(f"Unknown CBF QP solver {solver!r}; use 'osqp' or 'scipy'.")


# ---------------------------------------------------------------------------
# CBFFilter
# ---------------------------------------------------------------------------
class CBFFilter:
    """
    Continuous-time CBF safety filter.

    For each barrier i:  L_f h_i + L_g h_i * u_safe >= -alpha * h_i
    where u_safe = u_rl + u_cbf. The QP minimises ||u_cbf||^2 (with optional
    slack on the barrier inequality when ``use_slack``) subject to per-motor
    thrust bounds [u_min, u_max] (set via :py:meth:`set_thrust_limits` to
    match the deployment's action affine).

    State convention: x = [p(3), q(4 wxyz), v(3 world), omega(3 body)] (13).
    Inputs: u_rl, u_safe are per-rotor thrusts in Newtons.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        quadrotor_model_config_path: Optional[Union[str, Path]] = None,
    ):
        cfg = _load_yaml_section(config_path, _DEFAULT_CBF_CONFIG, "cbf")

        self._alpha = float(cfg.get("alpha", 0.5))
        # Honour discrete_cbf flag by warning + falling back to continuous-time
        # (the embedded backend only supports the continuous-time formulation).
        if bool(cfg.get("discrete_cbf", False)):
            logger.warning(
                "cbf_core_lib: discrete_cbf=true requested, but the embedded "
                "backend is continuous-time only. Using continuous-time CBF."
            )

        solver = str(cfg.get("solver", "osqp")).lower()
        if solver == "acados":
            logger.warning(
                "cbf_core_lib: solver='acados' requested, but acados is not "
                "bundled here. Using OSQP instead."
            )
            solver = "osqp"
        if solver not in ("osqp", "scipy"):
            raise ValueError(f"Unknown CBF solver {solver!r}")
        self._solver = solver

        # Surface a clear up-front warning if the requested backend isn't
        # importable; otherwise the user only finds out via a silent QP
        # failure the first time a barrier becomes active.
        if solver == "osqp":
            try:
                import osqp  # noqa: F401
                import scipy.sparse  # noqa: F401
            except ImportError as _imp_err:
                logger.warning(
                    "cbf_core_lib: solver='osqp' requested but `osqp` "
                    "(and/or scipy.sparse) is not installed (%s). The filter "
                    "will silently fall back to SLSQP, which is unreliable on "
                    "slack QPs (K_lin=%g). Install with `pip install osqp scipy`.",
                    _imp_err, self._K_lin,
                )
        if bool(cfg.get("use_slack", False)):
            # SLSQP can choke on the K_lin/K_quad scale mix; warn if that's
            # all we have available.
            try:
                import osqp  # noqa: F401
            except ImportError:
                if self._K_lin > 1e3:
                    logger.warning(
                        "cbf_core_lib: use_slack=true with K_lin=%g but OSQP "
                        "missing -- expect 'Positive directional derivative "
                        "for linesearch' SLSQP failures on active barriers.",
                        self._K_lin,
                    )

        self._use_slack = bool(cfg.get("use_slack", False))
        self._K_lin = float(cfg.get("K_lin", 1.0e6))
        self._K_quad = float(cfg.get("K_quad", 0.0))
        self._max_iter = int(cfg.get("max_iter", 4000))

        quad_cfg = (
            quadrotor_model_config_path
            if quadrotor_model_config_path is not None
            else cfg.get("quadrotor_model_path")
        )
        self._model = QuadrotorModel(config_path=quad_cfg)

        r_uav = float(cfg.get("r_uav", 0.0))
        self._barriers: List[HOCBFBarrier] = []
        for b in cfg.get("position_barriers", []):
            n = np.asarray(b["n"], dtype=np.float64)
            q = float(b["q"]) - r_uav  # paper Sec 3.1: effective bound is q - r_uav
            kv = float(b.get("kv", 0.0))
            name = str(b.get("name", "")) or f"barrier_{len(self._barriers)}"
            self._barriers.append(HOCBFBarrier(n, q, kv=kv, name=name))

        if not self._barriers:
            logger.warning("cbf_core_lib: no position_barriers configured!")

        self._last_qp_failed = False
        self._last_u_cbf: Optional[np.ndarray] = None
        self._last_slack: Optional[dict] = None

    # ------------------------------------------------------------------ getters
    @property
    def model(self) -> QuadrotorModel:
        return self._model

    @property
    def barriers(self) -> List[HOCBFBarrier]:
        return list(self._barriers)

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def use_slack(self) -> bool:
        return self._use_slack

    @property
    def last_qp_failed(self) -> bool:
        return self._last_qp_failed

    @property
    def last_u_cbf(self) -> Optional[np.ndarray]:
        return None if self._last_u_cbf is None else self._last_u_cbf.copy()

    @property
    def last_slack(self) -> Optional[dict]:
        return None if self._last_slack is None else dict(self._last_slack)

    @property
    def last_qp_failure_reason(self) -> Optional[str]:
        return _last_qp_failure_reason

    def set_thrust_limits(self, thrust_min: float, thrust_max: float) -> None:
        """Align CBF feasible set with what the deployment can actually command."""
        self._model.set_thrust_limits(thrust_min, thrust_max)

    # ----------------------------------------------------------------- filter
    def filter(
        self,
        x: np.ndarray,
        u_rl: np.ndarray,
    ) -> np.ndarray:
        """
        Returns u_safe = u_rl + u_cbf (4,) [N] satisfying barrier inequalities
        and per-motor thrust bounds. On QP infeasibility, falls back to the
        unclamped u_rl projected onto the box (and sets ``last_qp_failed``).
        """
        x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
        u_rl = np.asarray(u_rl, dtype=np.float64).ravel()[:INPUT_DIM]
        u_min, u_max = self._model.get_thrust_limits()
        n_u = INPUT_DIM

        ineq_list_A: List[np.ndarray] = []
        ineq_list_b: List[float] = []
        for bar in self._barriers:
            L_f_h, L_g_h = compute_hocbf_derivatives(self._model, x, bar)
            h_val = bar.h(x[POS], x[VEL])
            # Constraint: L_g_h u_cbf >= -alpha*h - L_f_h - L_g_h u_rl
            #             -L_g_h u_cbf <= +alpha*h + L_f_h + L_g_h u_rl
            rhs = -self._alpha * h_val - L_f_h - float(np.dot(L_g_h, u_rl))
            ineq_list_A.append(-np.asarray(L_g_h, dtype=np.float64))
            ineq_list_b.append(-rhs)

        lb_u = np.asarray(u_min - u_rl, dtype=np.float64)
        ub_u = np.asarray(u_max - u_rl, dtype=np.float64)

        if self._use_slack and ineq_list_A:
            nh = len(ineq_list_A)
            n_x = n_u + nh
            H_slack = np.zeros((n_x, n_x), dtype=np.float64)
            H_slack[:n_u, :n_u] = 2.0 * np.eye(n_u, dtype=np.float64)
            for j in range(nh):
                H_slack[n_u + j, n_u + j] = 2.0 * self._K_quad
            g_slack = np.zeros(n_x, dtype=np.float64)
            g_slack[n_u:] = self._K_lin
            A_slack = np.zeros((nh, n_x), dtype=np.float64)
            for j in range(nh):
                A_slack[j, :n_u] = ineq_list_A[j]
                A_slack[j, n_u + j] = -1.0  # row j: A_j u_cbf - eps_j <= b_j
            b_slack = np.asarray(ineq_list_b, dtype=np.float64)
            lb_slack = np.concatenate([lb_u, np.zeros(nh, dtype=np.float64)])
            ub_slack = np.concatenate([ub_u, np.full(nh, 1e10, dtype=np.float64)])
            x_sol = solve_cbf_qp(
                H_slack, g_slack, A_slack, b_slack, lb_slack, ub_slack,
                solver=self._solver, max_iter=self._max_iter,
            )
            if x_sol is None:
                u_cbf = None
            else:
                u_cbf = x_sol[:n_u].astype(np.float64)
                eps = x_sol[n_u:n_u + nh]
                self._last_slack = {
                    b.name: float(e) for b, e in zip(self._barriers, eps)
                }
        else:
            H = 2.0 * np.eye(n_u, dtype=np.float64)
            g = np.zeros(n_u, dtype=np.float64)
            A_ineq = (
                np.vstack(ineq_list_A) if ineq_list_A else np.empty((0, n_u))
            )
            b_ineq = np.asarray(ineq_list_b, dtype=np.float64)
            u_cbf = solve_cbf_qp(
                H, g, A_ineq, b_ineq, lb_u, ub_u,
                solver=self._solver, max_iter=self._max_iter,
            )
            self._last_slack = None

        if u_cbf is None:
            self._last_u_cbf = None
            self._last_slack = None
            self._last_qp_failed = True
            # Project u_rl onto the box to at least keep motors physical.
            return self._model.clamp_motor_thrusts(u_rl)

        self._last_u_cbf = np.asarray(u_cbf, dtype=np.float64)
        self._last_qp_failed = False
        return u_rl + self._last_u_cbf

    # ----------------------------------------------------- diagnostic helpers
    def barrier_values(self, x: np.ndarray) -> dict:
        """{barrier_name: h(p, v)} for the given state. Useful for logging."""
        x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
        p = x[POS]
        v = x[VEL]
        return {b.name: float(b.h(p, v)) for b in self._barriers}

    def barrier_normal_velocities(self, x: np.ndarray) -> dict:
        """
        {barrier_name: n . v}  -- signed approach speed along the barrier
        normal. Negative => moving toward the barrier (h shrinks faster when
        kv > 0). Useful to diagnose why h is decreasing.
        """
        x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
        v = x[VEL]
        return {b.name: float(np.dot(b.n, v)) for b in self._barriers}


__all__ = [
    "POS", "ATT", "VEL", "OME", "STATE_DIM", "INPUT_DIM",
    "QuadrotorModel",
    "HOCBFBarrier",
    "compute_hocbf_derivatives",
    "CBFFilter",
    "solve_cbf_qp",
    "quaternion_normalize",
    "R_from_q",
]
