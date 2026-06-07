"""
CBF (Control Barrier Function) safety filter for quadrotor.
Implements Cheng et al. "End-to-End Safe RL through Barrier Functions" (AAAI 2019).

Velocity-aware barriers: h(p, v) = n'p + q + kv*(n'v) >= 0.

Discrete CBF (paper Eq 8/14): h_{t+1} >= (1-eta)*h_t - epsilon
  min ||u_CBF||^2 + K*epsilon  s.t. barrier + actuator constraints.
  eta in [0,1]: 0 = Lyapunov (barrier never decreases), 1 = just stay safe.
Also supports continuous-time: L_f h + L_g h u >= -alpha*h.

QP solver: acados (NONLINEAR_LS + GAUSS_NEWTON, same as MPC), osqp, or scipy.
"""
from pathlib import Path
import contextlib
import os
import sys
import warnings

_SCRIPT_DIR = Path(__file__).resolve().parent

# Suppress acados "AcadosOcpDims.N has been migrated to N_horizon" so it doesn't flood the terminal
warnings.filterwarnings("ignore", message=".*N_horizon.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*AcadosOcpDims.*", category=UserWarning)


@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily suppress stderr to hide acados migration/deprecation print spam."""
    fd = sys.stderr.fileno()
    with os.fdopen(os.dup(fd), "w") as old_stderr:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
        try:
            yield
        finally:
            os.dup2(old_stderr.fileno(), fd)


@contextlib.contextmanager
def _suppress_output():
    """Suppress both stdout and stderr (acados C code prints to both)."""
    out_fd = sys.stdout.fileno()
    err_fd = sys.stderr.fileno()
    with os.fdopen(os.dup(out_fd), "w") as old_out, \
         os.fdopen(os.dup(err_fd), "w") as old_err:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), out_fd)
            os.dup2(devnull.fileno(), err_fd)
        try:
            yield
        finally:
            os.dup2(old_out.fileno(), out_fd)
            os.dup2(old_err.fileno(), err_fd)
_REPO_ROOT = _SCRIPT_DIR.parent
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import List, Optional, Tuple, Union

import numpy as np
import yaml

from scripts.quadrotor_model import (
    POS,
    ATT,
    VEL,
    QuadrotorModel,
    R_from_q,
    quaternion_normalize,
    STATE_DIM,
    INPUT_DIM,
)

_DEFAULT_CBF_CONFIG_PATH = _REPO_ROOT / "configs" / "cbf_config.yaml"


def _load_cbf_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    path = Path(config_path) if config_path else _DEFAULT_CBF_CONFIG_PATH
    if not path.is_absolute():
        path = _REPO_ROOT / path
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "cbf" not in cfg:
        raise KeyError(f"Expected 'cbf' key in {path}")
    return cfg["cbf"]


class HOCBFBarrier:
    """
    Velocity-aware barrier: h(p, v) = n'p + q + kv * (n'v).
    Safety is h >= 0. As velocity toward the barrier (n'v < 0) increases, h decreases,
    so the filter intervenes earlier when approaching boundaries.
    """

    def __init__(self, n: np.ndarray, q: float, kv: float = 0.0, name: str = ""):
        self._n = np.asarray(n, dtype=np.float64).ravel()[:3]
        self._q = float(q)
        self._kv = float(kv)
        self._name = name or "barrier"

    def h(self, p: np.ndarray, v: np.ndarray) -> float:
        """
        Barrier value at position p (3,) and velocity v (3,).
        Safety is defined as h >= 0.
        As velocity toward the barrier (n'v < 0) increases, h decreases.
        """
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


class PositionBarrier(HOCBFBarrier):
    """Legacy position-only barrier: h(p, v) = n'p + q (kv=0)."""

    def __init__(self, n: np.ndarray, q: float, name: str = ""):
        super().__init__(n, q, kv=0.0, name=name)


def compute_position_barrier_lie_derivatives(
    model: QuadrotorModel,
    x: np.ndarray,
    barrier: "HOCBFBarrier",
) -> Tuple[float, np.ndarray]:
    """
    For h(p) = n'p + q: L_f h = n'v, L_g h = (1/m)*(n' R e_z)*[1,1,1,1].
    Returns (L_f h, L_g h) with L_g h shape (4,).
    """
    return compute_hocbf_derivatives(model, x, barrier)


def compute_hocbf_derivatives(
    model: QuadrotorModel,
    x: np.ndarray,
    barrier: "HOCBFBarrier",
) -> Tuple[float, np.ndarray]:
    """
    Velocity-aware barrier h(p,v) = n'p + q + kv*(n'v).
    h_dot = n'v + kv*(n'a), a = (1/m)*R*[0,0,F] + g.
    L_f h = n'v + kv*(n'g), L_g h = (kv/m)*(n' R e_z)*[1,1,1,1].
    For kv=0 (position-only): L_f h = n'v, L_g h = (1/m)*(n' R e_z)*[1,1,1,1].
    Returns (L_f h, L_g h) with L_g h shape (4,).
    """
    x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
    v = x[VEL]
    q = quaternion_normalize(x[ATT])
    R = R_from_q(q)
    n = barrier.n
    kv = barrier.kv
    g_vec = np.array([0.0, 0.0, model.gravity], dtype=np.float64)
    nRez = float(np.dot(n, R[:, 2]))
    if kv == 0:
        L_f_h = float(np.dot(n, v))
        L_g_h = (1.0 / model.mass) * nRez * np.ones(4, dtype=np.float64)
    else:
        L_f_h = float(np.dot(n, v)) + kv * float(np.dot(n, g_vec))
        L_g_h = (kv / model.mass) * nRez * np.ones(4, dtype=np.float64)
    return L_f_h, L_g_h


def _position_jacobian_wrt_u(
    model: QuadrotorModel,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    integrate: str = "euler",
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Jacobian of position p_next = step(x, u, dt)[POS] w.r.t. u at (x, u).
    Returns shape (3, 4): dp_next_du. Uses numerical differentiation.
    If model has motor lag, sets motor state to u before each step so linearization is at u.
    """
    x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
    u = np.asarray(u, dtype=np.float64).ravel()[:INPUT_DIM]
    if model.use_motor_lag:
        model.set_motor_state(u)
    x_next0 = model.step(x, u, dt, integrate=integrate)
    dp_du = np.zeros((3, INPUT_DIM), dtype=np.float64)
    for i in range(INPUT_DIM):
        u_plus = u.copy()
        u_plus[i] += eps
        if model.use_motor_lag:
            model.set_motor_state(u)
        x_next_plus = model.step(x, u_plus, dt, integrate=integrate)
        dp_du[:, i] = (x_next_plus[POS] - x_next0[POS]) / eps
    return dp_du


def _position_and_velocity_jacobian_wrt_u(
    model: QuadrotorModel,
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    integrate: str = "euler",
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Jacobians of p_next and v_next = step(x, u, dt)[POS], step(...)[VEL] w.r.t. u at (x, u).
    Returns (dp_next_du, dv_next_du), each shape (3, 4). Uses numerical differentiation.
    """
    x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
    u = np.asarray(u, dtype=np.float64).ravel()[:INPUT_DIM]
    if model.use_motor_lag:
        model.set_motor_state(u)
    x_next0 = model.step(x, u, dt, integrate=integrate)
    dp_du = np.zeros((3, INPUT_DIM), dtype=np.float64)
    dv_du = np.zeros((3, INPUT_DIM), dtype=np.float64)
    for i in range(INPUT_DIM):
        u_plus = u.copy()
        u_plus[i] += eps
        if model.use_motor_lag:
            model.set_motor_state(u)
        x_next_plus = model.step(x, u_plus, dt, integrate=integrate)
        dp_du[:, i] = (x_next_plus[POS] - x_next0[POS]) / eps
        dv_du[:, i] = (x_next_plus[VEL] - x_next0[VEL]) / eps
    return dp_du, dv_du


def compute_discrete_cbf_inequality_from_model_step(
    model: QuadrotorModel,
    x: np.ndarray,
    u_rl: np.ndarray,
    dt: float,
    barrier: "HOCBFBarrier",
    eta: float,
    integrate: str = "euler",
) -> Tuple[np.ndarray, float]:
    """
    Discrete CBF (Cheng et al. Eq 8): h_{t+1} >= (1-eta)*h_t.
    Uses the quadrotor model's exact discrete dynamics (model.step) and
    linearizes the constraint in u around u_rl for the QP.

    h_next(u) ≈ h_next_nom + L·(u - u_rl)  →  L·u_CBF >= (1-eta)*h_k - h_next_nom

    Returns (A_row, b_scalar) for QP: A_row @ u_CBF <= b_scalar.
    """
    x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
    u_rl = np.asarray(u_rl, dtype=np.float64).ravel()[:INPUT_DIM]
    n = barrier.n
    kv = barrier.kv
    h_k = barrier.h(x[POS], x[VEL])
    if model.use_motor_lag:
        model.set_motor_state(u_rl)
    x_next_nom = model.step(x, u_rl, dt, integrate=integrate)
    h_next_nom = barrier.h(x_next_nom[POS], x_next_nom[VEL])
    dp_next_du, dv_next_du = _position_and_velocity_jacobian_wrt_u(
        model, x, u_rl, dt, integrate=integrate
    )
    L = np.dot(n, dp_next_du).ravel() + kv * np.dot(n, dv_next_du).ravel()
    rhs = (1.0 - eta) * h_k - h_next_nom
    A_row = -np.asarray(L, dtype=np.float64)
    b_scalar = -float(rhs)
    return A_row, b_scalar


def compute_discrete_cbf_inequality(
    model: QuadrotorModel,
    x: np.ndarray,
    barrier: "HOCBFBarrier",
    u_rl: np.ndarray,
    dt: float,
    eta: float,
) -> Tuple[np.ndarray, float]:
    """
    Discrete CBF (Cheng et al. Eq 8) via Lie-derivative linearization:
      h_{t+1} ≈ h_t + L_f_h*dt + 0.5*dt^2*n'g + 0.5*dt^2*L_g_h*u_safe
    Require h_{t+1} >= (1-eta)*h_t:
      0.5*dt^2*L_g_h*u_cbf >= (1-eta)*h_t - h_t - L_f_h*dt - 0.5*dt^2*n'g - 0.5*dt^2*L_g_h*u_rl
    Returns (A_row, b_scalar) for QP: A_row @ u_CBF <= b_scalar.
    """
    x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
    n = barrier.n
    h_k = barrier.h(x[POS], x[VEL])
    L_f_h, L_g_h = compute_hocbf_derivatives(model, x, barrier)
    g_vec = np.array([0.0, 0.0, model.gravity], dtype=np.float64)
    n_dot_g = float(np.dot(n, g_vec))
    rhs_disc = (1.0 - eta) * h_k - h_k - L_f_h * dt - 0.5 * (dt ** 2) * n_dot_g
    scale = 0.5 * (dt ** 2)
    rhs_for_Lg_u = rhs_disc / scale
    rhs_cbf = rhs_for_Lg_u - float(np.dot(L_g_h, u_rl))
    A_row = -np.asarray(L_g_h, dtype=np.float64)
    b_scalar = -rhs_cbf
    return A_row, b_scalar


def solve_cbf_qp_osqp(
    H: np.ndarray,
    g: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_iter: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Solve QP min 0.5 x'Hx + g'x s.t. A_ineq x <= b_ineq, lb <= x <= ub. Returns x or None if infeasible."""
    global _last_qp_failure_reason
    if max_iter is None:
        max_iter = 4000
    try:
        import osqp
        from scipy import sparse as scipy_sparse
    except ImportError:
        _last_qp_failure_reason = "OSQP not installed"
        return None
    n = H.shape[0]
    # OSQP: min 0.5 x'Px + q'x, l <= Ax <= u. We have A_ineq x <= b_ineq and lb <= x <= ub.
    # Stack: A = [A_ineq; I], l = [-inf,...; lb], u = [b_ineq; ub].
    # OSQP expects scipy sparse matrices (no osqp.csc_matrix).
    P = scipy_sparse.csc_matrix(H)
    q_arr = np.asarray(g, dtype=np.float64)
    A_ineq = np.asarray(A_ineq, dtype=np.float64)
    if A_ineq.size == 0:
        A = np.eye(n, dtype=np.float64)
        l = np.asarray(lb, dtype=np.float64)
        u = np.asarray(ub, dtype=np.float64)
    else:
        A = np.vstack([A_ineq, np.eye(n)])
        l = np.concatenate([np.full(A_ineq.shape[0], -1e30), np.asarray(lb, dtype=np.float64)])
        u = np.concatenate([np.asarray(b_ineq, dtype=np.float64).ravel(), np.asarray(ub, dtype=np.float64)])
    A = scipy_sparse.csc_matrix(A)
    prob = osqp.OSQP()
    prob.setup(P, q_arr, A, l, u, verbose=False, max_iter=max_iter)
    res = prob.solve()
    if res.info.status in ("solved", "solved inaccurate"):
        return np.asarray(res.x, dtype=np.float64)
    extra = f", iter={getattr(res.info, 'iter', None)}, obj_val={getattr(res.info, 'obj_val', None)}"
    _last_qp_failure_reason = f"OSQP: status={res.info.status!r}{extra}"
    return None


def solve_cbf_qp_scipy(
    H: np.ndarray,
    g: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> Optional[np.ndarray]:
    """Solve QP with scipy.optimize.minimize (SLSQP) or quad_prog style."""
    global _last_qp_failure_reason
    from scipy.optimize import minimize

    n = H.shape[0]

    def obj(x):
        return 0.5 * float(x @ H @ x) + float(g @ x)

    # A_ineq x <= b_ineq  =>  -A_ineq x >= -b_ineq  for scipy (inequality is >= 0 for some interfaces).
    # scipy minimize: ineq: A_ineq @ x - b_ineq <= 0.
    constraints = []
    if A_ineq.size > 0:
        A_ineq = np.asarray(A_ineq, dtype=np.float64)
        b_ineq = np.asarray(b_ineq, dtype=np.float64).ravel()
        for i in range(A_ineq.shape[0]):
            a = A_ineq[i]
            b = b_ineq[i] if b_ineq.size > i else b_ineq.flat[0]
            constraints.append({"type": "ineq", "fun": lambda x, ai=a, bi=b: bi - np.dot(ai, x)})
    bounds = list(zip(np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)))
    x0 = np.clip(np.zeros(n), lb, ub)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        return np.asarray(res.x, dtype=np.float64)
    _last_qp_failure_reason = f"scipy SLSQP: {getattr(res, 'message', str(res))}"
    return None


# Module-level: last QP failure reason (set by solver on failure; cleared on success). For debug per-step display.
_last_qp_failure_reason: Optional[str] = None

# Module-level cache for acados solver (keyed by nh) to avoid regenerating every call.
_ACADOS_SOLVER_CACHE: dict = {}
_ACADOS_CACHE_DIR = _REPO_ROOT / "build" / "cbf_acados"


def _build_acados_cbf_solver(nh: int):
    """Build and cache acados OCP solver for CBF QP: min u'u s.t. A_ineq*u <= b_ineq, lb <= u <= ub."""
    if nh in _ACADOS_SOLVER_CACHE:
        return _ACADOS_SOLVER_CACHE[nh]
    try:
        import casadi as cs
        from acados_template import AcadosOcp, AcadosOcpSolver
    except ImportError:
        try:
            from acados import AcadosOcp, AcadosOcpSolver
            import casadi as cs
        except ImportError:
            return None
    nu = 4
    np_param = nh * 4 + nh  # A_ineq flat (nh*4) + b_ineq (nh)
    u = cs.SX.sym("u", nu)
    p = cs.SX.sym("p", np_param)
    b_vec = p[nh * nu:]
    # Build constraint row-by-row (avoids CasADi column-major reshape pitfall)
    con_h_rows = []
    for i in range(nh):
        a_row = p[i * nu: (i + 1) * nu]
        con_h_rows.append(cs.dot(a_row, u) - b_vec[i])
    con_h = cs.vertcat(*con_h_rows)

    x_dummy = cs.SX.sym("x_dummy")

    ocp = AcadosOcp()
    ocp.model.name = f"cbf_qp_nh{nh}"
    ocp.model.x = cs.vertcat(x_dummy)
    ocp.model.u = u
    ocp.model.p = p
    ocp.model.disc_dyn_expr = ocp.model.x  # x_next = x
    # With N=1, con_h_expr applies to intermediate stages (1..N-1) which is empty.
    # Use con_h_expr_0 for the initial (and only) shooting stage.
    ocp.model.con_h_expr_0 = con_h

    # NONLINEAR_LS cost: 0.5*(y-yref)'*W*(y-yref) with y=u, yref=0, W=2I
    # gives 0.5*u'*2I*u = ||u_cbf||^2  (same as MPC cost formulation)
    ocp.model.cost_y_expr = u
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.W = 2.0 * np.eye(nu)
    ocp.cost.yref = np.zeros(nu)
    # Terminal cost (dummy state, zero weight)
    ocp.model.cost_y_expr_e = cs.vertcat(x_dummy)
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = np.zeros((1, 1))
    ocp.cost.yref_e = np.zeros(1)

    ocp.dims.nh_0 = nh
    ocp.constraints.lh_0 = np.full(nh, -1e9)
    ocp.constraints.uh_0 = np.zeros(nh)
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.lbu = np.full(4, -1e10)
    ocp.constraints.ubu = np.full(4, 1e10)
    ocp.parameter_values = np.zeros(np_param)

    if hasattr(ocp.solver_options, "N_horizon"):
        ocp.solver_options.N_horizon = 1
    else:
        ocp.dims.N = 1
    ocp.solver_options.tf = 1.0
    ocp.solver_options.integrator_type = "DISCRETE"
    # All solver settings identical to MPC (mpc_controller.py)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.levenberg_marquardt = 0.0
    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.nlp_solver_tol_eq = 1e-6
    ocp.solver_options.nlp_solver_tol_ineq = 1e-6
    ocp.solver_options.nlp_solver_tol_comp = 1e-4

    _ACADOS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    code_export_dir = str(_ACADOS_CACHE_DIR / f"cbf_nh{nh}")
    if hasattr(ocp, 'code_gen_opts') and hasattr(ocp.code_gen_opts, 'code_export_directory'):
        ocp.code_gen_opts.code_export_directory = code_export_dir
    else:
        ocp.code_export_directory = code_export_dir
    json_path = os.path.join(code_export_dir, "acados_ocp.json")
    try:
        with _suppress_stderr():
            solver = AcadosOcpSolver(ocp, json_file=json_path, build=True, generate=True)
    except Exception:
        return None
    _ACADOS_SOLVER_CACHE[nh] = (solver, np_param)
    return _ACADOS_SOLVER_CACHE[nh]


def solve_cbf_qp_acados(
    H: np.ndarray,
    g: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> Optional[np.ndarray]:
    """Solve QP min 0.5 x'Hx + g'x s.t. A_ineq x <= b_ineq, lb <= x <= ub using acados (HPIPM). Returns x or None."""
    global _last_qp_failure_reason
    nh = A_ineq.shape[0] if A_ineq.size > 0 else 0
    n = H.shape[0]
    if nh == 0:
        return None
    cached = _build_acados_cbf_solver(nh)
    if cached is None:
        return None
    solver, np_param = cached
    p_val = np.concatenate([A_ineq.ravel(), np.asarray(b_ineq, dtype=np.float64).ravel()])
    with _suppress_output():
        solver.set(0, "p", p_val)
        solver.set(1, "p", p_val)
        solver.constraints_set(0, "lbu", lb)
        solver.constraints_set(0, "ubu", ub)
        solver.set(0, "x", np.array([0.0]))
        solver.set(0, "u", np.zeros(4))
        solver.set(1, "x", np.array([0.0]))
        status = solver.solve()
    if status != 0:
        _last_qp_failure_reason = f"acados: solver status={status}"
        return None
    with _suppress_output():
        u_cbf = solver.get(0, "u")
    return np.asarray(u_cbf, dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Acados-based slack CBF solver (extended variable [u_CBF; epsilon])
# ---------------------------------------------------------------------------
_ACADOS_SLACK_SOLVER_CACHE: dict = {}
_ACADOS_SLACK_CACHE_DIR = _REPO_ROOT / "build" / "cbf_acados_slack"
# When False, reuse existing generated acados slack solver .so without forcing a rebuild.
_CBF_ACADOS_SLACK_FORCE_REBUILD = True


def set_cbf_acados_slack_force_rebuild(force: bool) -> None:
    """If False, skip acados code generation/rebuild when the solver .so already exists."""
    global _CBF_ACADOS_SLACK_FORCE_REBUILD
    _CBF_ACADOS_SLACK_FORCE_REBUILD = bool(force)


def _build_acados_cbf_slack_solver(nh: int, K_lin: float, K_quad: float):
    """Build acados OCP solver for slack CBF QP (NONLINEAR_LS, same as MPC):
       min ||u_CBF||^2 + K_lin*sum(eps) + K_quad*sum(eps^2)
       s.t. A_ineq*u_CBF - epsilon <= b_ineq, epsilon >= 0, lb <= u_CBF <= ub.
    Extended decision variable: [u_CBF(4), epsilon(nh)].
    Implemented via completing the square in NONLINEAR_LS so GAUSS_NEWTON is valid."""
    if nh in _ACADOS_SLACK_SOLVER_CACHE:
        return _ACADOS_SLACK_SOLVER_CACHE[nh]
    try:
        import casadi as cs
        from acados_template import AcadosOcp, AcadosOcpSolver
    except ImportError:
        try:
            from acados import AcadosOcp, AcadosOcpSolver
            import casadi as cs
        except ImportError:
            return None

    nu = 4
    n_ext = nu + nh
    np_param = nh * nu + nh
    u_ext = cs.SX.sym("u_ext", n_ext)
    p = cs.SX.sym("p", np_param)
    b_vec = p[nh * nu:]
    u_cbf = u_ext[:nu]
    eps = u_ext[nu:]

    # Build constraint row-by-row (avoids CasADi column-major reshape pitfall)
    con_h_rows = []
    for i in range(nh):
        a_row = p[i * nu: (i + 1) * nu]
        con_h_rows.append(cs.dot(a_row, u_cbf) - eps[i] - b_vec[i])
    con_h = cs.vertcat(*con_h_rows)

    x_dummy = cs.SX.sym("x_dummy")

    ocp = AcadosOcp()
    ocp.model.name = f"cbf_slack_nh{nh}"
    ocp.model.x = cs.vertcat(x_dummy)
    ocp.model.u = u_ext
    ocp.model.p = p
    ocp.model.disc_dyn_expr = ocp.model.x
    # With N=1, use con_h_expr_0 for the initial (and only) shooting stage.
    ocp.model.con_h_expr_0 = con_h

    # NONLINEAR_LS cost: 0.5*(y-yref)'*W*(y-yref)
    # For u_cbf: W=2I, yref=0  →  ||u_cbf||^2
    # For eps:   complete the square to get K_lin*eps + K_quad*eps^2:
    #   K_quad*(eps + K_lin/(2*K_quad))^2 = K_quad*eps^2 + K_lin*eps + const
    #   → W_eps = 2*K_quad, yref_eps = -K_lin/(2*K_quad)
    # When K_quad=0, use a small value to approximate pure linear penalty.
    K_q_eff = max(K_quad, 1e-4)
    eps_ref = -K_lin / (2.0 * K_q_eff)
    ocp.model.cost_y_expr = u_ext
    ocp.cost.cost_type = "NONLINEAR_LS"
    W_diag = np.concatenate([2.0 * np.ones(nu), 2.0 * K_q_eff * np.ones(nh)])
    ocp.cost.W = np.diag(W_diag)
    yref = np.zeros(n_ext)
    yref[nu:] = eps_ref
    ocp.cost.yref = yref
    ocp.model.cost_y_expr_e = cs.vertcat(x_dummy)
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W_e = np.zeros((1, 1))
    ocp.cost.yref_e = np.zeros(1)

    ocp.dims.nh_0 = nh
    ocp.constraints.lh_0 = np.full(nh, -1e9)
    ocp.constraints.uh_0 = np.zeros(nh)
    ocp.constraints.idxbu = np.arange(n_ext)
    ocp.constraints.lbu = np.concatenate([np.full(nu, -1e10), np.zeros(nh)])
    ocp.constraints.ubu = np.full(n_ext, 1e10)
    ocp.parameter_values = np.zeros(np_param)

    if hasattr(ocp.solver_options, "N_horizon"):
        ocp.solver_options.N_horizon = 1
    else:
        ocp.dims.N = 1
    ocp.solver_options.tf = 1.0
    ocp.solver_options.integrator_type = "DISCRETE"
    # All solver settings identical to MPC (mpc_controller.py)
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.print_level = 0
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.levenberg_marquardt = 0.0
    ocp.solver_options.nlp_solver_tol_stat = 1e-6
    ocp.solver_options.nlp_solver_tol_eq = 1e-6
    ocp.solver_options.nlp_solver_tol_ineq = 1e-6
    ocp.solver_options.nlp_solver_tol_comp = 1e-4

    _ACADOS_SLACK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    code_export_dir = str(_ACADOS_SLACK_CACHE_DIR / f"cbf_slack_nh{nh}")
    if hasattr(ocp, 'code_gen_opts') and hasattr(ocp.code_gen_opts, 'code_export_directory'):
        ocp.code_gen_opts.code_export_directory = code_export_dir
    else:
        ocp.code_export_directory = code_export_dir
    json_path = os.path.join(code_export_dir, "acados_ocp.json")
    model_so_name = f"libacados_ocp_solver_cbf_slack_nh{nh}.so"
    so_path = os.path.join(code_export_dir, model_so_name)
    reuse = (not _CBF_ACADOS_SLACK_FORCE_REBUILD) and os.path.isfile(so_path)
    try:
        with _suppress_stderr():
            if reuse:
                solver = AcadosOcpSolver(ocp, json_file=json_path, build=False, generate=False)
            else:
                solver = AcadosOcpSolver(ocp, json_file=json_path, build=True, generate=True)
    except Exception:
        if reuse:
            try:
                with _suppress_stderr():
                    solver = AcadosOcpSolver(ocp, json_file=json_path, build=True, generate=True)
            except Exception:
                return None
        else:
            return None
    _ACADOS_SLACK_SOLVER_CACHE[nh] = (solver, np_param, n_ext)
    return _ACADOS_SLACK_SOLVER_CACHE[nh]


def solve_cbf_qp_acados_slack(
    A_ineq_rows: list,
    b_ineq_vals: list,
    lb_u: np.ndarray,
    ub_u: np.ndarray,
    K_lin: float,
    K_quad: float = 0.0,
) -> Optional[np.ndarray]:
    """Solve slack CBF QP via acados. Returns [u_CBF(4), epsilon(nh)] or None."""
    global _last_qp_failure_reason
    nh = len(A_ineq_rows)
    if nh == 0:
        return None
    cached = _build_acados_cbf_slack_solver(nh, K_lin, K_quad)
    if cached is None:
        return None
    solver, np_param, n_ext = cached
    nu = 4
    A_ineq = np.vstack(A_ineq_rows).astype(np.float64)
    b_ineq = np.asarray(b_ineq_vals, dtype=np.float64).ravel()
    p_val = np.concatenate([A_ineq.ravel(), b_ineq])
    # Box bounds: [lb_u; 0] to [ub_u; 1e10]
    lb_ext = np.concatenate([np.asarray(lb_u, dtype=np.float64), np.zeros(nh)])
    ub_ext = np.concatenate([np.asarray(ub_u, dtype=np.float64), np.full(nh, 1e10)])
    with _suppress_output():
        solver.set(0, "p", p_val)
        solver.set(1, "p", p_val)
        solver.constraints_set(0, "lbu", lb_ext)
        solver.constraints_set(0, "ubu", ub_ext)
        solver.set(0, "x", np.array([0.0]))
        solver.set(0, "u", np.zeros(n_ext))
        solver.set(1, "x", np.array([0.0]))
        status = solver.solve()
    if status != 0:
        _last_qp_failure_reason = f"acados slack: solver status={status}"
        return None
    with _suppress_output():
        x_sol = solver.get(0, "u")
    return np.asarray(x_sol, dtype=np.float64).ravel()


def solve_cbf_qp(
    H: np.ndarray,
    g: np.ndarray,
    A_ineq: np.ndarray,
    b_ineq: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    solver: str = "osqp",
    max_iter: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Dispatch QP to requested solver. Returns solution vector or None if infeasible/error."""
    global _last_qp_failure_reason
    _last_qp_failure_reason = None
    if solver == "acados":
        x = solve_cbf_qp_acados(H, g, A_ineq, b_ineq, lb, ub)
        if x is not None:
            return x
        solver = "osqp"
    if solver == "osqp":
        x = solve_cbf_qp_osqp(H, g, A_ineq, b_ineq, lb, ub, max_iter=max_iter)
    elif solver == "scipy":
        x = solve_cbf_qp_scipy(H, g, A_ineq, b_ineq, lb, ub)
    else:
        x = solve_cbf_qp_osqp(H, g, A_ineq, b_ineq, lb, ub, max_iter=max_iter)
    if x is not None:
        _last_qp_failure_reason = None
    return x


class CBFFilter:
    """
    CBF safety filter (Cheng et al. AAAI 2019): given state x and RL action u_RL,
    returns u_safe = u_RL + u_CBF that satisfies barrier conditions and actuator limits.
    Discrete CBF (Eq 8): h_{t+1} >= (1-eta)*h_t with fixed eta in [0,1].
    Also supports continuous-time CBF: L_f h + L_g h u >= -alpha*h.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        quadrotor_model_path: Optional[Union[str, Path]] = None,
    ):
        """
        quadrotor_model_path : if set, overrides cbf.quadrotor_model_path in the YAML
        (e.g. mismatched model for sim-vs-controller ablations while sim uses truth dynamics).
        """
        cfg = _load_cbf_config(config_path)
        self._alpha = float(cfg.get("alpha", 0.5))
        self._solver = str(cfg.get("solver", "osqp")).lower()
        self._discrete_cbf = bool(cfg.get("discrete_cbf", False))
        self._dt = float(cfg.get("dt", 0.02))
        self._eta = float(cfg.get("eta", 0.01))
        self._discrete_cbf_use_model_step = bool(cfg.get("discrete_cbf_use_model_step", True))
        integ = str(cfg.get("integrate", "euler")).lower()
        self._integrate = integ if integ in ("euler", "rk4") else "euler"
        quad_cfg = (
            quadrotor_model_path
            if quadrotor_model_path is not None
            else cfg.get("quadrotor_model_path")
        )
        self._model = QuadrotorModel(config_path=quad_cfg)
        r_uav = float(cfg.get("r_uav", 0.0))
        self._barriers: List[HOCBFBarrier] = []
        for b in cfg.get("position_barriers", []):
            n = np.array(b["n"], dtype=np.float64)
            q = float(b["q"]) - r_uav  # paper 3.1: effective bound uses q - r_uav
            kv = float(b.get("kv", 0.5))  # velocity-aware gain; 0.5 gives earlier intervention when approaching
            name = b.get("name", "")
            self._barriers.append(HOCBFBarrier(n, q, kv=kv, name=name))
        self._last_qp_failed = False
        self._use_slack = bool(cfg.get("use_slack", False))
        self._K_lin = float(cfg.get("K_lin", 1e6))
        self._K_quad = float(cfg.get("K_quad", 0.0))
        self._max_iter = int(cfg.get("max_iter", 4000))  # OSQP max iterations (when solver is osqp)
        self._last_u_cbf: Optional[np.ndarray] = None
        self._last_slack: Optional[dict] = None  # barrier name -> slack value (when use_slack)

    @property
    def model(self) -> QuadrotorModel:
        return self._model

    def set_thrust_limits(self, thrust_min: float, thrust_max: float) -> None:
        """Override per-motor thrust bounds on the internal QuadrotorModel.

        Call with ``effective_thrust_limits(scfg)`` from ``_action_scaling`` to
        align the CBF feasible set with the range actually deliverable through
        the Flightmare normalised-action interface.  The updated limits affect
        both the QP box constraints in ``filter()`` and ``clamp_motor_thrusts``.
        """
        self._model.set_thrust_limits(thrust_min, thrust_max)

    @property
    def barriers(self) -> List[HOCBFBarrier]:
        return list(self._barriers)

    @property
    def last_qp_failed(self) -> bool:
        """True if the last filter() call fell back to u_rl due to QP infeasibility."""
        return getattr(self, "_last_qp_failed", False)

    @property
    def last_u_cbf(self) -> Optional[np.ndarray]:
        """CBF correction u_CBF from last filter() call (None if QP failed)."""
        return getattr(self, "_last_u_cbf", None)

    @property
    def last_slack(self) -> Optional[dict]:
        """Slack values per barrier from last filter() (None if not use_slack or QP failed)."""
        return getattr(self, "_last_slack", None)

    @property
    def last_qp_failure_reason(self) -> Optional[str]:
        """Reason for last QP failure (None if last solve succeeded). For debug per-step display."""
        return _last_qp_failure_reason

    def filter(
        self,
        x: np.ndarray,
        u_rl: np.ndarray,
    ) -> np.ndarray:
        """
        Compute u_safe = u_RL + u_CBF where u_CBF minimizes ||u_CBF||^2 subject to
        L_f h + L_g h u_safe >= -alpha*h for each barrier and u_min <= u_safe <= u_max.
        x: state (13,) [p, q, v, omega]. u_rl: nominal motor thrusts (4,) [N].
        Returns u_safe (4,) [N].
        """
        x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
        u_rl = np.asarray(u_rl, dtype=np.float64).ravel()[:INPUT_DIM]
        u_min, u_max = self._model.get_thrust_limits()

        # QP in u_CBF: u_safe = u_rl + u_CBF, min ||u_CBF||^2
        # s.t. L_f h + L_g h (u_rl + u_CBF) >= -alpha*h  =>  L_g h u_CBF >= -alpha*h - L_f h - L_g h u_rl
        # and u_min <= u_rl + u_CBF <= u_max  =>  u_min - u_rl <= u_CBF <= u_max - u_rl.
        n_u = INPUT_DIM
        H = 2.0 * np.eye(n_u, dtype=np.float64)
        g = np.zeros(n_u, dtype=np.float64)

        ineq_list_A = []
        ineq_list_b = []
        for bar in self._barriers:
            if self._discrete_cbf:
                if self._discrete_cbf_use_model_step:
                    A_row, b_scalar = compute_discrete_cbf_inequality_from_model_step(
                        self._model, x, u_rl, self._dt, bar,
                        eta=self._eta,
                        integrate=self._integrate,
                    )
                else:
                    A_row, b_scalar = compute_discrete_cbf_inequality(
                        self._model, x, bar, u_rl,
                        self._dt, eta=self._eta,
                    )
                ineq_list_A.append(A_row)
                ineq_list_b.append(b_scalar)
            else:
                L_f_h, L_g_h = compute_hocbf_derivatives(self._model, x, bar)
                h_val = bar.h(x[POS], x[VEL])
                rhs = -self._alpha * h_val - L_f_h - float(np.dot(L_g_h, u_rl))
                # L_g h u_CBF >= rhs  =>  -L_g h u_CBF <= -rhs
                ineq_list_A.append(-np.asarray(L_g_h, dtype=np.float64))
                ineq_list_b.append(-rhs)

        lb_u = np.asarray(u_min - u_rl, dtype=np.float64)
        ub_u = np.asarray(u_max - u_rl, dtype=np.float64)

        if self._use_slack and ineq_list_A:
            nh = len(ineq_list_A)
            x_sol = None
            if self._solver == "acados":
                x_sol = solve_cbf_qp_acados_slack(ineq_list_A, ineq_list_b, lb_u, ub_u, self._K_lin, self._K_quad)
            if x_sol is None:
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
                    A_slack[j, n_u + j] = -1.0
                b_slack = np.array(ineq_list_b, dtype=np.float64)
                lb_slack = np.concatenate([lb_u, np.zeros(nh, dtype=np.float64)])
                ub_slack = np.concatenate([ub_u, np.full(nh, 1e10, dtype=np.float64)])
                x_sol = solve_cbf_qp(H_slack, g_slack, A_slack, b_slack, lb_slack, ub_slack, solver="osqp", max_iter=self._max_iter)
                if x_sol is None:
                    x_sol = solve_cbf_qp(H_slack, g_slack, A_slack, b_slack, lb_slack, ub_slack, solver="scipy")
            u_cbf = x_sol[:n_u].astype(np.float64) if x_sol is not None else None
            if u_cbf is not None:
                eps = x_sol[n_u:n_u + nh]
                self._last_slack = {b.name: float(e) for b, e in zip(self._barriers, eps)}
            else:
                self._last_slack = None
        else:
            A_ineq = np.vstack(ineq_list_A) if ineq_list_A else np.empty((0, n_u))
            b_ineq = np.array(ineq_list_b, dtype=np.float64)
            u_cbf = solve_cbf_qp(H, g, A_ineq, b_ineq, lb_u, ub_u, solver=self._solver, max_iter=self._max_iter)
            self._last_slack = None

        if u_cbf is None:
            self._last_u_cbf = None
            self._last_slack = None
            self._last_qp_failed = True
            return self._model.clamp_motor_thrusts(u_rl)
        self._last_u_cbf = u_cbf
        self._last_qp_failed = False
        u_safe = u_rl + u_cbf
        return u_safe


def _diagnose_acados_vs_osqp():
    """Compare acados and OSQP solutions on the same CBF QP for several states."""
    print("=" * 70)
    print("ACADOS vs OSQP CBF QP diagnosis")
    print("=" * 70)

    flt = CBFFilter()
    model = flt.model
    barriers = flt.barriers
    alpha = flt._alpha
    u_min, u_max = model.get_thrust_limits()
    hover = np.ones(4) * (model.mass * (-model.gravity) / 4.0)

    test_cases = [
        ("hover z=3 vz=0",    [0, 0, 3],  [0, 0, 0],    hover),
        ("hover z=5 vz=0",    [0, 0, 5],  [0, 0, 0],    hover),
        ("ascend z=5 vz=2",   [0, 0, 5],  [0, 0, 2],    hover * 1.3),
        ("ascend z=7 vz=1",   [0, 0, 7],  [0, 0, 1],    hover * 1.1),
        ("near ceil z=7.5 vz=0.5", [0, 0, 7.5], [0, 0, 0.5], hover * 1.05),
        ("at ceil z=8 vz=0",  [0, 0, 8],  [0, 0, 0],    hover),
        ("past ceil z=9 vz=0",[0, 0, 9],  [0, 0, 0],    hover),
    ]

    for label, pos, vel, u_rl in test_cases:
        x = np.zeros(STATE_DIM)
        x[ATT] = [1.0, 0.0, 0.0, 0.0]
        x[POS] = np.array(pos, dtype=np.float64)
        x[VEL] = np.array(vel, dtype=np.float64)
        u_rl = np.array(u_rl, dtype=np.float64)

        ineq_A, ineq_b = [], []
        for bar in barriers:
            L_f_h, L_g_h = compute_hocbf_derivatives(model, x, bar)
            h_val = bar.h(x[POS], x[VEL])
            rhs = -alpha * h_val - L_f_h - float(np.dot(L_g_h, u_rl))
            ineq_A.append(-np.asarray(L_g_h, dtype=np.float64))
            ineq_b.append(-rhs)

        A_ineq = np.vstack(ineq_A)
        b_ineq = np.array(ineq_b, dtype=np.float64)
        H = 2.0 * np.eye(4)
        g_vec = np.zeros(4)
        lb = np.asarray(u_min - u_rl, dtype=np.float64)
        ub = np.asarray(u_max - u_rl, dtype=np.float64)

        u_osqp = solve_cbf_qp_osqp(H, g_vec, A_ineq, b_ineq, lb, ub, max_iter=4000)
        u_acados = solve_cbf_qp_acados(H, g_vec, A_ineq, b_ineq, lb, ub)

        # Check constraint satisfaction
        def check_constraints(u_sol, name):
            if u_sol is None:
                return f"{name}: FAILED (None)"
            viol = A_ineq @ u_sol - b_ineq
            max_viol = np.max(viol)
            box_lo = np.min(u_sol - lb)
            box_hi = np.min(ub - u_sol)
            return (f"{name}: |u|={np.linalg.norm(u_sol):.5f}  "
                    f"u=[{u_sol[0]:+.4f},{u_sol[1]:+.4f},{u_sol[2]:+.4f},{u_sol[3]:+.4f}]  "
                    f"max_con_viol={max_viol:+.6f}  box_ok={box_lo >= -1e-6 and box_hi >= -1e-6}")

        h_ceil = barriers[1].h(x[POS], x[VEL])
        print(f"\n--- {label}  h_ceil={h_ceil:+.3f} ---")
        print(f"  {check_constraints(u_osqp, 'OSQP  ')}")
        print(f"  {check_constraints(u_acados, 'ACADOS')}")
        if u_osqp is not None and u_acados is not None:
            diff = np.linalg.norm(u_osqp - u_acados)
            print(f"  DIFF: {diff:.6f}")


def main() -> None:
    """Quick test: filter a random action and check barrier values."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnose":
        _diagnose_acados_vs_osqp()
        return

    flt = CBFFilter()
    x = np.zeros(STATE_DIM)
    x[ATT] = [1.0, 0.0, 0.0, 0.0]
    x[POS] = [0.0, 0.0, 5.0]
    x[VEL] = [0.0, 0.0, 0.0]
    u_rl = np.ones(4) * (flt.model.mass * (-flt.model.gravity) / 4.0)
    u_safe = flt.filter(x, u_rl)
    print("CBF filter test:")
    print("  u_rl (hover):", u_rl)
    print("  u_safe:", u_safe)
    for b in flt.barriers:
        print(f"  h({b._name}):", b.h(x[POS], x[VEL]))


if __name__ == "__main__":
    main()
