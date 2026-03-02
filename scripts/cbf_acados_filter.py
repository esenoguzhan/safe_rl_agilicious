"""
CBF safety filter using CasADi symbolic QP setup and acados (HPIPM) QP solver.

Same role as CBFFilter in cbf_filter.py: given state x and RL action u_rl,
solve min ||u_CBF||^2 s.t. discrete barrier constraints and actuator limits,
return u_safe = u_rl + u_CBF.

Barrier constraint computation reuses the same model.step()-based numerical
approach as the original CBFFilter, ensuring identical constraint matrices.
The QP is solved via acados (HPIPM) with an OSQP+slack fallback for
infeasible cases.

Usage:
    from scripts.cbf_acados_filter import AcadosCBFFilter
    flt = AcadosCBFFilter()                     # loads configs/cbf_config.yaml
    u_safe = flt.filter(x_state, u_rl)          # returns (4,) motor thrusts [N]
"""
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

warnings.filterwarnings("ignore", message=".*N_horizon.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*AcadosOcpDims.*", category=UserWarning)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.quadrotor_model import (
    POS, ATT, VEL, OME,
    QuadrotorModel,
    STATE_DIM, INPUT_DIM,
)
from scripts.cbf_filter import (
    HOCBFBarrier,
    compute_discrete_cbf_inequality_from_model_step,
    compute_discrete_cbf_inequality,
    compute_hocbf_derivatives,
    solve_cbf_qp,
    _load_cbf_config,
)
from scripts.mpc_controller import _suppress_stdout, _suppress_stderr


# ---------------------------------------------------------------------------
# Acados QP solver (CasADi-based setup, acados HPIPM backend)
# ---------------------------------------------------------------------------

_CODEGEN_DIR = _REPO_ROOT / "c_generated_code_cbf_acados"
_SOLVER_CACHE: dict = {}


def _get_acados_qp_solver(nh: int):
    """Build (or return cached) acados OCP solver for the CBF QP.

    Uses CasADi symbolic expressions to define the QP structure, compiled
    once via acados code generation.

    QP: min u'u  s.t.  A_ineq*u <= b_ineq,  lb <= u <= ub
    where A_ineq (nh x 4) and b_ineq (nh,) are passed as runtime parameters.
    """
    if nh in _SOLVER_CACHE:
        return _SOLVER_CACHE[nh]

    import casadi as cs
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

    nu = INPUT_DIM
    np_param = nh * nu + nh
    u = cs.SX.sym("u", nu)
    p = cs.SX.sym("p", np_param)

    A_flat = p[:nh * nu]
    b_vec = p[nh * nu:]
    A_mat = cs.reshape(A_flat, nh, nu)
    con_h = cs.mtimes(A_mat, u) - b_vec  # <= 0

    ocp = AcadosOcp()
    model = AcadosModel()
    model.name = f"cbf_acados_nh{nh}"
    model.x = cs.SX.sym("x_dummy", 1)
    model.u = u
    model.p = p
    model.disc_dyn_expr = model.x
    model.cost_expr_ext_cost = cs.dot(u, u)
    model.con_h_expr = con_h
    ocp.model = model

    ocp.dims.nh = nh
    ocp.constraints.lh = np.full(nh, -1e15)
    ocp.constraints.uh = np.zeros(nh)
    ocp.constraints.idxbu = np.arange(nu)
    ocp.constraints.lbu = np.full(nu, -1e10)
    ocp.constraints.ubu = np.full(nu, 1e10)
    ocp.parameter_values = np.zeros(np_param)

    ocp.solver_options.tf = 1.0
    try:
        ocp.solver_options.N_horizon = 1
    except AttributeError:
        ocp.dims.N = 1
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.nlp_solver_max_iter = 1
    ocp.solver_options.tol = 1e-8
    ocp.solver_options.print_level = 0

    _CODEGEN_DIR.mkdir(parents=True, exist_ok=True)
    code_dir = str(_CODEGEN_DIR / f"nh{nh}")
    json_file = os.path.join(code_dir, "acados_ocp.json")

    with _suppress_stdout():
        solver = AcadosOcpSolver(ocp, json_file=json_file)

    _SOLVER_CACHE[nh] = solver
    return solver


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AcadosCBFFilter:
    """
    CBF safety filter using acados (HPIPM) QP solver with CasADi-based setup.

    Same interface as CBFFilter: call filter(x, u_rl) to get u_safe.

    Barrier constraints are computed identically to CBFFilter (using
    model.step() and numerical Jacobians) to ensure matching behavior.
    The QP is solved via acados HPIPM, falling back to OSQP with slack
    variables when the hard QP is infeasible.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        barriers: Optional[List[dict]] = None,
    ):
        cfg = _load_cbf_config(config_path)

        self._alpha = float(cfg.get("alpha", 0.5))
        self._dt = float(cfg.get("dt", 0.02))
        self._discrete_cbf = bool(cfg.get("discrete_cbf", True))
        self._discrete_cbf_use_model_step = bool(cfg.get("discrete_cbf_use_model_step", True))
        self._gamma_min = float(cfg.get("gamma_min", 0.1))
        self._gamma_max = float(cfg.get("gamma_max", 0.8))
        self._sigma_param = float(cfg.get("sigma_param", 10.0))
        integ = str(cfg.get("integrate", "euler")).lower()
        self._integrate = integ if integ in ("euler", "rk4") else "euler"
        r_uav = float(cfg.get("r_uav", 0.0))

        self._use_slack = bool(cfg.get("use_slack", False))
        self._K_lin = float(cfg.get("K_lin", 1e6))
        self._max_iter = int(cfg.get("max_iter", 4000))

        quad_cfg_path = cfg.get("quadrotor_model_path")
        self._model = QuadrotorModel(config_path=quad_cfg_path)

        barrier_defs = barriers if barriers is not None else cfg.get("position_barriers", [])
        self._barriers: List[HOCBFBarrier] = []
        for b in barrier_defs:
            n = np.array(b["n"], dtype=np.float64)
            q = float(b["q"]) - r_uav
            kv = float(b.get("kv", 0.5))
            name = b.get("name", "")
            self._barriers.append(HOCBFBarrier(n, q, kv=kv, name=name))

        nh = len(self._barriers)
        self._nh = nh
        if nh > 0:
            self._acados_solver = _get_acados_qp_solver(nh)
        else:
            self._acados_solver = None

        self._last_qp_failed = False
        self._last_u_cbf: Optional[np.ndarray] = None

    # -- properties (match CBFFilter interface) --

    @property
    def model(self) -> QuadrotorModel:
        return self._model

    @property
    def barriers(self) -> List[HOCBFBarrier]:
        return list(self._barriers)

    @property
    def last_qp_failed(self) -> bool:
        return self._last_qp_failed

    @property
    def last_u_cbf(self) -> Optional[np.ndarray]:
        return self._last_u_cbf

    # -- filter --

    def filter(self, x: np.ndarray, u_rl: np.ndarray) -> np.ndarray:
        """
        Compute u_safe = u_rl + u_CBF.

        Barrier constraints are computed identically to the original CBFFilter
        (model.step() + numerical Jacobians). The QP is solved via acados
        (HPIPM), falling back to OSQP+slack if infeasible.
        """
        x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
        u_rl = np.asarray(u_rl, dtype=np.float64).ravel()[:INPUT_DIM]
        u_min, u_max = self._model.get_thrust_limits()

        if self._nh == 0:
            return self._model.clamp_motor_thrusts(u_rl)

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
                        self._gamma_min, self._gamma_max, self._sigma_param,
                        integrate=self._integrate,
                    )
                else:
                    A_row, b_scalar = compute_discrete_cbf_inequality(
                        self._model, x, bar, u_rl,
                        self._dt, self._gamma_min, self._gamma_max, self._sigma_param,
                    )
                ineq_list_A.append(A_row)
                ineq_list_b.append(b_scalar)
            else:
                L_f_h, L_g_h = compute_hocbf_derivatives(self._model, x, bar)
                h_val = bar.h(x[POS], x[VEL])
                rhs = -self._alpha * h_val - L_f_h - float(np.dot(L_g_h, u_rl))
                ineq_list_A.append(-np.asarray(L_g_h, dtype=np.float64))
                ineq_list_b.append(-rhs)

        lb_u = np.asarray(u_min - u_rl, dtype=np.float64)
        ub_u = np.asarray(u_max - u_rl, dtype=np.float64)

        A_ineq = np.vstack(ineq_list_A) if ineq_list_A else np.empty((0, n_u))
        b_ineq = np.array(ineq_list_b, dtype=np.float64)

        if self._use_slack and len(ineq_list_A) > 0:
            # When slack is enabled, use OSQP+slack as primary solver
            # (matches original CBFFilter which always uses OSQP for the
            # augmented slack QP, even when solver config says "acados").
            u_cbf = self._solve_osqp_slack(
                H, g, ineq_list_A, ineq_list_b, lb_u, ub_u,
            )
        else:
            # No slack: use acados HPIPM as primary solver
            u_cbf = self._solve_acados(A_ineq, b_ineq, lb_u, ub_u)
            # Fallback: OSQP without slack
            if u_cbf is None:
                u_cbf = solve_cbf_qp(
                    H, g, A_ineq, b_ineq, lb_u, ub_u,
                    solver="osqp", max_iter=self._max_iter,
                )

        if u_cbf is None:
            self._last_u_cbf = None
            self._last_qp_failed = True
            warnings.warn(
                "AcadosCBFFilter QP infeasible; using raw RL action."
            )
            return self._model.clamp_motor_thrusts(u_rl)

        self._last_u_cbf = u_cbf
        self._last_qp_failed = False
        return u_rl + u_cbf

    # -- QP solvers --

    def _solve_acados(
        self,
        A_ineq: np.ndarray,
        b_ineq: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Solve hard QP via acados HPIPM. Returns u_CBF or None."""
        if self._acados_solver is None:
            return None
        # CasADi reshape uses Fortran (column-major) ordering,
        # so A_ineq must be packed in column-major order.
        p_val = np.concatenate([A_ineq.ravel(order='F'), b_ineq.ravel()])
        with _suppress_stderr():
            self._acados_solver.set(0, "p", p_val)
            self._acados_solver.constraints_set(0, "lbu", lb.astype(np.float64))
            self._acados_solver.constraints_set(0, "ubu", ub.astype(np.float64))
            self._acados_solver.set(0, "x", np.array([0.0]))
            status = self._acados_solver.solve()
        if status != 0:
            return None
        with _suppress_stderr():
            u_cbf = self._acados_solver.get(0, "u")
        u_cbf = np.asarray(u_cbf, dtype=np.float64).ravel()
        # Verify constraint satisfaction (HPIPM may return status 0
        # with violated constraints for infeasible/near-infeasible QPs)
        violation = A_ineq @ u_cbf - b_ineq
        if np.any(violation > 1e-4):
            return None
        if np.any(u_cbf < lb - 1e-6) or np.any(u_cbf > ub + 1e-6):
            return None
        return u_cbf

    def _solve_osqp_slack(
        self,
        H: np.ndarray,
        g: np.ndarray,
        ineq_A_list: list,
        ineq_b_list: list,
        lb_u: np.ndarray,
        ub_u: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Solve augmented QP with slack variables via OSQP (matches original CBFFilter)."""
        n_u = INPUT_DIM
        nh = len(ineq_A_list)
        n_x = n_u + nh
        H_slack = np.zeros((n_x, n_x), dtype=np.float64)
        H_slack[:n_u, :n_u] = 2.0 * np.eye(n_u, dtype=np.float64)
        g_slack = np.zeros(n_x, dtype=np.float64)
        g_slack[n_u:] = self._K_lin
        A_slack = np.zeros((nh, n_x), dtype=np.float64)
        for j in range(nh):
            A_slack[j, :n_u] = ineq_A_list[j]
            A_slack[j, n_u + j] = -1.0
        b_slack = np.array(ineq_b_list, dtype=np.float64)
        lb_slack = np.concatenate([lb_u, np.zeros(nh, dtype=np.float64)])
        ub_slack = np.concatenate([ub_u, np.full(nh, 1e10, dtype=np.float64)])
        x_sol = solve_cbf_qp(
            H_slack, g_slack, A_slack, b_slack, lb_slack, ub_slack,
            solver="osqp", max_iter=self._max_iter,
        )
        if x_sol is None:
            x_sol = solve_cbf_qp(
                H_slack, g_slack, A_slack, b_slack, lb_slack, ub_slack,
                solver="scipy",
            )
        if x_sol is not None:
            return x_sol[:n_u].astype(np.float64)
        return None


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def main():
    print("Building AcadosCBFFilter ...")
    flt = AcadosCBFFilter()
    x = np.zeros(STATE_DIM)
    x[ATT] = [1.0, 0.0, 0.0, 0.0]
    x[POS] = [0.0, 0.0, 5.0]
    u_hover = np.ones(4) * (flt.model.mass * (-flt.model.gravity) / 4.0)

    u_safe = flt.filter(x, u_hover)
    print(f"  u_rl (hover): {u_hover}")
    print(f"  u_safe:       {u_safe}")
    print(f"  u_CBF:        {flt.last_u_cbf}")
    print(f"  QP failed:    {flt.last_qp_failed}")
    for b in flt.barriers:
        print(f"  h({b.name}): {b.h(x[POS], x[VEL]):.4f}")

    # Compare with original CBFFilter
    from scripts.cbf_filter import CBFFilter
    cbf_orig = CBFFilter()
    u_safe_orig = cbf_orig.filter(x, u_hover)
    print(f"\n  Original CBF u_safe:   {u_safe_orig}")
    print(f"  Original CBF u_CBF:    {cbf_orig.last_u_cbf}")
    print(f"  Match: {np.allclose(u_safe, u_safe_orig, atol=1e-6)}")
    if not np.allclose(u_safe, u_safe_orig, atol=1e-6):
        print(f"  Max diff: {np.max(np.abs(u_safe - u_safe_orig)):.2e}")
    print("Done.")


if __name__ == "__main__":
    main()
