"""
Nonlinear Model Predictive Controller for quadrotor goal tracking.
Uses CasADi for symbolic dynamics and acados for efficient OCP solving.

Dynamics match scripts/quadrotor_model.py exactly:
  State:   x = [p(3), q(4), v(3), omega(3)]  = 13 dims
  Control: u = motor thrusts (4) in [N]

Physical parameters are loaded from configs/quadrotor_model.yaml (shared with CBF).

Usage:
    from scripts.mpc_controller import MPCController
    mpc = MPCController()                         # loads configs/mpc_config.yaml
    u_opt = mpc.solve(x_current, goal_pos)        # returns 4 motor thrusts [N]
"""
import contextlib
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Union

import casadi as ca
import numpy as np
import yaml

warnings.filterwarnings("ignore", message=".*N_horizon.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*AcadosOcpDims.*", category=UserWarning)

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

sys.path.insert(0, str(_REPO_ROOT))
from scripts.quadrotor_model import load_config as load_quad_config, STATE_DIM, INPUT_DIM


def _load_mpc_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    path = Path(config_path) if config_path else _REPO_ROOT / "configs" / "mpc_config.yaml"
    if not path.is_absolute():
        path = _REPO_ROOT / path
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "mpc" not in cfg:
        raise KeyError(f"Expected 'mpc' key in {path}")
    return cfg["mpc"]


def _R_from_q_casadi(q: ca.SX) -> ca.SX:
    """Rotation matrix (body-to-world) from quaternion q = (w, x, y, z), CasADi symbolic."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)),
        ca.horzcat(    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)),
        ca.horzcat(    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)),
    )


def _build_casadi_dynamics(quad_params: dict):
    """
    Build CasADi symbolic continuous-time dynamics f(x, u) matching
    QuadrotorModel.dynamics_derivative.
    Returns (x_sym, u_sym, xdot_sym, params_dict).
    """
    mass = float(quad_params["mass"])
    arm_l = float(quad_params["arm_l"])
    inertia_scale = np.array(quad_params.get("inertia_scale", [4.5, 4.5, 7.0]))
    J_diag = (mass / 12.0) * (arm_l ** 2) * inertia_scale
    J = np.diag(J_diag)
    J_inv = np.linalg.inv(J)
    kappa = float(quad_params["kappa"])
    gravity = float(quad_params.get("gravity", -9.81))

    # Allocation matrix B: [F; tau_x; tau_y; tau_z] = B @ u
    sqrt_half = np.sqrt(0.5)
    t_BM = arm_l * sqrt_half * np.array([
        [1, -1, -1, 1],
        [-1, -1, 1, 1],
        [0, 0, 0, 0],
    ])
    k_row = kappa * np.array([[1, -1, 1, -1]])
    B = np.vstack([np.ones((1, 4)), t_BM[:2, :], k_row])

    # Thrust limits
    thrust_map = np.array(quad_params["thrust_map"])
    a, b, c = thrust_map[0], thrust_map[1], thrust_map[2]
    motor_omega_max = float(quad_params["motor_omega_max"])
    thrust_min = 0.0
    thrust_max = float(a * motor_omega_max**2 + b * motor_omega_max + c)

    # CasADi symbolic variables
    x = ca.SX.sym("x", STATE_DIM)
    u = ca.SX.sym("u", INPUT_DIM)

    p = x[0:3]
    q = x[3:7]
    v = x[7:10]
    omega = x[10:13]

    # Total thrust and body torque
    force_torque = ca.mtimes(ca.DM(B), u)
    F_total = force_torque[0]
    tau_body = force_torque[1:4]

    # p_dot = v
    p_dot = v

    # q_dot = 0.5 * Q_right(q_omega) @ q
    # Q_right for q_omega = (0, wx, wy, wz):
    #   [[0,  -wx, -wy, -wz],
    #    [wx,   0,  wz, -wy],
    #    [wy, -wz,   0,  wx],
    #    [wz,  wy, -wx,   0]]
    wx, wy, wz = omega[0], omega[1], omega[2]
    Q_omega = ca.vertcat(
        ca.horzcat(0,   -wx, -wy, -wz),
        ca.horzcat(wx,    0,  wz, -wy),
        ca.horzcat(wy,  -wz,   0,  wx),
        ca.horzcat(wz,   wy, -wx,   0),
    )
    q_dot = 0.5 * ca.mtimes(Q_omega, q)

    # v_dot = (1/m) * R(q) @ [0, 0, F_total] + [0, 0, g]
    R = _R_from_q_casadi(q)
    thrust_body_vec = ca.vertcat(0, 0, F_total)
    v_dot = (1.0 / mass) * ca.mtimes(R, thrust_body_vec) + ca.vertcat(0, 0, gravity)

    # omega_dot = J_inv @ (tau - omega x (J @ omega))
    J_omega = ca.mtimes(ca.DM(J), omega)
    cross = ca.vertcat(
        omega[1] * J_omega[2] - omega[2] * J_omega[1],
        omega[2] * J_omega[0] - omega[0] * J_omega[2],
        omega[0] * J_omega[1] - omega[1] * J_omega[0],
    )
    omega_dot = ca.mtimes(ca.DM(J_inv), tau_body - cross)

    xdot = ca.vertcat(p_dot, q_dot, v_dot, omega_dot)

    params = {
        "mass": mass, "gravity": gravity, "B": B,
        "thrust_min": thrust_min, "thrust_max": thrust_max,
        "J_diag": J_diag,
    }
    return x, u, xdot, params


@contextlib.contextmanager
def _suppress_output():
    """Suppress both stdout and stderr (acados C code prints MINSTEP etc. to both)."""
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


class MPCController:
    """
    Nonlinear MPC for quadrotor position tracking using acados.
    Solves an OCP at each timestep and returns the first optimal control.
    """

    def __init__(
        self,
        mpc_config_path: Optional[Union[str, Path]] = None,
        quad_config_path: Optional[Union[str, Path]] = None,
        pos_min: Optional[np.ndarray] = None,
        pos_max: Optional[np.ndarray] = None,
        constrained: bool = True,
        solver_label: str = "",
        thrust_limits: Optional[tuple] = None,
    ):
        """
        Parameters
        ----------
        pos_min, pos_max : optional (3,) arrays; only the z components (index 2) are
            used for state box constraints. x,y are not constrained in the OCP.
        constrained : if False, z (and all position) state box constraints are disabled
        solver_label : appended to codegen directory name to allow multiple
                       solver variants (e.g. "free" vs "constrained")
        thrust_limits : optional (thrust_min, thrust_max) tuple in [N] per motor.
            Overrides the physical limits derived from the thrust-map formula.
            Pass ``effective_thrust_limits(scfg)`` from ``_action_scaling`` to
            align the MPC feasible set with the range actually deliverable
            through the Flightmare normalised-action interface.
        """
        mpc_cfg = _load_mpc_config(mpc_config_path)

        quad_cfg_path = mpc_cfg.get("quadrotor_model_path") or quad_config_path
        quad_params = load_quad_config(quad_cfg_path)

        self._N = int(mpc_cfg["N"])
        self._dt = float(mpc_cfg["dt"])
        self._goal = np.array(mpc_cfg.get("goal_position", [0.0, 0.0, 5.0]))
        self._constrained = constrained

        # Build CasADi dynamics
        x_sym, u_sym, xdot_sym, dyn_params = _build_casadi_dynamics(quad_params)
        self._mass = dyn_params["mass"]
        self._gravity = dyn_params["gravity"]
        self._thrust_min = dyn_params["thrust_min"]
        self._thrust_max = dyn_params["thrust_max"]

        # Override thrust limits with effective interface range if provided.
        if thrust_limits is not None:
            t_min, t_max = float(thrust_limits[0]), float(thrust_limits[1])
            if t_max > t_min:
                self._thrust_min = t_min
                self._thrust_max = t_max

        # Hover thrust per motor
        self._u_hover = np.full(INPUT_DIM, (-self._mass * self._gravity) / 4.0)

        # Cost weights
        Q_pos = np.array(mpc_cfg.get("Q_pos", [10., 10., 10.]))
        Q_quat = np.array(mpc_cfg.get("Q_quat", [5., 1., 1., 1.]))
        Q_vel = np.array(mpc_cfg.get("Q_vel", [1., 1., 1.]))
        Q_omega = np.array(mpc_cfg.get("Q_omega", [0.1, 0.1, 0.1]))
        R_diag = np.array(mpc_cfg.get("R", [0.01, 0.01, 0.01, 0.01]))
        terminal_w = float(mpc_cfg.get("terminal_weight", 5.0))

        # z-only position box (indices [2]); x,y entries kept for config/API compatibility
        if pos_min is not None:
            self._pos_min = np.asarray(pos_min, dtype=np.float64).ravel()[:3]
        else:
            self._pos_min = np.array(mpc_cfg.get("pos_min", [-20., -20., 0.]), dtype=np.float64)
        if pos_max is not None:
            self._pos_max = np.asarray(pos_max, dtype=np.float64).ravel()[:3]
        else:
            self._pos_max = np.array(mpc_cfg.get("pos_max", [20., 20., 20.]), dtype=np.float64)
        self._z_lbx = float(self._pos_min[2])
        self._z_ubx = float(self._pos_max[2])

        # Solver settings
        nlp_solver = mpc_cfg.get("nlp_solver", "SQP_RTI")
        qp_solver = mpc_cfg.get("qp_solver", "PARTIAL_CONDENSING_HPIPM")
        integrator = mpc_cfg.get("integrator", "ERK")
        max_iter = int(mpc_cfg.get("max_iter", 100))
        integrator_steps = int(mpc_cfg.get("integrator_steps", 4))

        # --- Build acados OCP ---
        from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

        model = AcadosModel()
        _suffix = f"_{solver_label}" if solver_label else ""
        model.name = f"quadrotor_mpc{_suffix}"
        model.x = x_sym
        model.u = u_sym
        model.f_expl_expr = xdot_sym
        xdot_impl = ca.SX.sym("xdot", STATE_DIM)
        model.f_impl_expr = xdot_impl - xdot_sym
        model.xdot = xdot_impl

        ocp = AcadosOcp()
        ocp.model = model
        try:
            ocp.solver_options.N_horizon = self._N
        except AttributeError:
            ocp.dims.N = self._N
        ocp.solver_options.tf = self._N * self._dt

        # -- Cost: NONLINEAR_LS --
        # Stage residual y = [p(3), q(4), v(3), omega(3), u(4)] = 17
        y_expr = ca.vertcat(x_sym, u_sym)
        model.cost_y_expr = y_expr
        ocp.cost.cost_type = "NONLINEAR_LS"

        ny = STATE_DIM + INPUT_DIM  # 17
        W = np.diag(np.concatenate([Q_pos, Q_quat, Q_vel, Q_omega, R_diag]))
        ocp.cost.W = W
        ocp.cost.yref = np.zeros(ny)

        # Terminal residual y_e = [p(3), q(4), v(3), omega(3)] = 13
        model.cost_y_expr_e = x_sym
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        W_e = terminal_w * np.diag(np.concatenate([Q_pos, Q_quat, Q_vel, Q_omega]))
        ocp.cost.W_e = W_e
        ocp.cost.yref_e = np.zeros(STATE_DIM)

        # -- Control bounds --
        ocp.constraints.lbu = np.full(INPUT_DIM, self._thrust_min)
        ocp.constraints.ubu = np.full(INPUT_DIM, self._thrust_max)
        ocp.constraints.idxbu = np.arange(INPUT_DIM)

        # -- State box on z only (index 2); same ground/ceiling as CBF when barriers + r_uav are aligned
        self._use_slack = bool(mpc_cfg.get("use_slack", False))
        self._K_lin = float(mpc_cfg.get("K_lin", 1e6))
        self._K_quad = float(mpc_cfg.get("K_quad", 0.0))
        if self._constrained:
            n_pos = 1
            zmin = np.array([self._z_lbx], dtype=np.float64)
            zmax = np.array([self._z_ubx], dtype=np.float64)
            ocp.constraints.lbx = zmin
            ocp.constraints.ubx = zmax
            ocp.constraints.idxbx = np.array([2], dtype=int)
            ocp.constraints.lbx_e = zmin
            ocp.constraints.ubx_e = zmax
            ocp.constraints.idxbx_e = np.array([2], dtype=int)

            if self._use_slack:
                ocp.constraints.idxsbx = np.array([0], dtype=int)
                ocp.cost.zl = self._K_lin * np.ones(n_pos)
                ocp.cost.zu = self._K_lin * np.ones(n_pos)
                ocp.cost.Zl = self._K_quad * np.ones(n_pos)
                ocp.cost.Zu = self._K_quad * np.ones(n_pos)
                ocp.constraints.idxsbx_e = np.array([0], dtype=int)
                ocp.cost.zl_e = self._K_lin * np.ones(n_pos)
                ocp.cost.zu_e = self._K_lin * np.ones(n_pos)
                ocp.cost.Zl_e = self._K_quad * np.ones(n_pos)
                ocp.cost.Zu_e = self._K_quad * np.ones(n_pos)

        # -- Initial state constraint (set at runtime) --
        ocp.constraints.x0 = np.zeros(STATE_DIM)

        # -- Solver options --
        ocp.solver_options.nlp_solver_type = nlp_solver
        ocp.solver_options.qp_solver = qp_solver
        ocp.solver_options.integrator_type = integrator
        ocp.solver_options.sim_method_num_steps = integrator_steps
        ocp.solver_options.nlp_solver_max_iter = max_iter
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.print_level = 0
        ocp.solver_options.qp_solver_iter_max = 200
        # Levenberg-Marquardt regularization for better convergence from far-off initial guesses
        ocp.solver_options.levenberg_marquardt = 1e-2

        # Code generation directory (label differentiates constrained/free variants)
        codegen_dir = _REPO_ROOT / f"c_generated_code_mpc{_suffix}"
        try:
            ocp.code_gen_opts.code_export_directory = str(codegen_dir)
        except AttributeError:
            ocp.code_export_directory = str(codegen_dir)

        json_file = str(codegen_dir / f"acados_ocp_{model.name}.json")

        with _suppress_output():
            self._solver = AcadosOcpSolver(ocp, json_file=json_file)

        # Store default reference
        self._yref = np.zeros(ny)
        self._yref_e = np.zeros(STATE_DIM)
        self._set_reference(self._goal)

        # Timing stats
        self.last_solve_time_ms = 0.0
        self.last_status = 0

    def _set_reference(self, goal_pos: np.ndarray):
        """Update stage and terminal references for position tracking."""
        x_ref = np.zeros(STATE_DIM)
        x_ref[0:3] = goal_pos
        x_ref[3] = 1.0  # qw = 1 (upright)

        ny = STATE_DIM + INPUT_DIM
        yref = np.zeros(ny)
        yref[:STATE_DIM] = x_ref
        yref[STATE_DIM:] = self._u_hover

        yref_e = x_ref.copy()

        for k in range(self._N):
            self._solver.set(k, "yref", yref)
        self._solver.set(self._N, "yref", yref_e)

        self._yref = yref
        self._yref_e = yref_e
        self._goal = goal_pos.copy()

    def solve(
        self,
        x_current: np.ndarray,
        goal_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve the OCP for the current state and return optimal motor thrusts (4,) [N].

        Parameters
        ----------
        x_current : (13,) current state [p, q, v, omega]
        goal_pos  : (3,) optional goal override (updates internal reference)

        Returns
        -------
        u_opt : (4,) motor thrusts in [N]
        """
        x0 = np.asarray(x_current, dtype=np.float64).ravel()[:STATE_DIM]

        # Normalize quaternion in initial state
        qnorm = np.linalg.norm(x0[3:7])
        if qnorm > 1e-10:
            x0[3:7] /= qnorm
        if x0[3] < 0:
            x0[3:7] *= -1.0

        if goal_pos is not None and not np.allclose(goal_pos, self._goal):
            self._set_reference(np.asarray(goal_pos, dtype=np.float64).ravel()[:3])

        self._solver.set(0, "lbx", x0)
        self._solver.set(0, "ubx", x0)

        import time
        t0 = time.perf_counter()
        with _suppress_output():
            status = self._solver.solve()
        self.last_solve_time_ms = (time.perf_counter() - t0) * 1000.0
        self.last_status = status

        u_opt = self._solver.get(0, "u")

        # Warm-start: shift solution for next call
        for k in range(self._N - 1):
            x_k1 = self._solver.get(k + 1, "x")
            u_k1 = self._solver.get(k + 1, "u")
            self._solver.set(k, "x", x_k1)
            self._solver.set(k, "u", u_k1)
        # Last stage: repeat terminal
        x_N = self._solver.get(self._N, "x")
        self._solver.set(self._N - 1, "x", x_N)
        self._solver.set(self._N - 1, "u", u_opt)  # fallback
        self._solver.set(self._N, "x", x_N)

        return np.clip(u_opt, self._thrust_min, self._thrust_max)

    def reset(self, x0: Optional[np.ndarray] = None, warmup_iter: int = 5):
        """
        Reset solver warm-start and run warmup SQP iterations so the
        predicted trajectory is reasonable before the first real solve.
        """
        if x0 is not None:
            x0 = np.asarray(x0, dtype=np.float64).ravel()[:STATE_DIM]
            qn = np.linalg.norm(x0[3:7])
            if qn > 1e-10:
                x0[3:7] /= qn
            if x0[3] < 0:
                x0[3:7] *= -1.0
        else:
            x0 = np.zeros(STATE_DIM)
            x0[3] = 1.0
            x0[2] = self._goal[2]

        for k in range(self._N + 1):
            self._solver.set(k, "x", x0)
        for k in range(self._N):
            self._solver.set(k, "u", self._u_hover)

        # Run warmup iterations to build a sensible predicted trajectory
        if warmup_iter > 0:
            self._solver.set(0, "lbx", x0)
            self._solver.set(0, "ubx", x0)
            with _suppress_output():
                for _ in range(warmup_iter):
                    self._solver.solve()

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def N(self) -> int:
        return self._N

    @property
    def hover_thrust(self) -> np.ndarray:
        return self._u_hover.copy()

    @property
    def thrust_limits(self):
        return (self._thrust_min, self._thrust_max)

    @property
    def constrained(self) -> bool:
        return self._constrained

    @property
    def pos_limits(self):
        return (self._pos_min.copy(), self._pos_max.copy())


def main():
    """Quick sanity check: create MPC, solve one step from hover."""
    print("Building MPC solver (first run generates C code) ...")
    mpc = MPCController()
    print(f"  horizon N={mpc.N}, dt={mpc.dt}s, thrust_limits={mpc.thrust_limits}")
    print(f"  hover_thrust={mpc.hover_thrust}")

    x0 = np.zeros(STATE_DIM)
    x0[3] = 1.0
    x0[2] = 2.0  # start at z=2, goal at z=5
    mpc.reset(x0)

    goal = np.array([0.0, 0.0, 5.0])
    u = mpc.solve(x0, goal)
    print(f"  x0: pos={x0[:3]}, q={x0[3:7]}")
    print(f"  goal: {goal}")
    print(f"  u_opt (N): {u}")
    print(f"  solve time: {mpc.last_solve_time_ms:.2f} ms, status: {mpc.last_status}")

    from scripts.quadrotor_model import QuadrotorModel
    model = QuadrotorModel()
    x1 = model.step(x0, u, mpc.dt, integrate="rk4")
    print(f"  after one step: pos={x1[:3]}, |q|={np.linalg.norm(x1[3:7]):.6f}")
    print("MPC sanity check passed.")


if __name__ == "__main__":
    main()
