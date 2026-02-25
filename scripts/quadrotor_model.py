"""
Standalone quadrotor model compatible with Flightmare and quaternion representation.
All parameters are loaded from configs/quadrotor_model.yaml.
Designed for use with control barrier functions (CBF) / safety filter (e.g. ACADIS);
this module provides the model only (no CBF logic).

State: x = [p(3), q(4), v(3), omega(3)] = 13 dims
  - p: position (m), q: quaternion (w,x,y,z), v: velocity (m/s), omega: body rates (rad/s)
Control: u = motor thrusts (4) in [N]

Motor lag (optional): set use_motor_lag: true in config. step(x, u_cmd, dt) then applies
first-order lag u_actual = exp(-dt/tau)*u_actual + (1-exp(-dt/tau))*u_cmd before dynamics.
dynamics_derivative(x, u) always uses u as applied thrust (no lag), so CBF/Lie derivatives are unchanged.
"""
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import yaml

# State indices (Flightmare QuadState compatible)
POS = slice(0, 3)
ATT = slice(3, 7)   # quaternion (w, x, y, z)
VEL = slice(7, 10)
OME = slice(10, 13)
STATE_DIM = 13
INPUT_DIM = 4

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "quadrotor_model.yaml"


def Q_right(q: np.ndarray) -> np.ndarray:
    """4x4 matrix such that q_dot = 0.5 * Q_right(q_omega) @ q for quaternion derivative.
    Matches flightlib/common/math.cpp Q_right. q is (w, x, y, z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [w, -x, -y, -z],
        [x,  w,  z, -y],
        [y, -z,  w,  x],
        [z,  y, -x,  w],
    ], dtype=np.float64)


def R_from_q(q: np.ndarray) -> np.ndarray:
    """Rotation matrix body-to-world from unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Return unit quaternion; optionally enforce w >= 0 (Flightmare convention)."""
    q = np.asarray(q, dtype=np.float64).ravel()
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / n
    if q[0] < 0:
        q = -q
    return q


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Load quadrotor_model section from YAML. If config_path is None, use configs/quadrotor_model.yaml."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = _REPO_ROOT / path
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "quadrotor_model" not in cfg:
        raise KeyError(f"Expected 'quadrotor_model' key in {path}")
    return cfg["quadrotor_model"]


class QuadrotorModel:
    """
    Quadrotor dynamics model: state 13-dim [p, q, v, omega], control 4 motor thrusts [N].
    Compatible with Flightmare state layout and allocation. Parameters from quadrotor_model.yaml.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        params = load_config(config_path)
        self._mass = float(params["mass"])
        self._arm_l = float(params["arm_l"])
        inertia_scale = np.array(params.get("inertia_scale", [4.5, 4.5, 7.0]), dtype=np.float64)
        self._J = (self._mass / 12.0) * (self._arm_l ** 2) * np.diag(inertia_scale)
        self._J_inv = np.linalg.inv(self._J)

        self._motor_omega_min = float(params["motor_omega_min"])
        self._motor_omega_max = float(params["motor_omega_max"])
        self._motor_tau = float(params["motor_tau"])
        self._use_motor_lag = bool(params.get("use_motor_lag", False))
        self._thrust_map = np.array(params["thrust_map"], dtype=np.float64)  # [a, b, c]
        self._kappa = float(params["kappa"])
        self._omega_max = np.array(params["omega_max"], dtype=np.float64)
        self._gravity = float(params.get("gravity", -9.81))
        self._gz = np.array([0.0, 0.0, self._gravity], dtype=np.float64)

        # Per-motor thrust bounds from T = a*omega^2 + b*omega + c
        a, b, c = self._thrust_map[0], self._thrust_map[1], self._thrust_map[2]
        for om in [self._motor_omega_min, self._motor_omega_max]:
            if om * om * a + om * b + c < 0 and a != 0:
                pass  # clamp to 0 in practice
        self._thrust_min = 0.0
        self._thrust_max = float(
            a * (self._motor_omega_max ** 2) + b * self._motor_omega_max + c
        )
        if self._thrust_max < self._thrust_min:
            self._thrust_max = self._thrust_min + 1e-6

        # Allocation matrix B: [F; tau] = B @ motor_thrusts (F = total thrust, tau = body torque)
        # t_BM from Flightmare: arm_l * sqrt(0.5) * [[1,-1,-1,1], [-1,-1,1,1], [0,0,0,0]]
        sqrt_half = np.sqrt(0.5)
        t_BM = self._arm_l * sqrt_half * np.array([
            [1, -1, -1, 1],
            [-1, -1, 1, 1],
            [0, 0, 0, 0],
        ], dtype=np.float64)
        k_row = self._kappa * np.array([[1, -1, 1, -1]], dtype=np.float64)
        self._B = np.vstack([
            np.ones((1, 4), dtype=np.float64),
            t_BM[:2, :],
            k_row,
        ])
        self._B_inv = np.linalg.inv(self._B)

        # Motor lag state: actual thrust (4,) used when use_motor_lag is True. None = use u_cmd as-is on first step.
        self._u_motor: Optional[np.ndarray] = None

    @property
    def state_dim(self) -> int:
        return STATE_DIM

    @property
    def input_dim(self) -> int:
        return INPUT_DIM

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def gravity(self) -> float:
        return self._gravity

    def get_allocation_matrix(self) -> np.ndarray:
        """B such that [F; tau_x; tau_y; tau_z] = B @ motor_thrusts. (4, 4)."""
        return self._B.copy()

    def get_allocation_inverse(self) -> np.ndarray:
        """Inverse of allocation matrix for mapping desired thrust/torque to motor thrusts."""
        return self._B_inv.copy()

    def get_thrust_limits(self) -> Tuple[float, float]:
        """Per-motor thrust bounds (min, max) in [N]."""
        return (self._thrust_min, self._thrust_max)

    def get_omega_max(self) -> np.ndarray:
        """Body rate limits (rad/s) [x, y, z]."""
        return self._omega_max.copy()

    def get_J(self) -> np.ndarray:
        """Inertia matrix (3, 3) kg*m^2."""
        return self._J.copy()

    @property
    def motor_tau(self) -> float:
        """Motor first-order time constant (s)."""
        return self._motor_tau

    @property
    def use_motor_lag(self) -> bool:
        """Whether step() applies first-order motor lag to commanded thrust."""
        return self._use_motor_lag

    def set_motor_state(self, u: np.ndarray) -> None:
        """Set internal motor thrust state (used when use_motor_lag is True). Call on reset to match u_cmd."""
        self._u_motor = self.clamp_motor_thrusts(np.asarray(u, dtype=np.float64).ravel()[:INPUT_DIM]).copy()

    def get_motor_state(self) -> Optional[np.ndarray]:
        """Current motor thrust state (4,) if use_motor_lag and state has been set, else None."""
        return self._u_motor.copy() if self._u_motor is not None else None

    def clamp_motor_thrusts(self, u: np.ndarray) -> np.ndarray:
        """Clamp motor thrusts to [thrust_min, thrust_max] per motor."""
        u = np.asarray(u, dtype=np.float64).ravel()[:INPUT_DIM]
        return np.clip(u, self._thrust_min, self._thrust_max)

    def thrust_torque_from_motor_thrusts(self, motor_thrusts: np.ndarray) -> np.ndarray:
        """[F, tau_x, tau_y, tau_z] = B @ motor_thrusts (F in N, tau in N*m)."""
        u = np.asarray(motor_thrusts, dtype=np.float64).ravel()[:INPUT_DIM]
        return self._B @ u

    def motor_thrusts_from_thrust_torque(
        self, F: float, tau: np.ndarray, clamp: bool = True
    ) -> np.ndarray:
        """Desired total thrust F (N) and body torque tau (3,) -> motor thrusts (4,)."""
        tau = np.asarray(tau, dtype=np.float64).ravel()[:3]
        ft = np.concatenate([[F], tau])
        u = self._B_inv @ ft
        if clamp:
            u = self.clamp_motor_thrusts(u)
        return u

    def dynamics_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State derivative dx/dt for state x (13,) and motor thrusts u (4,).
        x = [p, q, v, omega], u = motor thrusts [N].
        Returns dx_dt (13,).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        u = np.asarray(u, dtype=np.float64).ravel()[:INPUT_DIM]
        if x.size != STATE_DIM:
            raise ValueError(f"state must have length {STATE_DIM}, got {x.size}")

        p = x[POS]
        q = x[ATT]
        v = x[VEL]
        omega = x[OME]

        # Normalize quaternion
        q = quaternion_normalize(q)

        # [F; tau] = B @ u
        F_total = np.sum(u)
        force_torque = self._B @ u
        tau_body = force_torque[1:4]

        # p_dot = v
        p_dot = v

        # q_dot = 0.5 * Q_right(q_omega) @ q, q_omega = (0, omega_x, omega_y, omega_z)
        q_omega = np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float64)
        q_dot = 0.5 * (Q_right(q_omega) @ q)

        # v_dot = (1/m) * R(q) @ [0, 0, F] + g
        R = R_from_q(q)
        thrust_body = np.array([0.0, 0.0, F_total], dtype=np.float64)
        acc = (1.0 / self._mass) * (R @ thrust_body) + self._gz
        v_dot = acc

        # omega_dot = J^{-1} @ (tau - omega x (J @ omega))
        J_omega = self._J @ omega
        omega_dot = self._J_inv @ (tau_body - np.cross(omega, J_omega))

        dx = np.zeros(STATE_DIM, dtype=np.float64)
        dx[POS] = p_dot
        dx[ATT] = q_dot
        dx[VEL] = v_dot
        dx[OME] = omega_dot
        return dx

    def step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
        integrate: str = "euler",
    ) -> np.ndarray:
        """
        Integrate dynamics over dt. integrate in ('euler', 'rk4').
        If use_motor_lag is True, first-order lag is applied: u_actual = alpha*u_actual + (1-alpha)*u_cmd
        with alpha = exp(-dt/motor_tau). Dynamics use u_actual; dynamics_derivative() is unchanged (instant u).
        Returns new state (13,). Quaternion is normalized.
        """
        u_cmd = self.clamp_motor_thrusts(u)
        if self._use_motor_lag:
            tau = self._motor_tau
            if tau <= 0:
                u_apply = u_cmd
                self._u_motor = u_cmd.copy()
            else:
                alpha = np.exp(-dt / tau)
                if self._u_motor is None:
                    self._u_motor = u_cmd.copy()
                u_apply = alpha * self._u_motor + (1.0 - alpha) * u_cmd
                u_apply = self.clamp_motor_thrusts(u_apply)
                self._u_motor = u_apply.copy()
        else:
            u_apply = u_cmd
        if integrate == "euler":
            dx = self.dynamics_derivative(x, u_apply)
            x_next = x + dt * dx
        elif integrate == "rk4":
            k1 = self.dynamics_derivative(x, u_apply)
            k2 = self.dynamics_derivative(x + 0.5 * dt * k1, u_apply)
            k3 = self.dynamics_derivative(x + 0.5 * dt * k2, u_apply)
            k4 = self.dynamics_derivative(x + dt * k3, u_apply)
            x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"integrate must be 'euler' or 'rk4', got {integrate!r}")
        x_next[ATT] = quaternion_normalize(x_next[ATT])
        return x_next

    def observation_from_state(self, x: np.ndarray, goal_pos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Observation vector compatible with Flightmare QuadrotorEnv:
        [pos_error(3), q(4), v(3), omega(3)] if goal_pos given, else [p(3), q(4), v(3), omega(3)].
        """
        x = np.asarray(x, dtype=np.float64).ravel()[:STATE_DIM]
        q = quaternion_normalize(x[ATT])
        if goal_pos is not None:
            pos_part = np.asarray(goal_pos, dtype=np.float64).ravel()[:3] - x[POS]
        else:
            pos_part = x[POS]
        return np.concatenate([pos_part, q, x[VEL], x[OME]]).astype(np.float32)

    def state_from_observation(
        self, obs: np.ndarray, goal_pos: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        If observation is [pos_error, q, v, omega] and goal_pos is set, state position = goal - pos_error.
        If goal_pos is None, assumes obs = [p, q, v, omega].
        """
        obs = np.asarray(obs, dtype=np.float64).ravel()
        if obs.size < STATE_DIM:
            raise ValueError(f"observation must have at least {STATE_DIM} elements, got {obs.size}")
        obs = obs[:STATE_DIM]
        if goal_pos is not None:
            goal_pos = np.asarray(goal_pos, dtype=np.float64).ravel()[:3]
            p = goal_pos - obs[0:3]
        else:
            p = obs[0:3]
        q = quaternion_normalize(obs[3:7])
        v = obs[7:10]
        omega = obs[10:13]
        return np.concatenate([p, q, v, omega]).astype(np.float64)


def main() -> None:
    """Quick sanity check: load model, run one step, print state derivative."""
    model = QuadrotorModel()
    x = np.zeros(STATE_DIM)
    x[ATT] = [1.0, 0.0, 0.0, 0.0]  # upright
    x[POS] = [0.0, 0.0, 5.0]
    # Hover thrust per motor
    hover_per_motor = -model.mass * model.gravity / 4.0
    u = np.ones(4) * hover_per_motor
    dx = model.dynamics_derivative(x, u)
    print("QuadrotorModel sanity check:")
    print("  state_dim:", model.state_dim, "input_dim:", model.input_dim)
    print("  thrust_limits:", model.get_thrust_limits())
    print("  dx (norm):", np.linalg.norm(dx))
    print("  |q_dot|:", np.linalg.norm(dx[ATT]))
    x1 = model.step(x, u, 0.02, integrate="rk4")
    print("  after 0.02s step, pos:", x1[POS], "|q|:", np.linalg.norm(x1[ATT]))


if __name__ == "__main__":
    main()
