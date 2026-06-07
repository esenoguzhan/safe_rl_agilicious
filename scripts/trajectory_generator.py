"""
Online random trajectory generator for RL training.

Generates dynamically-feasible random trajectories in memory using:
  - Sum-of-periodic kernels for smooth random path priors
  - Quintic spline interpolation to dense time grid
  - Differential-flatness-based attitude/rate computation

Designed to be called per-env on episode reset, producing (T, D) arrays
for pos, vel, quat, omega — the same format as _load_trajectory().

No file I/O, no CasADi dependency.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline


# ---------------------------------------------------------------------------
# Vectorised quaternion helpers (wxyz, batch-capable)
# ---------------------------------------------------------------------------

def _quat_inv_batch(q):
    qi = q.copy()
    qi[..., 1:] *= -1.0
    return qi


def _quat_mul_batch(q, r):
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    rw, rx, ry, rz = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        rw*qw - rx*qx - ry*qy - rz*qz,
        rw*qx + rx*qw - ry*qz + rz*qy,
        rw*qy + rx*qz + ry*qw - rz*qx,
        rw*qz - rx*qy + ry*qx + rz*qw,
    ], axis=-1)


def _euler_to_quat_batch(yaw):
    """yaw array -> quaternion array (wxyz). roll=pitch=0."""
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    out = np.zeros((len(yaw), 4), dtype=yaw.dtype)
    out[:, 0] = cy
    out[:, 3] = sy
    return out


def _q_to_rot_batch(q):
    """Batch quaternion (N, 4) wxyz -> rotation matrices (N, 3, 3)."""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=q.dtype)
    R[:, 0, 0] = 1 - 2*(qy**2 + qz**2)
    R[:, 0, 1] = 2*(qx*qy - qw*qz)
    R[:, 0, 2] = 2*(qx*qz + qw*qy)
    R[:, 1, 0] = 2*(qx*qy + qw*qz)
    R[:, 1, 1] = 1 - 2*(qx**2 + qz**2)
    R[:, 1, 2] = 2*(qy*qz - qw*qx)
    R[:, 2, 0] = 2*(qx*qz - qw*qy)
    R[:, 2, 1] = 2*(qy*qz + qw*qx)
    R[:, 2, 2] = 1 - 2*(qx**2 + qy**2)
    return R


# ---------------------------------------------------------------------------
# Kernel sampling (replaces sklearn GP for speed)
# ---------------------------------------------------------------------------

def _exp_sine_squared_kernel(X, length_scale, periodicity):
    """Compute ExpSineSquared kernel matrix.  X: (n, 1) -> (n, n)."""
    d = X - X.T
    return np.exp(-2.0 * np.sin(np.pi * np.abs(d) / periodicity)**2
                  / length_scale**2)


def _sample_gp_prior(t_col, rng, num_kernels, ls_min, ls_max, p_min, p_max):
    """Sample from a sum-of-ExpSineSquared GP prior at points t_col (n, 1).

    Returns (n,) sample.
    """
    n = len(t_col)
    K = np.zeros((n, n), dtype=np.float64)
    for _ in range(num_kernels + 1):
        ls = rng.uniform(ls_min, ls_max)
        per = rng.uniform(p_min, p_max)
        K += _exp_sine_squared_kernel(t_col, ls, per)

    K += 1e-8 * np.eye(n)
    L = np.linalg.cholesky(K)
    return L @ rng.standard_normal(n)


# ---------------------------------------------------------------------------
# Time mapping (smooth start/end ramp)
# ---------------------------------------------------------------------------

def _time_mapping_poly_coeffs(continuity_order=4):
    n = 2 * continuity_order + 1
    A = np.zeros((n, n))
    A[0, -1] = 1.0
    p = np.poly1d(np.ones(n))
    for i_der in range(2 * continuity_order):
        if i_der % 2 == 0:
            A[i_der + 1, n - i_der // 2 - 2] = np.polyder(p, i_der // 2 + 1).coeffs[-1]
        else:
            coeffs = np.polyder(p, (i_der + 1) // 2).coeffs
            A[i_der + 1, :len(coeffs)] = coeffs
    b = np.zeros(n)
    b[2] = 1.0
    return np.linalg.solve(A, b)


def _time_mapping(speedup_duration, dt, continuity_order=4):
    t_vec = np.arange(0.0, speedup_duration, dt)
    x = _time_mapping_poly_coeffs(continuity_order)
    n = 2 * continuity_order + 1
    y = np.zeros_like(t_vec)
    for i in range(n):
        y += speedup_duration * x[i] * np.power(t_vec / speedup_duration, n - 1 - i)
    return y


# ---------------------------------------------------------------------------
# Attitude computation from differential flatness (fully vectorised)
# ---------------------------------------------------------------------------

def _compute_attitude_and_rates(pos, vel, acc, dt, mass):
    """Compute quaternion and body rates from flat outputs (vectorised)."""
    T = len(pos)
    gravity = 9.81

    thrust = acc + np.array([[0.0, 0.0, gravity]])
    z_b = thrust / (np.linalg.norm(thrust, axis=1, keepdims=True) + 1e-12)

    e_z = np.array([[0.0, 0.0, 1.0]])
    q_w = 1.0 + np.sum(e_z * z_b, axis=1)
    q_xyz = np.cross(e_z, z_b)
    att = 0.5 * np.column_stack([q_w, q_xyz])
    att = att / (np.linalg.norm(att, axis=1, keepdims=True) + 1e-12)

    # Velocity-aligned yaw (vectorised)
    R_inv = _q_to_rot_batch(_quat_inv_batch(att))
    vel_body = np.einsum("nij,nj->ni", R_inv, vel)
    yaw_d = np.arctan2(vel_body[:, 1], vel_body[:, 0])
    q_yaw = _euler_to_quat_batch(yaw_d)
    att = _quat_mul_batch(att, q_yaw)
    att = att / (np.linalg.norm(att, axis=1, keepdims=True) + 1e-12)

    # Remove quaternion sign flips
    for i in range(1, T):
        if np.dot(att[i - 1], att[i]) < 0:
            att[i] *= -1.0

    # Body rates from quaternion derivative (vectorised)
    q_dot = np.gradient(att, axis=0) / dt
    omega = 2.0 * _quat_mul_batch(_quat_inv_batch(att), q_dot)[:, 1:]

    return att.astype(np.float32), omega.astype(np.float32)


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class RandomTrajectoryGenerator:
    """Generates random dynamically-feasible quadrotor trajectories.

    Parameters (from trajectory_cfg dict):
        duration     : float  — trajectory length in seconds
        sim_dt       : float  — output sample period
        bound_min    : [3]    — min corner of bounding box
        bound_max    : [3]    — max corner of bounding box
        num_kernels  : int    — GP kernel count (more = more complex)
        length_scale_range : [2] — [min, max] GP length scale
        period_range : [2]    — [min, max] GP periodicity
        t_speedup    : float  — ramp duration for smooth start/end
        dt_gen       : float  — internal dense sampling dt (default 0.005)
        mass         : float  — quadrotor mass in kg (default 0.774)
    """

    def __init__(self, trajectory_cfg):
        self.duration = float(trajectory_cfg["duration"])
        self.sim_dt = float(trajectory_cfg["sim_dt"])
        self.bound_min = np.array(trajectory_cfg.get("bound_min", [-3, -2, 0.5]),
                                  dtype=np.float64)
        self.bound_max = np.array(trajectory_cfg.get("bound_max", [3, 2, 3]),
                                  dtype=np.float64)
        self.num_kernels = int(trajectory_cfg.get("num_kernels", 2))
        ls_range = trajectory_cfg.get("length_scale_range", [0.10, 0.14])
        self.ls_min, self.ls_max = float(ls_range[0]), float(ls_range[1])
        p_range = trajectory_cfg.get("period_range", [20.0, 50.0])
        self.p_min, self.p_max = float(p_range[0]), float(p_range[1])
        self.t_speedup = float(trajectory_cfg.get("t_speedup", 2.0))
        self.dt_gen = float(trajectory_cfg.get("dt_gen", 0.005))
        self.mass = float(trajectory_cfg.get("mass", 0.774))

        self._t_dense = np.arange(0.0, self.duration, self.dt_gen)
        self._t_coarse = np.arange(0.0, self.duration, 0.1)
        self._t_coarse_col = self._t_coarse[:, np.newaxis]

        startup = _time_mapping(self.t_speedup, self.dt_gen)
        t_mid = self._t_dense[
            (self.t_speedup / 2.0 <= self._t_dense) &
            (self._t_dense < self.duration - 1.5 * self.t_speedup)
        ]
        if len(t_mid) == 0:
            self._scaled_time = self._t_dense
        else:
            end_ramp = t_mid[-1] + self.t_speedup / 2.0 - np.flip(startup)
            self._scaled_time = np.concatenate([startup, t_mid, end_ramp])
        n_dense = len(self._t_dense)
        if len(self._scaled_time) > n_dense:
            self._scaled_time = self._scaled_time[:n_dense]
        elif len(self._scaled_time) < n_dense:
            pad = np.full(n_dense - len(self._scaled_time),
                          self._scaled_time[-1])
            self._scaled_time = np.concatenate([self._scaled_time, pad])

        self._subsample = max(1, round(self.sim_dt / self.dt_gen))

    def generate(self, rng=None):
        """Generate one random trajectory.

        Parameters
        ----------
        rng : numpy.random.Generator or None

        Returns
        -------
        pos  : (T, 3) float32
        vel  : (T, 3) float32
        quat : (T, 4) float32  (wxyz)
        omega: (T, 3) float32
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample coarse GP priors for x, y, z
        x_s = _sample_gp_prior(self._t_coarse_col, rng,
                                self.num_kernels, self.ls_min, self.ls_max,
                                self.p_min, self.p_max)
        y_s = _sample_gp_prior(self._t_coarse_col, rng,
                                self.num_kernels, self.ls_min, self.ls_max,
                                self.p_min, self.p_max)
        z_s = _sample_gp_prior(self._t_coarse_col, rng,
                                self.num_kernels, self.ls_min, self.ls_max,
                                self.p_min, self.p_max)

        pos_coarse = np.column_stack([x_s, y_s, z_s])

        # Scale to bounding box
        mn = pos_coarse.min(axis=0)
        mx = pos_coarse.max(axis=0)
        rng_vals = np.clip(mx - mn, 1e-6, None)
        centered = pos_coarse - (mx + mn) / 2.0
        scaled = centered * (self.bound_max - self.bound_min) / rng_vals
        pos_coarse = scaled + (self.bound_max + self.bound_min) / 2.0

        # Fit quintic splines and evaluate at dense scaled time
        spl_x = UnivariateSpline(self._t_coarse, pos_coarse[:, 0], k=5, s=0)
        spl_y = UnivariateSpline(self._t_coarse, pos_coarse[:, 1], k=5, s=0)
        spl_z = UnivariateSpline(self._t_coarse, pos_coarse[:, 2], k=5, s=0)

        st = self._scaled_time
        pos_dense = np.column_stack([spl_x(st), spl_y(st), spl_z(st)])
        vel_dense = np.gradient(pos_dense, axis=0) / self.dt_gen
        acc_dense = np.gradient(vel_dense, axis=0) / self.dt_gen

        att, omega = _compute_attitude_and_rates(
            pos_dense, vel_dense, acc_dense, self.dt_gen, self.mass)

        s = self._subsample
        return (pos_dense[::s].astype(np.float32),
                vel_dense[::s].astype(np.float32),
                att[::s],
                omega[::s])
