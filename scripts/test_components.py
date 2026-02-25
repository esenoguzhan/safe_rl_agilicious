#!/usr/bin/env python3
"""
Validate quadrotor model, CBF filter, and CBF wrapper individually.
Run: python scripts/test_components.py
Or:  pytest scripts/test_components.py -v
"""
from pathlib import Path
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np


def _assert_allclose(a, b, rtol=1e-5, atol=1e-8, msg=""):
    a, b = np.asarray(a), np.asarray(b)
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(f"{msg}\n  got {a}\n  expected ~ {b}")


def _assert_true(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


# -----------------------------------------------------------------------------
# 1. Quadrotor model
# -----------------------------------------------------------------------------

def test_quadrotor_model_config_and_constants():
    """Load config and check model constants."""
    from scripts.quadrotor_model import load_config, QuadrotorModel, STATE_DIM, INPUT_DIM

    cfg = load_config()
    _assert_true("mass" in cfg and "arm_l" in cfg, "config must have mass, arm_l")
    model = QuadrotorModel()
    _assert_true(model.state_dim == STATE_DIM == 13, "state_dim == 13")
    _assert_true(model.input_dim == INPUT_DIM == 4, "input_dim == 4")
    _assert_true(model.mass > 0 and model.mass < 10, "reasonable mass")
    thrust_min, thrust_max = model.get_thrust_limits()
    _assert_true(thrust_max > thrust_min, "thrust_max > thrust_min")
    J = model.get_J()
    _assert_true(J.shape == (3, 3) and np.allclose(J, J.T), "J is 3x3 symmetric")
    B = model.get_allocation_matrix()
    _assert_true(B.shape == (4, 4), "B is 4x4")
    _assert_allclose(B[0, :], np.ones(4), msg="first row of B is ones")
    print("  [OK] quadrotor_model: config, state_dim, input_dim, J, B, thrust_limits")


def test_quadrotor_model_quaternion_helpers():
    """Q_right, R_from_q, quaternion_normalize."""
    from scripts.quadrotor_model import Q_right, R_from_q, quaternion_normalize

    q = np.array([1.0, 0.0, 0.0, 0.0])
    _assert_allclose(np.linalg.norm(quaternion_normalize(q)), 1.0)
    _assert_allclose(quaternion_normalize(q), q)
    q2 = np.array([0.5, 0.5, 0.5, 0.5])
    q2n = quaternion_normalize(q2)
    _assert_allclose(np.linalg.norm(q2n), 1.0)

    R = R_from_q(q)
    _assert_allclose(R, np.eye(3), msg="R(identity quat) = I")
    q90z = np.array([np.cos(0.5 * np.pi / 2), 0, 0, np.sin(0.5 * np.pi / 2)])
    R90 = R_from_q(quaternion_normalize(q90z))
    _assert_allclose(R90 @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-6)

    Qr = Q_right(q)
    _assert_true(Qr.shape == (4, 4))
    print("  [OK] quadrotor_model: Q_right, R_from_q, quaternion_normalize")


def test_quadrotor_model_quaternion_normalize_zero_vector():
    """Edge case: zero-norm quaternion (malformed obs) returns identity [1,0,0,0] and does not crash."""
    from scripts.quadrotor_model import quaternion_normalize

    q_zero = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    out = quaternion_normalize(q_zero)
    _assert_true(out is not None and out.shape == (4,), "normalizer must return 4-vector")
    _assert_allclose(out, np.array([1.0, 0.0, 0.0, 0.0]), msg="zero quat -> identity [1,0,0,0]")
    _assert_allclose(np.linalg.norm(out), 1.0)
    # Very small norm (below 1e-10) should also yield identity
    q_tiny = np.array([1e-12, 0.0, 0.0, 0.0], dtype=np.float64)
    out_tiny = quaternion_normalize(q_tiny)
    _assert_allclose(out_tiny, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)
    print("  [OK] quadrotor_model: quaternion_normalize zero/small-norm -> identity, no crash")


def test_quadrotor_model_hover_derivative():
    """At hover (u = hover thrust), derivative should be ~0."""
    from scripts.quadrotor_model import QuadrotorModel, STATE_DIM, POS, ATT

    model = QuadrotorModel()
    x = np.zeros(STATE_DIM)
    x[ATT] = [1.0, 0.0, 0.0, 0.0]
    x[POS] = [0.0, 0.0, 5.0]
    hover = model.mass * (-model.gravity) / 4.0
    u = np.ones(4) * hover
    dx = model.dynamics_derivative(x, u)
    _assert_allclose(dx, np.zeros(STATE_DIM), atol=1e-6, msg="hover: dx ~ 0")
    print("  [OK] quadrotor_model: hover derivative ~ 0")


def test_quadrotor_model_step():
    """Integration step preserves quaternion norm and moves position under thrust."""
    from scripts.quadrotor_model import QuadrotorModel, STATE_DIM, POS, ATT

    model = QuadrotorModel()
    x = np.zeros(STATE_DIM)
    x[ATT] = [1.0, 0.0, 0.0, 0.0]
    x[POS] = [0.0, 0.0, 5.0]
    hover = model.mass * (-model.gravity) / 4.0
    u = np.ones(4) * hover
    dt = 0.02
    x1 = model.step(x, u, dt, integrate="euler")
    x2 = model.step(x, u, dt, integrate="rk4")
    _assert_allclose(np.linalg.norm(x1[ATT]), 1.0, atol=1e-6)
    _assert_allclose(np.linalg.norm(x2[ATT]), 1.0, atol=1e-6)
    _assert_allclose(x1[POS], x[POS], atol=1e-5)
    _assert_allclose(x2[POS], x[POS], atol=1e-5)
    # Slight upward thrust > hover should increase z
    u_up = np.ones(4) * (hover * 1.1)
    x3 = model.step(x, u_up, dt, integrate="rk4")
    _assert_true(x3[POS][2] > x[POS][2], "extra thrust should increase z")
    print("  [OK] quadrotor_model: step (euler/rk4), quat norm, position change")


def test_quadrotor_model_observation_state_roundtrip():
    """observation_from_state and state_from_observation roundtrip."""
    from scripts.quadrotor_model import QuadrotorModel, STATE_DIM, ATT, VEL, POS

    model = QuadrotorModel()
    x = np.zeros(STATE_DIM)
    x[ATT] = [1.0, 0.0, 0.0, 0.0]
    x[POS] = [1.0, -2.0, 5.0]
    x[VEL] = [0.5, 0.0, 0.0]
    goal = np.array([0.0, 0.0, 5.0])
    obs = model.observation_from_state(x, goal_pos=goal)
    _assert_true(obs.shape == (13,) or obs.size >= 13)
    x_back = model.state_from_observation(obs, goal_pos=goal)
    _assert_allclose(x_back[POS], x[POS], atol=1e-5)
    _assert_allclose(x_back[VEL], x[VEL], atol=1e-5)
    _assert_allclose(np.linalg.norm(x_back[ATT]), 1.0)
    print("  [OK] quadrotor_model: observation_from_state / state_from_observation roundtrip")


def test_quadrotor_model_allocation():
    """Thrust/torque from motor thrusts and inverse."""
    from scripts.quadrotor_model import QuadrotorModel

    model = QuadrotorModel()
    u = np.ones(4) * 2.0
    ft = model.thrust_torque_from_motor_thrusts(u)
    _assert_true(ft.shape == (4,))
    _assert_allclose(ft[0], 4 * 2.0)
    u2 = model.motor_thrusts_from_thrust_torque(ft[0], ft[1:4], clamp=True)
    _assert_allclose(u2, u, atol=1e-5)
    print("  [OK] quadrotor_model: thrust_torque_from_motor_thrusts, motor_thrusts_from_thrust_torque")


# -----------------------------------------------------------------------------
# 2. CBF filter
# -----------------------------------------------------------------------------

def test_cbf_position_barrier():
    """PositionBarrier (kv=0) and HOCBFBarrier h(p, v) = n'p + q + kv*(n'v)."""
    from scripts.cbf_filter import PositionBarrier, HOCBFBarrier

    # ground: z >= 0  =>  n = [0,0,1], q = 0. PositionBarrier has kv=0.
    bar = PositionBarrier([0, 0, 1], 0, "ground")
    v_zero = np.zeros(3)
    _assert_allclose(bar.h([0, 0, 5], v_zero), 5.0)
    _assert_allclose(bar.h([0, 0, 0], v_zero), 0.0)
    _assert_true(bar.h([0, 0, -1], v_zero) < 0)
    # velocity-aware: approaching ground (v_z < 0) decreases h
    hocbf = HOCBFBarrier([0, 0, 1], 0.1, kv=0.5, name="ground")
    _assert_true(hocbf.h([0, 0, 0.1], [0, 0, -1.0]) < hocbf.h([0, 0, 0.1], [0, 0, 0]))
    print("  [OK] cbf_filter: PositionBarrier / HOCBFBarrier h(p, v)")


def test_cbf_lie_derivatives():
    """L_f h and L_g h for position barrier."""
    from scripts.cbf_filter import PositionBarrier, compute_position_barrier_lie_derivatives
    from scripts.quadrotor_model import QuadrotorModel, STATE_DIM, ATT, VEL

    model = QuadrotorModel()
    bar = PositionBarrier([0, 0, 1], 0)
    x = np.zeros(STATE_DIM)
    x[ATT] = [1.0, 0.0, 0.0, 0.0]
    x[VEL] = [0.0, 0.0, 1.0]
    L_f, L_g = compute_position_barrier_lie_derivatives(model, x, bar)
    _assert_allclose(L_f, 1.0)
    _assert_true(L_g.shape == (4,))
    _assert_allclose(L_g, np.ones(4) / model.mass)
    print("  [OK] cbf_filter: compute_position_barrier_lie_derivatives")


def test_cbf_qp_solver():
    """QP solver returns feasible solution (tries osqp then scipy)."""
    from scripts.cbf_filter import solve_cbf_qp

    n = 4
    H = 2.0 * np.eye(n)
    g = np.zeros(n)
    A_ineq = np.empty((0, n))
    b_ineq = np.empty(0)
    lb = np.array([-1.0, -1.0, -1.0, -1.0])
    ub = np.array([1.0, 1.0, 1.0, 1.0])

    x = solve_cbf_qp(H, g, A_ineq, b_ineq, lb, ub, solver="osqp")
    if x is None:
        x = solve_cbf_qp(H, g, A_ineq, b_ineq, lb, ub, solver="scipy")
    _assert_true(x is not None, "QP (no ineq) should return solution (install osqp or use scipy)")
    _assert_allclose(x, np.zeros(n), atol=1e-5)

    A_ineq = np.array([[-1.0, -1.0, 0.0, 0.0]])
    b_ineq = np.array([-1.0])
    x = solve_cbf_qp(H, g, A_ineq, b_ineq, lb, ub, solver="osqp")
    if x is None:
        x = solve_cbf_qp(H, g, A_ineq, b_ineq, lb, ub, solver="scipy")
    _assert_true(x is not None, "QP (one ineq) should return solution")
    _assert_true(x[0] + x[1] >= 0.99, f"constraint x0+x1>=1: got x={x}, sum={x[0]+x[1]}")
    print("  [OK] cbf_filter: solve_cbf_qp (osqp/scipy)")


def test_cbf_qp_infeasible_returns_none():
    """Edge case: impossible constraints (x>=10 and x<=2 with box [-1,1]) -> solver returns None."""
    from scripts.cbf_filter import solve_cbf_qp

    n = 4
    H = 2.0 * np.eye(n)
    g = np.zeros(n)
    # x[0] <= -10  and  x[0] >= 2  with lb=-1, ub=1  -> impossible
    A_ineq = np.array([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    b_ineq = np.array([-10.0, -2.0])  # x0 <= -10 and -x0 <= -2 => x0 >= 2
    lb = np.array([-1.0, -1.0, -1.0, -1.0])
    ub = np.array([1.0, 1.0, 1.0, 1.0])
    x_osqp = solve_cbf_qp(H, g, A_ineq, b_ineq, lb, ub, solver="osqp")
    x_scipy = solve_cbf_qp(H, g, A_ineq, b_ineq, lb, ub, solver="scipy")
    # Infeasible: at least one solver should return None; or any returned point cannot satisfy both constraints
    ok = (
        x_osqp is None
        or x_scipy is None
        or (x_osqp is not None and (x_osqp[0] > 1.5 or x_osqp[0] < -9.5))  # violates one constraint
        or (x_scipy is not None and (x_scipy[0] > 1.5 or x_scipy[0] < -9.5))
    )
    _assert_true(ok, "infeasible QP: solver(s) should return None or a point violating constraints")
    print("  [OK] cbf_filter: infeasible QP -> solver returns None or invalid point")


def test_cbf_filter_infeasible_graceful():
    """Edge case: when QP is infeasible (e.g. cannot save from crash), filter returns clipped u_RL, no crash."""
    from scripts.cbf_filter import CBFFilter
    from scripts.quadrotor_model import STATE_DIM, INPUT_DIM

    flt = CBFFilter()
    u_min, u_max = flt.model.get_thrust_limits()
    u_rl = np.ones(4) * (u_min + u_max) / 2.0  # nominal
    # State that can make QP infeasible: on its side (L_g h ~ 0) + low + fast downward
    x = np.zeros(STATE_DIM)
    # 90° pitch: body-z horizontal -> n' R e_z = 0 for ground barrier
    x[3:7] = [np.cos(0.5 * np.pi / 2), 0, np.sin(0.5 * np.pi / 2), 0]  # rotate 90° around y
    x[0:3] = [0, 0, 0.3]
    x[7:10] = [0, 0, -5.0]
    u_safe = flt.filter(x, u_rl)
    _assert_true(u_safe is not None and u_safe.shape == (INPUT_DIM,), "filter must return 4-vector")
    _assert_true(np.all(u_safe >= u_min - 1e-6) and np.all(u_safe <= u_max + 1e-6), "u_safe within thrust limits")
    print("  [OK] cbf_filter: infeasible QP -> filter returns clipped u_RL, no crash")


def test_cbf_filter_safe_state():
    """At safe state with hover thrust, filter returns same thrust."""
    from scripts.cbf_filter import CBFFilter
    from scripts.quadrotor_model import STATE_DIM

    flt = CBFFilter()
    x = np.zeros(STATE_DIM)
    x[3:7] = [1, 0, 0, 0]
    x[0:3] = [0, 0, 5]
    u_rl = np.ones(4) * (flt.model.mass * (-flt.model.gravity) / 4.0)
    u_safe = flt.filter(x, u_rl)
    _assert_allclose(u_safe, u_rl, atol=1e-4)
    print("  [OK] cbf_filter: CBFFilter at safe state returns ~ u_RL")


def test_cbf_filter_near_ground():
    """Near ground with downward velocity, filter may increase thrust."""
    from scripts.cbf_filter import CBFFilter
    from scripts.quadrotor_model import STATE_DIM

    flt = CBFFilter()
    x = np.zeros(STATE_DIM)
    x[3:7] = [1, 0, 0, 0]
    x[0:3] = [0, 0, 0.5]
    x[7:10] = [0, 0, -1.0]
    u_rl = np.ones(4) * (flt.model.mass * (-flt.model.gravity) / 4.0) * 0.9
    u_safe = flt.filter(x, u_rl)
    _assert_true(np.all(u_safe >= 0))
    total_safe = np.sum(u_safe)
    total_rl = np.sum(u_rl)
    _assert_true(total_safe >= total_rl - 0.01)
    print("  [OK] cbf_filter: CBFFilter near ground produces valid u_safe")


def test_cbf_filter_attitude_loss_of_control_authority():
    """Edge case: drone 90° on its side; for ground barrier L_g h ~ 0 (vertical thrust can't stop vertical fall)."""
    from scripts.cbf_filter import PositionBarrier, compute_position_barrier_lie_derivatives, CBFFilter
    from scripts.quadrotor_model import QuadrotorModel, R_from_q, quaternion_normalize, STATE_DIM, ATT, POS

    model = QuadrotorModel()
    ground = PositionBarrier([0, 0, 1], 0, "ground")
    # Upright: body z = world z -> n' R e_z = 1
    x_upright = np.zeros(STATE_DIM)
    x_upright[ATT] = [1.0, 0.0, 0.0, 0.0]
    L_f_u, L_g_u = compute_position_barrier_lie_derivatives(model, x_upright, ground)
    _assert_true(abs(L_g_u[0]) > 0.1, "upright: L_g h should be significant")

    # 90° pitch (nose down): body z = world -x -> R e_z has no world-z component -> n' R e_z = 0
    half = 0.5 * np.pi / 2
    q_90y = quaternion_normalize(np.array([np.cos(half), 0, np.sin(half), 0]))
    x_side = np.zeros(STATE_DIM)
    x_side[ATT] = q_90y
    x_side[POS] = [0, 0, 1.0]
    L_f_s, L_g_s = compute_position_barrier_lie_derivatives(model, x_side, ground)
    _assert_true(np.abs(L_g_s[0]) < 0.01, "90° on side: L_g h (control authority) should be ~0")

    # Filter with this state must still return valid thrust (clipped u_RL when QP infeasible)
    flt = CBFFilter()
    u_rl = np.ones(4) * (model.mass * (-model.gravity) / 4.0)
    u_safe = flt.filter(x_side, u_rl)
    _assert_true(u_safe is not None and len(u_safe) == 4)
    thrust_min, thrust_max = model.get_thrust_limits()
    _assert_true(np.all(u_safe >= thrust_min - 1e-6) and np.all(u_safe <= thrust_max + 1e-6))
    print("  [OK] cbf_filter: attitude-induced L_g h~0; filter still returns valid u_safe")


# -----------------------------------------------------------------------------
# 3. CBF wrapper (with mock env)
# -----------------------------------------------------------------------------

def test_cbf_wrapper_mock_env():
    """CBFWrapper with a minimal mock VecEnv."""
    from scripts.cbf_wrapper import CBFWrapper

    class MockVenv:
        def __init__(self, n=2):
            self._num_envs = n
            self._obs = np.zeros((n, 13), dtype=np.float32)
            self._last_step = None

        @property
        def num_envs(self):
            return self._num_envs

        @property
        def observation_space(self):
            import gymnasium as gym
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        @property
        def action_space(self):
            import gymnasium as gym
            return gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        def reset(self, **kwargs):
            self._obs = np.zeros((self._num_envs, 13), dtype=np.float32)
            self._obs[:, 3] = 1.0
            self._obs[:, 2] = 5.0
            return self._obs.copy()

        def step_async(self, actions):
            self._last_step = actions

        def step_wait(self):
            self._obs = self._obs + 0.01
            return self._obs.copy(), np.zeros(self._num_envs), np.zeros(self._num_envs, dtype=bool), [{}] * self._num_envs

    venv = MockVenv(2)
    goal = [0.0, 0.0, 5.0]
    wrapper = CBFWrapper(venv, goal_position=goal)
    _assert_true(wrapper.num_envs == 2)
    obs = wrapper.reset()
    _assert_true(hasattr(wrapper, "_last_obs") and wrapper._last_obs is not None)
    act = np.zeros((2, 4), dtype=np.float32)
    wrapper.step_async(act)
    obs2, r, d, i = wrapper.step_wait()
    _assert_true(obs2.shape == (2, 13))
    _assert_true(venv._last_step is not None and venv._last_step.shape == (2, 4))
    print("  [OK] cbf_wrapper: CBFWrapper with mock env (reset, step_async, step_wait)")


# -----------------------------------------------------------------------------
# 4. Pipeline (Flightmare / val compatibility)
# -----------------------------------------------------------------------------

def _flightmare_available():
    """Return True if flightlib/flightgym can be imported (Flightmare env)."""
    try:
        from flightgym import QuadrotorEnv_v1
        return True
    except ModuleNotFoundError:
        pass
    try:
        from flightlib import QuadrotorEnv_v1
        return True
    except ModuleNotFoundError:
        pass
    return False


def test_pipeline_flightmare_env_build_and_step():
    """Build Flightmare env (same stack as val/train), reset and run a few steps. Skips if flightlib not available."""
    if not _flightmare_available():
        if "pytest" in sys.modules:
            import pytest
            pytest.skip("flightlib/flightgym not available — install from flightmare/flightlib")
        print("  [SKIP] pipeline: flightlib/flightgym not available")
        return
    import copy
    from scripts.config_loader import load_config
    from scripts.env_wrapper import FlightlibVecEnv

    config_path = _REPO_ROOT / "configs" / "drone_ppo_default.yaml"
    if not config_path.is_file():
        if "pytest" in sys.modules:
            import pytest
            pytest.skip(f"config not found: {config_path}")
        print(f"  [SKIP] pipeline: config not found {config_path}")
        return

    cfg = load_config(str(config_path))
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("env", {}).setdefault("vec_env", {})["num_envs"] = 1
    cfg.setdefault("paths", {})
    cfg["paths"] = {}  # so prepare_env_run_dir returns None and we use vec_config_str (no file write)

    from scripts import train as train_module
    env = train_module._make_env(cfg)
    _assert_true(env.num_envs >= 1)
    _assert_true(env.observation_space.shape[0] >= 13)
    _assert_true(env.action_space.shape[0] >= 4)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    _assert_true(obs.shape[0] >= 1 and obs.shape[1] >= 13)

    for _ in range(5):
        act = np.random.uniform(-1.0, 1.0, (env.num_envs, env.action_space.shape[0])).astype(np.float32)
        obs, rewards, dones, infos = env.step(act)
        _assert_true(obs.shape[0] == env.num_envs and obs.shape[1] >= 13)
        _assert_true(rewards.shape == (env.num_envs,))
        _assert_true(dones.shape == (env.num_envs,))
        _assert_true(len(infos) == env.num_envs)

    env.close()
    print("  [OK] pipeline: Flightmare env build, reset, step (val/train compatible)")


def test_pipeline_val_with_cbf_wrapper():
    """Run val-style loop with CBF wrapper: env + CBFWrapper, reset, several steps with random actions. Skips if flightlib not available."""
    if not _flightmare_available():
        if "pytest" in sys.modules:
            import pytest
            pytest.skip("flightlib/flightgym not available — install from flightmare/flightlib")
        print("  [SKIP] pipeline: flightlib/flightgym not available")
        return
    import copy
    from scripts.config_loader import load_config
    from scripts.cbf_wrapper import CBFWrapper

    config_path = _REPO_ROOT / "configs" / "drone_ppo_default.yaml"
    if not config_path.is_file():
        if "pytest" in sys.modules:
            import pytest
            pytest.skip(f"config not found: {config_path}")
        print(f"  [SKIP] pipeline: config not found {config_path}")
        return

    cfg = load_config(str(config_path))
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("env", {}).setdefault("vec_env", {})["num_envs"] = 1
    cfg["paths"] = {}

    from scripts import train as train_module
    env = train_module._make_env(cfg)
    env_cfg = cfg.get("env", {})
    goal_pos = env_cfg.get("goal_position", [0.0, 0.0, 5.0])
    qd = env_cfg.get("quadrotor_dynamics", {})
    mass = float(qd.get("mass", 0.774))
    g = 9.81
    act_mean = np.full(4, (mass * g) / 4.0, dtype=np.float32)
    act_std = np.full(4, (mass * 2 * g) / 4.0, dtype=np.float32)
    env = CBFWrapper(env, goal_position=goal_pos, act_mean=act_mean, act_std=act_std)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    _assert_true(obs.shape[0] >= 1 and obs.shape[1] >= 13)

    for _ in range(5):
        act = np.random.uniform(-1.0, 1.0, (env.num_envs, env.action_space.shape[0])).astype(np.float32)
        obs, rewards, dones, infos = env.step(act)
        _assert_true(obs.shape[0] == env.num_envs and obs.shape[1] >= 13)
        _assert_true(rewards.shape == (env.num_envs,))
        _assert_true(dones.shape == (env.num_envs,))

    env.close()
    print("  [OK] pipeline: val-style env + CBF wrapper, reset, step (no crash)")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

def run_all():
    """Run all tests and print summary."""
    tests = [
        ("Quadrotor model", [
            test_quadrotor_model_config_and_constants,
            test_quadrotor_model_quaternion_helpers,
            test_quadrotor_model_quaternion_normalize_zero_vector,
            test_quadrotor_model_hover_derivative,
            test_quadrotor_model_step,
            test_quadrotor_model_observation_state_roundtrip,
            test_quadrotor_model_allocation,
        ]),
        ("CBF filter", [
            test_cbf_position_barrier,
            test_cbf_lie_derivatives,
            test_cbf_qp_solver,
            test_cbf_qp_infeasible_returns_none,
            test_cbf_filter_infeasible_graceful,
            test_cbf_filter_attitude_loss_of_control_authority,
            test_cbf_filter_safe_state,
            test_cbf_filter_near_ground,
        ]),
        ("CBF wrapper", [
            test_cbf_wrapper_mock_env,
        ]),
        ("Pipeline (Flightmare / val)", [
            test_pipeline_flightmare_env_build_and_step,
            test_pipeline_val_with_cbf_wrapper,
        ]),
    ]
    failed = []
    for group, fns in tests:
        print(f"\n--- {group} ---")
        for f in fns:
            try:
                f()
            except Exception as e:
                failed.append((f.__name__, str(e)))
                print(f"  [FAIL] {f.__name__}: {e}")
    if failed:
        print(f"\nTotal: {len(failed)} failure(s)")
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)
    print("\nAll component tests passed.")


if __name__ == "__main__":
    run_all()
