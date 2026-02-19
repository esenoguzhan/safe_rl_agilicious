"""
Load PPO drone config YAML and optionally write flightlib vec_env / quadrotor_env
configs to a run directory for use with flightmare_context.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the main PPO drone config YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _ensure_run_dir(run_dir: str) -> Path:
    run_path = Path(run_dir)
    config_dir = run_path / "flightlib" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return run_path


_VEC_ENV_DEFAULTS = {
    "seed": 1,
    "scene_id": 0,
    "num_envs": 4,
    "num_threads": 2,
    "render": False,
}


def _vec_env_yaml_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build vec_env section for flightlib from config env.vec_env."""
    env = cfg.get("env", {})
    vec = env.get("vec_env", {})
    merged = {**_VEC_ENV_DEFAULTS, **{k: vec[k] for k in vec if k in _VEC_ENV_DEFAULTS}}
    # ensure render is bool for C++
    r = merged.get("render")
    if isinstance(r, str):
        merged["render"] = r.lower() in ("yes", "true", "1")
    return {"env": merged}


def _quadrotor_env_yaml_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Build full quadrotor_env YAML dict from config env.quadrotor_env and env.quadrotor_dynamics."""
    env = cfg.get("env", {})
    qe = env.get("quadrotor_env")
    qd = env.get("quadrotor_dynamics")
    rl = env.get("rl")
    out = {}
    if qe:
        out["quadrotor_env"] = qe
    if qd:
        out["quadrotor_dynamics"] = qd
    if rl:
        out["rl"] = rl
    return out if out else None


def write_env_configs(cfg: Dict[str, Any], run_dir: str) -> str:
    """
    Write vec_env.yaml and quadrotor_env.yaml under run_dir/flightlib/configs/
    from config sections env.vec_env and env.quadrotor_env.
    Returns run_dir (absolute) for use with flightmare_context.
    """
    run_path = _ensure_run_dir(run_dir)
    config_dir = run_path / "flightlib" / "configs"

    vec = _vec_env_yaml_from_config(cfg)
    if vec:
        with open(config_dir / "vec_env.yaml", "w") as f:
            yaml.dump(vec, f, default_flow_style=False, sort_keys=False)

    quad = _quadrotor_env_yaml_from_config(cfg)
    if quad:
        with open(config_dir / "quadrotor_env.yaml", "w") as f:
            yaml.dump(quad, f, default_flow_style=False, sort_keys=False)

    return str(run_path.resolve())


def get_vec_env_config_string(cfg: Dict[str, Any]) -> str:
    """Build vec_env YAML as a string for QuadrotorEnv_v1(cfg_string, from_file=False)."""
    vec = _vec_env_yaml_from_config(cfg)
    return yaml.dump(vec, default_flow_style=False, sort_keys=False)


def prepare_env_run_dir(cfg: Dict[str, Any], paths: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Only when config has env.quadrotor_env (or quadrotor_dynamics or rl), write both
    vec_env and quadrotor_env to run_dir and return it for flightmare_context.
    When only env.vec_env is set, return None so the C++ loads quadrotor_env.yaml
    from the original FLIGHTMARE_PATH (avoids "bad file" for missing quadrotor_env.yaml).
    """
    paths = paths or cfg.get("paths", {})
    run_dir = paths.get("log_dir") or paths.get("save_dir") or paths.get("env_config_dir")
    if not run_dir:
        return None
    env = cfg.get("env", {})
    has_quadrotor_config = env.get("quadrotor_env") or env.get("quadrotor_dynamics") or env.get("rl")
    if not has_quadrotor_config:
        return None
    return write_env_configs(cfg, run_dir)
