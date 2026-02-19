"""
Context manager for FLIGHTMARE_PATH.
Saves and restores the environment variable so config overrides don't leak.
"""
import os
from contextlib import contextmanager
from typing import Optional


@contextmanager
def flightmare_context(run_dir: str):
    """
    Set FLIGHTMARE_PATH to run_dir only inside this block; restore on exit.

    Use when creating QuadrotorEnv_v1 from configs written under run_dir so that
    C++ loads vec_env.yaml and quadrotor_env.yaml from there.

    Example:
        with flightmare_context(run_dir):
            env = QuadrotorEnv_v1("flightlib/configs/vec_env.yaml", from_file=True)
            ...
    """
    key = "FLIGHTMARE_PATH"
    prev: Optional[str] = os.environ.get(key)
    try:
        os.environ[key] = run_dir
        yield
    finally:
        if prev is not None:
            os.environ[key] = prev
        elif key in os.environ:
            os.environ.pop(key)
