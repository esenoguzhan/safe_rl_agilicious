#!/usr/bin/env python3
"""
Offline diagnostic for the SB3 PPO + VecNormalize bundle used by
rl_feedthrough_*.py.

Run on the host (no ROS needed) to answer:
  - Does the policy output near-hover thrusts at pos_err = 0?
  - Are the VecNormalize running stats sane (non-zero variance, finite)?
  - Which motor permutation (if any) makes the differential thrust
    pattern match the agilicious [FR, BL, BR, FL] convention when the
    policy was likely trained against flightmare's [FR, FL, BL, BR]?

Usage:
  python3 rl_probe_policy.py \
    --model-path ~/Desktop/agilicious_repo/agilicious/PPO_50000000_seq6dec_s6_tau_dr_ph500_rl50/best_model.zip \
    --vecnormalize-path ~/Desktop/agilicious_repo/agilicious/PPO_50000000_seq6dec_s6_tau_dr_ph500_rl50/vecnormalize.pkl
"""
import argparse
import os
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rl_feedthrough_core import (  # noqa: E402
    RlFeedthroughCore,
    default_paths_under_repo,
)


# Motor index conventions, indexed by physical position (FR, FL, BL, BR).
AGILICIOUS_ORDER = ("FR", "BL", "BR", "FL")        # Command.thrusts[0..3]
FLIGHTMARE_ORDER = ("FR", "FL", "BL", "BR")        # stock flightmare default


def perm_from_to(src_order, dst_order):
    """Return a permutation `p` such that dst_vec[i] == src_vec[p[i]]."""
    return tuple(src_order.index(name) for name in dst_order)


# Candidates to A/B against agilicious [FR, BL, BR, FL]
PERM_CANDIDATES = {
    "identity                   (FR,BL,BR,FL)": (0, 1, 2, 3),
    "flightmare→agilicious      (FR,FL,BL,BR -> FR,BL,BR,FL)":
        perm_from_to(FLIGHTMARE_ORDER, AGILICIOUS_ORDER),
    "swap-BL-BR (1,2)            ": (0, 2, 1, 3),
    "reverse                    ": (3, 2, 1, 0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="")
    p.add_argument("--vecnormalize-path", default="")
    p.add_argument("--quad-mass-kg", type=float, default=0.774)
    p.add_argument("--gravity-z", type=float, default=-9.81)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def build_obs(pos_err, q_wxyz=(1.0, 0.0, 0.0, 0.0),
              v_world=(0.0, 0.0, 0.0), w_body=(0.0, 0.0, 0.0),
              hist_20=None):
    base = np.concatenate([
        np.asarray(pos_err, dtype=np.float32),
        np.asarray(q_wxyz, dtype=np.float32),
        np.asarray(v_world, dtype=np.float32),
        np.asarray(w_body, dtype=np.float32),
    ]).astype(np.float32)
    if hist_20 is None:
        hist_20 = np.zeros(20, dtype=np.float32)
    return np.concatenate([base, np.asarray(hist_20, dtype=np.float32)])


def thrusts_from_action(act, mu, sig):
    a = np.clip(np.asarray(act, dtype=np.float64).reshape(-1), -1.0, 1.0)
    t = a * sig + mu
    return np.maximum(t, 0.0)


def fmt_arr(a, n=3):
    return "[" + ", ".join(f"{v:+.{n}f}" for v in np.asarray(a).reshape(-1)) + "]"


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main():
    args = parse_args()

    dm, dv = default_paths_under_repo(_SCRIPT_DIR)
    model_path = (args.model_path.strip()
                  or (dm if os.path.isfile(dm) else ""))
    vnorm_path = (args.vecnormalize_path.strip()
                  or (dv if os.path.isfile(dv) else ""))
    if not model_path or not os.path.isfile(model_path):
        print(f"ERROR: model not found: {model_path}")
        return 2

    core = RlFeedthroughCore(
        model_path=model_path,
        vecnormalize_path=vnorm_path,
        device=args.device,
        quad_mass_kg=args.quad_mass_kg,
        gravity_z=args.gravity_z,
    )
    if not core.load_policy():
        print("ERROR: load_policy() failed.")
        return 2

    mg = args.quad_mass_kg * abs(args.gravity_z)
    mu, sig = core._act_mean, core._act_std
    print(f"mass = {args.quad_mass_kg} kg   |g| = {abs(args.gravity_z)} m/s^2")
    print(f"m*g  = {mg:.3f} N   hover/4 = {mg/4:.3f} N")
    print(f"act_mean = {fmt_arr(mu)}")
    print(f"act_std  = {fmt_arr(sig)}")
    print(f"-> action= 0 -> thrust = {fmt_arr(thrusts_from_action(np.zeros(4), mu, sig))} N "
          f"(sum={float(thrusts_from_action(np.zeros(4), mu, sig).sum()):.3f} N)")

    # ----- VecNormalize stats -----
    section("VecNormalize obs running statistics (per-feature)")
    vn = core._vecnorm
    if vn is None:
        print("WARNING: VecNormalize was not loaded; obs are unnormalized.")
    else:
        mean = np.asarray(vn.obs_rms.mean).reshape(-1)
        var = np.asarray(vn.obs_rms.var).reshape(-1)
        std = np.sqrt(np.maximum(var, 1e-12))
        names = (
            ["pos_err_x", "pos_err_y", "pos_err_z",
             "q_w", "q_x", "q_y", "q_z",
             "v_x", "v_y", "v_z",
             "w_x", "w_y", "w_z"] +
            [f"hist_{i//4}_a{i%4}" for i in range(20)]
        )
        n = min(len(names), mean.size)
        for i in range(n):
            print(f"  {i:2d} {names[i]:<14s}  mean={mean[i]:+8.4f}  std={std[i]:8.4f}")
        print(f"  count = {float(vn.obs_rms.count):.0f}  "
              f"clip_obs = {getattr(vn, 'clip_obs', 'n/a')}  "
              f"epsilon = {getattr(vn, 'epsilon', 'n/a')}")

        red_flags = []
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            red_flags.append("non-finite values in obs_rms")
        if np.any(std[:13] < 1e-3):
            red_flags.append("near-zero std in core 13 dims (collapsed normalization)")
        if vn.obs_rms.count < 1e3:
            red_flags.append("very low obs_rms.count (VecNormalize barely accumulated)")
        if red_flags:
            print("  RED FLAGS: " + "; ".join(red_flags))

    # ----- Policy at perfect hover -----
    section("Probe 1: drone exactly at goal, identity attitude, zero vel/omega")
    obs = build_obs([0, 0, 0])
    a = core.predict_action(obs)
    t = thrusts_from_action(a, mu, sig)
    print(f"action            = {fmt_arr(a)}")
    print(f"thrusts (agi-idx) = {fmt_arr(t)} N   sum = {float(t.sum()):.3f} N "
          f"(target {mg:.3f} N)")
    print(f"agilicious order  : [{', '.join(AGILICIOUS_ORDER)}]")
    if abs(float(t.sum()) - mg) < 0.5:
        print("  -> total thrust ~ m*g: policy is producing a healthy hover.")
    else:
        print("  -> total thrust DEVIATES from m*g: policy may not be loading "
              "correctly, or VecNormalize stats are off.")

    # ----- Pos error sign sanity check -----
    section("Probe 2: drone 2 m below goal (z=3, goal_z=5)")
    print("Training convention: pos_err = goal - pos  (your script uses this).")
    print(f"  pos_err = (0, 0, +2): drone needs to climb -> total thrust > m*g.")
    obs = build_obs([0, 0, +2.0])
    a = core.predict_action(obs)
    t = thrusts_from_action(a, mu, sig)
    print(f"  action            = {fmt_arr(a)}")
    print(f"  thrusts (agi-idx) = {fmt_arr(t)} N   sum = {float(t.sum()):.3f} N")
    print(f"  -> climb? {'YES' if t.sum() > mg + 0.05 else ('NO' if t.sum() < mg - 0.05 else 'flat')}")

    print("\nFlipped sign sanity (should produce DESCENT if training really used g-p):")
    obs = build_obs([0, 0, -2.0])
    a = core.predict_action(obs)
    t = thrusts_from_action(a, mu, sig)
    print(f"  pos_err = (0, 0, -2): action = {fmt_arr(a)}")
    print(f"                       thrusts = {fmt_arr(t)} N   sum = {float(t.sum()):.3f} N")
    print(f"  -> climb? {'YES' if t.sum() > mg + 0.05 else ('NO' if t.sum() < mg - 0.05 else 'flat')}")

    # ----- Differential probe: how does the policy distribute thrust under tilt? -----
    section("Probe 3a: small roll-rate (w_x = +1 rad/s, body)  -> identifies LEFT vs RIGHT")
    obs = build_obs([0, 0, 0], w_body=[1.0, 0.0, 0.0])
    a = core.predict_action(obs)
    t = thrusts_from_action(a, mu, sig)
    diff = t - t.mean()
    print(f"action            = {fmt_arr(a)}")
    print(f"thrusts           = {fmt_arr(t)} N   sum = {float(t.sum()):.3f}")
    print(f"thrust − mean     = {fmt_arr(diff, n=4)}")
    boosted_x = sorted(np.argsort(diff)[-2:].tolist())
    print(f"=> policy BOOSTS indices {boosted_x} -> these are RIGHT-side motors")
    print(f"   policy CUTS   indices {sorted(np.argsort(diff)[:2].tolist())} -> these are LEFT-side motors")

    section("Probe 3b: small pitch-rate (w_y = +1 rad/s, body)  -> identifies FRONT vs BACK")
    obs = build_obs([0, 0, 0], w_body=[0.0, 1.0, 0.0])
    a = core.predict_action(obs)
    t = thrusts_from_action(a, mu, sig)
    diff = t - t.mean()
    print(f"action            = {fmt_arr(a)}")
    print(f"thrusts           = {fmt_arr(t)} N   sum = {float(t.sum()):.3f}")
    print(f"thrust − mean     = {fmt_arr(diff, n=4)}")
    boosted_y = sorted(np.argsort(diff)[-2:].tolist())
    print(f"=> policy BOOSTS indices {boosted_y} -> these are FRONT motors")
    print(f"   policy CUTS   indices {sorted(np.argsort(diff)[:2].tolist())} -> these are BACK motors")

    # Combine the two probes to nail down the motor identity at each policy index.
    section("Inferred motor identity per POLICY index, from probes 3a and 3b")
    front_idx = set(boosted_y)
    right_idx = set(boosted_x)
    name_per_idx = {}
    for i in range(4):
        is_front = i in front_idx
        is_right = i in right_idx
        name_per_idx[i] = ("F" if is_front else "B") + ("R" if is_right else "L")
    inferred = [name_per_idx[i] for i in range(4)]
    print(f"policy[0..3] = {inferred}")

    AGI_INDEX_OF = {name: i for i, name in enumerate(AGILICIOUS_ORDER)}
    if set(inferred) == set(AGILICIOUS_ORDER):
        # agi[i] = policy[ inferred.index(AGILICIOUS_ORDER[i]) ]
        perm = tuple(inferred.index(AGILICIOUS_ORDER[i]) for i in range(4))
        print(f"=> Recommended live flag: --motor-perm {','.join(str(p) for p in perm)}")
        print(f"   (this maps policy order {inferred} into agilicious order "
              f"[{', '.join(AGILICIOUS_ORDER)}])")
    else:
        print("=> WARNING: probes did not yield 4 distinct motors {FL,FR,BL,BR}. "
              "Either signs are flipped or the policy is not learning a clean roll/pitch "
              "response in this regime. Try the perm candidates in Probe 4.")

    # ----- Permutation A/B at pos_err = (0, 0, +2) -----
    section("Probe 4: candidate motor permutations under pos_err=(0,0,+2)")
    print("This shows what the thrust pattern looks like AFTER each candidate "
          "permutation is applied. Pick the perm that yields a near-symmetric "
          "thrust vector (all 4 close to each other) when the policy is asked "
          "to climb straight up — that is the most likely correct order.\n")
    obs = build_obs([0, 0, +2.0])
    a = core.predict_action(obs)
    raw = thrusts_from_action(a, mu, sig)

    print(f"raw policy thrusts: {fmt_arr(raw)}")
    for label, perm in PERM_CANDIDATES.items():
        permuted = raw[list(perm)]
        sd = float(np.std(permuted))
        print(f"  perm {perm}  std={sd:.3f}  {fmt_arr(permuted)}   {label}")

    print("\nNote: a pure climb command should produce near-equal thrusts across the "
          "four motors (low std). Any permutation that gives the LOWEST std is the "
          "best candidate for your training motor order.")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
