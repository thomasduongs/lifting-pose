#!/usr/bin/env python3
"""Sourced ligament stiffness profile for the MyoSim reference model + sensitivity bench.

Run with .venv-biomech/bin/python.  See docs/ligament-profile-contract.md.

Source: Boardman et al. 1996, J Shoulder Elbow Surg, DOI 10.1016/s1058-2746(96)80050-4
(https://pubmed.ncbi.nlm.nih.gov/8872921/): cadaver bone-ligament-bone LINEAR STIFFNESS means,
SGHL 17.4 N/mm (reported variation 1.5), CHL 36.7 N/mm (reported variation 5.9).
Only the four native SGHL/CHL routes are mapped.  MGHL/IGHL routes have no stiffness in this
source and stay disabled; nothing is extrapolated.  The reported variations are NOT established
confidence intervals.  These are aggregate reference values, not this subject's tissue; the
spring is a linear tension-only approximation on a simplified path (no toe region, thickness,
failure, rate dependence, or subject-specific slack).
"""
import argparse
import csv
import itertools
import json
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import mujoco
    import biomech_model as bm
    HAVE_BACKEND = True
except ImportError:  # global Python without MuJoCo
    HAVE_BACKEND = False

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

N_PER_MM_TO_N_PER_M = 1000.0
DOI = "10.1016/s1058-2746(96)80050-4"
PROFILES = {
    "boardman1996": {
        "citation": "Boardman ND et al., J Shoulder Elbow Surg 1996;5(4):249-254",
        "doi": DOI,
        "url": "https://pubmed.ncbi.nlm.nih.gov/8872921/",
        "quantity": "linear stiffness of cadaveric bone-ligament-bone complex (not modulus/stress)",
        "ligaments": {
            "SGHL": {"mean_N_per_mm": 17.4, "variation_N_per_mm": 1.5,
                     "routes": ("s_glenohum_ligament", "s_glenohum_ligament_l")},
            "CHL": {"mean_N_per_mm": 36.7, "variation_N_per_mm": 5.9,
                    "routes": ("coracohum_ligament", "coracohum_ligament_l")},
        },
        "unsupported_routes": ("m_glenohum_ligament", "i_glenohum_ligament",
                               "m_glenohum_ligament_l", "i_glenohum_ligament_l"),
    },
}
CAVEAT = ("Aggregate cadaver reference stiffness applied as a linear tension-only spring on a simplified "
          "native path: no toe region, tissue thickness, failure, rate dependence or subject-specific slack "
          "identification. Reported variation is not a confidence interval. Not this subject's tissue.")
VARIANTS = ("low", "mean", "high")


class LigamentError(Exception):
    pass


def _k_si(entry, variant):
    delta = {"low": -1.0, "mean": 0.0, "high": 1.0}[variant]
    return (entry["mean_N_per_mm"] + delta * entry["variation_N_per_mm"]) * N_PER_MM_TO_N_PER_M


def apply_profile(model, data, profile="boardman1996", slack_scale=1.0, variant="mean"):
    """Set tension-only springs on the 4 supported native routes IN PLACE; disable MGHL/IGHL.
    Source slack is preserved (springlength=[0, source_slack*slack_scale]), damping 0.
    ``variant`` low/mean/high = mean -/+ reported variation (sensitivity only, not a CI).
    Calls mj_forward and returns JSON-safe metadata."""
    if profile not in PROFILES:
        raise LigamentError(f"unknown ligament profile {profile!r}; choose from {sorted(PROFILES)}")
    if not (np.isfinite(slack_scale) and slack_scale > 0):
        raise LigamentError("slack_scale must be finite and > 0")
    if variant not in VARIANTS:
        raise LigamentError(f"variant must be one of {VARIANTS}")
    prof = PROFILES[profile]
    src = bm.ligament_config(model)["routes"]
    routes = {}
    for lig, entry in prof["ligaments"].items():
        k = _k_si(entry, variant)
        for name in entry["routes"]:
            i = model.tendon(name).id
            source_slack = float(src[name]["source_springlength_m"])
            slack = source_slack * slack_scale
            model.tendon_stiffness[i] = k
            model.tendon_stiffnesspoly[i] = 0.0
            model.tendon_damping[i] = 0.0
            model.tendon_dampingpoly[i] = 0.0
            model.tendon_lengthspring[i] = [0.0, slack]
            routes[name] = {"ligament": lig, "stiffness_N_per_m": k, "source_slack_m": source_slack,
                            "slack_length_m": slack, "damping": 0.0}
    for name in prof["unsupported_routes"]:
        i = model.tendon(name).id
        model.tendon_stiffness[i] = 0.0
        model.tendon_stiffnesspoly[i] = 0.0
        model.tendon_damping[i] = 0.0
        model.tendon_dampingpoly[i] = 0.0
    mujoco.mj_forward(model, data)
    return {
        "profile": profile, "variant": variant, "slack_scale": float(slack_scale),
        "source": {k: prof[k] for k in ("citation", "doi", "url", "quantity")},
        "conversion": "N/mm * 1000 = N/m (SGHL 17.4 -> 17400; CHL 36.7 -> 36700)",
        "reported_variation_N_per_mm": {l: e["variation_N_per_mm"] for l, e in prof["ligaments"].items()},
        "variation_note": "low/high = mean -/+ reported variation; NOT established confidence bounds",
        "enabled_routes": routes,
        "unsupported_routes_disabled": list(prof["unsupported_routes"]),
        "unsupported_reason": "no matching stiffness in this source; not extrapolated from SGHL/CHL",
        "law": "F = k*max(0, L - L_slack), tension-only; MuJoCo springlength=[0, L_slack]",
        "caveat": CAVEAT,
    }


def disabled_metadata():
    return {"profile": "disabled", "enabled_routes": {}, "status": "source model: ligament stiffness 0"}


# ----------------------------------------------------------------- sensitivity bench
PLANE = (0.0, 1.2, 2.27)   # elv_angle (plane of elevation), rad; 2.27 = model upper limit
ELV = (0.8, 1.6, 2.24)
ROT = (-1.57, -0.5, 0.5)
SLACK_MULTS = (0.9, 1.0, 1.1)


def _set_pose(model, data, elv, rot, plane=0.0):
    for sfx in ("_l", "_r"):
        for nm, v in (("shoulder_elv", elv), ("shoulder_rot", rot), ("elv_angle", plane)):
            j = model.joint(nm + sfx)
            lo, hi = model.jnt_range[j.id]
            data.qpos[model.jnt_qposadr[j.id]] = float(np.clip(v, lo, hi))
    data.qvel[:] = 0
    bm.enforce_couplings(model, data)
    mujoco.mj_forward(model, data)


def bench(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, configs = [], []
    for variant, mult in itertools.product(VARIANTS, SLACK_MULTS):
        model, data = bm.load_model()
        meta = apply_profile(model, data, "boardman1996", slack_scale=mult, variant=variant)
        configs.append({"variant": variant, "slack_mult": mult, "routes": meta["enabled_routes"]})
        dofs = {n: int(model.jnt_dofadr[model.joint(n).id]) for n in
                ("shoulder_elv_l", "shoulder_elv_r", "shoulder_rot_l", "shoulder_rot_r")}
        for plane, elv, rot in itertools.product(PLANE, ELV, ROT):
            _set_pose(model, data, elv, rot, plane)
            tau = bm.ligament_generalized_force(model, data)
            for s in bm.ligament_state(model, data):
                if s["name"] not in meta["enabled_routes"]:
                    continue
                side = "l" if s["name"].endswith("_l") else "r"
                rows.append({
                    "variant": variant, "slack_mult": mult, "plane_rad": plane, "elv_rad": elv, "rot_rad": rot,
                    "route": s["name"], "length_m": s["length_m"], "slack_m": s["slack_length_m"],
                    "stiffness_N_per_m": s["stiffness_N_per_m"], "strain": s["strain"],
                    "tension_N": s["tension_N"], "energy_J": s["energy_J"],
                    "tau_shoulder_elv_Nm": tau[dofs[f"shoulder_elv_{side}"]],
                    "tau_shoulder_rot_Nm": tau[dofs[f"shoulder_rot_{side}"]],
                    "tau_norm_Nm": float(np.linalg.norm(tau)),
                })
    # Consistency: passive force difference (spring on vs off) == -J^T * tension, at the tautest pose.
    tight = max(rows, key=lambda r: r["tension_N"])
    check = {"performed_at": {k: tight[k] for k in ("variant", "slack_mult", "plane_rad", "elv_rad", "rot_rad")}}
    off, d_off = bm.load_model()
    on, d_on = bm.load_model()
    apply_profile(on, d_on, "boardman1996", slack_scale=tight["slack_mult"], variant=tight["variant"])
    _set_pose(off, d_off, tight["elv_rad"], tight["rot_rad"], tight["plane_rad"])
    _set_pose(on, d_on, tight["elv_rad"], tight["rot_rad"], tight["plane_rad"])
    diff = d_on.qfrc_passive - d_off.qfrc_passive
    pred = bm.ligament_generalized_force(on, d_on)
    check.update({"max_abs_difference": float(np.abs(diff - pred).max()),
                  "max_abs_force": float(np.abs(pred).max()), "tolerance": 1e-8,
                  "passed": bool(np.abs(diff - pred).max() < 1e-8 and np.abs(pred).max() > 0),
                  "statement": "qfrc_passive(on)-qfrc_passive(off) vs -J^T*tension from native tendon Jacobian"})
    with open(out / "ligament_sensitivity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    units = {"length_m": "m", "slack_m": "m", "stiffness_N_per_m": "N/m", "strain": "dimensionless",
             "tension_N": "N", "energy_J": "J", "tau_*_Nm": "N*m (generalized force, J^T*F)",
             "angles": "rad"}
    report = {
        "source": {k: PROFILES["boardman1996"][k] for k in ("citation", "doi", "url", "quantity")},
        "caveat": CAVEAT, "units": units,
        "design": {"slack_multipliers": SLACK_MULTS, "stiffness_variants": VARIANTS,
                   "variant_definition": "mean -/+ reported variation; not confidence bounds",
                   "pose_sweep": {"elv_angle_rad": PLANE, "shoulder_elv_rad": ELV, "shoulder_rot_rad": ROT,
                                  "note": "both arms, elbow at standing, clipped to model limits"}},
        "unsupported_routes_disabled": list(PROFILES["boardman1996"]["unsupported_routes"]),
        "jacobian_consistency": check,
        "configs": configs,
        "summary": {"rows": len(rows), "max_tension_N": max(r["tension_N"] for r in rows),
                    "taut_rows": sum(r["tension_N"] > 0 for r in rows)},
        "claims": "Model-implied passive response of reference ligament springs; no patient/subject force claims.",
    }
    (out / "ligament_sensitivity.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description="Ligament profile sensitivity bench (Boardman 1996 SGHL/CHL)")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    if not HAVE_BACKEND:
        print("error: need .venv-biomech/bin/python (mujoco, myo_sim)", file=sys.stderr)
        return 1
    r = bench(args.output)
    print(f"rows={r['summary']['rows']} max_tension_N={r['summary']['max_tension_N']:.3g} "
          f"jacobian_check_passed={r['jacobian_consistency']['passed']}")
    return 0 if r["jacobian_consistency"]["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
