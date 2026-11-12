#!/usr/bin/env python3
"""Export the compiled MyoSim/MuJoCo anatomy, fitted motion and independent MuJoCo reference
fixtures for the standalone Rust crate ``crates/biomech-lite`` (docs/rust-biomech-contract.md).

Reads arrays straight from the compiled ``output/biomech-twin/model.mjb``; nothing is
reimplemented except the exact polynomial coupling chain rule for the reduced coordinates.
Run: ``.venv-biomech/bin/python scripts/export_biomech_lite.py [--output assets/biomech-lite]``

Coupling convention (matches native mjEQ_JOINT and biomech_model._couplings):
  master != null: q_dep = qpos0[dep] + sum_i coef[i] (q_master - qpos0[master])^i
  master == null (locked): q_dep = qpos0[dep] + coef[0]
Couplings are listed in dependency order (a master is written before it is read).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco  # noqa: E402

import biomech_model as bm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TWIN = ROOT / "output" / "biomech-twin"
DEFAULT_OUT = ROOT / "assets" / "biomech-lite"
DYNAMIC_NAMES = ["elv_angle_r", "shoulder_elv_r", "shoulder_rot_r", "elbow_flexion_r",
                 "elv_angle_l", "shoulder_elv_l", "shoulder_rot_l", "elbow_flexion_l"]
MUJOCO_URL = "https://github.com/google-deepmind/mujoco/tree/3.17.0"
SUPPORTED_JOINTS = {int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_SLIDE),
                    int(mujoco.mjtJoint.mjJNT_HINGE)}
SUPPORTED_WRAPS = {3, 4, 5}


class ExportError(Exception):
    pass


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def f(a):
    return np.asarray(a, dtype=np.float64).tolist()


def load_native(mjb=TWIN / "model.mjb"):
    return mujoco.MjModel.from_binary_path(str(mjb))


def check_supported(m):
    bad = sorted(set(int(t) for t in m.jnt_type) - SUPPORTED_JOINTS)
    if bad:
        raise ExportError("unsupported joint kinds %s (ball not implemented)" % bad)
    if m.nu == 0:
        raise ExportError("no actuators")
    if (m.actuator_trntype != int(mujoco.mjtTrn.mjTRN_TENDON)).any():
        raise ExportError("only tendon-transmission actuators are supported")
    if (m.actuator_gaintype != int(mujoco.mjtGain.mjGAIN_MUSCLE)).any() or \
            (m.actuator_biastype != int(mujoco.mjtBias.mjBIAS_MUSCLE)).any():
        raise ExportError("only muscle gain/bias actuators are supported")
    wb = sorted(set(int(t) for t in m.wrap_type) - SUPPORTED_WRAPS)
    if wb:
        raise ExportError("unsupported wrap kinds %s" % wb)
    if (m.actuator_gainprm[:, 9] != 0).any() or (m.actuator_biasprm[:, 9] != 0).any():
        raise ExportError("muscle prm[9] nonzero")
    if m.tendon_limited.any():
        raise ExportError("limited tendons unsupported")
    if (m.actuator_dyntype != int(mujoco.mjtDyn.mjDYN_MUSCLE)).any():
        raise ExportError("only muscle activation dynamics are supported")
    gear = np.zeros(6)
    gear[0] = 1.0
    if not np.allclose(m.actuator_gear, gear):
        raise ExportError("actuator gear must be [1,0,0,0,0,0]")
    if m.actuator_forcelimited.any() or m.actuator_actlimited.any() or m.actuator_actearly.any():
        raise ExportError("actuator force/act limits or actearly unsupported")
    if np.any(m.body_gravcomp != 0):
        raise ExportError("body gravcomp unsupported")
    if np.any(m.dof_frictionloss != 0):
        raise ExportError("dof frictionloss unsupported")
    if np.any(m.jnt_stiffness != 0):
        raise ExportError("joint spring stiffness unsupported")
    if np.any(m.tendon_frictionloss != 0):
        raise ExportError("tendon frictionloss unsupported")


def dynamic_joint_ids(m):
    return [int(m.joint(n).id) for n in DYNAMIC_NAMES]


def provenance(mjb_path, recording_sha):
    prov = {}
    pj = TWIN / "provenance.json"
    if pj.is_file():
        prov = json.loads(pj.read_text())
    return {
        "model": "MyoSim myofullbody 0.2.5 compiled with MuJoCo 3.17.0",
        "mjb": "output/biomech-twin/model.mjb", "mjb_sha256": sha256(mjb_path),
        "recording_sha256": recording_sha,
        "versions": {k: metadata.version(k) for k in ("myo-sim", "mujoco", "numpy")},
        "python": sys.version.split()[0], "platform": platform.platform(),
        "mujoco_source": MUJOCO_URL,
        "mujoco_files": ["src/engine/engine_util_misc.c", "src/engine/engine_core_smooth.c"],
        "myo_sim": "https://github.com/MyoHub/myo_sim (Apache-2.0)",
        "ligaments": prov.get("ligament_profile", {}).get("profile", "from compiled model arrays"),
        "coupling_convention": "master!=null: q=qpos0[dep]+poly(q_master-qpos0[master]); "
                               "master==null: q=qpos0[dep]+coef[0]",
        "anatomy_kind": "generic reference anatomy; not person-specific",
    }


def export_model_json(m, mjb_path, recording_sha):
    check_supported(m)
    c = bm._couplings(m)
    if c.unsupported:
        raise ExportError("unsupported equalities: %s" % (c.unsupported,))
    O = mujoco.mjtObj

    def nm(t, i):
        return mujoco.mj_id2name(m, t, i) or ""

    bodies = []
    for b in range(m.nbody):
        js = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == b]
        bodies.append({"name": nm(O.mjOBJ_BODY, b), "parent": int(m.body_parentid[b]),
                       "pos": f(m.body_pos[b]), "quat": f(m.body_quat[b]),
                       "ipos": f(m.body_ipos[b]), "iquat": f(m.body_iquat[b]),
                       "mass": float(m.body_mass[b]), "inertia": f(m.body_inertia[b]),
                       "joints": js})
    joints = []
    for j in range(m.njnt):
        da = int(m.jnt_dofadr[j])
        qa = int(m.jnt_qposadr[j])
        joints.append({"name": nm(O.mjOBJ_JOINT, j), "kind": int(m.jnt_type[j]),
                       "body": int(m.jnt_bodyid[j]), "qposadr": qa, "dofadr": da,
                       "pos": f(m.jnt_pos[j]), "axis": f(m.jnt_axis[j]),
                       "limited": bool(m.jnt_limited[j]), "range": f(m.jnt_range[j]),
                       "damping": float(m.dof_damping[da]), "armature": float(m.dof_armature[da]),
                       "stiffness": float(m.jnt_stiffness[j]),
                       "springref": float(m.qpos_spring[qa])})
    sites = [{"name": nm(O.mjOBJ_SITE, s), "body": int(m.site_bodyid[s]), "pos": f(m.site_pos[s])}
             for s in range(m.nsite)]
    geoms = [{"name": nm(O.mjOBJ_GEOM, g), "body": int(m.geom_bodyid[g]), "kind": int(m.geom_type[g]),
              "pos": f(m.geom_pos[g]), "quat": f(m.geom_quat[g]), "size": f(m.geom_size[g])}
             for g in range(m.ngeom)]
    couplings = []
    for e in c.eq_ids:
        mas = int(m.eq_obj2id[e])
        couplings.append({"dependent": int(m.eq_obj1id[e]), "master": mas if mas >= 0 else None,
                          "coef": f(m.eq_data[e, :5])})
    tendons = []
    for t in range(m.ntendon):
        a, n = int(m.tendon_adr[t]), int(m.tendon_num[t])
        path = [{"kind": int(m.wrap_type[w]), "object": int(m.wrap_objid[w]),
                 "prm": float(m.wrap_prm[w])} for w in range(a, a + n)]
        tendons.append({"name": nm(O.mjOBJ_TENDON, t), "path": path,
                        "stiffness": float(m.tendon_stiffness[t]), "damping": float(m.tendon_damping[t]),
                        "springlength": f(m.tendon_lengthspring[t])})
    muscles = [{"name": nm(O.mjOBJ_ACTUATOR, a), "tendon": int(m.actuator_trnid[a, 0]),
                "gainprm": f(m.actuator_gainprm[a, :9]), "biasprm": f(m.actuator_biasprm[a, :9]),
                "dynprm": f(m.actuator_dynprm[a, :3]), "lengthrange": f(m.actuator_lengthrange[a]),
                "acc0": float(m.actuator_acc0[a]), "ctrlrange": f(m.actuator_ctrlrange[a])}
               for a in range(m.nu)]
    standing = standing_qpos(m)
    return {
        "schema_version": 1,
        "source": provenance(mjb_path, recording_sha),
        "nq": int(m.nq), "nv": int(m.nv), "qpos0": f(m.qpos0), "standing_qpos": f(standing),
        "gravity": f(m.opt.gravity), "dynamic_joints": dynamic_joint_ids(m),
        "bodies": bodies, "joints": joints, "sites": sites, "geoms": geoms,
        "couplings": couplings, "tendons": tendons, "muscles": muscles,
    }


def standing_qpos(m):
    q = m.key_qpos[int(m.key("standing").id)].copy()
    bm.project_couplings_qpos(m, q)
    return q


def export_motion_json(m, recording_sha):
    d = np.load(TWIN / "motion.npz", allow_pickle=True)
    q = np.array(d["qpos"], dtype=np.float64)
    if q.shape[1] != m.nq:
        raise ExportError("motion qpos width != nq")
    return {"schema_version": 1, "timestamps_seconds": f(d["timestamps_seconds"]),
            "frame_indices": f(d["frame_indices"]), "qpos": q.tolist(),
            "source": {"file": "output/biomech-twin/motion.npz", "frames": int(len(q)),
                       "recording_sha256": recording_sha, "note": "irregular timestamps kept as fitted"}}


# ------------------------------------------------------------------ references
def reduction_matrix(m, qpos):
    """C (nv x 8): d(dof)/d(dynamic coordinate) through the exact polynomial coupling chain."""
    c = bm._couplings(m)
    C = np.zeros((m.nv, 8))
    for k, j in enumerate(dynamic_joint_ids(m)):
        C[int(m.jnt_dofadr[j]), k] = 1.0
    if len(c.locked_d):
        C[c.locked_d] = 0.0
    for r in range(len(c.dep_q)):
        x = qpos[c.mas_q[r]] - c.mas_ref[r]
        _, s = bm._poly(c.coef[r], x)
        C[c.dep_d[r]] = s * C[c.mas_d[r]]
    return C


def run_case(m, name, qpos, v8=None, act=None, ctrl=None):
    qpos = np.array(qpos, dtype=np.float64)
    bm.project_couplings_qpos(m, qpos)
    C = reduction_matrix(m, qpos)
    d = mujoco.MjData(m)
    d.qpos[:] = qpos
    if v8 is not None:
        d.qvel[:] = C @ np.asarray(v8, dtype=np.float64)
    d.act[:] = 0 if act is None else act
    d.ctrl[:] = 0 if ctrl is None else ctrl
    mujoco.mj_forward(m, d)  # NB: overwrites qacc; act_dot is read right after
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    g = mujoco.MjData(m)
    g.qpos[:] = qpos
    mujoco.mj_forward(m, g)  # qvel = 0 -> qfrc_bias is pure gravity demand
    v = np.zeros(8) if v8 is None else np.asarray(v8, dtype=np.float64)
    # Cdot*xdot by centered difference of C along xdot (coordinates moved by +/- h*xdot, re-projected)
    h = 1e-5
    jd = dynamic_joint_ids(m)
    qa = [int(m.jnt_qposadr[j]) for j in jd]
    cs = []
    for sgn in (1.0, -1.0):
        qs = qpos.copy()
        qs[qa] += sgn * h * v
        bm.project_couplings_qpos(m, qs)
        cs.append(reduction_matrix(m, qs))
    cdot_xdot = ((cs[0] - cs[1]) / (2 * h)) @ v
    reduced_bias = C.T @ (d.qfrc_bias + M @ cdot_xdot)
    return {
        "name": name, "qpos": f(qpos), "body_pos": f(d.xpos), "body_quat": f(d.xquat),
        "site_pos": f(d.site_xpos), "tendon_lengths": f(d.ten_length),
        "tendon_velocities": f(d.ten_velocity), "activation": f(d.act),
        "muscle_tensions": f(-d.actuator_force), "activation_derivative": f(d.act_dot),
        "control": f(d.ctrl), "reduced_mass": f(C.T @ M @ C), "reduced_gravity": f(C.T @ g.qfrc_bias),
        "reduced_velocity": f(v), "reduced_bias": f(reduced_bias),
        "reduced_muscle_torque": f(C.T @ d.qfrc_actuator),
    }


def build_cases(m, motion_q):
    rng = np.random.default_rng(20260101)
    jd = dynamic_joint_ids(m)
    qadr = [int(m.jnt_qposadr[j]) for j in jd]
    lo = np.array([m.jnt_range[j, 0] for j in jd])
    hi = np.array([m.jnt_range[j, 1] for j in jd])
    stand = standing_qpos(m)

    def pose(vals):
        q = stand.copy()
        q[qadr] = vals
        return q

    def frac(fr):
        return lo + (hi - lo) * np.asarray(fr)

    cases = [run_case(m, "standing", stand)]
    for i, fr in enumerate((0.2, 0.4, 0.6, 0.8)):
        cases.append(run_case(m, "shoulder_elv_sweep_%d" % i, pose(frac([0.5, fr, 0.5, 0.3, 0.5, fr, 0.5, 0.3]))))
    for i, fr in enumerate((0.15, 0.5, 0.85)):
        cases.append(run_case(m, "elbow_sweep_%d" % i, pose(frac([0.5, 0.5, 0.5, fr, 0.5, 0.5, 0.5, fr]))))
    for i, fr in enumerate((0.2, 0.5, 0.8)):
        cases.append(run_case(m, "shoulder_rot_sweep_%d" % i, pose(frac([0.5, 0.5, fr, 0.4, 0.5, 0.5, fr, 0.4]))))
    for i, fr in enumerate((0.1, 0.5, 0.9)):
        cases.append(run_case(m, "elv_angle_sweep_%d" % i, pose(frac([fr, 0.6, 0.5, 0.4, fr, 0.6, 0.5, 0.4]))))
    for i in range(4):
        cases.append(run_case(m, "asymmetric_%d" % i, pose(frac(0.1 + 0.8 * rng.random(8)))))
    n = len(motion_q)
    for k in (0, n // 4, n // 2, 3 * n // 4, n - 30, n - 10, n - 1):
        cases.append(run_case(m, "recorded_frame_%d" % k, motion_q[k]))
    nu = m.nu
    for tag, sgn in (("shortening", -1.0), ("lengthening", 1.0)):
        for k in (n // 2, n - 1):
            v8 = sgn * np.array([0.6, 0.8, 0.5, 1.0, 0.4, 0.7, 0.6, 0.9])
            cases.append(run_case(m, "velocity_%s_frame_%d" % (tag, k), motion_q[k], v8=v8,
                                  act=np.full(nu, 0.3), ctrl=np.full(nu, 0.3)))
    cases.append(run_case(m, "velocity_random_standing", stand, v8=rng.normal(0, 1, 8),
                          act=rng.random(nu), ctrl=rng.random(nu)))
    for tag, a, u in (("partial", 0.4, 0.7), ("full", 1.0, 1.0), ("decay", 0.8, 0.0)):
        cases.append(run_case(m, "activation_%s_standing" % tag, stand,
                              act=np.full(nu, a), ctrl=np.full(nu, u)))
    cases.append(run_case(m, "activation_random_recorded", motion_q[n - 5], v8=0.5 * rng.normal(0, 1, 8),
                          act=rng.random(nu), ctrl=rng.random(nu)))
    return cases


def copy_licenses(out):
    dst = out / "provenance"
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted((TWIN / "provenance").glob("*")):
        shutil.copyfile(p, dst / p.name)
    (out / "NOTICE.md").write_text(
        "# biomech-lite data notices\n\n"
        "Model and fixtures are compiled from MyoSim 0.2.5 `myofullbody` (Apache-2.0, "
        "https://github.com/MyoHub/myo_sim) with MuJoCo 3.17.0 (Apache-2.0, "
        + MUJOCO_URL + "). Components: MoBL arm (Holzbaur et al. 2005), Rajagopal et al. 2016 legs, "
        "MyoBack torso (Walia et al. 2025), SimTK lumbarspine. Derived wrapping/muscle code in the Rust "
        "crate ports MuJoCo `engine_util_misc.c` and `engine_core_smooth.c`.\n"
        "Original LICENSE/NOTICE/third-party files are in `provenance/`.\n\n"
        "Reference (generic) anatomy only; not person-specific. Fitted motion is a monocular "
        "kinematic fit, not measured forces.\n")


def export_all(out=DEFAULT_OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    mjb = TWIN / "model.mjb"
    pj = TWIN / "provenance.json"
    rec = json.loads(pj.read_text()).get("source_recording_sha256", "") if pj.is_file() else ""
    m = load_native(mjb)
    model = export_model_json(m, mjb, rec)
    motion = export_motion_json(m, rec)
    cases = build_cases(m, np.array(motion["qpos"]))

    def dump(o):
        return json.dumps(o, separators=(",", ":"), allow_nan=False)

    (out / "model.json").write_text(dump(model))
    (out / "motion.json").write_text(dump(motion))
    (out / "reference.json").write_text(dump({"schema_version": 1, "source": model["source"],
                                              "cases": cases}))
    copy_licenses(out)
    return {"out": str(out), "cases": len(cases),
            "sizes": {p.name: p.stat().st_size for p in out.glob("*.json")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)
    try:
        print(json.dumps(export_all(a.output), indent=1))
    except ExportError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
