#!/usr/bin/env python3
"""Sourced full-body musculoskeletal reference model (MyoSim ``myofullbody`` on MuJoCo).

This is a thin, explicit backend over the *unmodified* upstream MyoSim model:

* anatomy: MyoArm (MoBL, Holzbaur et al. 2005) bilateral arms + hands, MyoLeg
  (Rajagopal et al. 2016) legs, MyoTorso (MyoBack, Walia et al. 2025) lumbar
  spine/abdomen, and a rigid (zero-DOF) MyoHead scaffold. Pinned versions live in
  ``scripts/requirements-biomech.txt``.
* muscles: MuJoCo's built-in muscle model (first-order activation dynamics and
  normalized force-length-velocity curves, ``gaintype/biastype=muscle``) acting on
  spatial musculotendon paths with wrapping. The series tendon is *stiff*
  (no tendon compliance state). Nothing here solves soft-tissue elasticity,
  intramuscular fascicle geometry, cartilage, ligament injury or subject-specific
  bone/muscle geometry. This is REFERENCE anatomy, not personalized anatomy.
* ligaments: upstream ships eight real glenohumeral/coracohumeral ligament routes
  (``s/m/i_glenohum_ligament``, ``coracohum_ligament``; ``_l`` for left) with
  stiffness 0, damping 0 and no limits, i.e. they produce NO force by default.
  ``load_model(ligament_stiffness>0)`` turns exactly those existing routes into
  tension-only linear springs using MuJoCo's spring dead-band
  (``springlength=[0, L_slack]``): zero force for L <= L_slack, ``k (L - L_slack)``
  tension above. Stiffness and slack are EXPLORATORY PRIORS supplied by the
  user, not measured tissue properties. No other ligaments are modeled.

Public API (see docs/biomechanics-contract.md):
    load_model(ligament_stiffness=0.0, ligament_slack_scale=1.0,
               ligament_slack_reference="source") -> (model, data)
    enforce_couplings(model, data, qvel=True) -> list[str]  (unsupported eq names)
    project_couplings_qpos(model, qpos, qvel=None) -> None   (array version, in place)
    coupling_residuals(model, qpos) -> ndarray
    marker_positions(model, data) -> dict[str, ndarray(3,)]
    ligament_state(model, data) -> list[dict]
    inventory(model) -> dict (JSON-safe)
    export_model(out_dir, ...) -> dict (paths)

CLI: ``.venv-biomech/bin/python scripts/biomech_model.py --output output/biomech-twin``
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import re
import shutil
import sys
import warnings
import weakref
from importlib import metadata
from pathlib import Path

import numpy as np

import mujoco

MODEL_NAME = "myofullbody"
KEYFRAME = "standing"
LIGAMENT_NAMES = (
    "s_glenohum_ligament", "m_glenohum_ligament", "i_glenohum_ligament", "coracohum_ligament",
    "s_glenohum_ligament_l", "m_glenohum_ligament_l", "i_glenohum_ligament_l", "coracohum_ligament_l",
)
SLACK_REFERENCES = ("source", "standing")
DEFAULT_OUTPUT = Path("output/biomech-reference")
PINNED = {"myo-sim": "0.2.5", "mujoco": "3.17.0"}

MARKER_DEFINITIONS = {
    # name: (kind, operands, description). Every operand is a native model element.
    "left_shoulder": ("joint_anchor", ["shoulder_elv_l"],
                      "glenohumeral rotation center (anchor of native shoulder_elv_l = humerus_l origin)"),
    "right_shoulder": ("joint_anchor", ["shoulder_elv_r"],
                       "glenohumeral rotation center (anchor of native shoulder_elv_r = humerus_r origin)"),
    "left_elbow": ("site_mean", ["Elbow_Lateral_marker_l", "Elbow_Medial_marker_l"],
                   "midpoint of native lateral/medial epicondyle markers"),
    "right_elbow": ("site_mean", ["Elbow_Lateral_marker_r", "Elbow_Medial_marker_r"],
                    "midpoint of native lateral/medial epicondyle markers"),
    "left_wrist": ("joint_anchor", ["flexion_l"],
                   "wrist rotation center (anchor of native flexion_l/deviation_l in the lunate frame)"),
    "right_wrist": ("joint_anchor", ["flexion_r"],
                    "wrist rotation center (anchor of native flexion_r/deviation_r in the lunate frame)"),
    "left_wrist_styloids": ("site_mean", ["Radius_marker_l", "Ulna_marker_l"],
                            "midpoint of native radial/ulnar styloid markers (surface wrist)"),
    "right_wrist_styloids": ("site_mean", ["Radius_marker_r", "Ulna_marker_r"],
                             "midpoint of native radial/ulnar styloid markers (surface wrist)"),
    "thorax": ("joint_anchor_mean", ["shoulder_elv_l", "shoulder_elv_r"],
               "midpoint of the two glenohumeral centers (matches the recording's shoulder-center 'thorax')"),
    "c7": ("site_mean", ["C7_marker_l", "C7_marker_r"], "native C7 spinous-process markers"),
    "head": ("body_origin", ["head"],
             "native head body frame origin (rigid head, no neck DOF in source)"),
    "skull_center": ("body_mesh_aabb_center", ["head"],
                     "center of the axis-aligned bounds of the head body's native skull/jaw mesh vertices"),
    "pelvis": ("site_mean", ["ASI_l", "ASI_r", "PSI_l", "PSI_r"], "mean of native ASIS/PSIS markers"),
    "left_hip": ("site", ["HJC_l"], "native hip joint center marker"),
    "right_hip": ("site", ["HJC_r"], "native hip joint center marker"),
    "left_knee": ("site", ["KJC_l"], "native knee joint center marker"),
    "right_knee": ("site", ["KJC_r"], "native knee joint center marker"),
    "left_ankle": ("site", ["AJC_l"], "native ankle joint center marker"),
    "right_ankle": ("site", ["AJC_r"], "native ankle joint center marker"),
}

LANDMARK_MISMATCH_NOTE = (
    "MediaPipe/H36M keypoints are image-derived surface/learned landmarks; these markers are "
    "anatomical joint centers of the reference skeleton. Shoulder: MediaPipe marks the acromial "
    "surface, the model uses the glenohumeral center (several cm medial/inferior). Elbow: the "
    "epicondyle midpoint is near the flexion axis; MediaPipe's elbow is a surface estimate. Wrist: "
    "model wrist = carpal rotation center (lunate), ~1-2 cm distal to the styloid line that "
    "MediaPipe approximates; 'wrist_styloids' is provided for comparison. Head: the model has no "
    "nose/ear landmarks; 'head' is the native head frame and 'skull_center' a mesh-bounds center, "
    "neither equals the recording's ear/nose head point. Use a fitted offset or low weight for head."
)

SCOPE_STATEMENT = {
    "anatomy_kind": "reference (generic adult) anatomy, NOT personalized/subject-specific anatomy",
    "muscle_model": "MuJoCo muscle actuator: activation dynamics + normalized force-length-velocity "
                    "(Hill-type) on spatial musculotendon paths; stiff (inextensible) series tendon",
    "not_modeled": [
        "tendon/aponeurosis compliance and fiber-tendon equilibrium",
        "3D continuum muscle mechanics, fascicle architecture or intramuscular strain (cf. Blemker 2005)",
        "subject-specific bone geometry, muscle volumes or moment arms",
        "cartilage, joint contact mechanics, ligament injury",
        "ligaments other than the 8 upstream glenohumeral/coracohumeral routes",
        "neck muscles and neck DOF (head is rigidly attached to the torso in the source)",
    ],
    "ligaments": "8 upstream routes with zero stiffness/damping/limits in source; optional "
                 "tension-only springs on those routes are user-chosen exploratory priors",
}


class BiomechError(Exception):
    pass


# ----------------------------------------------------------------- loading
def _quiet_spec():
    import myo_sim

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = myo_sim.load_spec(MODEL_NAME, build_kwargs={"postures": True})
    return spec


def _compile(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec.compile()


def _validate_ligament_options(stiffness, slack_scale, slack_reference):
    try:
        stiffness = float(stiffness)
        slack_scale = float(slack_scale)
    except (TypeError, ValueError) as exc:
        raise BiomechError(f"ligament options must be numbers: {exc}") from exc
    if not np.isfinite(stiffness) or stiffness < 0:
        raise BiomechError(f"ligament_stiffness must be finite and >= 0 N/m, got {stiffness}")
    if not np.isfinite(slack_scale) or slack_scale <= 0:
        raise BiomechError(f"ligament_slack_scale must be finite and > 0, got {slack_scale}")
    if slack_reference not in SLACK_REFERENCES:
        raise BiomechError(f"ligament_slack_reference must be one of {SLACK_REFERENCES}")
    return stiffness, slack_scale


def _reset_standing(model, data):
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
    if kid < 0:
        raise BiomechError(f"source model has no {KEYFRAME!r} keyframe")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    enforce_couplings(model, data)
    mujoco.mj_forward(model, data)


def build_spec(ligament_stiffness=0.0, ligament_slack_scale=1.0, ligament_slack_reference="source"):
    """Return (spec, ligament_config). The spec is the source spec with only the 8 ligament
    tendons' spring parameters changed when ``ligament_stiffness > 0``."""
    k, scale = _validate_ligament_options(ligament_stiffness, ligament_slack_scale,
                                          ligament_slack_reference)
    spec = _quiet_spec()
    by_name = {t.name: t for t in spec.tendons}
    missing = [n for n in LIGAMENT_NAMES if n not in by_name]
    if missing:
        raise BiomechError(f"source ligament routes missing: {missing}")
    source_slack = {n: float(np.atleast_1d(by_name[n].springlength)[0]) for n in LIGAMENT_NAMES}
    standing_len = None
    if ligament_slack_reference == "standing":
        m0 = _compile(spec)
        d0 = mujoco.MjData(m0)
        _reset_standing(m0, d0)
        standing_len = {n: float(d0.ten_length[m0.tendon(n).id]) for n in LIGAMENT_NAMES}
    config = {
        "enabled": k > 0,
        "stiffness_N_per_m": k,
        "damping_Ns_per_m": 0.0,
        "slack_scale": scale,
        "slack_reference": ligament_slack_reference,
        "law": "tension-only linear: F = k*max(0, L - L_slack); MuJoCo springlength=[0, L_slack]",
        "status": ("exploratory prior; stiffness and slack are user-chosen, not measured"
                   if k > 0 else "disabled (source: stiffness 0, damping 0, unlimited)"),
        "routes": {},
    }
    for n in LIGAMENT_NAMES:
        ref = standing_len[n] if standing_len else source_slack[n]
        slack = scale * ref
        config["routes"][n] = {
            "source_springlength_m": source_slack[n],
            "slack_length_m": slack if k > 0 else None,
        }
        if k > 0:
            t = by_name[n]
            # MuJoCo>=3.5 specs take [linear, poly2, poly3]; higher-order terms stay zero
            t.stiffness = [k, 0.0, 0.0]
            t.damping = [0.0, 0.0, 0.0]
            t.springlength = [0.0, slack]
    return spec, config


def load_model(ligament_stiffness=0.0, ligament_slack_scale=1.0, ligament_slack_reference="source"):
    """Compile the source full-body model and return (model, data) at the standing keyframe
    with native couplings enforced.

    ligament_stiffness (N/m, >=0): 0 keeps the source (no ligament force). >0 enables
        tension-only springs on the 8 existing ligament routes only.
    ligament_slack_scale (>0): multiplies the reference slack length.
    ligament_slack_reference: "source" uses upstream ``springlength`` scalars;
        "standing" uses route length at the standing keyframe (an explicit model choice,
        NOT a measured calibration).
    All 416 muscle actuators are kept exactly as compiled from the source.
    """
    spec, config = build_spec(ligament_stiffness, ligament_slack_scale, ligament_slack_reference)
    model = _compile(spec)
    data = mujoco.MjData(model)
    _reset_standing(model, data)
    _MODEL_META[model] = config
    return model, data


_MODEL_META: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def ligament_config(model):
    """Actual ligament settings, preserving source metadata when a profile changes the model."""
    cached = _MODEL_META.get(model)
    if cached is not None:
        k = cached["stiffness_N_per_m"]
        unchanged = all(
            model.tendon_stiffness[model.tendon(n).id] == k
            and (not k or np.allclose(model.tendon_lengthspring[model.tendon(n).id],
                                     [0.0, cached["routes"][n]["slack_length_m"]], atol=1e-12))
            for n in LIGAMENT_NAMES
        )
        if unchanged:
            return cached
    routes = {}
    stiffnesses = []
    for n in LIGAMENT_NAMES:
        i = model.tendon(n).id
        k = float(model.tendon_stiffness[i])
        stiffnesses.append(k)
        routes[n] = {
            "source_springlength_m": cached["routes"][n]["source_springlength_m"] if cached else None,
            "slack_length_m": float(model.tendon_lengthspring[i, 1]) if k > 0 else None,
            "stiffness_N_per_m": k,
            "damping_Ns_per_m": float(model.tendon_damping[i]),
        }
    uniform = len(set(stiffnesses)) == 1
    return {"enabled": max(stiffnesses) > 0,
            "stiffness_N_per_m": stiffnesses[0] if uniform else None,
            "stiffness_is_uniform": uniform,
            "stiffness_N_per_m_max": max(stiffnesses), "routes": routes,
            "status": "actual per-route model arrays; construction metadata is not treated as current stiffness"}


# ----------------------------------------------------------------- couplings
@dataclasses.dataclass(frozen=True)
class _Couplings:
    locked_q: np.ndarray
    locked_d: np.ndarray
    locked_val: np.ndarray
    dep_q: np.ndarray
    dep_d: np.ndarray
    dep_ref: np.ndarray
    mas_q: np.ndarray
    mas_d: np.ndarray
    mas_ref: np.ndarray
    coef: np.ndarray
    eq_ids: np.ndarray
    unsupported: tuple
    dependent_joints: tuple
    master_joints: tuple


_COUPLING_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _couplings(model) -> _Couplings:
    key = model
    c = _COUPLING_CACHE.get(key)
    if c is not None:
        return c
    active = np.asarray(model.eq_active0, dtype=bool)
    unsupported = []
    rows = []
    for e in range(model.neq):
        if not active[e]:
            continue
        if int(model.eq_type[e]) != int(mujoco.mjtEq.mjEQ_JOINT):
            unsupported.append(model.eq(e).name or f"eq{e}:{mujoco.mjtEq(model.eq_type[e]).name}")
            continue
        dep = int(model.eq_obj1id[e])
        if int(model.jnt_type[dep]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                            int(mujoco.mjtJoint.mjJNT_SLIDE)):
            unsupported.append(f"eq{e}: non-scalar dependent joint")
            continue
        rows.append(e)
    deps = [int(model.eq_obj1id[e]) for e in rows]
    if len(set(deps)) != len(deps):
        raise BiomechError("a joint is the dependent of more than one joint equality")
    # dependency order: a coupling runs after the coupling that writes its master
    dep_row = {int(model.eq_obj1id[e]): e for e in rows}
    order, state = [], {}

    def visit(e, stack=()):
        if state.get(e) == 2:
            return
        if e in stack:
            raise BiomechError("joint equalities form a cycle")
        m = int(model.eq_obj2id[e])
        if m >= 0 and m in dep_row:
            visit(dep_row[m], stack + (e,))
        state[e] = 2
        order.append(e)

    for e in rows:
        visit(e)
    locked = [e for e in order if model.eq_obj2id[e] < 0]
    coupled = [e for e in order if model.eq_obj2id[e] >= 0]
    qadr, dadr = model.jnt_qposadr, model.jnt_dofadr
    ld = np.array([model.eq_obj1id[e] for e in locked], dtype=int)
    cd = np.array([model.eq_obj1id[e] for e in coupled], dtype=int)
    cm = np.array([model.eq_obj2id[e] for e in coupled], dtype=int)
    c = _Couplings(
        locked_q=qadr[ld].astype(int) if len(ld) else np.zeros(0, int),
        locked_d=dadr[ld].astype(int) if len(ld) else np.zeros(0, int),
        locked_val=(model.qpos0[qadr[ld]] + model.eq_data[locked, 0]) if len(ld) else np.zeros(0),
        dep_q=qadr[cd].astype(int), dep_d=dadr[cd].astype(int),
        dep_ref=model.qpos0[qadr[cd]].copy(),
        mas_q=qadr[cm].astype(int), mas_d=dadr[cm].astype(int),
        mas_ref=model.qpos0[qadr[cm]].copy(),
        coef=model.eq_data[coupled, :5].copy(),
        eq_ids=np.array(locked + coupled, dtype=int),
        unsupported=tuple(unsupported),
        dependent_joints=tuple(model.joint(int(j)).name for j in list(ld) + list(cd)),
        master_joints=tuple(sorted({model.joint(int(j)).name for j in cm})),
    )
    _COUPLING_CACHE[key] = c
    return c


def _poly(coef, x):
    v = coef[4] * x ** 4 + coef[3] * x ** 3 + coef[2] * x ** 2 + coef[1] * x + coef[0]
    s = 4 * coef[4] * x ** 3 + 3 * coef[3] * x ** 2 + 2 * coef[2] * x + coef[1]
    return v, s


def project_couplings_qpos(model, qpos, qvel=None):
    """In place: write every dependent/locked joint of ``qpos`` (and ``qvel``) from its master
    using the native mjEQ_JOINT quartic ``y - y0 = sum c_i (x - x0)^i`` (y0, x0 = qpos0).
    Rows run sequentially in dependency order, so chained couplings are exact."""
    c = _couplings(model)
    qpos[..., c.locked_q] = c.locked_val
    if qvel is not None:
        qvel[..., c.locked_d] = 0.0
    for r in range(len(c.dep_q)):
        x = qpos[..., c.mas_q[r]] - c.mas_ref[r]
        v, s = _poly(c.coef[r], x)
        qpos[..., c.dep_q[r]] = c.dep_ref[r] + v
        if qvel is not None:
            qvel[..., c.dep_d[r]] = s * qvel[..., c.mas_d[r]]


def enforce_couplings(model, data, qvel=True):
    """Update dependent scalar joints of ``data.qpos`` (and ``data.qvel``) from native joint
    equality polynomials. Masters are never moved. Does not call mj_forward.
    Returns the names of active equalities that are NOT joint polynomials (left unenforced)."""
    project_couplings_qpos(model, data.qpos, data.qvel if qvel else None)
    return list(_couplings(model).unsupported)


def coupling_residuals(model, qpos):
    """Residual ``y - y0 - p(x - x0)`` (rad or m) for each enforced coupling, eq-id order of
    ``_couplings(model).eq_ids``. Independent of the projection code path order."""
    c = _couplings(model)
    out = []
    for e in c.eq_ids:
        dq = model.jnt_qposadr[model.eq_obj1id[e]]
        y = qpos[dq] - model.qpos0[dq]
        if model.eq_obj2id[e] >= 0:
            mq = model.jnt_qposadr[model.eq_obj2id[e]]
            p, _ = _poly(model.eq_data[e, :5], qpos[mq] - model.qpos0[mq])
        else:
            p = model.eq_data[e, 0]
        out.append(y - p)
    return np.asarray(out)


def independent_dofs(model):
    """Names of scalar joints not written by any coupling (root free joint excluded)."""
    dep = set(_couplings(model).dependent_joints)
    return [model.joint(j).name for j in range(model.njnt)
            if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE) and model.joint(j).name not in dep]


# ----------------------------------------------------------------- markers
_MESH_AABB_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _body_mesh_aabb_local(model, body_id):
    per_model = _MESH_AABB_CACHE.setdefault(model, {})
    key = body_id
    if key not in per_model:
        pts = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != body_id or int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            mid = model.geom_dataid[g]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[a:a + n].astype(float)
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, model.geom_quat[g])
            pts.append(v @ R.reshape(3, 3).T + model.geom_pos[g])
        if not pts:
            raise BiomechError(f"body {model.body(body_id).name} has no mesh geoms")
        p = np.concatenate(pts)
        per_model[key] = (0.5 * (p.min(0) + p.max(0)), p.min(0), p.max(0))
    return per_model[key]


def marker_positions(model, data, names=None):
    """World xyz (m, MuJoCo world frame: +z up, subject faces -y, subject's right is -x)
    for anatomical markers defined only from native model elements (MARKER_DEFINITIONS).
    Requires kinematics to be current (call mj_kinematics/mj_forward first)."""
    out = {}
    for name in names or MARKER_DEFINITIONS:
        kind, ops, _ = MARKER_DEFINITIONS[name]
        if kind == "joint_anchor":
            p = data.xanchor[model.joint(ops[0]).id]
        elif kind == "joint_anchor_mean":
            p = np.mean([data.xanchor[model.joint(o).id] for o in ops], axis=0)
        elif kind == "site":
            p = data.site_xpos[model.site(ops[0]).id]
        elif kind == "site_mean":
            p = np.mean([data.site_xpos[model.site(o).id] for o in ops], axis=0)
        elif kind == "body_origin":
            p = data.xpos[model.body(ops[0]).id]
        elif kind == "body_mesh_aabb_center":
            b = model.body(ops[0]).id
            c, _, _ = _body_mesh_aabb_local(model, b)
            p = data.xpos[b] + data.xmat[b].reshape(3, 3) @ c
        else:  # pragma: no cover
            raise BiomechError(f"unknown marker kind {kind}")
        out[name] = np.array(p, dtype=float)
    return out


# ----------------------------------------------------------------- ligaments
def ligament_state(model, data):
    """Per ligament route: length, slack, strain, tension (N, >=0), elastic energy (J) under the
    model's spring parameters, and the engine's generalized spring force contribution norm.
    Requires mj_forward (ten_length/ten_J current)."""
    out = []
    for n in LIGAMENT_NAMES:
        i = model.tendon(n).id
        L = float(data.ten_length[i])
        k = float(model.tendon_stiffness[i])
        if hasattr(model, "tendon_stiffnesspoly") and np.any(model.tendon_stiffnesspoly[i] != 0):
            raise BiomechError(f"{n}: nonlinear stiffness terms are not supported by this helper")
        lo, hi = (float(x) for x in model.tendon_lengthspring[i])
        if k > 0:
            ext = (L - hi) if L > hi else ((L - lo) if L < lo else 0.0)
            tension = k * max(ext, 0.0)
            energy = 0.5 * k * ext * ext
        else:
            tension = energy = 0.0
        out.append({
            "name": n, "length_m": L, "stiffness_N_per_m": k,
            "slack_length_m": hi if k > 0 else None,
            "strain": (L - hi) / hi if k > 0 and hi > 0 else None,
            "tension_N": tension, "energy_J": energy, "taut": bool(tension > 0),
        })
    return out


def tendon_jacobian(model, data, tendon):
    """Dense d(length)/d(qvel) row (nv,) of a tendon (id or name); needs mj_forward or
    mj_fwdPosition. MuJoCo 3.17 stores ``data.ten_J`` SPARSE (model.ten_J_rowadr/rownnz/colind);
    do not index it as (ntendon, nv). Muscle moment arm about dof j = -J[j] (MuJoCo sign:
    actuator_moment = J, muscle force <= 0)."""
    t = model.tendon(tendon).id if isinstance(tendon, str) else int(tendon)
    J = np.zeros(model.nv)
    if hasattr(model, "ten_J_rowadr"):
        a, n = model.ten_J_rowadr[t], model.ten_J_rownnz[t]
        J[model.ten_J_colind[a:a + n]] = data.ten_J[a:a + n]
    else:  # older dense layout
        J[:] = np.asarray(data.ten_J).reshape(model.ntendon, model.nv)[t]
    return J


def ligament_generalized_force(model, data):
    """J_lig^T * F_lig (nv,), N or N*m, from the tendon Jacobian (data.ten_J, needs mj_forward).
    Equals the ligament share of data.qfrc_spring; sign: positive tension pulls length shorter."""
    f = np.zeros(model.nv)
    for s in ligament_state(model, data):
        if s["tension_N"] > 0:
            f -= s["tension_N"] * tendon_jacobian(model, data, s["name"])
    return f


# ----------------------------------------------------------------- inventory
def _source_of_body(model, b):
    names = []
    while b > 0:
        names.append(model.body(b).name)
        b = model.body_parentid[b]
    if "myoarm_r_root" in names or "myoarm_l_root" in names:
        return "MyoArm/MoBL (Holzbaur 2005)"
    if "myolegs_root" in names:
        return "MyoLeg/Rajagopal 2016"
    if "head_attach" in names:
        return "MyoHead rigid HAT scaffold"
    return "MyoTorso/MyoBack (Walia 2025; lumbarspine SimTK)"


def _actuator_source(model, a):
    t = model.actuator_trnid[a, 0]
    adr = model.tendon_adr[t]
    for w in range(adr, adr + model.tendon_num[t]):
        if int(model.wrap_type[w]) == int(mujoco.mjtWrap.mjWRAP_SITE):
            return _source_of_body(model, model.site_bodyid[model.wrap_objid[w]])
    return "unknown"


def inventory(model, data=None):
    """JSON-safe inventory of what the compiled model actually contains."""
    if data is None:
        data = mujoco.MjData(model)
        _reset_standing(model, data)
    c = _couplings(model)
    jt = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    joints = []
    dep = set(c.dependent_joints)
    for j in range(model.njnt):
        joints.append({
            "name": model.joint(j).name, "type": jt[int(model.jnt_type[j])],
            "body": model.body(model.jnt_bodyid[j]).name,
            "limited": bool(model.jnt_limited[j]),
            "range": [float(x) for x in model.jnt_range[j]],
            "units": "m" if model.jnt_type[j] == 2 else ("mixed" if model.jnt_type[j] == 0 else "rad"),
            "qposadr": int(model.jnt_qposadr[j]), "dofadr": int(model.jnt_dofadr[j]),
            "role": ("root" if model.jnt_type[j] == 0 else "dependent" if model.joint(j).name in dep
                     else "independent"),
            "standing_qpos": float(data.qpos[model.jnt_qposadr[j]]) if model.jnt_type[j] != 0 else None,
            "source": _source_of_body(model, int(model.jnt_bodyid[j])),
        })
    couplings = []
    for e in c.eq_ids:
        couplings.append({
            "name": model.eq(int(e)).name,
            "dependent": model.joint(int(model.eq_obj1id[e])).name,
            "master": model.joint(int(model.eq_obj2id[e])).name if model.eq_obj2id[e] >= 0 else None,
            "coef_c0_c4": [float(x) for x in model.eq_data[e, :5]],
            "law": "dep - dep0 = sum_i c_i (master - master0)^i  (locked: dep - dep0 = c0)",
        })
    muscles = []
    for a in range(model.nu):
        g = model.actuator_gainprm[a]
        lr = model.actuator_lengthrange[a]
        L0 = (lr[1] - lr[0]) / (g[1] - g[0])
        t = int(model.actuator_trnid[a, 0])
        muscles.append({
            "name": model.actuator(a).name, "tendon": model.tendon(t).name,
            "source": _actuator_source(model, a),
            "F0_N": float(g[2]), "scale": float(g[3]),
            "operating_range_L_over_L0": [float(g[0]), float(g[1])],
            "lmin": float(g[4]), "lmax": float(g[5]), "vmax_L0_per_s": float(g[6]),
            "fpmax": float(g[7]), "fvmax": float(g[8]),
            "tau_act_s": float(model.actuator_dynprm[a, 0]),
            "tau_deact_s": float(model.actuator_dynprm[a, 1]),
            "tausmooth": float(model.actuator_dynprm[a, 2]),
            "mtu_length_range_m": [float(lr[0]), float(lr[1])],
            "optimal_length_L0_m": float(L0),
            "standing_mtu_length_m": float(data.actuator_length[a]),
            "standing_normalized_length": float(g[0] + (data.actuator_length[a] - lr[0]) / L0),
            "ctrlrange": [float(x) for x in model.actuator_ctrlrange[a]],
        })
    ligs = []
    cfg = ligament_config(model)
    for s in ligament_state(model, data):
        i = model.tendon(s["name"]).id
        adr, num = model.tendon_adr[i], model.tendon_num[i]
        path = []
        for w in range(adr, adr + num):
            wt = int(model.wrap_type[w])
            o = int(model.wrap_objid[w])
            if wt == int(mujoco.mjtWrap.mjWRAP_SITE):
                path.append({"site": model.site(o).name, "body": model.body(model.site_bodyid[o]).name})
            elif wt in (int(mujoco.mjtWrap.mjWRAP_SPHERE), int(mujoco.mjtWrap.mjWRAP_CYLINDER)):
                path.append({"wrap_geom": model.geom(o).name,
                             "body": model.body(model.geom_bodyid[o]).name})
        ligs.append({**s, "path": path,
                     "source_springlength_m": cfg.get("routes", {}).get(s["name"], {}).get(
                         "source_springlength_m"),
                     "standing_length_m": s["length_m"]})
    n_mesh_geom = int((model.geom_type == mujoco.mjtGeom.mjGEOM_MESH).sum())
    return {
        "model": MODEL_NAME, "keyframe": KEYFRAME,
        "counts": {
            "bodies_including_world": int(model.nbody), "joints": int(model.njnt),
            "nq": int(model.nq), "nv": int(model.nv),
            "muscle_actuators": int(sum(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)),
            "actuators": int(model.nu), "spatial_tendons": int(model.ntendon),
            "ligament_routes": len(LIGAMENT_NAMES), "meshes": int(model.nmesh),
            "mesh_geoms": n_mesh_geom, "geoms": int(model.ngeom), "sites": int(model.nsite),
            "equalities": int(model.neq), "joint_couplings": len(c.dep_q),
            "locked_joints": len(c.locked_q),
            "independent_scalar_dofs": len(independent_dofs(model)),
            "root_free_joint_dofs": 6,
        },
        "unsupported_equalities": list(c.unsupported),
        "independent_joints": independent_dofs(model),
        "bodies": [model.body(b).name for b in range(model.nbody)],
        "joints": joints, "couplings": couplings,
        "actuators": [model.actuator(a).name for a in range(model.nu)],
        "muscles": muscles,
        "ligaments": {"config": cfg, "routes": ligs},
        "markers": {k: {"kind": v[0], "operands": v[1], "description": v[2]}
                    for k, v in MARKER_DEFINITIONS.items()},
        "marker_landmark_mismatch": LANDMARK_MISMATCH_NOTE,
        "frames": "MuJoCo world: meters, +z up, gravity -z; standing subject faces -y, right side -x",
        "geometry_scope": {
            "meshes": [model.mesh(i).name for i in range(model.nmesh)],
            "mesh_vertices_total": int(model.nmeshvert),
            "note": "rigid bone surface meshes from source models (visual/inertial); no soft tissue",
        },
        "timestep_s": float(model.opt.timestep),
        "scope": SCOPE_STATEMENT,
    }


# ----------------------------------------------------------------- export
def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def _copy_provenance(out_dir):
    import myo_sim

    pkg = Path(myo_sim.__file__).resolve().parent
    prov = out_dir / "provenance"
    prov.mkdir(parents=True, exist_ok=True)
    copied = {}
    for dist, sub in (("myo-sim", "myo_sim"), ("mujoco", "mujoco")):
        d = metadata.distribution(dist)
        for f in d.files or []:
            s = str(f)
            if "/licenses/" in s or s.upper().endswith(("LICENSE", "NOTICE", "LICENSES_THIRD_PARTY.MD")):
                src = Path(d.locate_file(f))
                if src.is_file():
                    dst = prov / f"{sub}-{src.name}"
                    shutil.copyfile(src, dst)
                    copied[dst.name] = _sha(dst)
    for rel in ("models/arm/README.md", "models/arm/README_hand.md", "models/leg/README.md",
                "models/torso/README.md", "models/head/README.md", "build/README.md"):
        src = pkg / rel
        if src.is_file():
            dst = prov / ("myo_sim-" + rel.replace("/", "_"))
            shutil.copyfile(src, dst)
            copied[dst.name] = _sha(dst)
    return copied


def _export_portable_xml(spec, model, out_dir):
    """Write model.xml with mesh/texture files copied under assets/; return (path, n_files)."""
    import myo_sim

    src_root = Path(spec.meshdir) if spec.meshdir else Path(myo_sim.MODELS_DIR)
    assets = out_dir / "assets"
    n = 0
    for item in list(spec.meshes) + list(spec.textures) + list(spec.hfields):
        f = getattr(item, "file", "")
        if not f:
            continue
        src = src_root / f
        dst = assets / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copyfile(src, dst)
            n += 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml = spec.to_xml()
    # Re-point the compiler asset dirs at the copied assets (relative to model.xml).
    for attr, val in (("meshdir", spec.meshdir), ("texturedir", spec.texturedir)):
        if val:
            xml = xml.replace(f'{attr}="{val}"', f'{attr}="assets"')
    # MjSpec.to_xml emits nested *unnamed* <default> blocks for fragments attached with an empty
    # prefix (upstream warning "may prevent the model from being saved correctly"); MJCF rejects
    # those. Give each a unique class name: children keep inheriting, nothing references it.
    # The bilateral arms are the same fragment attached twice, so their default trees repeat
    # verbatim (same class names); an identical repeat is dropped, a differing one is an error.
    lines, top_indent, out, seen, n_anon, i = xml.split("\n"), None, [], set(), 0, 0
    while i < len(lines):
        ln = lines[i]
        ind = len(ln) - len(ln.lstrip())
        if ln.strip() == "<default>" and top_indent is None:
            top_indent = ind
        elif ln.strip() == "<default>" and ind > top_indent:
            j = i + 1
            while not (lines[j].strip() == "</default>" and len(lines[j]) - len(lines[j].lstrip()) == ind):
                j += 1
            body = "\n".join(lines[i + 1:j])
            key = re.sub(r'material="[^"]*"', "", body)  # left arm differs only by material
            if key in seen:
                i = j + 1
                continue
            names = re.findall(r'class="([^"]+)"', body)
            emitted = "\n".join(out)
            if any(f'class="{n}"' in emitted for n in names):
                raise BiomechError("conflicting duplicate default classes in exported XML")
            seen.add(key)
            ln = ln.replace("<default>", f'<default class="myosim_unnamed_{n_anon}">')
            n_anon += 1
        out.append(ln)
        i += 1
    xml = "\n".join(out)
    path = out_dir / "model.xml"
    path.write_text(xml)
    return path, n


def compare_models(a, b, tol=1e-9):
    """Return list of mismatching structural/parameter fields between two compiled models."""
    bad = []
    for f in ("nq", "nv", "nu", "nbody", "njnt", "ntendon", "nmesh", "neq", "nkey", "nsite"):
        if getattr(a, f) != getattr(b, f):
            bad.append(f)
    if bad:
        return bad
    for f in ("qpos0", "body_mass", "body_pos", "body_inertia", "jnt_range", "jnt_limited",
              "dof_damping", "dof_armature", "actuator_gainprm", "actuator_biasprm",
              "actuator_dynprm", "actuator_lengthrange", "actuator_ctrlrange",
              "tendon_stiffness", "tendon_lengthspring", "tendon_damping", "tendon_limited",
              "eq_data", "eq_active0", "key_qpos", "site_pos", "geom_size", "geom_pos",
              "geom_contype", "geom_conaffinity", "geom_matid", "mesh_vert"):
        x, y = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
        if x.shape != y.shape or not np.allclose(x, y, atol=tol, rtol=1e-7):
            bad.append(f)
    return bad


def export_model(out_dir=DEFAULT_OUTPUT, ligament_stiffness=0.0, ligament_slack_scale=1.0,
                 ligament_slack_reference="source"):
    """Write model.mjb, portable model.xml (+assets), anatomy.json, provenance.json and
    provenance/ (licenses, upstream READMEs). Returns dict of written paths/checks."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec, cfg = build_spec(ligament_stiffness, ligament_slack_scale, ligament_slack_reference)
    model = _compile(spec)
    data = mujoco.MjData(model)
    _reset_standing(model, data)
    _MODEL_META[model] = cfg
    mjb = out_dir / "model.mjb"
    mujoco.mj_saveModel(model, str(mjb), None)
    xml_path, n_assets = _export_portable_xml(spec, model, out_dir)
    checks = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_mjb = mujoco.MjModel.from_binary_path(str(mjb))
        try:
            m_xml = mujoco.MjModel.from_xml_path(str(xml_path))
            if "geom_matid" in compare_models(model, m_xml, tol=1e-6):
                # dropped duplicate arm default only differed in (visual) material: make the
                # material explicit on the affected named geoms
                xml = xml_path.read_text()
                for g in np.flatnonzero(model.geom_matid != m_xml.geom_matid):
                    gname, mid = model.geom(int(g)).name, int(model.geom_matid[g])
                    if gname and mid >= 0:
                        xml = re.sub(rf'(<geom name="{re.escape(gname)}")',
                                     rf'\1 material="{model.material(mid).name}"', xml, count=1)
                xml_path.write_text(xml)
                m_xml = mujoco.MjModel.from_xml_path(str(xml_path))
            checks["xml_reload_mismatches"] = compare_models(model, m_xml, tol=1e-6)
        except Exception as exc:  # documented, not hidden
            checks["xml_reload_error"] = str(exc)
    checks["mjb_reload_mismatches"] = compare_models(model, m_mjb)
    inv = inventory(model, data)
    (out_dir / "anatomy.json").write_text(json.dumps(inv, indent=1, default=_json_default))
    copied = _copy_provenance(out_dir)
    prov = {
        "model": MODEL_NAME,
        "versions": {k: metadata.version(k) for k in ("myo-sim", "mujoco", "numpy")},
        "pinned_expected": PINNED,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "sources": {
            "myo_sim": "https://github.com/MyoHub/myo_sim (PyPI myo-sim, Apache-2.0)",
            "arm": "MoBL upper extremity, Holzbaur et al. 2005, doi:10.1007/s10439-005-3320-7",
            "legs": "Rajagopal et al. 2016, doi:10.1109/TBME.2016.2586891",
            "torso": "MyoBack, Walia et al. 2025 (IROS); SimTK lumbarspine",
            "engine": "MuJoCo (Apache-2.0), https://mujoco.org",
        },
        "recipe": [
            ".venv-biomech/bin/python -m pip install -r scripts/requirements-biomech.txt",
            ".venv-biomech/bin/python scripts/biomech_model.py --output " + str(out_dir)
            + f" --ligament-stiffness {cfg['stiffness_N_per_m']} --ligament-slack-scale "
            f"{cfg['slack_scale']} --ligament-slack-reference {cfg['slack_reference']}",
            "equivalently: myo_sim.load_spec('myofullbody', build_kwargs={'postures': True}), "
            "set the 8 ligament tendons' stiffness/springlength as in ligament config, compile",
        ],
        "ligaments": cfg,
        "files": {"model.mjb": _sha(mjb), "model.xml": _sha(xml_path),
                  "anatomy.json": _sha(out_dir / "anatomy.json"), **copied},
        "xml_assets_copied": n_assets,
        "reload_checks": checks,
        "notes": [
            "model.mjb is MuJoCo-version specific; model.xml + assets/ is the portable form.",
            "Upstream package files were not modified; meshes are copied verbatim.",
            "Saved model and analysis model share identical ligament configuration.",
        ],
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=1, default=_json_default))
    return {"out_dir": str(out_dir), "mjb": str(mjb), "xml": str(xml_path), "checks": checks,
            "counts": inv["counts"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--ligament-stiffness", type=float, default=0.0,
                    help="N/m; 0 = source (no ligament force). Exploratory prior when >0.")
    ap.add_argument("--ligament-slack-scale", type=float, default=1.0)
    ap.add_argument("--ligament-slack-reference", choices=SLACK_REFERENCES, default="source")
    a = ap.parse_args(argv)
    try:
        r = export_model(a.output, a.ligament_stiffness, a.ligament_slack_scale,
                         a.ligament_slack_reference)
    except BiomechError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(r, indent=1, default=_json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
