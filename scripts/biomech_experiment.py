#!/usr/bin/env python3
"""Isometric muscle bench experiment on the sourced MyoSim full-body model (MuJoCo).

What this runs (all on the real engine, source muscle definitions unchanged):

1. Angle sweep: one INDEPENDENT joint is swept across its source range with every other
   coordinate at the standing keyframe and native couplings enforced at every angle. At each
   angle and activation level (default 0, .25, .5, 1; joint velocity 0) it records MTU length,
   MuJoCo normalized muscle length, tension, moment arms and joint torque.
2. Activation step: at one fixed pose the muscle's control steps 0->1 (then back to 0); the
   engine integrates activation dynamics with mj_step while the kinematics are clamped
   (qpos reset, qvel = 0 every step). It shows activation/force lag. Isometric bench test only,
   NOT a free-body motion claim.
3. Ligament sweep: length / tension / elastic energy / generalized force of the 8 upstream
   glenohumeral ligament routes while sweeping a shoulder coordinate. With the source model
   (stiffness 0) forces are zero by construction; with --ligament-stiffness they are the
   user's exploratory tension-only prior (biomech_model.load_model).

Sign conventions (exported explicitly; MuJoCo native signs differ):
    tension_N      = -actuator_force  (>= 0 for a pulling muscle)
    moment_arm_m   = -dL/dq           (L = musculotendon length)
    torque_Nm      = moment_arm_m * tension_N  (= native qfrc_actuator on that dof)
  "partial" = isolated coordinate derivative (other qpos fixed);
  "total"   = derivative along the native coupling manifold (dependent joints move with q).
  Both are checked against central finite differences of the engine's tendon length.

Scope: MuJoCo's muscle is a rigid-tendon Hill-type actuator; its "normalized length" is
MTU-based (via the compiled lengthrange), NOT an OpenSim fiber length. Parameters are the
generic reference model's, not the subject's; nothing here is measured from video.

CLI: .venv-biomech/bin/python scripts/biomech_experiment.py --output output/biomech-experiment \
        --muscle BIClong [--joint elbow_flexion_r]
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biomech_model as bm  # noqa: E402

DEFAULT_ACTIVATIONS = (0.0, 0.25, 0.5, 1.0)
FD_STEP = 1e-6
SIDE_SUFFIXES = ("", "_r", "_l")


class ExperimentError(Exception):
    pass


# ----------------------------------------------------------------- discovery
def _side(name):
    return "l" if name.endswith("_l") else ("r" if name.endswith("_r") else None)


def resolve_muscle(model, query, joint=None):
    """Return the actuator id whose source name matches ``query`` (case-insensitive, optional
    _r/_l side suffix). Side is taken from ``joint`` when ambiguous; unsuffixed MyoArm names
    are the right arm (left copies end in _l)."""
    names = [model.actuator(a).name for a in range(model.nu)]
    exact = [a for a, n in enumerate(names) if n == query]
    if exact and (joint is None or _side(query) is not None):
        return exact[0]
    q = query.lower()
    cands = [a for a, n in enumerate(names) if any(n.lower() == q + s for s in SIDE_SUFFIXES)]
    if len(cands) > 1:
        want = _side(joint) if joint else "r"
        side_of = {a: (_side(names[a]) or "r") for a in cands}
        cands = [a for a in cands if side_of[a] == want] or cands
    if len(cands) != 1:
        close = difflib.get_close_matches(query, names, n=8, cutoff=0.5)
        raise ExperimentError(f"muscle {query!r} not found uniquely; close source names: {close}")
    return cands[0]


def _dof_of(model, joint):
    j = model.joint(joint).id
    if int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
        raise ExperimentError(f"{joint} is not a scalar joint")
    return j, int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])


def standing(model):
    d = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, d, model.key(bm.KEYFRAME).id)
    bm.enforce_couplings(model, d)
    mujoco.mj_forward(model, d)
    return d


def resolve_joint(model, actuator, joint=None):
    """Validate (or choose) the swept joint: must be an independent scalar joint. When not
    given, pick the independent joint with the largest |total moment arm| over 7 samples of
    its range (other coordinates standing), among joints the muscle spans at standing."""
    indep = bm.independent_dofs(model)
    t = int(model.actuator_trnid[actuator, 0])
    if joint is not None:
        try:
            model.joint(joint)
        except KeyError as exc:
            close = difflib.get_close_matches(joint, indep, n=8, cutoff=0.5)
            raise ExperimentError(f"joint {joint!r} not in model; close: {close}") from exc
        if joint not in indep:
            raise ExperimentError(f"{joint} is a dependent (coupled/locked) joint; sweep its master")
        return joint
    d = standing(model)
    J = bm.tendon_jacobian(model, d, t)
    spanned = [n for n in indep if abs(J[model.joint(n).dofadr[0]]) > 1e-9]
    if not spanned:
        raise ExperimentError("muscle has no moment arm about any independent joint at standing")
    score = {}
    for n in spanned:
        dof = model.joint(n).dofadr[0]
        best_n = 0.0
        for ang in np.linspace(*model.jnt_range[model.joint(n).id], 7):
            dp = _pose(model, d.qpos, {n: ang})
            mujoco.mj_fwdPosition(model, dp)
            best_n = max(best_n, abs(bm.tendon_jacobian(model, dp, t)
                                     @ coupled_direction(model, dp.qpos, dof)))
        score[n] = best_n
    return max(score, key=score.get)


# ----------------------------------------------------------------- kinematics helpers
def _pose(model, base_qpos, values):
    d = mujoco.MjData(model)
    d.qpos[:] = base_qpos
    for name, v in values.items():
        d.qpos[model.joint(name).qposadr[0]] = v
    bm.enforce_couplings(model, d, qvel=False)
    return d


def coupled_direction(model, qpos, dof):
    """qvel (nv,) for unit velocity of ``dof`` with dependents moving along native couplings."""
    qv = np.zeros(model.nv)
    qv[dof] = 1.0
    bm.project_couplings_qpos(model, qpos.copy(), qv)
    return qv


def _tendon_length_at(model, qpos, tendon):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    mujoco.mj_fwdPosition(model, d)
    return float(d.ten_length[tendon])


def fd_moment_arms(model, base_qpos, joint, angle, tendon, h=FD_STEP):
    """Central finite-difference moment arms (-dL/dq): (partial, total along couplings)."""
    _, qa, _ = _dof_of(model, joint)
    d0 = _pose(model, base_qpos, {joint: angle})
    Lp, Lm = [], []
    for s, store in ((1, Lp), (-1, Lm)):
        q = d0.qpos.copy()
        q[qa] += s * h
        store.append(_tendon_length_at(model, q, tendon))  # partial: dependents frozen
        dc = _pose(model, base_qpos, {joint: angle + s * h})
        store.append(_tendon_length_at(model, dc.qpos, tendon))  # total: couplings re-enforced
    partial = -(Lp[0] - Lm[0]) / (2 * h)
    total = -(Lp[1] - Lm[1]) / (2 * h)
    return partial, total


def muscle_parameters(model, a):
    g = model.actuator_gainprm[a]
    lr = model.actuator_lengthrange[a]
    return {
        "name": model.actuator(a).name, "tendon": model.tendon(int(model.actuator_trnid[a, 0])).name,
        "F0_N": float(g[2]), "operating_range_L_over_L0": [float(g[0]), float(g[1])],
        "lmin": float(g[4]), "lmax": float(g[5]), "vmax_L0_per_s": float(g[6]),
        "fpmax": float(g[7]), "fvmax": float(g[8]),
        "tau_act_s": float(model.actuator_dynprm[a, 0]),
        "tau_deact_s": float(model.actuator_dynprm[a, 1]),
        "tausmooth": float(model.actuator_dynprm[a, 2]),
        "mtu_length_range_m": [float(lr[0]), float(lr[1])],
        "L0_m": float((lr[1] - lr[0]) / (g[1] - g[0])),
        "note": "MuJoCo muscle: L0=(lengthrange span)/(range span); MTU-based, rigid tendon",
    }


def normalized_length(model, a, mtu_length):
    g = model.actuator_gainprm[a]
    lr = model.actuator_lengthrange[a]
    L0 = (lr[1] - lr[0]) / (g[1] - g[0])
    return g[0] + (mtu_length - lr[0]) / L0


# ----------------------------------------------------------------- 1. angle sweep
def angle_sweep(model, actuator, joint, n_angles=41, activations=DEFAULT_ACTIVATIONS,
                fixed=None, fd_check=True):
    """Isometric sweep. Returns dict of arrays: angles (A,), activations (K,), and (A,K) /
    (A,) quantities. ``fixed`` = {joint: value} applied before the swept joint."""
    _, qa, dof = _dof_of(model, joint)
    lo, hi = model.jnt_range[model.joint(joint).id]
    angles = np.linspace(lo, hi, n_angles)
    acts = np.asarray(activations, float)
    t = int(model.actuator_trnid[actuator, 0])
    base = standing(model).qpos.copy()
    if fixed:
        base = _pose(model, base, fixed).qpos.copy()
    A, K = len(angles), len(acts)
    out = {k: np.zeros((A, K)) for k in ("tension_N", "active_tension_N", "torque_partial_Nm",
                                         "torque_total_Nm", "engine_active_qfrc_Nm",
                                         "active_torque_partial_Nm")}
    for k in ("mtu_length_m", "normalized_length", "moment_arm_partial_m", "moment_arm_total_m",
              "fd_moment_arm_partial_m", "fd_moment_arm_total_m", "passive_tension_N",
              "passive_all_muscles_torque_Nm", "passive_ligament_torque_Nm", "coupling_residual_max"):
        out[k] = np.full(A, np.nan)
    for i, ang in enumerate(angles):
        d = _pose(model, base, {joint: ang})
        d.qvel[:] = 0.0
        d.act[:] = 0.0
        mujoco.mj_forward(model, d)
        J = bm.tendon_jacobian(model, d, t)
        qv = coupled_direction(model, d.qpos, dof)
        out["mtu_length_m"][i] = d.actuator_length[actuator]
        out["normalized_length"][i] = normalized_length(model, actuator, d.actuator_length[actuator])
        out["moment_arm_partial_m"][i] = -J[dof]
        out["moment_arm_total_m"][i] = -J @ qv
        out["coupling_residual_max"][i] = np.abs(bm.coupling_residuals(model, d.qpos)).max(initial=0)
        # passive mobility: all muscles at zero activation + ligament springs, on the swept dof
        out["passive_all_muscles_torque_Nm"][i] = d.qfrc_actuator @ qv
        out["passive_ligament_torque_Nm"][i] = bm.ligament_generalized_force(model, d) @ qv
        if fd_check:
            p, tt = fd_moment_arms(model, base, joint, ang, t)
            out["fd_moment_arm_partial_m"][i], out["fd_moment_arm_total_m"][i] = p, tt
        q_rest = float(d.qfrc_actuator[dof])  # all muscles passive (act 0)
        f_rest = float(d.actuator_force[actuator])
        for k_, a_ in enumerate(acts):
            d.act[:] = 0.0
            d.act[model.actuator_actadr[actuator]] = a_
            mujoco.mj_forward(model, d)
            ten = -float(d.actuator_force[actuator])
            out["tension_N"][i, k_] = ten
            out["torque_partial_Nm"][i, k_] = out["moment_arm_partial_m"][i] * ten
            out["torque_total_Nm"][i, k_] = out["moment_arm_total_m"][i] * ten
            # independent engine path: change of the ASSEMBLED native qfrc_actuator on the dof
            # caused by this muscle's activation (other muscles stay passive), compared in the
            # summary with moment_arm_partial * active tension
            out["engine_active_qfrc_Nm"][i, k_] = float(d.qfrc_actuator[dof]) - q_rest
            out["active_torque_partial_Nm"][i, k_] = out["moment_arm_partial_m"][i] * (ten + f_rest)
        out["passive_tension_N"][i] = out["tension_N"][i, 0] if acts[0] == 0 else np.nan
    if acts[0] == 0:
        out["active_tension_N"] = out["tension_N"] - out["tension_N"][:, :1]
    else:
        out["active_tension_N"][:] = np.nan
    out["angles"] = angles
    out["activations"] = acts
    return out


# ----------------------------------------------------------------- 2. activation step
def reference_activation(u, a, prm, dt):
    """Documented MuJoCo muscle activation ODE (Millard et al. 2013 form, tausmooth=0):
    tau = tau_act*(0.5+1.5a) if u>a else tau_deact/(0.5+1.5a); da/dt=(u-a)/tau, Euler."""
    ta, td = prm[0], prm[1]
    tau = ta * (0.5 + 1.5 * a) if u - a > 0 else td / (0.5 + 1.5 * a)
    return float(np.clip(a + dt * (u - a) / tau, 0.0, 1.0))


def activation_step(model, actuator, joint, angle, on_s=0.3, off_s=0.3, fixed=None):
    """Clamp kinematics at (joint=angle), step ctrl 0->1 at t=0 and 1->0 at t=on_s; integrate
    with mj_step. Returns dict of time series and lag metrics."""
    base = standing(model).qpos.copy()
    if fixed:
        base = _pose(model, base, fixed).qpos.copy()
    d = _pose(model, base, {joint: angle})
    clamp_q = d.qpos.copy()
    _, _, dof = _dof_of(model, joint)
    t_id = int(model.actuator_trnid[actuator, 0])
    dt = model.opt.timestep
    n_on, n_off = int(round(on_s / dt)), int(round(off_s / dt))
    adr = model.actuator_actadr[actuator]
    d.act[:] = 0.0
    d.ctrl[:] = 0.0
    T, U, ACT, TEN, TORQ, REF, QERR = [], [], [], [], [], [], []
    a_ref = 0.0
    for step in range(n_on + n_off + 1):
        u = 1.0 if step < n_on else 0.0
        d.qpos[:] = clamp_q
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        d.ctrl[actuator] = u
        mujoco.mj_forward(model, d)
        J = bm.tendon_jacobian(model, d, t_id)
        T.append(d.time)
        U.append(u)
        ACT.append(float(d.act[adr]))
        REF.append(a_ref)
        TEN.append(-float(d.actuator_force[actuator]))
        TORQ.append(-J[dof] * TEN[-1])
        a_ref = reference_activation(u, ACT[-1], model.actuator_dynprm[actuator], dt)
        mujoco.mj_step(model, d)  # integrates act (engine); qpos/qvel discarded next step
        QERR.append(float(np.abs(d.qpos - clamp_q).max()))
    T, ACT, TEN = np.array(T), np.array(ACT), np.array(TEN)
    f_pass, f_full = TEN[0], TEN[n_on - 1]
    rise = f_pass + 0.632 * (f_full - f_pass)
    fall = f_full - 0.632 * (f_full - f_pass)
    t63_on = float(T[np.argmax(TEN[:n_on] >= rise)]) if np.any(TEN[:n_on] >= rise) else None
    t90_on = (float(T[np.argmax(TEN[:n_on] >= f_pass + 0.9 * (f_full - f_pass))])
              if np.any(TEN[:n_on] >= f_pass + 0.9 * (f_full - f_pass)) else None)
    off = TEN[n_on:]
    t63_off = float(T[n_on + np.argmax(off <= fall)] - T[n_on]) if np.any(off <= fall) else None
    return {
        "time_s": T, "ctrl": np.array(U), "activation": ACT, "reference_activation": np.array(REF),
        "tension_N": TEN, "torque_Nm": np.array(TORQ),
        "metrics": {
            "angle": float(angle), "joint": joint, "dt_s": float(dt),
            "passive_tension_N": float(f_pass), "steady_tension_N": float(f_full),
            "activation_at_step_end": float(ACT[n_on - 1]),
            "t63_rise_s": t63_on, "t90_rise_s": t90_on, "t63_fall_s": t63_off,
            "max_activation_vs_reference_ode": float(np.abs(ACT - np.array(REF)).max()),
            "max_kinematic_drift_within_step": float(max(QERR)),
            "clamp": "qpos reset and qvel=0 before every step; only act evolves",
        },
    }


# ----------------------------------------------------------------- 3. ligament sweep
def ligament_sweep(model, joint="shoulder_rot_r", n_angles=41, fixed=None):
    """Length/tension/energy of all 8 ligament routes and their generalized force on ``joint``
    (along couplings) across its range. Also dE/dq vs generalized force consistency."""
    _, _, dof = _dof_of(model, joint)
    lo, hi = model.jnt_range[model.joint(joint).id]
    angles = np.linspace(lo, hi, n_angles)
    base = standing(model).qpos.copy()
    if fixed:
        base = _pose(model, base, fixed).qpos.copy()
    n = len(bm.LIGAMENT_NAMES)
    L, F, E = (np.zeros((n_angles, n)) for _ in range(3))
    Q = np.zeros(n_angles)
    dEdq = np.zeros(n_angles)

    def energy(ang):
        d = _pose(model, base, {joint: ang})
        mujoco.mj_fwdPosition(model, d)
        return sum(s["energy_J"] for s in bm.ligament_state(model, d))

    for i, ang in enumerate(angles):
        d = _pose(model, base, {joint: ang})
        mujoco.mj_forward(model, d)
        st = bm.ligament_state(model, d)
        L[i] = [s["length_m"] for s in st]
        F[i] = [s["tension_N"] for s in st]
        E[i] = [s["energy_J"] for s in st]
        Q[i] = bm.ligament_generalized_force(model, d) @ coupled_direction(model, d.qpos, dof)
        dEdq[i] = (energy(ang + FD_STEP) - energy(ang - FD_STEP)) / (2 * FD_STEP)
    slack = [s["slack_length_m"] for s in bm.ligament_state(model, standing(model))]
    return {"angles": angles, "joint": joint, "names": list(bm.LIGAMENT_NAMES),
            "length_m": L, "tension_N": F, "energy_J": E, "generalized_force": Q,
            "slack_length_m": slack, "neg_dE_dq": -dEdq,
            "stiffness_N_per_m": float(max(model.tendon_stiffness[model.tendon(x).id]
                                           for x in bm.LIGAMENT_NAMES))}


# ----------------------------------------------------------------- reporting
def _summary(model, actuator, joint, sw, extra=None):
    ma = sw["moment_arm_total_m"]
    full = sw["activations"].tolist().index(1.0) if 1.0 in sw["activations"] else -1
    tq = sw["torque_total_Nm"][:, full]
    fd_err_p = np.nanmax(np.abs(sw["moment_arm_partial_m"] - sw["fd_moment_arm_partial_m"]))
    fd_err_t = np.nanmax(np.abs(sw["moment_arm_total_m"] - sw["fd_moment_arm_total_m"]))
    eng = np.nanmax(np.abs(sw["active_torque_partial_Nm"] - sw["engine_active_qfrc_Nm"]))
    sig = np.abs(ma) > 1e-3  # 1 mm: below this the sign is not meaningful
    pos, neg = float(np.mean(ma[sig] > 0)) if sig.any() else 0.0, float(np.mean(ma[sig] < 0)) if sig.any() else 0.0
    role = {
        "dominant_direction": ("+q (moment arm > 0)" if pos >= neg else "-q (moment arm < 0)")
        if sig.any() else "negligible (|moment arm| < 1 mm everywhere)",
        "fraction_of_range_positive": float(np.mean(ma > 1e-3)),
        "fraction_of_range_negative": float(np.mean(ma < -1e-3)),
        "fraction_of_range_below_1mm": float(np.mean(~sig)),
        "joint_positive_direction": "source joint axis convention (e.g. +elbow_flexion_r = flexion)",
    }
    i_pk = int(np.argmax(np.abs(tq)))
    return {
        "muscle": model.actuator(actuator).name, "joint": joint,
        "joint_range_rad": [float(x) for x in model.jnt_range[model.joint(joint).id]],
        "role_about_joint": role,
        "moment_arm_total_m": {"min": float(ma.min()), "max": float(ma.max()),
                               "at_min_angle": float(sw["angles"][ma.argmin()]),
                               "at_max_angle": float(sw["angles"][ma.argmax()])},
        "partial_vs_total_moment_arm_max_abs_diff_m":
            float(np.nanmax(np.abs(sw["moment_arm_partial_m"] - sw["moment_arm_total_m"]))),
        "fd_check_max_abs_error_m": {"partial": float(fd_err_p), "total": float(fd_err_t)},
        "active_torque_vs_engine_qfrc_actuator_max_abs_diff_Nm": float(eng),
        "peak_full_activation_torque_Nm": float(tq[i_pk]),
        "peak_angle_rad": float(sw["angles"][i_pk]),
        "max_tension_N": float(np.nanmax(sw["tension_N"])),
        "normalized_length_range": [float(np.nanmin(sw["normalized_length"])),
                                    float(np.nanmax(sw["normalized_length"]))],
        "max_coupling_residual": float(np.nanmax(sw["coupling_residual_max"])),
        **(extra or {}),
    }


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def pronation_comparison(model, actuator, joint, n_angles):
    """Moment arm about an elbow joint at forearm pro_sup min/0/max (reported raw)."""
    side = _side(joint) or "r"
    ps = f"pro_sup_{side}"
    ps_dof = model.joint(ps).dofadr[0]
    elbow_mid = float(np.clip(np.pi / 2, *model.jnt_range[model.joint(joint).id]))
    d = _pose(model, standing(model).qpos, {joint: elbow_mid})
    mujoco.mj_fwdPosition(model, d)

    def arm_about_ps(a_id):
        return float(-bm.tendon_jacobian(model, d, int(model.actuator_trnid[a_id, 0]))[ps_dof])

    pronators = {}
    for q in ("PQ", "PT"):
        try:
            pronators[model.actuator(resolve_muscle(model, q, joint)).name] = None
        except ExperimentError:
            pass
    for n in pronators:
        pronators[n] = arm_about_ps(model.actuator(n).id)
    pr = [v for v in pronators.values() if abs(v) > 1e-4]
    sup_dir = None
    if pr and (all(v > 0 for v in pr) or all(v < 0 for v in pr)):
        sup_dir = "-pro_sup" if pr[0] > 0 else "+pro_sup"
    res = {"pro_sup_joint": ps, "elbow_angle_for_signs_rad": elbow_mid,
           "muscle_moment_arm_about_pro_sup_m": arm_about_ps(actuator),
           "pronator_moment_arms_about_pro_sup_m": pronators,
           "inferred_supination_direction": sup_dir,
           "note": ("pro_sup values are the raw source MoBL coordinate. Supination direction is "
                    "INFERRED as opposite to the source pronators' (PQ/PT) moment-arm sign; a "
                    "muscle with positive moment arm about pro_sup drives it toward +pro_sup."),
           "curves": {}}
    lo, hi = model.jnt_range[model.joint(ps).id]
    for v in (lo, 0.0, hi):
        sw = angle_sweep(model, actuator, joint, n_angles, (0.0,), fixed={ps: v}, fd_check=False)
        res["curves"][f"{v:+.4f}"] = {"angles": sw["angles"].tolist(),
                                      "moment_arm_total_m": sw["moment_arm_total_m"].tolist(),
                                      "mean_m": float(sw["moment_arm_total_m"].mean())}
    return res


def run(out_dir, muscle="BIClong", joint=None, n_angles=41, activations=DEFAULT_ACTIVATIONS,
        step_angle=None, on_s=0.3, off_s=0.3, ligament_stiffness=0.0, ligament_slack_scale=1.0,
        ligament_slack_reference="source", ligament_joint="shoulder_rot_r"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = bm.load_model(ligament_stiffness, ligament_slack_scale, ligament_slack_reference)
    a = resolve_muscle(model, muscle, joint)
    joint = resolve_joint(model, a, joint)
    sw = angle_sweep(model, a, joint, n_angles, activations)
    lo, hi = model.jnt_range[model.joint(joint).id]
    if step_angle is None:
        step_angle = float(sw["angles"][int(np.argmax(np.abs(sw["moment_arm_total_m"])))])
    if not lo <= step_angle <= hi:
        raise ExperimentError(f"step angle {step_angle} outside source range [{lo}, {hi}]")
    st = activation_step(model, a, joint, step_angle, on_s, off_s)
    lig = ligament_sweep(model, ligament_joint, n_angles)
    pro = pronation_comparison(model, a, joint, n_angles) if joint.startswith("elbow_flexion") else None

    acts = sw["activations"]
    rows = []
    for i, ang in enumerate(sw["angles"]):
        for k, av in enumerate(acts):
            rows.append([ang, np.degrees(ang), av, sw["mtu_length_m"][i], sw["normalized_length"][i],
                         sw["tension_N"][i, k], sw["active_tension_N"][i, k], sw["passive_tension_N"][i],
                         sw["moment_arm_partial_m"][i], sw["moment_arm_total_m"][i],
                         sw["fd_moment_arm_partial_m"][i], sw["fd_moment_arm_total_m"][i],
                         sw["torque_partial_Nm"][i, k], sw["torque_total_Nm"][i, k],
                         sw["active_torque_partial_Nm"][i, k], sw["engine_active_qfrc_Nm"][i, k],
                         sw["passive_all_muscles_torque_Nm"][i],
                         sw["passive_ligament_torque_Nm"][i]])
    _write_csv(out / "sweep.csv",
               ["angle_rad", "angle_deg", "activation", "mtu_length_m", "normalized_length",
                "tension_N", "active_tension_N", "passive_tension_N", "moment_arm_partial_m",
                "moment_arm_total_m", "fd_moment_arm_partial_m", "fd_moment_arm_total_m",
                "torque_partial_Nm", "torque_total_Nm", "active_torque_partial_Nm",
                "engine_active_qfrc_Nm",
                "passive_all_muscles_torque_Nm", "passive_ligament_torque_Nm"], rows)
    _write_csv(out / "activation_step.csv",
               ["time_s", "ctrl", "activation", "reference_activation", "tension_N", "torque_Nm"],
               zip(st["time_s"], st["ctrl"], st["activation"], st["reference_activation"],
                   st["tension_N"], st["torque_Nm"]))
    lrows = []
    for i, ang in enumerate(lig["angles"]):
        lrows.append([ang, *lig["length_m"][i], *lig["tension_N"][i], *lig["energy_J"][i],
                      lig["generalized_force"][i], lig["neg_dE_dq"][i]])
    _write_csv(out / "ligaments.csv",
               ["angle_rad"] + [f"{n}_length_m" for n in lig["names"]]
               + [f"{n}_tension_N" for n in lig["names"]] + [f"{n}_energy_J" for n in lig["names"]]
               + ["generalized_force_Nm", "neg_dEdq_Nm"], lrows)
    np.savez(out / "experiment.npz",
             **{f"sweep_{k}": v for k, v in sw.items()},
             **{f"step_{k}": v for k, v in st.items() if k != "metrics"},
             **{f"lig_{k}": np.asarray(v) for k, v in lig.items() if k != "joint"})
    lig_on = lig["stiffness_N_per_m"] > 0
    taut = lig["tension_N"] > 0
    diag = {
        "experiment": "isometric bench test on sourced reference model (not subject-specific)",
        "muscle_parameters": muscle_parameters(model, a),
        "sweep": _summary(model, a, joint, sw),
        "activation_step": st["metrics"],
        "ligaments": {
            "joint": ligament_joint, "config": bm.ligament_config(model),
            "stiffness_N_per_m": lig["stiffness_N_per_m"],
            "taut_fraction_per_route": dict(zip(lig["names"], taut.mean(0).tolist())),
            "max_tension_N_per_route": dict(zip(lig["names"], lig["tension_N"].max(0).tolist())),
            "length_range_m_per_route": {n: [float(lig["length_m"][:, j].min()), float(lig["length_m"][:, j].max())]
                                         for j, n in enumerate(lig["names"])},
            "slack_length_m": dict(zip(lig["names"], lig["slack_length_m"])),
            "energy_consistency_max_abs_Nm": float(np.abs(lig["generalized_force"] - lig["neg_dE_dq"]).max()),
            "energy_consistency_note": "generalized ligament force along couplings vs -dE/dq "
                                       "(central difference, h=1e-6 rad)",
            "status": ("exploratory tension-only prior (user stiffness/slack), not measured"
                       if lig_on else "source model: ligament routes carry no force (stiffness 0)"),
        },
        "forearm_rotation_comparison": pro,
        "units": {"angle": "rad", "length": "m", "force": "N", "torque": "N*m", "time": "s",
                  "energy": "J", "activation": "dimensionless 0..1"},
        "sign_conventions": {
            "tension_N": "-actuator_force (>=0 pulling)", "moment_arm_m": "-dL/dq",
            "torque_Nm": "moment_arm_m*tension_N (= native qfrc_actuator component)",
            "partial": "isolated coordinate derivative", "total": "along native couplings",
        },
        "limitations": [
            "MuJoCo muscle: rigid tendon, MTU-normalized length; no fiber-tendon equilibrium",
            "generic reference parameters (MyoSim/MoBL, Rajagopal, MyoBack), not personalized",
            "joint velocity is zero (isometric); force-velocity effects not exercised",
            "torque is the generalized force on the swept coordinate, gravity/contacts excluded",
            "no comparison with digitized experimental moment-arm data (qualitative only, cf. "
            "Murray, Delp & Buchanan 1995, doi:10.1016/0021-9290(94)00114-j)",
        ],
        "versions": {k: metadata.version(k) for k in ("mujoco", "myo-sim", "numpy")},
        "command": " ".join([".venv-biomech/bin/python", "scripts/biomech_experiment.py"] + sys.argv[1:])
        if Path(sys.argv[0]).name == "biomech_experiment.py" else None,
        "files": ["sweep.csv", "activation_step.csv", "ligaments.csv", "experiment.npz"],
    }
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=1, default=bm._json_default))
    return diag


def main(argv=None):
    ap = argparse.ArgumentParser(description="Isometric MuJoCo muscle bench experiment")
    ap.add_argument("--output", required=True)
    ap.add_argument("--muscle", default="BIClong", help="source actuator name (case-insensitive)")
    ap.add_argument("--joint", default=None, help="independent joint to sweep (default: auto)")
    ap.add_argument("--n-angles", type=int, default=41)
    ap.add_argument("--activations", default="0,0.25,0.5,1")
    ap.add_argument("--step-angle", type=float, default=None, help="rad; default = max moment arm")
    ap.add_argument("--step-on", type=float, default=0.3, help="s at ctrl=1")
    ap.add_argument("--step-off", type=float, default=0.3, help="s at ctrl=0 afterwards")
    ap.add_argument("--ligament-stiffness", type=float, default=0.0)
    ap.add_argument("--ligament-slack-scale", type=float, default=1.0)
    ap.add_argument("--ligament-slack-reference", choices=bm.SLACK_REFERENCES, default="source")
    ap.add_argument("--ligament-joint", default="shoulder_rot_r")
    a = ap.parse_args(argv)
    try:
        acts = tuple(float(x) for x in a.activations.split(","))
        if any(not 0 <= x <= 1 for x in acts):
            raise ExperimentError("activations must be in [0, 1]")
        if a.n_angles < 3:
            raise ExperimentError("--n-angles must be >= 3")
        diag = run(a.output, a.muscle, a.joint, a.n_angles, acts, a.step_angle, a.step_on,
                   a.step_off, a.ligament_stiffness, a.ligament_slack_scale,
                   a.ligament_slack_reference, a.ligament_joint)
    except (ExperimentError, bm.BiomechError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"sweep": diag["sweep"], "activation_step": diag["activation_step"]},
                     indent=1, default=bm._json_default))
    return 0


if __name__ == "__main__":
    sys.exit(main())
