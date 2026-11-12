#!/usr/bin/env python3
"""Fit a recorded .twin to constrained anatomical shoulder/elbow joints of the
MyoSim reference full-body model, and export motion + mechanics diagnostics.

Run with .venv-biomech/bin/python.  See docs/biomechanics-contract.md.

What is fit: for each arm the four *independent* coordinates elv_angle,
shoulder_elv, shoulder_rot, elbow_flexion.  Every dependent joint (sternoclavicular,
scapula, acromioclavicular, shoulder1_r2, ...) is recomputed by the model's native
equality polynomials (biomech_model.enforce_couplings) on EVERY residual evaluation.
Held at the standing keyframe (explicit priors): pro_sup, wrists, fingers, trunk,
lower body, root.  Recorded per-finger quaternions are never copied.

Alignment: recorded joint positions come from reconstruct_twin forward kinematics
(camera frame x right, y down, z forward).  Targets are expressed relative to the
recorded shoulder in a per-frame torso frame (lateral = left-right shoulder,
up = camera -y orthogonalised, forward = lateral x up; right-handed, no mirroring)
and mapped to the same construction in the model frame.  One per-subject scale
(model arm length / recorded arm length) is applied.  This is an alignment of
RECORDED coordinates to REFERENCE anatomy, not a measurement of personal anatomy.
Torso lean/twist is therefore NOT fit (torso held at its standing prior).
"""
import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reconstruct_twin as rt  # noqa: E402
import biomech_ligaments as lig  # noqa: E402

warnings.filterwarnings("ignore")

SIDES = ("left", "right")
INDEP_STEMS = ("elv_angle", "shoulder_elv", "shoulder_rot", "elbow_flexion")
DEPENDENT_STEMS = ("sternoclavicular_r2", "sternoclavicular_r3", "unrotscap_r3", "unrotscap_r2",
                   "acromioclavicular_r1", "acromioclavicular_r2", "acromioclavicular_r3", "shoulder1_r2",
                   "unrothum_r1", "unrothum_r2", "unrothum_r3")
CONF_FLOOR = 0.02        # continuous weight floor (never drops a frame)
CONTINUITY = 0.1     # meters-equivalent per radian of change from previous frame
NEUTRAL_PRIOR = 0.01     # meters-equivalent per radian from standing posture (weak)
LIMIT_PENALTY = 5.0      # hinge penalty on dependent-joint limit violation (per rad)
TARGET_NAMES = ("thorax", "head", "left_shoulder", "left_elbow", "left_wrist",
                "right_shoulder", "right_elbow", "right_wrist")


IMG_MIN_SCORE = 0.3      # minimum logged keypoint score to use a 2D observation
IMG_MARGIN_PX = 2.0      # observations closer than this to / outside the image border are rejected
SUPPORT_MIN_CONF = 0.2   # 3D target counted as "supported" (reporting / rmse) at or above this confidence
AX_GAIN = 8.0            # extra continuity on axially weakly observable coordinates (soft, not a cap)
IMG_JOINTS = ("elbow", "wrist")


class FitError(Exception):
    pass


def load_image_observations(path, subject_segment):
    """Read saved 2D keypoints from the bundle's fixed member poses.jsonl (no extraction).
    Returns (obs, status): obs[frame] = dict(width, height, pts{name:(x,y,score)}) for left/right
    elbow/wrist from ``observed_keypoints`` (network output before temporal stabilisation).
    Legacy twin.json or bundles without poses.jsonl return ({}, reason) -- an explicit fallback."""
    import zipfile
    path = Path(path)
    if not zipfile.is_zipfile(path):
        return {}, "unavailable: legacy JSON has no saved image observations"
    wanted = {f"{s}_{j}" for s in SIDES for j in IMG_JOINTS}
    obs = {}
    with zipfile.ZipFile(path) as z:
        if "poses.jsonl" not in z.namelist():
            return {}, "unavailable: bundle has no poses.jsonl"
        with z.open("poses.jsonl") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for pr in rec.get("people") or []:
                    if pr.get("person_index") != 0:
                        continue
                    seg = (pr.get("twin") or {}).get("subject_segment")
                    if seg is not None and subject_segment is not None and seg != subject_segment:
                        continue
                    kps = pr.get("observed_keypoints") or pr.get("keypoints") or []
                    pts = {k["name"]: (float(k["x"]), float(k["y"]), float(k["score"]))
                           for k in kps if k.get("name") in wanted}
                    obs[rec["frame"]] = {"width": rec.get("width"), "height": rec.get("height"), "pts": pts}
    return obs, "ok"


def side_suffix(side):
    return "_l" if side == "left" else "_r"


# ---------------------------------------------------------------- recording side
def recorded_positions(twin, subject, names, parents, frames):
    """FK joint positions (F,J,3) in camera frame and confidences (F,J)."""
    person = subject.get("person") or {}
    offsets = np.asarray(person.get("rest_offsets"), dtype=float)
    J = len(names)
    if offsets.shape != (J, 3) or not rt._finite(offsets):
        raise FitError("person.rest_offsets must be finite (J,3)")
    P = np.full((len(frames), J, 3), np.nan)
    C = np.zeros((len(frames), J))
    ok = np.zeros(len(frames), dtype=bool)
    reasons = [None] * len(frames)
    for k, f in enumerate(frames):
        try:
            rp = np.asarray(f.get("root_position"), dtype=float)
            if rp.shape != (3,) or not rt._finite(rp):
                raise rt.TwinError("invalid root_position")
            local, conf = rt.frame_rotations(f, names)
            pos, _ = rt.forward_kinematics(parents, offsets, local,
                                           rt.quat_to_matrix(f.get("root_orientation_xyzw")), rp)
            P[k], C[k], ok[k] = pos, conf, True
        except (rt.TwinError, ValueError, TypeError) as e:
            reasons[k] = str(e)
    return P, C, ok, reasons


def torso_basis(left_sh, right_sh):
    """Columns (lateral-left, up, forward) in camera coordinates; None if degenerate."""
    lat = left_sh - right_sh
    n = np.linalg.norm(lat)
    if n < 1e-6:
        return None
    lat = lat / n
    up = np.array([0.0, -1.0, 0.0])  # camera y is down
    up = up - lat * (up @ lat)
    m = np.linalg.norm(up)
    if m < 1e-3:
        return None
    up /= m
    fwd = np.cross(lat, up)
    return np.stack([lat, up, fwd], axis=1)


YAW_MIN_CONF = 0.5       # both shoulder confidences needed for a heading estimate
YAW_MIN_HORIZ = 0.3      # |horizontal shoulder line| / |shoulder line| below this is degenerate (foreshortened)


def heading_yaw(P, C, ix, valid):
    """Per-frame torso heading (rad, about world +z) RELATIVE to the first supported frame.

    Heading comes from the horizontal projection of the estimated 3D shoulder line (left-right,
    by name) in the camera frame (x right, y down, z forward; right-handed, 'up' = camera -y).
    A rotation theta about camera up maps to the same theta about model +z (proper camera->world
    map, no reflection).  Unsupported frames hold the previous heading.  Returns
    (yaw (F,), supported (F,) bool, horiz_fraction (F,)).  This is derived from estimated
    shoulder 3D (depth is monocular and uncertain), NOT measured 6-DOF tracking."""
    F = P.shape[0]
    up = np.array([0.0, -1.0, 0.0])
    yaw = np.zeros(F)
    supported = np.zeros(F, dtype=bool)
    frac = np.zeros(F)
    ref = None
    last = 0.0
    for k in range(F):
        yaw[k] = last
        if not valid[k]:
            continue
        lat = P[k, ix["left_shoulder"]] - P[k, ix["right_shoulder"]]
        n = np.linalg.norm(lat)
        if n < 1e-6:
            continue
        lh = lat - up * (lat @ up)
        frac[k] = np.linalg.norm(lh) / n
        if min(C[k, ix["left_shoulder"]], C[k, ix["right_shoulder"]]) < YAW_MIN_CONF or frac[k] < YAW_MIN_HORIZ:
            continue
        lh = lh / np.linalg.norm(lh)
        if ref is None:
            ref = lh
        theta = float(np.arctan2(up @ np.cross(ref, lh), ref @ lh))
        yaw[k] = last = theta
        supported[k] = True
    return yaw, supported, frac


def yaw_quat(theta, q_base):
    """Rotation by theta about world +z composed (left) with base wxyz quaternion."""
    qz = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = q_base
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def model_basis(mk):
    lat = np.asarray(mk["left_shoulder"]) - np.asarray(mk["right_shoulder"])
    lat = lat / np.linalg.norm(lat)
    up = np.array([0.0, 0.0, 1.0])  # MuJoCo gravity is -z in this model
    up = up - lat * (up @ lat)
    up /= np.linalg.norm(up)
    return np.stack([lat, up, np.cross(lat, up)], axis=1)


# ---------------------------------------------------------------- model side
class ArmFitter:
    def __init__(self, bm, model, data, mujoco):
        self.bm, self.model, self.data, self.mj = bm, model, data, mujoco
        self.std_qpos = data.qpos.copy()
        self.qadr, self.names, self.lo, self.hi = {}, {}, {}, {}
        for side in SIDES:
            sfx = side_suffix(side)
            for stem in INDEP_STEMS + DEPENDENT_STEMS:
                nm = stem + sfx
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
                if jid < 0:
                    raise FitError(f"model lacks joint {nm}")
                self.qadr[nm] = int(model.jnt_qposadr[jid])
                if model.jnt_limited[jid]:
                    self.lo[nm], self.hi[nm] = [float(x) for x in model.jnt_range[jid]]
                else:
                    self.lo[nm], self.hi[nm] = -np.inf, np.inf
        self.indep = [s + side_suffix(sd) for sd in SIDES for s in INDEP_STEMS]
        self.dep = [s + side_suffix(sd) for sd in SIDES for s in DEPENDENT_STEMS]
        self.x0 = np.array([self.std_qpos[self.qadr[n]] for n in self.indep])
        self.lb = np.array([self.lo[n] for n in self.indep])
        self.ub = np.array([self.hi[n] for n in self.indep])
        self.markers_standing = bm.marker_positions(model, data)
        self.basis = model_basis(self.markers_standing)

    def set_x(self, x):
        for n, v in zip(self.indep, x):
            self.data.qpos[self.qadr[n]] = v
        self.unsupported_eq = self.bm.enforce_couplings(self.model, self.data)
        self.mj.mj_kinematics(self.model, self.data)  # marker_positions needs current kinematics

    def markers(self, x):
        self.set_x(x)
        return self.bm.marker_positions(self.model, self.data)

    def dependent_violation(self):
        v = []
        for n in self.dep:
            q = self.data.qpos[self.qadr[n]]
            v.append(max(self.lo[n] - q, 0.0, q - self.hi[n]))
        return np.array(v)

    def arm_length(self):
        mk = self.markers_standing
        return float(np.mean([np.linalg.norm(mk[f"{s}_elbow"] - mk[f"{s}_shoulder"]) +
                              np.linalg.norm(mk[f"{s}_wrist"] - mk[f"{s}_elbow"]) for s in SIDES]))


SEED_ELV = (0.2, 0.8, 1.4, 2.0, 2.6, 3.0)
SEED_PLANE = (-1.2, 0.0, 1.2, 2.0)
RESEED_RMS = 0.05  # weighted arm residual (m) above which the global seed grid is tried again


class ImageContext:
    """Per-frame data for the reprojection residual (all in the recording's camera frame).
    obs (F,2,2,2) px, w2d (F,2,2) in [0,1] (0 = no valid observation), sh_cam (F,2,3) recorded shoulder,
    B_cam (F,3,3) torso basis columns (lat,up,fwd), focal (F,), cx/cy (F,), scale model/recording."""
    def __init__(self, obs, w2d, sh_cam, B_cam, focal, cx, cy, scale):
        self.obs, self.w2d, self.sh_cam, self.B_cam = obs, w2d, sh_cam, B_cam
        self.focal, self.cx, self.cy, self.scale = focal, cx, cy, scale

    def project(self, k, si, v_model, basis):
        """Predicted pixel of a model arm vector v (shoulder->joint, model world): returns (u,v,z)."""
        rel = basis.T @ v_model / self.scale
        pc = self.sh_cam[k, si] + self.B_cam[k] @ rel
        z = max(float(pc[2]), 0.2)
        return self.cx[k] + self.focal[k] * pc[0] / z, self.cy[k] + self.focal[k] * pc[1] / z, z


def run_fit(fitter, targets, weights, valid, prev_seed, max_nfev, img=None, support3d=None):
    """Sequential per-frame robust fit; arms are independent given the torso (4 coordinates each).
    targets (F,2,2,3): side, [elbow, wrist] rel. to shoulder in model frame; weights (F,2,2) are
    3D-target weights.  ``img`` adds a confidence-weighted reprojection residual (metres = z/f * px).
    An arm with neither supported 3D targets nor valid 2D observations is HELD (not solved)."""
    from scipy.optimize import least_squares
    F = targets.shape[0]
    nI = len(fitter.indep)
    n_arm = len(INDEP_STEMS)
    X = np.zeros((F, nI))
    rmse = np.zeros(F)
    status = [None] * F
    x_prev = fitter.x0.copy()
    if support3d is None:
        support3d = weights > CONF_FLOOR + 1e-9
    has2d = (img.w2d > 0) if img is not None else np.zeros_like(support3d)

    def solve_arm(k, si, x_all, seeds):
        sl = slice(si * n_arm, (si + 1) * n_arm)
        s = SIDES[si]
        w = np.sqrt(weights[k, si])  # (2,)
        w2 = np.sqrt(img.w2d[k, si]) if img is not None else np.zeros(2)

        def resid(xa):
            x = x_all.copy()
            x[sl] = xa
            mk = fitter.markers(x)
            sh = mk[f"{s}_shoulder"]
            parts = []
            for ji, j in enumerate(IMG_JOINTS):
                v = mk[f"{s}_{j}"] - sh
                parts.append(w[ji] * (v - targets[k, si, ji]))
                if w2[ji] > 0:
                    u, vv, z = img.project(k, si, v, fitter.basis)
                    parts.append(w2[ji] * (z / img.focal[k]) * (np.array([u, vv]) - img.obs[k, si, ji]))
            viol = fitter.dependent_violation()[si * len(DEPENDENT_STEMS):(si + 1) * len(DEPENDENT_STEMS)]
            # Soft extra continuity where a coordinate is axially unobservable: shoulder-plane angle
            # when the arm hangs (elevation ~ 0) and humeral rotation when the elbow is straight.
            mult = np.ones(n_arm)
            mult[0] += AX_GAIN * np.exp(-(xa[1] / 0.25) ** 2)
            mult[2] += AX_GAIN * np.exp(-(xa[3] / 0.3) ** 2)
            parts += [CONTINUITY * mult * (xa - x_prev[sl]), NEUTRAL_PRIOR * (xa - fitter.x0[sl]),
                      LIMIT_PENALTY * viol]
            return np.concatenate(parts)
        best = None
        for s0 in seeds:
            try:
                sol = least_squares(resid, np.clip(s0, fitter.lb[sl], fitter.ub[sl]),
                                    bounds=(fitter.lb[sl], fitter.ub[sl]), loss="soft_l1",
                                    f_scale=0.03, max_nfev=max_nfev, x_scale=0.5)
            except (ValueError, FloatingPointError):
                continue
            if best is None or sol.cost < best.cost:
                best = sol
        return best

    def grid_seeds(si):
        sl = slice(si * n_arm, (si + 1) * n_arm)
        base = fitter.x0[sl]
        out = []
        for e in SEED_ELV:
            for pl in SEED_PLANE:
                s0 = base.copy()
                s0[0], s0[1], s0[3] = pl, e, 0.3
                out.append(s0)
        return out

    for k in range(F):
        if not valid[k]:
            X[k], rmse[k], status[k] = x_prev, np.nan, "held_no_target"
            continue
        x_new = x_prev.copy()
        stats = []
        for si in range(2):
            sl = slice(si * n_arm, (si + 1) * n_arm)
            observed = bool(support3d[k, si].any() or has2d[k, si].any())
            if not observed:
                stats.append("held_no_data")  # no supported 3D target, no valid 2D observation
                continue
            best = solve_arm(k, si, x_prev, [x_prev[sl]])
            rms = np.sqrt(2 * best.cost / max(1, len(best.fun))) if best is not None else np.inf
            if best is None or k == 0 or rms > RESEED_RMS:
                alt = solve_arm(k, si, x_prev, grid_seeds(si))
                if alt is not None and (best is None or alt.cost < best.cost - 1e-9):
                    best = alt
            if best is None:
                stats.append("optimizer_failed_held")
                continue
            x_new[sl] = best.x
            stats.append(f"ok:{best.status}")
        X[k] = x_new
        x_prev = x_new
        mk = fitter.markers(x_new)
        err = []
        for si, s in enumerate(SIDES):
            for ji, j in enumerate(IMG_JOINTS):
                if support3d[k, si, ji]:
                    err.append(np.linalg.norm(mk[f"{s}_{j}"] - mk[f"{s}_shoulder"] - targets[k, si, ji]))
        rmse[k] = float(np.sqrt(np.mean(np.square(err)))) if err else np.nan
        status[k] = "/".join(stats)
    return X, rmse, status


def reprojection_errors(fitter, X, img):
    """Pixel residuals (F,2,2) of final states against valid 2D observations (NaN where none)."""
    F = X.shape[0]
    out = np.full((F, 2, 2), np.nan)
    for k in range(F):
        if not (img.w2d[k] > 0).any():
            continue
        mk = fitter.markers(X[k])
        for si, s in enumerate(SIDES):
            for ji, j in enumerate(IMG_JOINTS):
                if img.w2d[k, si, ji] > 0:
                    u, v, _ = img.project(k, si, mk[f"{s}_{j}"] - mk[f"{s}_shoulder"], fitter.basis)
                    out[k, si, ji] = np.hypot(u - img.obs[k, si, ji, 0], v - img.obs[k, si, ji, 1])
    return out


def prescribed_derivatives(mujoco, m, qpos, ts):
    """qvel, qacc (F,nv) from nonuniform timestamps. Interior: (h2*v_back + h1*v_fwd)/(h1+h2) with
    v = mj_differentiatePos(...)/h (valid for the quaternion root); endpoints one-sided.
    qacc applies the same weighted scheme to qvel."""
    F, nv = qpos.shape[0], m.nv
    qvel = np.zeros((F, nv))
    if F < 2:
        return qvel, np.zeros((F, nv))
    seg = np.zeros((F - 1, nv))
    for i in range(F - 1):
        mujoco.mj_differentiatePos(m, seg[i], ts[i + 1] - ts[i], qpos[i], qpos[i + 1])
    qvel[0], qvel[-1] = seg[0], seg[-1]  # first-order one-sided
    if F >= 3:  # second-order one-sided endpoints (exact for quadratic motion on unequal steps)
        ha, hb = ts[1] - ts[0], ts[2] - ts[1]
        qvel[0] = seg[0] + (seg[0] - seg[1]) * ha / (ha + hb)
        ha, hb = ts[-1] - ts[-2], ts[-2] - ts[-3]
        qvel[-1] = seg[-1] + (seg[-1] - seg[-2]) * ha / (ha + hb)
    for k in range(1, F - 1):
        h1, h2 = ts[k] - ts[k - 1], ts[k + 1] - ts[k]
        qvel[k] = (h2 * seg[k - 1] + h1 * seg[k]) / (h1 + h2)
    dv = np.diff(qvel, axis=0) / np.diff(ts)[:, None]
    qacc = np.zeros((F, nv))
    qacc[0], qacc[-1] = dv[0], dv[-1]
    for k in range(1, F - 1):
        h1, h2 = ts[k] - ts[k - 1], ts[k + 1] - ts[k]
        qacc[k] = (h2 * dv[k - 1] + h1 * dv[k]) / (h1 + h2)
    return qvel, qacc


def mechanics(mujoco, model, qpos, ts):
    """Musculotendon lengths/velocities/Jacobians, passive and inverse-dynamics generalized forces."""
    m = model
    d = mujoco.MjData(m)
    F = qpos.shape[0]
    nu, nv = m.nu, m.nv
    qvel, qacc = prescribed_derivatives(mujoco, m, qpos, ts)
    L, V = np.zeros((F, nu)), np.zeros((F, nu))
    MA = np.zeros((F, nu, nv), dtype=np.float32)
    passive_joint = np.zeros((F, nv))
    passive_muscle = np.zeros((F, nv))
    inverse = np.zeros((F, nv))
    import biomech_model as bm
    lig_tension = np.zeros((F, len(bm.LIGAMENT_NAMES)))
    lig_tau = np.zeros((F, nv))
    saved_flags = m.opt.disableflags
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)  # no contacts in inverse dynamics
    try:
        for k in range(F):
            d.qpos[:], d.qvel[:], d.qacc[:] = qpos[k], qvel[k], qacc[k]
            d.act[:] = 0.0
            d.ctrl[:] = 0.0
            mujoco.mj_forward(m, d)  # overwrites d.qacc with forward-dynamics acceleration
            L[k] = d.actuator_length
            dense = np.zeros((nu, nv))  # actuator_moment is stored sparse (CSR) in MuJoCo >= 3
            mujoco.mju_sparse2dense(dense, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
            MA[k] = dense
            V[k] = MA[k] @ qvel[k]  # d(length)/dt = J qvel
            passive_joint[k] = d.qfrc_passive      # joint damping/stiffness + enabled tendon springs
            passive_muscle[k] = d.qfrc_actuator    # act=0: passive (parallel-elastic) muscle force only
            lig_tension[k] = [x["tension_N"] for x in bm.ligament_state(m, d)]
            lig_tau[k] = bm.ligament_generalized_force(m, d)
            d.qacc[:] = qacc[k]  # prescribed acceleration for inverse dynamics (NOT forward-dynamics qacc)
            mujoco.mj_inverse(m, d)
            inverse[k] = d.qfrc_inverse
    finally:
        m.opt.disableflags = saved_flags
    return dict(
        actuator_length_m=L, actuator_velocity_mps=V,
        actuator_moment_dLdq=MA,
        qvel=qvel, qacc=qacc,
        qvel_qacc_method=np.array("weighted backward/forward mj_differentiatePos (h2*vback+h1*vfwd)/(h1+h2); "
                                  "second-order one-sided at endpoints; qacc = same scheme on qvel"),
        qfrc_passive_joint_and_tendon_spring=passive_joint,
        qfrc_passive_muscle_act0=passive_muscle,
        qfrc_inverse_dynamics_demand=inverse,
        ligament_route_names=np.array(bm.LIGAMENT_NAMES),
        ligament_tension_N=lig_tension,
        qfrc_ligament_springs=lig_tau,
        units=np.array("length m; velocity m/s; moment=d(length)/d(q) m or m/rad (moment arm = -value by MuJoCo "
                       "sign convention); generalized force N or N*m; qvel rad/s or m/s; qacc rad/s^2 or m/s^2"),
        note=np.array("Inverse dynamics of PRESCRIBED fitted kinematics, contacts excluded, root free (derived yaw makes root forces non-physical). It is "
                      "required generalized force demand incl. equality-constraint reactions, NOT identified "
                      "muscle, tendon or tissue force and NOT measured from video. Activations are not "
                      "estimated; passive muscle uses activation 0."),
    )


# ---------------------------------------------------------------- per subject
def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_subject(twin, subject, args, out_dir, bm, mujoco, input_sha, names, parents):
    frames = subject.get("frames") or []
    if not frames:
        raise FitError("subject has no frames")
    for f in frames:
        if not isinstance(f.get("frame"), int) or isinstance(f.get("frame"), bool) or f["frame"] < 0:
            raise FitError("every frame needs a non-negative integer 'frame'")
    if len({f["frame"] for f in frames}) != len(frames):
        raise FitError("duplicate frame indices")
    frames = sorted(frames, key=lambda f: f["frame"])[:: max(1, args.stride)]
    if args.max_frames:
        frames = frames[: args.max_frames]
    fidx = np.array([f["frame"] for f in frames])
    stamps, timing_source = rt.timestamps_for(twin, fidx)
    P, C, ok, reasons = recorded_positions(twin, subject, names, parents, frames)
    ix = {n: names.index(n) for n in TARGET_NAMES if n in names}
    missing = [n for n in TARGET_NAMES if n not in ix]
    if missing:
        raise FitError(f"twin lacks joints {missing}")

    profile = getattr(args, "ligament_profile", "disabled")
    if profile != "disabled" and args.ligament_stiffness != 0:
        raise FitError("--ligament-stiffness (exploratory scalar) cannot be combined with --ligament-profile")
    model, data = bm.load_model(ligament_stiffness=args.ligament_stiffness,
                                ligament_slack_scale=args.ligament_slack_scale)
    if profile != "disabled":  # applied BEFORE fitting, mechanics and model.mjb export (identical model everywhere)
        lig_meta = lig.apply_profile(model, data, profile, slack_scale=args.ligament_slack_scale)
    else:
        lig_meta = lig.disabled_metadata()
    fitter = ArmFitter(bm, model, data, mujoco)
    F = len(frames)

    # Per-frame torso frame and shoulder-relative targets (recorded units).
    rel = np.zeros((F, 2, 2, 3))
    W = np.zeros((F, 2, 2))
    Bs = np.tile(np.eye(3), (F, 1, 1))
    valid = ok.copy()
    for k in range(F):
        if not ok[k]:
            continue
        B = torso_basis(P[k, ix["left_shoulder"]], P[k, ix["right_shoulder"]])
        if B is None:
            valid[k] = False
            reasons[k] = "degenerate shoulder line"
            continue
        Bs[k] = B
        for si, s in enumerate(SIDES):
            sh = P[k, ix[f"{s}_shoulder"]]
            for ji, j in enumerate(("elbow", "wrist")):
                rel[k, si, ji] = B.T @ (P[k, ix[f"{s}_{j}"]] - sh)  # components in (left, up, fwd)
                W[k, si, ji] = CONF_FLOOR + (1 - CONF_FLOOR) * C[k, ix[f"{s}_{j}"]]
    # Per-subject scale: confident arm lengths only.
    lens, lw = [], []
    for k in np.where(valid)[0]:
        for si in range(2):
            lens.append(np.linalg.norm(rel[k, si, 0]) + np.linalg.norm(rel[k, si, 1] - rel[k, si, 0]))
            lw.append(min(W[k, si, 0], W[k, si, 1]))
    if not lens or max(lw) < 0.2:
        raise FitError("no frame with confident arm observations to determine scale")
    lens, lw = np.asarray(lens), np.asarray(lw)
    sel = lw >= max(0.5, np.percentile(lw, 50))
    rec_len = float(np.median(lens[sel]))
    scale = fitter.arm_length() / rec_len
    # Map (left, up, fwd) components into model world coordinates.
    targets = (rel * scale) @ fitter.basis.T

    # Supported 3D targets (reported/used for rmse) vs floor-weighted depth prior.
    conf3 = np.zeros((F, 2, 2))
    for si, sd in enumerate(SIDES):
        for ji, j in enumerate(IMG_JOINTS):
            conf3[:, si, ji] = C[:, ix[f"{sd}_{j}"]]
    support3d = (conf3 >= SUPPORT_MIN_CONF) & valid[:, None, None]

    # Saved 2D image observations (measured in-image; depth along the ray stays inferred).
    img, img_info = None, {"status": "disabled by --no-image-observations"}
    if not getattr(args, "no_image_observations", False):
        obs, status_txt = load_image_observations(args.input, subject.get("subject_segment"))
        img_info = {"status": status_txt}
        if obs:
            o2 = np.zeros((F, 2, 2, 2))
            w2 = np.zeros((F, 2, 2))
            focal, cxs, cys = np.ones(F), np.zeros(F), np.zeros(F)
            rej = {"missing_keypoint": 0, "no_log_row_or_intrinsics": 0, "low_score": 0, "out_of_image": 0}
            above1 = 0
            for k in range(F):
                if not valid[k]:
                    continue
                row = obs.get(int(fidx[k]))
                fl = frames[k].get("focal_length_px")
                if row is None or not row.get("width") or not row.get("height") or not (
                        isinstance(fl, (int, float)) and np.isfinite(fl) and fl > 0):
                    rej["no_log_row_or_intrinsics"] += 4
                    continue
                focal[k], cxs[k], cys[k] = fl, row["width"] / 2.0, row["height"] / 2.0
                for si, sd in enumerate(SIDES):
                    for ji, j in enumerate(IMG_JOINTS):
                        pt = row["pts"].get(f"{sd}_{j}")
                        if pt is None or not np.all(np.isfinite(pt)):
                            rej["missing_keypoint"] += 1
                        elif pt[2] < IMG_MIN_SCORE:
                            rej["low_score"] += 1
                        elif not (IMG_MARGIN_PX <= pt[0] <= row["width"] - IMG_MARGIN_PX and
                                  IMG_MARGIN_PX <= pt[1] <= row["height"] - IMG_MARGIN_PX):
                            rej["out_of_image"] += 1
                        else:
                            o2[k, si, ji] = pt[:2]
                            w2[k, si, ji] = min(pt[2], 1.0)
                            above1 += pt[2] > 1.0
            sh_cam = np.stack([P[:, ix["left_shoulder"]], P[:, ix["right_shoulder"]]], axis=1)
            sh_cam = np.nan_to_num(sh_cam)
            img = ImageContext(o2, w2, sh_cam, Bs, focal, cxs, cys, scale)
            img_info.update({
                "source": "poses.jsonl observed_keypoints (network output before stabilisation), read in place",
                "min_score": IMG_MIN_SCORE, "margin_px": IMG_MARGIN_PX,
                "camera": "pinhole; focal_length_px from each twin frame (may be assumed, see "
                          "focal_length_assumed), principal point = image centre; NOT a camera calibration",
                "residual": "metres = (z/f) * pixel residual, soft_l1 shared with 3D terms",
                "rejected_observations": rej,
                "scores_above_one_clipped_to_one": int(above1),
                "provenance": "2D positions are image measurements; depth along each ray remains inferred "
                              "(3D-target prior + continuity).",
            })

    X, rmse, status = run_fit(fitter, targets, W, valid, None, args.max_nfev, img=img, support3d=support3d)
    reproj = reprojection_errors(fitter, X, img) if img is not None else np.full((F, 2, 2), np.nan)
    support2d = (img.w2d > 0) if img is not None else np.zeros((F, 2, 2), bool)
    with np.errstate(all="ignore"):
        reproj_rms = np.sqrt(np.nanmean(reproj.reshape(F, -1) ** 2, axis=1))

    # Evaluate final states: qpos, markers, limit/coupling checks.
    nq = model.nq
    qpos = np.tile(fitter.std_qpos, (F, 1))
    marker_names = list(bm.marker_positions(model, data).keys())
    markers = np.zeros((F, len(marker_names), 3))
    dep_violation = np.zeros(F)
    indep_violation = np.zeros(F)
    use_yaw = not getattr(args, "no_root_yaw", False)
    if use_yaw:
        if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise FitError("expected a free root joint as joint 0")
        yaw, yaw_supported, yaw_frac = heading_yaw(P, C, ix, valid)
    else:
        yaw, yaw_supported, yaw_frac = np.zeros(F), np.zeros(F, bool), np.zeros(F)
    q_root0 = fitter.std_qpos[3:7].copy()
    for k in range(F):
        fitter.set_x(X[k])
        if use_yaw:  # derived heading applied AFTER the arm fit; root translation stays fixed
            data.qpos[3:7] = yaw_quat(yaw[k], q_root0)
            mujoco.mj_kinematics(model, data)
        qpos[k] = data.qpos
        mk = bm.marker_positions(model, data)
        markers[k] = np.array([mk[n] for n in marker_names])
        dep_violation[k] = fitter.dependent_violation().max()
        indep_violation[k] = np.maximum(np.maximum(fitter.lb - X[k], X[k] - fitter.ub), 0).max()
    # Coupling check: re-apply and compare.
    coup_resid = 0.0
    for k in range(min(F, 20)):
        data.qpos[:] = qpos[k]
        before = data.qpos.copy()
        bm.enforce_couplings(model, data)
        coup_resid = max(coup_resid, float(np.abs(data.qpos - before).max()))

    cres = [bm.coupling_residuals(model, q) for q in qpos[:50]]
    coup_native = float(max(np.abs(c).max() for c in cres if c.size)) if any(c.size for c in cres) else None
    if coup_native is None:
        print("warning: no native joint couplings were enforced (biomech_model reports none)", file=sys.stderr)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = fitter.std_qpos
    mujoco.mj_forward(model, data)
    lig_state = bm.ligament_state(model, data)
    jn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    conf_out = C.astype(np.float32)
    sdir = Path(out_dir)
    sdir.mkdir(parents=True, exist_ok=True)
    nv = model.nv
    fitted_mask = np.zeros(nv, dtype=bool)
    for n in fitter.indep:
        fitted_mask[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] = True
    dependent_mask = np.zeros(nv, dtype=bool)
    for n in fitter.dep:
        dependent_mask[model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]] = True
    derived_mask = np.zeros(nv, dtype=bool)
    if use_yaw:
        derived_mask[3:6] = True  # root rotational DOFs: only yaw (about +z) is derived and varies
    np.savez_compressed(
        sdir / "motion.npz", derived_dof_mask=derived_mask, root_yaw_rad=yaw,
        root_yaw_supported=yaw_supported, root_yaw_horizontal_fraction=yaw_frac, qpos=qpos, timestamps_seconds=stamps, frame_indices=fidx,
        joint_names=np.array(names), source_joint_confidence=conf_out,
        marker_positions=markers, marker_names=np.array(marker_names), fit_rmse=rmse,
        fitted_dof_mask=fitted_mask, dependent_dof_mask=dependent_mask,
        qpos_joint_names=np.array(jn),
        target_valid=valid, support_3d=support3d, support_2d=support2d,
        reprojection_rmse_px=reproj_rms, fit_status=np.array(status),
    )
    mujoco.mj_saveModel(model, str(sdir / "model.mjb"))

    mech = None
    if args.mechanics:
        mech = mechanics(mujoco, model, qpos, stamps)
        np.savez_compressed(sdir / "mechanics.npz", **mech)

    # Jumps: per-frame joint-angle change rate among fitted coordinates.
    dt = np.diff(stamps)
    rate = np.abs(np.diff(X, axis=0)) / dt[:, None] if F > 1 else np.zeros((0, X.shape[1]))
    finite = rmse[np.isfinite(rmse)]
    support = support3d.sum(axis=(1, 2)) + support2d.sum(axis=(1, 2))
    held = np.array([("held_no_data" in st) for st in status])
    rp = reproj[np.isfinite(reproj)]
    report = {
        "input": {"sha256": input_sha, "units": (subject.get("person") or {}).get("units"),
                  "subject_segment": subject.get("subject_segment")},
        "frames": {"count": int(F), "stride": args.stride, "timing_source": timing_source,
                   "unusable_frames": [{"frame": int(fidx[k]), "reason": reasons[k]}
                                       for k in range(F) if not valid[k]]},
        "model": {"nq": int(model.nq), "nv": int(model.nv),
                  "ligament_stiffness": args.ligament_stiffness,
                  "ligament_slack_scale": args.ligament_slack_scale,
                  "ligament_profile": lig_meta,
                  "ligament_native_inventory_at_standing": lig_state,
                  "snapshot": "model.mjb (MuJoCo %s; version-specific)" % mujoco.__version__},
        "observed_dofs": {n: "fit to elbow/wrist position targets" for n in fitter.indep},
        "dependent_dofs_from_native_couplings": fitter.dep,
        "held_fixed_at_standing_keyframe": ["root translation and pitch/roll (yaw is derived, see root_heading)", "trunk/lumbar/neck", "pro_sup (forearm twist)",
                                            "wrists", "fingers", "lower body", "scapula/clavicle independent motion"],
        "alignment": {
            "recording_frame": "camera x right, y down, z forward (inferred meters)",
            "per_frame_torso_frame": "lateral=left-right shoulder; up=camera -y orthogonalised; forward=lateral x up",
            "model_basis_columns_lateral_up_forward": fitter.basis.tolist(),
            "handedness": "proper rotation, no mirroring; left/right matched by joint name",
            "recorded_arm_length": rec_len, "model_arm_length_m": fitter.arm_length(),
            "scale_model_over_recording": scale,
            "targets": "elbow and wrist relative to shoulder; shoulder translation absorbed by model scapula",
            "not_fit": "torso lean/twist, root translation, lower body; this is NOT personalised anatomy",
        },
        "root_heading": {
            "enabled": use_yaw,
            "method": "yaw about world +z from horizontal projection of estimated left-right shoulder line, "
                      "relative to first frame with both shoulder confidences >= %.1f" % YAW_MIN_CONF,
            "supported_frames": int(yaw_supported.sum()),
            "held_frames": int((~yaw_supported).sum()),
            "yaw_range_rad": [float(yaw.min()), float(yaw.max())],
            "max_yaw_rate_rad_per_s": float(np.abs(np.diff(yaw) / np.diff(stamps)).max()) if F > 1 else 0.0,
            "status": "DERIVED from estimated shoulder 3D (monocular depth, sign/foreshortening uncertain); "
                      "not measured 6-DOF tracking. Root translation, pelvis tilt, legs, head fixed at reference.",
            "arm_fit_invariance": "arm targets are torso-relative, so heading does not change the arm fit",
        },
        "image_observations": img_info,
        "remaining_ambiguities": [
            "depth along each camera ray is inferred, not measured",
            "humeral rotation (shoulder_rot) is weakly observable with a straight elbow",
            "shoulder plane angle (elv_angle) is undefined near zero elevation",
            "focal length may be assumed; no camera calibration is claimed"],
        "weights": {"conf_floor": CONF_FLOOR, "continuity_m_per_rad": CONTINUITY,
                    "neutral_prior_m_per_rad": NEUTRAL_PRIOR, "loss": "soft_l1 f_scale=0.03 m",
                    "elbow_target_confidence": "joint_rotations confidence of the elbow joint",
                    "wrist_target_confidence": "joint_rotations confidence of the wrist joint"},
        "fit": {"rmse_m_model_scale_mean": float(finite.mean()) if finite.size else None,
                "rmse_m_median": float(np.median(finite)) if finite.size else None,
                "rmse_m_p95": float(np.percentile(finite, 95)) if finite.size else None,
                "rmse_m_max": float(finite.max()) if finite.size else None,
                "frames_with_no_supported_observation": int((support == 0).sum()),
                "frames_with_any_arm_held_no_data": int(held.sum()),
                "frames_with_supported_3d": int(support3d.any(axis=(1, 2)).sum()),
                "frames_with_valid_2d": int(support2d.any(axis=(1, 2)).sum()),
                "arm_joint_observations_3d": int(support3d.sum()),
                "arm_joint_observations_2d": int(support2d.sum()),
                "reprojection_px": {"count": int(rp.size),
                                    "rms": float(np.sqrt(np.mean(rp ** 2))) if rp.size else None,
                                    "median": float(np.median(rp)) if rp.size else None,
                                    "p95": float(np.percentile(rp, 95)) if rp.size else None,
                                    "max": float(rp.max()) if rp.size else None},
                "rmse_note": "rmse_m_* = 3D-target residual over supported (conf>=%.1f) targets only; not "
                             "comparable to FEM skinned-vertex error; held/no-data arms report no residual" % SUPPORT_MIN_CONF,
                "max_joint_rate_rad_per_s": float(rate.max()) if rate.size else 0.0,
                "frames_with_rate_over_20_rad_per_s": int((rate.max(axis=1) > 20).sum()) if rate.size else 0,
                "optimizer_status_counts": {s: status.count(s) for s in sorted(set(status))}},
        "checks": {"max_dependent_joint_limit_violation_rad": float(dep_violation.max()),
                   "max_independent_bound_violation_rad": float(indep_violation.max()),
                   "max_coupling_reapply_change": coup_resid,
                   "max_native_coupling_residual": coup_native,
                   "unsupported_equalities_left_unenforced": list(fitter.unsupported_eq)},
        "mechanics": None if mech is None else {
            "file": "mechanics.npz", "contacts": "excluded", "activation": "0 (passive only; not estimated)",
            "claims": "kinematic demand under prescribed motion; no subject-specific force identification"},
        "limitations": [
            "Reference anatomy (MyoSim), not subject-specific bones, muscle or ligament.",
            "Forearm twist, wrists, fingers, scapular independent motion and lower body are priors.",
            "Fitted from RGB-derived joint estimates; shoulder_rot is weakly observable at straight elbow.",
            "No EMG/GRF/load data: muscle activation and tissue forces are not observed.",
        ],
    }
    (sdir / "fit_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("input", help=".twin or twin.json")
    p.add_argument("--output", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--ligament-stiffness", type=float, default=0.0)
    p.add_argument("--ligament-slack-scale", type=float, default=1.0)
    p.add_argument("--ligament-profile", choices=("disabled", "boardman1996"), default="disabled",
                   help="sourced SGHL/CHL stiffness (Boardman 1996); incompatible with nonzero --ligament-stiffness")
    p.add_argument("--max-nfev", type=int, default=60)
    p.add_argument("--no-root-yaw", action="store_true",
                   help="keep the root at the standing heading instead of deriving torso yaw")
    p.add_argument("--no-image-observations", action="store_true",
                   help="do not use saved 2D keypoints from poses.jsonl (legacy 3D-only fit)")
    p.add_argument("--mechanics", action="store_true", help="also write mechanics.npz")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        import mujoco
        import biomech_model as bm
        twin, sha = rt.load_twin(args.input)
        names, parents = rt.validate_hierarchy(twin)
        subjects = twin.get("subjects") or []
        if not subjects:
            raise FitError("twin has no subjects")
        out = Path(args.output)
        reports = []
        for i, sub in enumerate(subjects):
            sd = out if len(subjects) == 1 else out / rt.safe_name(sub.get("subject_segment"), i)
            reports.append(fit_subject(twin, sub, args, sd, bm, mujoco, sha, names, parents))
            r = reports[-1]["fit"]
            print(f"subject {i}: frames={reports[-1]['frames']['count']} rmse_mean={r['rmse_m_mean'] if 'rmse_m_mean' in r else r['rmse_m_model_scale_mean']} "
                  f"max_dep_viol={reports[-1]['checks']['max_dependent_joint_limit_violation_rad']:.2e}")
    except (FitError, rt.TwinError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
