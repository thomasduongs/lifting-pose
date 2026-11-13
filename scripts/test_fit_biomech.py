"""Tests for fit_biomech. Run: .venv-biomech/bin/python -m unittest scripts/test_fit_biomech.py
Uses the real MuJoCo engine and MyoSim model at small scale (a few frames)."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import mujoco  # noqa: F401
    import biomech_model as bm
    import fit_biomech as fb
    HAVE = True
except ImportError:  # run with the biomech venv
    HAVE = False
from test_skeleton_recording import make_twin


def zrot_quat(deg):
    h = np.radians(deg) / 2
    return [0.0, 0.0, float(np.sin(h)), float(np.cos(h))]


def set_rot(frame, joint, quat):
    for e in frame["joint_rotations"]:
        if e["joint"] == joint:
            e["quaternion_xyzw"] = quat


def run_cli(twin, out, extra=()):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.json"
        p.write_text(json.dumps(twin))
        rc = fb.main([str(p), "--output", str(out), "--max-nfev", "40", *extra])
    assert rc == 0
    return np.load(Path(out) / "motion.npz"), json.loads((Path(out) / "fit_report.json").read_text())


@unittest.skipUnless(HAVE, "needs .venv-biomech (mujoco, myo_sim, biomech_model)")
class FitBiomechTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data = bm.load_model()
        cls.fitter = fb.ArmFitter(bm, cls.model, cls.data, mujoco)

    def test_known_pose_recovery_respects_limits_and_couplings(self):
        f = self.fitter
        names = f.indep
        truth = []
        for x in ([0.3, 1.0, 0.2, 1.2, 0.5, 0.8, 0.0, 0.6],
                  [-0.4, 0.5, 0.1, 0.4, 0.2, 1.4, -0.2, 1.0]):
            truth.append(np.array(x))
        tg = np.zeros((2, 2, 2, 3))
        for k, x in enumerate(truth):
            mk = f.markers(x)
            for si, s in enumerate(fb.SIDES):
                for ji, j in enumerate(("elbow", "wrist")):
                    tg[k, si, ji] = mk[f"{s}_{j}"] - mk[f"{s}_shoulder"]
        # Independent target construction: same geometry, independent of the optimizer.
        X, rmse, _ = fb.run_fit(f, tg, np.ones((2, 2, 2)), np.ones(2, bool), None, 200)
        for k in range(2):
            mk = f.markers(X[k])
            for si, s in enumerate(fb.SIDES):
                for ji, j in enumerate(("elbow", "wrist")):
                    err = np.linalg.norm(mk[f"{s}_{j}"] - mk[f"{s}_shoulder"] - tg[k, si, ji])
                    self.assertLess(err, 0.02, (k, s, j, err))  # regularisation allows a few mm
            self.assertTrue(np.all(X[k] >= f.lb - 1e-9) and np.all(X[k] <= f.ub + 1e-9))
            f.set_x(X[k])
            self.assertEqual(f.dependent_violation().max(), 0.0)
            before = f.data.qpos.copy()
            bm.enforce_couplings(f.model, f.data)
            self.assertLess(np.abs(f.data.qpos - before).max(), 1e-9)
        # Elbow flexion (well observed) recovered per arm.
        for si in range(2):
            i = names.index(("elbow_flexion" + fb.side_suffix(fb.SIDES[si])))
            self.assertLess(abs(X[0][i] - truth[0][i]), 0.1)

    def test_cli_no_mirroring_irregular_time_missing_data_no_hand_copy(self):
        twin = make_twin(n=4, conf=0.95)
        for f in twin["subjects"][0]["frames"]:
            set_rot(f, "left_shoulder", zrot_quat(-90))  # left arm up in image (camera -y)
            set_rot(f, "right_shoulder", zrot_quat(0))
        # Frame 2: all arm confidences zero (missing data).
        for e in twin["subjects"][0]["frames"][2]["joint_rotations"]:
            if "elbow" in e["joint"] or "wrist" in e["joint"]:
                e["confidence"] = 0.0
        with tempfile.TemporaryDirectory() as td:
            m1, rep = run_cli(twin, Path(td) / "a")
            twin2 = copy.deepcopy(twin)
            for f in twin2["subjects"][0]["frames"]:  # glitching finger quaternions must be ignored
                f["hands"] = [{"side": "left", "confidence": 1.0,
                               "bone_rotations_xyzw": [[0.7, 0.0, 0.7, 0.1]] * 21}]
            m2, _ = run_cli(twin2, Path(td) / "b")
        names = list(m1["qpos_joint_names"])
        adr = lambda n: self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        self.assertEqual(m1["qpos"].shape, (4, self.model.nq))      # every frame exported
        np.testing.assert_allclose(m1["timestamps_seconds"], [0.0, 0.1, 0.35, 0.4])  # irregular preserved
        np.testing.assert_array_equal(m1["frame_indices"], [0, 1, 2, 3])
        self.assertTrue(np.all(np.isfinite(m1["qpos"])))
        # Left raised higher than right: no mirrored handedness.
        el, er = m1["qpos"][1, adr("shoulder_elv_l")], m1["qpos"][1, adr("shoulder_elv_r")]
        self.assertGreater(el, er + 0.5, (el, er))
        # Missing frame holds neighbouring pose rather than being dropped or invented.
        self.assertTrue(np.isnan(m1["fit_rmse"][2]))  # no confident target -> no claimed residual
        np.testing.assert_allclose(m1["qpos"][2], m1["qpos"][1], atol=0.05)
        self.assertEqual(int(m1["fitted_dof_mask"].sum()), 8)
        # Hands never copied: identical output with and without hand data; fingers = standing prior.
        np.testing.assert_allclose(m1["qpos"], m2["qpos"])
        std = self.fitter.std_qpos
        free = np.ones(self.model.nq, bool)
        for n in self.fitter.indep + self.fitter.dep:
            free[adr(n)] = False
        np.testing.assert_allclose(m1["qpos"][:, free], np.tile(std[free], (4, 1)))
        self.assertEqual(rep["frames"]["count"], 4)
        self.assertEqual(rep["checks"]["max_dependent_joint_limit_violation_rad"], 0.0)
        self.assertGreater(rep["alignment"]["scale_model_over_recording"], 0)

    def test_dependent_joints_follow_native_couplings(self):
        f = self.fitter
        x = f.x0.copy()
        x[f.indep.index("shoulder_elv_l")] = 2.0
        f.set_x(x)
        a = f.qadr["sternoclavicular_r2_l"]
        self.assertGreater(abs(f.data.qpos[a] - f.std_qpos[a]), 0.1)  # dependent moved with elevation
        res = bm.coupling_residuals(f.model, f.data.qpos)
        self.assertGreater(res.size, 0)
        self.assertLess(np.abs(res).max(), 1e-9)
        self.assertEqual(list(f.unsupported_eq), [])

    def test_root_yaw_equivariance_and_handedness(self):
        deg = 40.0
        h = np.radians(deg) / 2
        up_quat = [0.0, -float(np.sin(h)), 0.0, float(np.cos(h))]  # +40 deg about camera up (-y)
        base = make_twin(n=4, conf=0.95)
        turned = copy.deepcopy(base)
        for k in (2, 3):
            turned["subjects"][0]["frames"][k]["root_orientation_xyzw"] = up_quat
        with tempfile.TemporaryDirectory() as td:
            a, ra = run_cli(base, Path(td) / "a")
            b, rb = run_cli(turned, Path(td) / "b")
            c, _ = run_cli(turned, Path(td) / "c", extra=("--no-root-yaw",))
        adr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
               for n in self.fitter.indep + self.fitter.dep]
        # Arm fit invariant under rigid heading change.
        np.testing.assert_allclose(a["qpos"][:, adr], b["qpos"][:, adr], atol=1e-6)
        # Static recording: no yaw. Turned frames: +40 deg about +z (same sense as camera up).
        np.testing.assert_allclose(a["root_yaw_rad"], 0.0, atol=1e-9)
        np.testing.assert_allclose(b["root_yaw_rad"], [0, 0, np.radians(deg), np.radians(deg)], atol=1e-6)
        q = b["qpos"][2, 3:7]
        self.assertAlmostEqual(2 * np.arctan2(q[3], q[0]), np.radians(deg), places=6)
        np.testing.assert_allclose(b["qpos"][:, :3], np.tile(b["qpos"][0, :3], (4, 1)))        # root translation fixed
        # Recomputed markers: left-right shoulder line rotated by +40 deg about +z, no reflection.
        names = list(b["marker_names"])
        lat = lambda m: m[names.index("left_shoulder")] - m[names.index("right_shoulder")]
        l0, l2 = lat(b["marker_positions"][0]), lat(b["marker_positions"][2])
        ang = np.arctan2(l0[0] * l2[1] - l0[1] * l2[0], l0[:2] @ l2[:2])
        self.assertAlmostEqual(ang, np.radians(deg), places=3)
        self.assertEqual(int(b["derived_dof_mask"].sum()), 3)
        self.assertFalse(np.any(b["derived_dof_mask"] & b["fitted_dof_mask"]))
        np.testing.assert_allclose(c["root_yaw_rad"], 0.0)
        self.assertEqual(rb["root_heading"]["supported_frames"], 4)

    def test_yaw_held_when_shoulders_unsupported(self):
        twin = make_twin(n=4, conf=0.95)
        h = np.radians(30) / 2
        twin["subjects"][0]["frames"][1]["root_orientation_xyzw"] = [0.0, -float(np.sin(h)), 0.0, float(np.cos(h))]
        for e in twin["subjects"][0]["frames"][2]["joint_rotations"]:
            if "shoulder" in e["joint"]:
                e["confidence"] = 0.0
        with tempfile.TemporaryDirectory() as td:
            m, rep = run_cli(twin, Path(td))
        y = m["root_yaw_rad"]
        self.assertAlmostEqual(y[1], np.radians(30), places=6)
        self.assertAlmostEqual(y[2], y[1])           # held, not invented
        self.assertFalse(bool(m["root_yaw_supported"][2]))

    def test_inverse_dynamics_uses_prescribed_acceleration(self):
        m, d0 = bm.load_model()
        adr = m.jnt_qposadr[m.joint("elbow_flexion_r").id]
        dof = m.jnt_dofadr[m.joint("elbow_flexion_r").id]
        ts = np.array([0.0, 0.1, 0.25, 0.45, 0.7])  # unequal steps
        base = np.tile(d0.qpos, (5, 1))
        lin, quad = base.copy(), base.copy()
        lin[:, adr] = 0.5 + 1.0 * ts
        quad[:, adr] = 0.5 + 0.5 * 4.0 * ts ** 2        # constant acceleration 4 rad/s^2
        mq = fb.mechanics(mujoco, m, quad, ts)
        ml = fb.mechanics(mujoco, m, lin, ts)
        # exact derivatives of polynomial motion recovered with unequal timestamps
        np.testing.assert_allclose(mq["qacc"][1:-1, dof], 4.0, atol=1e-8)
        np.testing.assert_allclose(ml["qvel"][:, dof], 1.0, atol=1e-8)
        np.testing.assert_allclose(mq["qvel"][2, dof], 4.0 * ts[2], atol=1e-8)
        # demand differs with imposed acceleration and matches a direct mj_inverse
        self.assertGreater(abs(mq["qfrc_inverse_dynamics_demand"][2, dof] -
                               ml["qfrc_inverse_dynamics_demand"][2, dof]), 1e-4)
        d = mujoco.MjData(m)
        m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        try:
            d.qpos[:], d.qvel[:] = quad[2], mq["qvel"][2]
            d.act[:] = 0
            d.ctrl[:] = 0
            mujoco.mj_forward(m, d)
            d.qacc[:] = mq["qacc"][2]
            mujoco.mj_inverse(m, d)
        finally:
            m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        np.testing.assert_allclose(mq["qfrc_inverse_dynamics_demand"][2], d.qfrc_inverse, atol=1e-9)

    # ---------------------------------------------------------------- image observations
    def _bundle(self, twin, td, pts_override=None):
        """Write a .twin zip whose poses.jsonl holds ideal projections of the twin's own FK."""
        import zipfile
        import skeleton_recording as rt
        names, parents = rt.validate_hierarchy(twin)
        off = np.asarray(twin["subjects"][0]["person"]["rest_offsets"], float)
        lines = []
        for f in twin["subjects"][0]["frames"]:
            f["focal_length_px"] = 1280.0
            local, _ = rt.frame_rotations(f, names)
            pos, _ = rt.forward_kinematics(parents, off, local, rt.quat_to_matrix(f["root_orientation_xyzw"]),
                                           np.asarray(f["root_position"], float))
            kps = []
            for nm in ("left_elbow", "left_wrist", "right_elbow", "right_wrist"):
                P = pos[names.index(nm)]
                kp = {"name": nm, "x": 640 + 1280 * P[0] / P[2], "y": 360 + 1280 * P[1] / P[2], "score": 0.95}
                if pts_override and (f["frame"], nm) in pts_override:
                    kp.update(pts_override[(f["frame"], nm)])
                kps.append(kp)
            lines.append(json.dumps({"frame": f["frame"], "width": 1280, "height": 720, "people": [
                {"person_index": 0, "twin": {"subject_segment": 0}, "observed_keypoints": kps}]}))
        path = Path(td) / "t.twin"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("twin.json", json.dumps(twin))
            z.writestr("poses.jsonl", "\n".join(lines))
        return path

    def _raised_twin(self, conf_after):
        twin = make_twin(n=4, conf=0.95)
        for k, f in enumerate(twin["subjects"][0]["frames"]):
            if k >= 1:
                set_rot(f, "left_shoulder", zrot_quat(-60))
                for e in f["joint_rotations"]:
                    if "elbow" in e["joint"] or "wrist" in e["joint"]:
                        e["confidence"] = conf_after
        return twin

    def test_image_observations_recover_arm_without_3d_confidence(self):
        adr = lambda n: self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ref_twin = self._raised_twin(0.95)                  # full 3D confidence = reference answer
            pr = self._bundle(copy.deepcopy(ref_twin), td)
            self.assertEqual(fb.main([str(pr), "--output", str(td / "ref"), "--max-nfev", "60"]), 0)
            blind = self._raised_twin(0.0)                      # native 3D confidence removed on frames 1-3
            pb = self._bundle(copy.deepcopy(blind), td)
            self.assertEqual(fb.main([str(pb), "--output", str(td / "img"), "--max-nfev", "60"]), 0)
            self.assertEqual(fb.main([str(pb), "--output", str(td / "noimg"), "--max-nfev", "60",
                                      "--no-image-observations"]), 0)
            ref = np.load(td / "ref" / "motion.npz")
            img = np.load(td / "img" / "motion.npz")
            noimg = np.load(td / "noimg" / "motion.npz")
            rep = json.loads((td / "img" / "fit_report.json").read_text())
            rep0 = json.loads((td / "noimg" / "fit_report.json").read_text())
            a = adr("shoulder_elv_l")
            truth = ref["qpos"][1:, a]
            self.assertGreater(abs(truth.mean() - ref["qpos"][0, a]), 0.3)     # the pose really changed
            np.testing.assert_allclose(img["qpos"][1:, a], truth, atol=0.15)    # recovered from pixels alone
            self.assertGreater(np.abs(noimg["qpos"][1:, a] - truth).max(), 0.3)  # without images: held, wrong
            ml = list(ref["marker_names"]).index("left_wrist")
            self.assertLess(np.linalg.norm(img["marker_positions"][1:, ml] - ref["marker_positions"][1:, ml],
                                           axis=1).max(), 0.06)
            self.assertTrue(img["support_2d"][1:].any() and not img["support_3d"][1:].any())
            self.assertLess(rep["fit"]["reprojection_px"]["median"], 12.0)  # ~1.4 cm at 1.7 m; arm proportions differ from model
            self.assertEqual(rep["image_observations"]["status"], "ok")
            self.assertEqual(rep0["image_observations"]["status"], "disabled by --no-image-observations")
            # No 3D support & no images => held/no-data semantics, not counted as observed.
            self.assertGreater(rep0["fit"]["frames_with_any_arm_held_no_data"], 0)
            self.assertEqual(rep["fit"]["frames_with_any_arm_held_no_data"], 0)
            # Native couplings / limits still respected.
            self.assertEqual(rep["checks"]["max_dependent_joint_limit_violation_rad"], 0.0)

    def test_image_observation_rejection_and_legacy_fallback(self):
        twin = make_twin(n=3, conf=0.95)
        override = {(0, "left_wrist"): {"score": 0.1},              # low confidence
                    (1, "right_wrist"): {"x": -80.0},               # outside the image
                    (2, "left_elbow"): {"score": 0.99}}
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            p = self._bundle(copy.deepcopy(twin), td, override)
            self.assertEqual(fb.main([str(p), "--output", str(td / "o"), "--max-nfev", "20"]), 0)
            io = json.loads((td / "o" / "fit_report.json").read_text())["image_observations"]
            self.assertEqual(io["rejected_observations"]["low_score"], 1)
            self.assertEqual(io["rejected_observations"]["out_of_image"], 1)
            self.assertEqual(io["rejected_observations"]["missing_keypoint"], 0)
            self.assertIn("NOT a camera calibration", io["camera"])
            legacy = td / "legacy.json"                              # plain twin.json: explicit fallback
            legacy.write_text(json.dumps(twin))
            self.assertEqual(fb.main([str(legacy), "--output", str(td / "l"), "--max-nfev", "20"]), 0)
            st = json.loads((td / "l" / "fit_report.json").read_text())["image_observations"]["status"]
            self.assertTrue(st.startswith("unavailable"), st)

    def test_mechanics_outputs_finite(self):
        twin = make_twin(n=4, conf=0.95)
        with tempfile.TemporaryDirectory() as td:
            run_cli(twin, Path(td), extra=("--mechanics",))
            mech = np.load(Path(td) / "mechanics.npz")
            for k in ("actuator_length_m", "actuator_velocity_mps", "actuator_moment_dLdq",
                      "qfrc_passive_muscle_act0", "qfrc_inverse_dynamics_demand"):
                self.assertTrue(np.all(np.isfinite(mech[k])), k)
            self.assertEqual(mech["actuator_length_m"].shape[1], 416)


if __name__ == "__main__":
    unittest.main()
