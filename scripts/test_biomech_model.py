#!/usr/bin/env python3
"""Tests for scripts/biomech_model.py against the real MuJoCo engine and source model.

Run: .venv-biomech/bin/python -m unittest scripts/test_biomech_model.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
    import myo_sim  # noqa: F401
except ImportError as exc:  # keep the Python 3.9 FEM test environment usable
    raise unittest.SkipTest(
        f"biomechanics stack unavailable ({exc}); run with .venv-biomech/bin/python") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biomech_model as bm  # noqa: E402

# A pose (right shoulder elevation, plane of elevation, axial rotation) where the source
# ligament routes are longer than their upstream springlength (found by ROM sweep).
TAUT_POSE = {"shoulder_elv_r": 1.96, "elv_angle_r": 2.27, "shoulder_rot_r": -1.57}


def _set(model, data, values):
    for name, v in values.items():
        data.qpos[model.joint(name).qposadr[0]] = v


def _eq_rows(data):
    return data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.d = bm.load_model()

    def test_compiles_finite_with_source_counts(self):
        m, d = self.m, self.d
        self.assertEqual((m.nbody, m.njnt, m.nq, m.nv), (104, 123, 129, 128))
        self.assertEqual(m.nu, 416)
        self.assertTrue(np.all(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE))
        self.assertEqual(m.ntendon, 424)
        self.assertTrue(np.isfinite(d.qpos).all() and np.isfinite(d.qacc).all())
        self.assertTrue((m.body_mass >= 0).all())
        self.assertTrue(50 < m.body_mass.sum() < 120)  # generic adult reference mass, kg
        for b in ("humerus_r", "ulna_r", "radius_r", "femur_l", "torso", "head"):
            self.assertGreater(m.body(b).mass[0], 0)

    def test_standing_keyframe_and_joint_limits(self):
        m, d = self.m, self.d
        kid = m.key("standing").id
        lim = m.jnt_limited.astype(bool)
        q = d.qpos[m.jnt_qposadr[lim]]
        r = m.jnt_range[lim]
        self.assertTrue(np.all(q >= r[:, 0] - 1e-6) and np.all(q <= r[:, 1] + 1e-6))
        # enforcing couplings at the source keyframe changes nothing material
        self.assertLess(np.abs(d.qpos - m.key_qpos[kid]).max(), 1e-6)

    def test_couplings_match_engine_and_upstream(self):
        m = self.m
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("standing").id)
        _set(m, d, {"shoulder_elv_r": 1.3, "elv_angle_r": 0.6, "shoulder_elv_l": 0.7,
                    "elv_angle_l": -0.4, "knee_angle_r": 1.1, "knee_angle_l": 0.4,
                    "flex_extension": 0.3, "lat_bending": -0.1, "axial_rotation": 0.15})
        q_master = d.qpos.copy()
        rng = np.random.default_rng(0)
        d.qvel[:] = rng.normal(size=m.nv)
        self.assertEqual(bm.enforce_couplings(m, d), [])
        mujoco.mj_forward(m, d)
        eq = _eq_rows(d)
        self.assertEqual(eq.sum(), 51)
        # engine's own equality constraint violation (position and velocity)
        self.assertLess(np.abs(d.efc_pos[eq]).max(), 1e-9)
        self.assertLess(np.abs(d.efc_vel[eq]).max(), 1e-9)
        self.assertLess(np.abs(bm.coupling_residuals(m, d.qpos)).max(), 1e-12)
        # master joints untouched
        c = bm._couplings(m)
        for mq in c.mas_q:
            self.assertEqual(d.qpos[mq], q_master[mq])
        # independent check against the upstream MyoSim implementation
        from myo_sim.physics.joint_coupling import JointCouplings, project_joint_couplings
        q2 = q_master.copy()
        project_joint_couplings(JointCouplings.from_model(m), q2)
        np.testing.assert_allclose(q2, d.qpos, atol=1e-12)

    def test_couplings_actually_move_dependents(self):
        m = self.m
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("standing").id)
        before = d.qpos.copy()
        _set(m, d, {"shoulder_elv_r": 2.0})
        bm.enforce_couplings(m, d)
        sc = m.joint("sternoclavicular_r2_r").qposadr[0]
        self.assertGreater(abs(d.qpos[sc] - before[sc]), 1e-3)  # scapulohumeral rhythm

    def test_independent_dofs_exclude_dependents(self):
        ind = set(bm.independent_dofs(self.m))
        for n in ("shoulder_elv_r", "elv_angle_r", "shoulder_rot_r", "elbow_flexion_r",
                  "pro_sup_r", "knee_angle_r"):
            self.assertIn(n, ind)
        for n in ("shoulder1_r2_r", "unrothum_r1_r", "knee_angle_translation1_r", "Abs_t1"):
            self.assertNotIn(n, ind)

    def test_muscle_force_varies_with_activation(self):
        m = self.m
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("standing").id)
        names = [m.actuator(a).name for a in range(m.nu)]
        bic = [n for n in names if n.lower().startswith("biclong") and not n.endswith("_l")]
        self.assertEqual(len(bic), 1)
        a = m.actuator(bic[0]).id
        _set(m, d, {"elbow_flexion_r": 1.0})
        bm.enforce_couplings(m, d)
        forces = []
        for act in (0.0, 0.25, 0.5, 1.0):
            d.act[:] = 0.0
            d.act[m.actuator_actadr[a]] = act
            mujoco.mj_forward(m, d)
            forces.append(d.actuator_force[a])
        f = np.array(forces)
        self.assertTrue(np.all(f <= 0))  # muscles pull (MuJoCo sign convention)
        self.assertTrue(np.all(np.diff(f) < 0))  # more activation, more tension
        self.assertGreater(-f[-1], 0.1 * m.actuator_gainprm[a, 2])
        # force produces a joint moment through the tendon Jacobian (elbow flexor)
        dof = m.joint("elbow_flexion_r").dofadr[0]
        self.assertGreater(d.qfrc_actuator[dof], 0)

    def test_tendon_jacobian_matches_finite_difference(self):
        m = self.m
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("standing").id)
        _set(m, d, {"elbow_flexion_r": 0.8})
        bm.enforce_couplings(m, d)
        mujoco.mj_forward(m, d)
        t = m.actuator_trnid[m.actuator("BIClong").id, 0]
        J = bm.tendon_jacobian(m, d, int(t))
        dof = m.joint("elbow_flexion_r").dofadr[0]
        qa = m.joint("elbow_flexion_r").qposadr[0]
        h, L = 1e-6, []
        for s in (-1, 1):
            d2 = mujoco.MjData(m)
            d2.qpos[:] = d.qpos
            d2.qpos[qa] += s * h
            mujoco.mj_fwdPosition(m, d2)
            L.append(d2.ten_length[t])
        self.assertAlmostEqual(J[dof], (L[1] - L[0]) / (2 * h), places=6)
        self.assertLess(J[dof], 0)  # biceps shortens with elbow flexion: positive moment arm

    def test_markers_named_finite_and_anatomically_ordered(self):
        mk = bm.marker_positions(self.m, self.d)
        for n in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
                  "right_wrist", "thorax", "head"):
            self.assertIn(n, mk)
            self.assertEqual(mk[n].shape, (3,))
            self.assertTrue(np.isfinite(mk[n]).all())
        # model frame: subject's left is +x, up is +z; no mirrored handedness
        self.assertGreater(mk["left_shoulder"][0], mk["right_shoulder"][0])
        self.assertGreater(mk["left_wrist"][0], mk["right_wrist"][0])
        for s in ("left", "right"):
            self.assertGreater(mk[f"{s}_shoulder"][2], mk[f"{s}_elbow"][2])
            self.assertGreater(mk[f"{s}_elbow"][2], mk[f"{s}_wrist"][2])
            ua = np.linalg.norm(mk[f"{s}_shoulder"] - mk[f"{s}_elbow"])
            self.assertTrue(0.2 < ua < 0.4, ua)
        self.assertGreater(mk["head"][2], mk["thorax"][2])
        # wrist center is the native wrist joint anchor; elbow is near the native flexion axis
        np.testing.assert_allclose(mk["right_wrist"], self.d.xanchor[self.m.joint("flexion_r").id])
        self.assertLess(np.linalg.norm(
            mk["right_elbow"] - self.d.xanchor[self.m.joint("elbow_flexion_r").id]), 0.02)

    def test_markers_follow_motion(self):
        m = self.m
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("standing").id)
        mujoco.mj_kinematics(m, d)
        w0 = bm.marker_positions(m, d)["right_wrist"]
        _set(m, d, {"elbow_flexion_r": 1.5})
        mujoco.mj_kinematics(m, d)
        w1 = bm.marker_positions(m, d)["right_wrist"]
        self.assertGreater(w1[2] - w0[2], 0.1)

    def test_inventory_json_safe(self):
        inv = bm.inventory(self.m, self.d)
        json.dumps(inv)
        self.assertEqual(inv["counts"]["muscle_actuators"], 416)
        self.assertEqual(inv["counts"]["joint_couplings"] + inv["counts"]["locked_joints"], 51)
        self.assertEqual(inv["unsupported_equalities"], [])
        self.assertFalse(inv["ligaments"]["config"]["enabled"])
        self.assertEqual(len(inv["ligaments"]["routes"]), 8)
        srcs = {mu["source"] for mu in inv["muscles"]}
        self.assertTrue(any("MoBL" in s for s in srcs) and any("Rajagopal" in s for s in srcs))
        self.assertNotIn("unknown", srcs)
        for mu in inv["muscles"]:
            self.assertGreater(mu["F0_N"], 0)
            self.assertGreater(mu["optimal_length_L0_m"], 0)


class LigamentTest(unittest.TestCase):
    K = 2000.0  # N/m, arbitrary exploratory test value

    @classmethod
    def setUpClass(cls):
        cls.off, _ = bm.load_model()
        cls.on, _ = bm.load_model(ligament_stiffness=cls.K)
        cls.short, _ = bm.load_model(ligament_stiffness=cls.K, ligament_slack_scale=0.5)
        cls.stand, _ = bm.load_model(ligament_stiffness=cls.K, ligament_slack_reference="standing")

    def _pose(self, model, pose=None):
        d = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, d, model.key("standing").id)
        if pose:
            _set(model, d, pose)
        bm.enforce_couplings(model, d)
        mujoco.mj_forward(model, d)
        return d

    def test_source_has_no_ligament_mechanics(self):
        for n in bm.LIGAMENT_NAMES:
            i = self.off.tendon(n).id
            self.assertEqual(self.off.tendon_stiffness[i], 0)
            self.assertEqual(self.off.tendon_damping[i], 0)
        d = self._pose(self.off, TAUT_POSE)
        self.assertTrue(all(s["tension_N"] == 0 for s in bm.ligament_state(self.off, d)))

    def test_only_ligament_routes_changed_and_muscles_preserved(self):
        lig = [self.on.tendon(n).id for n in bm.LIGAMENT_NAMES]
        other = np.setdiff1d(np.arange(self.on.ntendon), lig)
        np.testing.assert_array_equal(self.on.tendon_stiffness[other], self.off.tendon_stiffness[other])
        np.testing.assert_array_equal(self.on.tendon_lengthspring[other],
                                      self.off.tendon_lengthspring[other])
        for f in ("actuator_gainprm", "actuator_biasprm", "actuator_dynprm", "actuator_lengthrange"):
            np.testing.assert_array_equal(getattr(self.on, f), getattr(self.off, f))
        self.assertTrue(np.all(self.on.tendon_lengthspring[lig, 0] == 0))

    def test_below_slack_is_exactly_zero(self):
        # source slack lengths exceed standing route lengths -> enabled springs are silent
        d_on, d_off = self._pose(self.on), self._pose(self.off)
        for s in bm.ligament_state(self.on, d_on):
            self.assertLess(s["length_m"], s["slack_length_m"])
            self.assertEqual(s["tension_N"], 0.0)
        np.testing.assert_allclose(d_on.qfrc_spring, d_off.qfrc_spring, atol=1e-12)
        np.testing.assert_allclose(d_on.qfrc_passive, d_off.qfrc_passive, atol=1e-12)

    def test_taut_produces_tension_matching_engine(self):
        for model in (self.on, self.short):
            pose = TAUT_POSE if model is self.on else None
            d_on, d_off = self._pose(model, pose), self._pose(self.off, pose)
            st = bm.ligament_state(model, d_on)
            taut = [s for s in st if s["taut"]]
            self.assertTrue(taut)
            for s in taut:
                self.assertAlmostEqual(s["tension_N"], self.K * (s["length_m"] - s["slack_length_m"]))
                self.assertGreater(s["energy_J"], 0)
            diff = d_on.qfrc_spring - d_off.qfrc_spring
            np.testing.assert_allclose(diff, bm.ligament_generalized_force(model, d_on), atol=1e-9)
            self.assertGreater(np.abs(diff).max(), 0)

    def test_standing_slack_reference_is_zero_at_standing(self):
        d = self._pose(self.stand)
        for s in bm.ligament_state(self.stand, d):
            self.assertAlmostEqual(s["length_m"], s["slack_length_m"], places=9)
            self.assertEqual(s["tension_N"], 0.0)
        d2 = self._pose(self.stand, TAUT_POSE)
        self.assertTrue(any(s["taut"] for s in bm.ligament_state(self.stand, d2)[:4]))

    def test_invalid_options_rejected(self):
        for kw in ({"ligament_stiffness": -1}, {"ligament_stiffness": float("nan")},
                   {"ligament_slack_scale": 0}, {"ligament_slack_reference": "measured"}):
            with self.assertRaises(bm.BiomechError):
                bm.load_model(**kw)


class ExportTest(unittest.TestCase):
    def test_export_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = bm.export_model(tmp, ligament_stiffness=1500.0, ligament_slack_scale=0.9)
            self.assertEqual(r["checks"]["mjb_reload_mismatches"], [])
            self.assertEqual(r["checks"].get("xml_reload_mismatches"), [])
            out = Path(tmp)
            for f in ("model.mjb", "model.xml", "anatomy.json", "provenance.json"):
                self.assertTrue((out / f).is_file(), f)
            prov = json.loads((out / "provenance.json").read_text())
            self.assertEqual(prov["versions"]["mujoco"], bm.PINNED["mujoco"])
            self.assertEqual(prov["versions"]["myo-sim"], bm.PINNED["myo-sim"])
            self.assertTrue(any("LICENSE" in k for k in prov["files"]))
            self.assertTrue(any("README" in k for k in prov["files"]))
            m = mujoco.MjModel.from_binary_path(str(out / "model.mjb"))
            lig = m.tendon("s_glenohum_ligament").id
            self.assertEqual(m.tendon_stiffness[lig], 1500.0)
            self.assertAlmostEqual(m.tendon_lengthspring[lig, 1], 0.9 * 0.0448)
            inv = json.loads((out / "anatomy.json").read_text())
            self.assertTrue(inv["ligaments"]["config"]["enabled"])


if __name__ == "__main__":
    unittest.main()
