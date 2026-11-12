import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_biomech_lite as ex  # noqa: E402
import mujoco  # noqa: E402


class ExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir1 = tempfile.TemporaryDirectory()
        cls.dir2 = tempfile.TemporaryDirectory()
        ex.export_all(cls.dir1.name)
        ex.export_all(cls.dir2.name)

        def load(n):
            return json.loads((Path(cls.dir1.name) / n).read_text())

        cls.model, cls.motion, cls.ref = load("model.json"), load("motion.json"), load("reference.json")

    def test_schema(self):
        m = self.model
        self.assertEqual((m["nq"], m["nv"], len(m["dynamic_joints"])), (129, 128, 8))
        self.assertEqual(len(m["muscles"]), 416)
        self.assertEqual(len(m["tendons"]), 424)
        self.assertEqual(len(m["qpos0"]), 129)
        self.assertEqual(len(self.motion["qpos"]), 222)
        self.assertEqual(len(self.motion["timestamps_seconds"]), 222)
        for k in ("mjb_sha256", "recording_sha256", "versions", "mujoco_source"):
            self.assertIn(k, m["source"])
        self.assertTrue((Path(self.dir1.name) / "NOTICE.md").is_file())
        self.assertTrue(list((Path(self.dir1.name) / "provenance").glob("*LICENSE*")))

    def test_deterministic(self):
        for n in ("model.json", "motion.json", "reference.json"):
            a = (Path(self.dir1.name) / n).read_text()
            b = (Path(self.dir2.name) / n).read_text()
            self.assertEqual(a, b, n)

    def test_cases(self):
        cases = self.ref["cases"]
        names = [c["name"] for c in cases]
        for key in ("standing", "velocity_shortening", "activation_full", "recorded_frame_221"):
            self.assertTrue(any(key in n for n in names), key)
        for c in cases:
            M = np.array(c["reduced_mass"])
            self.assertEqual(M.shape, (8, 8))
            self.assertLess(np.abs(M - M.T).max(), 1e-9)
            self.assertGreater(np.linalg.eigvalsh(M).min(), 0)
            self.assertTrue(min(c["muscle_tensions"]) >= -1e-9)
        for c in cases:
            for k in ("reduced_velocity", "reduced_bias", "reduced_muscle_torque", "reduced_passive_torque"):
                self.assertEqual(len(c[k]), 8, k)
        st = cases[0]
        np.testing.assert_allclose(st["reduced_bias"], st["reduced_gravity"], atol=1e-9)
        self.assertTrue(any(c["name"].startswith("ligament_loaded") and np.abs(c["reduced_passive_torque"]).max() > 1
                            for c in cases))
        v = [c for c in cases if c["name"].startswith("velocity_")]
        self.assertTrue(all(np.abs(np.array(c["reduced_bias"]) - c["reduced_gravity"]).max() > 1e-6 for c in v))
        self.assertTrue(all(np.abs(c["tendon_velocities"]).max() > 0 for c in v))
        self.assertTrue(any(np.abs(c["activation_derivative"]).max() > 0 for c in cases))

    def test_reduction_matrix_matches_finite_difference(self):
        m = ex.load_native()
        q = np.array(self.ref["cases"][0]["qpos"])
        C = ex.reduction_matrix(m, q)
        h = 1e-6
        for k, j in enumerate(ex.dynamic_joint_ids(m)):
            qa = int(m.jnt_qposadr[j])
            qp, qm = q.copy(), q.copy()
            qp[qa] += h
            qm[qa] -= h
            ex.bm.project_couplings_qpos(m, qp)
            ex.bm.project_couplings_qpos(m, qm)
            dq = np.zeros(m.nv)
            mujoco.mj_differentiatePos(m, dq, 2 * h, qm, qp)
            np.testing.assert_allclose(dq, C[:, k], atol=1e-6)

    def test_unsupported_ball_fails(self):
        m = ex.load_native()
        m.jnt_type[5] = int(mujoco.mjtJoint.mjJNT_BALL)
        with self.assertRaises(ex.ExportError):
            ex.check_supported(m)


if __name__ == "__main__":
    unittest.main()
