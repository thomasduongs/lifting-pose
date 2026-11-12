#!/usr/bin/env python3
"""Tests for scripts/biomech_experiment.py on the real engine (small grids).

Run: .venv-biomech/bin/python -m unittest scripts/test_biomech_experiment.py
"""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco  # noqa: F401
    import myo_sim  # noqa: F401
except ImportError as exc:
    raise unittest.SkipTest(
        f"biomechanics stack unavailable ({exc}); run with .venv-biomech/bin/python") from exc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import biomech_experiment as bx  # noqa: E402
import biomech_model as bm  # noqa: E402

K = 17400.0  # N/m exploratory test stiffness


class ExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, _ = bm.load_model()
        cls.mk, _ = bm.load_model(ligament_stiffness=K)
        cls.bic = bx.resolve_muscle(cls.m, "BIClong")
        cls.sw = bx.angle_sweep(cls.m, cls.bic, "elbow_flexion_r", n_angles=9)

    # ------------------------------------------------------------ discovery
    def test_muscle_discovery(self):
        m = self.m
        self.assertEqual(m.actuator(bx.resolve_muscle(m, "biclong")).name, "BIClong")
        self.assertEqual(m.actuator(bx.resolve_muscle(m, "BIClong", "elbow_flexion_l")).name,
                         "BIClong_l")
        self.assertEqual(m.actuator(bx.resolve_muscle(m, "glmax1", "hip_flexion_l")).name, "glmax1_l")
        with self.assertRaises(bx.ExperimentError) as cm:
            bx.resolve_muscle(m, "BICEPS_LONG")
        self.assertIn("close source names", str(cm.exception))

    def test_joint_resolution(self):
        self.assertEqual(bx.resolve_joint(self.m, self.bic), "elbow_flexion_r")
        with self.assertRaises(bx.ExperimentError):
            bx.resolve_joint(self.m, self.bic, "shoulder1_r2_r")  # dependent (coupled)
        with self.assertRaises(bx.ExperimentError):
            bx.resolve_joint(self.m, self.bic, "not_a_joint")

    # ------------------------------------------------------------ sweep
    def test_sweep_finite_and_si(self):
        sw = self.sw
        for k in ("tension_N", "mtu_length_m", "moment_arm_total_m", "torque_total_Nm",
                  "normalized_length", "passive_all_muscles_torque_Nm"):
            self.assertTrue(np.isfinite(sw[k]).all(), k)
        self.assertTrue((sw["tension_N"] >= 0).all())
        lo, hi = self.m.actuator_lengthrange[self.bic]
        self.assertTrue(np.all((sw["mtu_length_m"] > lo - 1e-3) & (sw["mtu_length_m"] < hi + 1e-3)))

    def test_force_increases_with_activation(self):
        t = self.sw["tension_N"]
        self.assertTrue(np.all(np.diff(t, axis=1) > 0))
        np.testing.assert_allclose(self.sw["passive_tension_N"], t[:, 0])
        F0 = self.m.actuator_gainprm[self.bic, 2]
        self.assertLess(t.max(), F0 * (1 + self.m.actuator_gainprm[self.bic, 7]) + 1e-6)

    def test_moment_arm_matches_finite_difference(self):
        sw = self.sw
        np.testing.assert_allclose(sw["moment_arm_partial_m"], sw["fd_moment_arm_partial_m"], atol=1e-7)
        np.testing.assert_allclose(sw["moment_arm_total_m"], sw["fd_moment_arm_total_m"], atol=1e-7)
        np.testing.assert_allclose(sw["active_torque_partial_Nm"], sw["engine_active_qfrc_Nm"],
                                   atol=1e-9)

    def test_flexor_and_extensor_signs(self):
        mid = slice(2, 8)
        self.assertTrue(np.all(self.sw["moment_arm_total_m"][mid] > 0.01))  # biceps flexor
        self.assertTrue(np.all(self.sw["torque_total_Nm"][mid, -1] > 0))
        tri = bx.resolve_muscle(self.m, "TRIlong")
        sw = bx.angle_sweep(self.m, tri, "elbow_flexion_r", n_angles=5, activations=(0, 1),
                            fd_check=False)
        self.assertTrue(np.all(sw["moment_arm_total_m"] < 0))  # extensor
        # moment arm varies with angle (not a constant lever)
        self.assertGreater(np.ptp(self.sw["moment_arm_total_m"]), 0.01)

    def test_coupled_joint_total_differs_from_partial(self):
        delt = bx.resolve_muscle(self.m, "DELT2")
        sw = bx.angle_sweep(self.m, delt, "shoulder_elv_r", n_angles=5, activations=(0, 1))
        self.assertGreater(np.abs(sw["moment_arm_total_m"] - sw["moment_arm_partial_m"]).max(), 1e-4)
        np.testing.assert_allclose(sw["moment_arm_total_m"], sw["fd_moment_arm_total_m"], atol=1e-7)
        np.testing.assert_allclose(sw["moment_arm_partial_m"], sw["fd_moment_arm_partial_m"], atol=1e-7)
        self.assertLess(np.nanmax(sw["coupling_residual_max"]), 1e-12)

    # ------------------------------------------------------------ activation step
    def test_activation_step_lag(self):
        ang = 1.5
        st = bx.activation_step(self.m, self.bic, "elbow_flexion_r", ang, on_s=0.1, off_s=0.1)
        n_on = int(round(0.1 / self.m.opt.timestep))
        a = st["activation"]
        self.assertEqual(a[0], 0.0)
        self.assertTrue(np.all(np.diff(a[:n_on]) >= 0))
        self.assertTrue(np.all(np.diff(a[n_on + 1:]) <= 0))
        self.assertLess(a[1], 0.5)  # lag: not instantaneous
        self.assertGreater(a[n_on - 1], 0.99)
        met = st["metrics"]
        self.assertGreater(met["t63_rise_s"], 0)
        self.assertGreater(met["t63_fall_s"], met["t63_rise_s"])  # deactivation slower (source taus)
        self.assertLess(met["max_activation_vs_reference_ode"], 1e-9)
        # clamped-step tension equals a static isometric evaluation at the same activation/angle
        d = bx._pose(self.m, bx.standing(self.m).qpos, {"elbow_flexion_r": ang})
        d.act[self.m.actuator_actadr[self.bic]] = met["activation_at_step_end"]
        mujoco.mj_forward(self.m, d)
        self.assertAlmostEqual(met["steady_tension_N"], -d.actuator_force[self.bic], delta=1e-6)

    # ------------------------------------------------------------ ligaments
    def test_ligament_sweep_tension_only(self):
        lig = bx.ligament_sweep(self.mk, "shoulder_rot_r", n_angles=15)
        L, F, slack = lig["length_m"], lig["tension_N"], np.array(lig["slack_length_m"])
        below = L <= slack
        self.assertTrue(np.all(F[below] == 0))
        np.testing.assert_allclose(F[~below], K * (L - slack)[~below], rtol=1e-9)
        self.assertTrue(below.any() and (~below).any())
        np.testing.assert_allclose(lig["generalized_force"], lig["neg_dE_dq"], atol=1e-5)
        off = bx.ligament_sweep(self.m, "shoulder_rot_r", n_angles=5)
        self.assertTrue(np.all(off["tension_N"] == 0))

    # ------------------------------------------------------------ CLI artifact
    def test_run_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc = bx.main(["--output", tmp, "--muscle", "biclong", "--joint", "elbow_flexion_r",
                          "--n-angles", "5", "--step-on", "0.05", "--step-off", "0.05",
                          "--ligament-stiffness", str(K)])
            self.assertEqual(rc, 0)
            out = Path(tmp)
            diag = json.loads((out / "diagnostics.json").read_text())
            self.assertEqual(diag["sweep"]["muscle"], "BIClong")
            self.assertLess(diag["sweep"]["fd_check_max_abs_error_m"]["total"], 1e-7)
            self.assertEqual(diag["forearm_rotation_comparison"]["inferred_supination_direction"],
                             "-pro_sup")
            with open(out / "sweep.csv") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 5 * 4)
            self.assertIn("moment_arm_total_m", rows[0])
            with np.load(out / "experiment.npz") as z:
                self.assertEqual(z["sweep_tension_N"].shape, (5, 4))
            self.assertEqual(bx.main(["--output", tmp, "--muscle", "nope"]), 2)


if __name__ == "__main__":
    unittest.main()
