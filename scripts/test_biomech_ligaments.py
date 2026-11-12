"""Tests for biomech_ligaments. Run with .venv-biomech/bin/python (skips on global Python)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import numpy as np
    import mujoco
    import biomech_model as bm
    import biomech_ligaments as lig
    import fit_biomech as fb
    from test_reconstruct_twin import make_twin
    HAVE = True
except ImportError:
    HAVE = False

SUPPORTED = ("s_glenohum_ligament", "s_glenohum_ligament_l", "coracohum_ligament", "coracohum_ligament_l")
UNSUPPORTED = ("m_glenohum_ligament", "m_glenohum_ligament_l", "i_glenohum_ligament", "i_glenohum_ligament_l")


@unittest.skipUnless(HAVE, "needs .venv-biomech/bin/python (mujoco, myo_sim)")
class LigamentTests(unittest.TestCase):
    def fresh(self, **kw):
        m, d = bm.load_model()
        meta = lig.apply_profile(m, d, **kw)
        return m, d, meta

    def test_conversion_and_mapping(self):
        m, d, meta = self.fresh()
        for n in SUPPORTED:
            i = m.tendon(n).id
            exp = 17400.0 if n.startswith("s_") else 36700.0
            self.assertAlmostEqual(m.tendon_stiffness[i], exp)
            self.assertEqual(m.tendon_damping[i], 0.0)
            self.assertEqual(m.tendon_lengthspring[i][0], 0.0)
            self.assertAlmostEqual(m.tendon_lengthspring[i][1], meta["enabled_routes"][n]["source_slack_m"])
        for n in UNSUPPORTED:
            self.assertEqual(m.tendon_stiffness[m.tendon(n).id], 0.0)
        self.assertEqual(sorted(meta["enabled_routes"]), sorted(SUPPORTED))
        self.assertEqual(sorted(meta["unsupported_routes_disabled"]), sorted(UNSUPPORTED))
        self.assertEqual(meta["source"]["doi"], "10.1016/s1058-2746(96)80050-4")
        json.dumps(meta)
        # muscles untouched
        self.assertEqual(m.nu, 416)
        cfg = bm.ligament_config(m)
        self.assertTrue(cfg["enabled"])
        self.assertFalse(cfg["stiffness_is_uniform"])
        self.assertIsNone(cfg["stiffness_N_per_m"])
        self.assertEqual(cfg["routes"][SUPPORTED[0]]["stiffness_N_per_m"], 17400.0)
        self.assertEqual(cfg["routes"][SUPPORTED[0]]["source_springlength_m"],
                         meta["enabled_routes"][SUPPORTED[0]]["source_slack_m"])

    def test_slack_scale_and_variants(self):
        m, d, meta = self.fresh(slack_scale=1.1, variant="low")
        n = "s_glenohum_ligament"
        self.assertAlmostEqual(m.tendon_stiffness[m.tendon(n).id], 15900.0)
        self.assertAlmostEqual(m.tendon_lengthspring[m.tendon(n).id][1], 0.0448 * 1.1)
        with self.assertRaises(lig.LigamentError):
            lig.apply_profile(m, d, "nope")

    def test_tension_only_slack_and_taut(self):
        m, d, meta = self.fresh()
        taut_any = False
        bm_pose = lambda plane, elv, rot: lig._set_pose(m, d, elv, rot, plane)
        bm_pose(0.0, 0.0, 0.0)  # standing: below slack
        for s in bm.ligament_state(m, d):
            if s["name"] in SUPPORTED:
                self.assertLess(s["length_m"], s["slack_length_m"])
                self.assertEqual(s["tension_N"], 0.0)
        bm_pose(2.27, 2.24, -1.57)
        for s in bm.ligament_state(m, d):
            if s["name"] in SUPPORTED and s["length_m"] > s["slack_length_m"]:
                taut_any = True
                self.assertAlmostEqual(s["tension_N"], s["stiffness_N_per_m"] * (s["length_m"] - s["slack_length_m"]))
        self.assertTrue(taut_any)

    def test_passive_force_consistency(self):
        off, d_off = bm.load_model()
        on, d_on, _ = self.fresh()
        for mm, dd in ((off, d_off), (on, d_on)):
            lig._set_pose(mm, dd, 2.24, -1.57, 2.27)
        diff = d_on.qfrc_passive - d_off.qfrc_passive
        pred = bm.ligament_generalized_force(on, d_on)
        self.assertGreater(np.abs(pred).max(), 0.0)
        np.testing.assert_allclose(diff, pred, atol=1e-8)

    def test_bench_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            r = lig.bench(td)
            self.assertTrue(r["jacobian_consistency"]["passed"])
            self.assertTrue((Path(td) / "ligament_sensitivity.csv").is_file())
            self.assertGreater(r["summary"]["taut_rows"], 0)
            self.assertEqual(len(r["unsupported_routes_disabled"]), 4)

    def test_fitter_profile_flag_and_incompatible_options(self):
        twin = make_twin(n=3, conf=0.95)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.json"
            p.write_text(json.dumps(twin))
            rc = fb.main([str(p), "--output", str(Path(td) / "bad"), "--ligament-profile", "boardman1996",
                          "--ligament-stiffness", "100"])
            self.assertEqual(rc, 1)
            rc = fb.main([str(p), "--output", str(Path(td) / "ok"), "--ligament-profile", "boardman1996",
                          "--max-nfev", "10", "--mechanics"])
            self.assertEqual(rc, 0)
            rep = json.loads((Path(td) / "ok" / "fit_report.json").read_text())
            self.assertEqual(rep["model"]["ligament_profile"]["profile"], "boardman1996")
            self.assertEqual(len(rep["model"]["ligament_native_inventory_at_standing"]), 8)
            # saved model.mjb carries the profile
            mdl = mujoco.MjModel.from_binary_path(str(Path(td) / "ok" / "model.mjb"))
            self.assertAlmostEqual(mdl.tendon_stiffness[mdl.tendon("coracohum_ligament").id], 36700.0)
            mech = np.load(Path(td) / "ok" / "mechanics.npz")
            self.assertEqual(mech["ligament_tension_N"].shape, (3, 8))


if __name__ == "__main__":
    unittest.main()
