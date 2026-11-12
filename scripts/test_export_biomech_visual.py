import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_biomech_visual as ev  # noqa: E402
import export_biomech_lite as ex  # noqa: E402
import mujoco  # noqa: E402


class VisualExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "visual.json"
        cls.summary = ev.export_visual(cls.out)
        cls.v = json.loads(cls.out.read_text())
        cls.m = ex.load_native()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_counts_and_schema(self):
        v = self.v
        self.assertEqual(v["schema_version"], 1)
        self.assertEqual(len(v["meshes"]), 104)
        self.assertEqual(len(v["geoms"]), 102)
        self.assertEqual(self.summary["meshes"], 104)
        self.assertEqual(self.summary["geoms"], 102)
        self.assertEqual(self.summary["reference_samples"], 3 * 102)
        self.assertEqual(self.summary["source_model_sha256"], v["source_model_sha256"])

    def test_source_hash_matches_model_and_mjb(self):
        mjb_hash = ex.sha256(ex.TWIN / "model.mjb")
        self.assertEqual(self.v["source_model_sha256"], mjb_hash)
        model_json = ex.TWIN.parent.parent / "assets" / "biomech-lite" / "model.json"
        if model_json.is_file():
            self.assertEqual(json.loads(model_json.read_text())["source"]["mjb_sha256"], mjb_hash)

    def test_mesh_arrays_are_native_and_valid(self):
        m, meshes = self.m, self.v["meshes"]
        for mid, mesh in enumerate(meshes):
            vn = int(m.mesh_vertnum[mid])
            self.assertEqual(len(mesh["vertices"]), vn)
            faces = np.asarray(mesh["faces"], dtype=np.int64)
            self.assertGreater(len(faces), 0)
            self.assertTrue(faces.min() >= 0 and faces.max() < vn)
            verts = np.asarray(mesh["vertices"], dtype=np.float64)
            self.assertTrue(np.isfinite(verts).all())
            np.testing.assert_array_equal(
                verts, np.asarray(m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + vn]))

    def test_geoms_native_ids_and_unobserved_set(self):
        m = self.m
        ids = [g["geom"] for g in self.v["geoms"]]
        self.assertEqual(ids, ev.mesh_geom_ids(m))
        for g in self.v["geoms"]:
            self.assertEqual(g["mesh"], int(m.geom_dataid[g["geom"]]))
            self.assertEqual(g["body"], int(m.geom_bodyid[g["geom"]]))
            self.assertTrue(all(math.isfinite(x) for x in g["rgba"]))
        lower = [g for g in self.v["geoms"] if g["unobserved"]]
        self.assertEqual(len(lower), 16)  # 16 geoms on the 13 LOWER_BODY bodies
        for g in lower:
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, g["body"])
            self.assertIn(name, ev.rb.LOWER_BODY)

    def test_reference_samples_match_native_kinematics(self):
        m = self.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.v["reference"]["qpos"]
        mujoco.mj_forward(m, d)
        self.assertEqual(len(self.v["reference"]["qpos"]), m.nq)
        for s in self.v["reference"]["samples"]:
            g, vid = s["geom"], s["vertex"]
            mid = int(m.geom_dataid[g])
            vn = int(m.mesh_vertnum[mid])
            self.assertIn(vid, {0, vn // 2, vn - 1})
            local = np.asarray(m.mesh_vert[m.mesh_vertadr[mid] + vid], dtype=np.float64)
            world = d.geom_xpos[g] + d.geom_xmat[g].reshape(3, 3) @ local
            np.testing.assert_allclose(s["world_position"], world, rtol=0, atol=1e-9)

    def test_deterministic_and_atomic(self):
        out2 = Path(self.tmp.name) / "again.json"
        ev.export_visual(out2)
        self.assertEqual(self.out.read_bytes(), out2.read_bytes())
        leftovers = [p for p in Path(self.tmp.name).iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
