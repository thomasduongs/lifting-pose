//! Sourced anatomy: the offline-exported MyoSim bone meshes (`assets/biomech-lite/visual.json`).
//!
//! Vertices are the compiled MuJoCo `mesh_vert` slices (already in the geom frame), so the
//! world position of a vertex is `kin.geom_pos[g] + kin.geom_mat[g] * v`; `mesh_pos/quat` are
//! never applied a second time. Mesh data: MyoSim 0.2.5 (Apache-2.0) and its cited sources.

use std::path::Path;

use biomech_lite::kinematics::{add3, cross3, dot3, mat_vec, sub3};
use biomech_lite::{Kinematics, Model};
use serde::Deserialize;

/// Native `mjGEOM_MESH`.
pub const GEOM_MESH: i32 = 7;

/// One native mesh (`mesh_vert` / `mesh_face` slices, local indices).
#[derive(Debug, Clone)]
pub struct Mesh {
    pub name: String,
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
    /// `+1.0` when the faces wind counter-clockwise seen from outside (positive signed
    /// volume), `-1.0` otherwise. Used for back-face culling; face data is never altered.
    pub winding: f64,
}

/// One native mesh geom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualGeom {
    /// Native geom id (index into `Model::geoms` / `Kinematics::geom_pos`).
    pub geom: usize,
    /// Native mesh id (index into [`VisualModel::meshes`]).
    pub mesh: usize,
    /// Native body id.
    pub body: usize,
    pub rgba: [f32; 4],
    /// Lower body: not observed in the recording, held at the reference pose.
    pub unobserved: bool,
}

/// Independent attachment fixture: the native engine's world position of one mesh vertex
/// at [`VisualModel::reference_qpos`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferenceSample {
    pub geom: usize,
    pub vertex: usize,
    pub world_position: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct VisualModel {
    pub source_model_sha256: String,
    /// Free-form source note (versions), for display.
    pub source_note: String,
    pub meshes: Vec<Mesh>,
    pub geoms: Vec<VisualGeom>,
    pub reference_qpos: Vec<f64>,
    pub reference_samples: Vec<ReferenceSample>,
    pub vertex_count: usize,
    /// Triangles over all mesh geoms (instanced meshes counted per geom).
    pub triangle_count: usize,
    /// native geom id -> index into `geoms`.
    geom_lookup: Vec<Option<usize>>,
}

#[derive(Deserialize)]
struct RawMesh {
    name: String,
    vertices: Vec<[f64; 3]>,
    faces: Vec<[i64; 3]>,
}

#[derive(Deserialize)]
struct RawGeom {
    geom: usize,
    mesh: usize,
    body: usize,
    rgba: [f64; 4],
    unobserved: bool,
}

#[derive(Deserialize)]
struct RawSample {
    geom: usize,
    vertex: usize,
    world_position: [f64; 3],
}

#[derive(Deserialize)]
struct RawReference {
    qpos: Vec<f64>,
    samples: Vec<RawSample>,
}

#[derive(Deserialize)]
struct RawVisual {
    schema_version: u32,
    source_model_sha256: String,
    #[serde(default)]
    source: serde_json::Value,
    meshes: Vec<RawMesh>,
    geoms: Vec<RawGeom>,
    reference: RawReference,
}

fn finite3(v: &[f64; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Signed volume of a triangle soup (divergence theorem), sign tells the winding.
fn signed_volume(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> f64 {
    let mut c = [0.0; 3];
    for v in vertices {
        c = add3(&c, v);
    }
    let n = vertices.len().max(1) as f64;
    c = [c[0] / n, c[1] / n, c[2] / n];
    faces.iter().map(|f| {
        let a = sub3(&vertices[f[0] as usize], &c);
        let b = sub3(&vertices[f[1] as usize], &c);
        let d = sub3(&vertices[f[2] as usize], &c);
        dot3(&a, &cross3(&b, &d))
    }).sum::<f64>() / 6.0
}

impl VisualModel {
    /// Load and validate `visual.json` against `model` (source hash, geom/body/mesh ids).
    pub fn load(path: impl AsRef<Path>, model: &Model) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read visual asset {}: {e}", path.display()))?;
        Self::from_json_str(&text, model).map_err(|e| format!("visual asset {}: {e}", path.display()))
    }

    /// Parse and validate visual JSON text.
    pub fn from_json_str(text: &str, model: &Model) -> Result<Self, String> {
        let raw: RawVisual = serde_json::from_str(text).map_err(|e| format!("invalid JSON: {e}"))?;
        Self::from_raw(raw, model)
    }

    fn from_raw(raw: RawVisual, model: &Model) -> Result<Self, String> {
        if raw.schema_version != 1 {
            return Err(format!("unsupported schema_version {}", raw.schema_version));
        }
        let want = model.source.get("mjb_sha256").and_then(|v| v.as_str())
            .ok_or("physics model has no source.mjb_sha256 to match the visual asset against")?;
        if !raw.source_model_sha256.eq_ignore_ascii_case(want) {
            return Err(format!("source_model_sha256 {} does not match model mjb_sha256 {want}; \
                re-export the visual asset", raw.source_model_sha256));
        }
        if raw.meshes.is_empty() {
            return Err("no meshes".into());
        }
        let mut meshes = Vec::with_capacity(raw.meshes.len());
        for (m, rm) in raw.meshes.into_iter().enumerate() {
            let ctx = format!("mesh {m} ({})", rm.name);
            if rm.vertices.is_empty() || rm.faces.is_empty() {
                return Err(format!("{ctx}: empty vertices or faces"));
            }
            if let Some(i) = rm.vertices.iter().position(|v| !finite3(v)) {
                return Err(format!("{ctx}: non-finite vertex {i}"));
            }
            let nv = rm.vertices.len() as i64;
            let mut faces = Vec::with_capacity(rm.faces.len());
            for (f, face) in rm.faces.iter().enumerate() {
                if face.iter().any(|&i| i < 0 || i >= nv) {
                    return Err(format!("{ctx}: face {f} index out of range 0..{nv}: {face:?}"));
                }
                faces.push([face[0] as u32, face[1] as u32, face[2] as u32]);
            }
            let winding = if signed_volume(&rm.vertices, &faces) < 0.0 { -1.0 } else { 1.0 };
            meshes.push(Mesh { name: rm.name, vertices: rm.vertices, faces, winding });
        }

        let mut geom_lookup = vec![None; model.geoms.len()];
        let mut geoms = Vec::with_capacity(raw.geoms.len());
        for (i, g) in raw.geoms.iter().enumerate() {
            let ctx = format!("geoms[{i}] (native geom {})", g.geom);
            let mg = model.geoms.get(g.geom).ok_or(format!("{ctx}: geom id out of range"))?;
            if mg.kind != GEOM_MESH {
                return Err(format!("{ctx}: model geom {} is kind {}, not a mesh", mg.name, mg.kind));
            }
            if mg.body != g.body {
                return Err(format!("{ctx}: body {} but model geom {} is on body {}", g.body, mg.name, mg.body));
            }
            if g.mesh >= meshes.len() {
                return Err(format!("{ctx}: mesh id {} out of range", g.mesh));
            }
            if !g.rgba.iter().all(|x| x.is_finite()) {
                return Err(format!("{ctx}: non-finite rgba"));
            }
            if geom_lookup[g.geom].is_some() {
                return Err(format!("{ctx}: duplicate geom"));
            }
            geom_lookup[g.geom] = Some(i);
            geoms.push(VisualGeom {
                geom: g.geom,
                mesh: g.mesh,
                body: g.body,
                rgba: g.rgba.map(|x| x as f32),
                unobserved: g.unobserved,
            });
        }
        if let Some((g, mg)) = model.geoms.iter().enumerate()
            .find(|(g, mg)| mg.kind == GEOM_MESH && geom_lookup[*g].is_none())
        {
            return Err(format!("model mesh geom {g} ({}) has no visual geometry", mg.name));
        }

        if raw.reference.qpos.len() != model.nq {
            return Err(format!("reference qpos length {} != nq {}", raw.reference.qpos.len(), model.nq));
        }
        if raw.reference.qpos.iter().any(|x| !x.is_finite()) {
            return Err("reference qpos not finite".into());
        }
        let mut samples = Vec::with_capacity(raw.reference.samples.len());
        for (i, s) in raw.reference.samples.iter().enumerate() {
            let vg = geom_lookup.get(s.geom).copied().flatten()
                .ok_or(format!("reference sample {i}: geom {} is not a visual geom", s.geom))?;
            let nv = meshes[geoms[vg].mesh].vertices.len();
            if s.vertex >= nv || !finite3(&s.world_position) {
                return Err(format!("reference sample {i}: vertex {} out of range or bad position", s.vertex));
            }
            samples.push(ReferenceSample { geom: s.geom, vertex: s.vertex, world_position: s.world_position });
        }

        let vertex_count = meshes.iter().map(|m| m.vertices.len()).sum();
        let triangle_count = geoms.iter().map(|g| meshes[g.mesh].faces.len()).sum();
        let source_note = match &raw.source {
            serde_json::Value::Object(o) => o.iter()
                .map(|(k, v)| format!("{k} {}", v.as_str().map(str::to_owned).unwrap_or_else(|| v.to_string())))
                .collect::<Vec<_>>().join(", "),
            _ => String::new(),
        };
        Ok(VisualModel {
            source_model_sha256: raw.source_model_sha256,
            source_note,
            meshes,
            geoms,
            reference_qpos: raw.reference.qpos,
            reference_samples: samples,
            vertex_count,
            triangle_count,
            geom_lookup,
        })
    }

    /// Visual geom for a native geom id.
    pub fn geom_by_native(&self, geom: usize) -> Option<&VisualGeom> {
        self.geom_lookup.get(geom).copied().flatten().map(|i| &self.geoms[i])
    }

    /// World position of local vertex `vertex` of `vg` at kinematics `kin`.
    pub fn world_vertex(&self, kin: &Kinematics, vg: &VisualGeom, vertex: usize) -> [f64; 3] {
        let v = &self.meshes[vg.mesh].vertices[vertex];
        add3(&kin.geom_pos[vg.geom], &mat_vec(&kin.geom_mat[vg.geom], v))
    }

    /// Check that `kin` has poses for every visual geom.
    pub fn check_kinematics(&self, kin: &Kinematics) -> Result<(), String> {
        let n = kin.geom_pos.len().min(kin.geom_mat.len());
        match self.geoms.iter().find(|g| g.geom >= n) {
            Some(g) => Err(format!("kinematics has {n} geoms; visual geom {} out of range", g.geom)),
            None => Ok(()),
        }
    }

    /// World-space axis-aligned bounds of the selected geoms' vertices.
    pub fn world_bounds(&self, kin: &Kinematics, mut include: impl FnMut(&VisualGeom) -> bool)
        -> Option<([f64; 3], [f64; 3])>
    {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for vg in self.geoms.iter().filter(|g| include(g)) {
            if vg.geom >= kin.geom_pos.len() {
                continue;
            }
            for i in 0..self.meshes[vg.mesh].vertices.len() {
                let p = self.world_vertex(kin, vg, i);
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
        }
        (lo[0] <= hi[0]).then_some((lo, hi))
    }
}
