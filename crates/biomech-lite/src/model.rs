//! Model JSON v1 (`assets/biomech-lite/model.json`), loading, validation and the exact
//! native joint-coupling parameterization.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::{
    GEOM_CYLINDER, GEOM_SPHERE, JNT_BALL, JNT_FREE, JNT_HINGE, JNT_SLIDE, WRAP_CYLINDER,
    WRAP_SITE, WRAP_SPHERE,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Body {
    pub name: String,
    pub parent: usize,
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub ipos: [f64; 3],
    pub iquat: [f64; 4],
    pub mass: f64,
    pub inertia: [f64; 3],
    pub joints: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Joint {
    pub name: String,
    pub kind: i32,
    pub body: usize,
    pub qposadr: usize,
    pub dofadr: usize,
    pub pos: [f64; 3],
    pub axis: [f64; 3],
    pub limited: bool,
    pub range: [f64; 2],
    pub damping: f64,
    pub armature: f64,
    pub stiffness: f64,
    pub springref: f64,
}

impl Joint {
    pub fn is_scalar(&self) -> bool {
        self.kind == JNT_HINGE || self.kind == JNT_SLIDE
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Site {
    pub name: String,
    pub body: usize,
    pub pos: [f64; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Geom {
    pub name: String,
    pub body: usize,
    pub kind: i32,
    pub pos: [f64; 3],
    pub quat: [f64; 4],
    pub size: [f64; 3],
}

/// Native `mjEQ_JOINT`: `q[dep] - qpos0[dep] = sum_i coef[i] (q[master] - qpos0[master])^i`;
/// with no master (locked joint) `q[dep] = qpos0[dep] + coef[0]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coupling {
    pub dependent: usize,
    pub master: Option<usize>,
    pub coef: [f64; 5],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WrapPoint {
    pub kind: i32,
    pub object: i32,
    pub prm: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tendon {
    pub name: String,
    pub path: Vec<WrapPoint>,
    pub stiffness: f64,
    pub damping: f64,
    pub springlength: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Muscle {
    pub name: String,
    pub tendon: usize,
    pub gainprm: [f64; 9],
    pub biasprm: [f64; 9],
    pub dynprm: [f64; 3],
    pub lengthrange: [f64; 2],
    pub acc0: f64,
    pub ctrlrange: [f64; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub schema_version: u32,
    #[serde(default)]
    pub source: serde_json::Value,
    pub nq: usize,
    pub nv: usize,
    pub qpos0: Vec<f64>,
    pub standing_qpos: Vec<f64>,
    pub gravity: [f64; 3],
    pub dynamic_joints: Vec<usize>,
    pub bodies: Vec<Body>,
    pub joints: Vec<Joint>,
    pub sites: Vec<Site>,
    pub geoms: Vec<Geom>,
    pub couplings: Vec<Coupling>,
    pub tendons: Vec<Tendon>,
    pub muscles: Vec<Muscle>,
}

fn finite(name: &str, v: &[f64]) -> Result<(), String> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(format!("{name}: non-finite value at index {i}"));
    }
    Ok(())
}

/// Value and slope of the native quartic `c0 + c1 x + ... + c4 x^4`.
pub fn coupling_poly(c: &[f64; 5], x: f64) -> (f64, f64) {
    let v = c[4] * x.powi(4) + c[3] * x.powi(3) + c[2] * x * x + c[1] * x + c[0];
    let s = 4.0 * c[4] * x.powi(3) + 3.0 * c[3] * x * x + 2.0 * c[2] * x + c[1];
    (v, s)
}

impl Model {
    /// Load and validate `model.json`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read model {}: {e}", path.display()))?;
        let m: Model = serde_json::from_str(&text)
            .map_err(|e| format!("invalid model JSON {}: {e}", path.display()))?;
        m.validate()?;
        Ok(m)
    }

    /// Structural and semantic checks. Unsupported native features fail here, clearly.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 {
            return Err(format!("unsupported schema_version {}", self.schema_version));
        }
        if self.qpos0.len() != self.nq || self.standing_qpos.len() != self.nq {
            return Err(format!("qpos0/standing_qpos length must equal nq={}", self.nq));
        }
        finite("qpos0", &self.qpos0)?;
        finite("standing_qpos", &self.standing_qpos)?;
        finite("gravity", &self.gravity)?;
        if self.bodies.is_empty() || self.bodies[0].parent != 0 || !self.bodies[0].joints.is_empty() {
            return Err("body 0 must be the jointless world body".into());
        }
        let (mut nq, mut nv) = (0usize, 0usize);
        let mut joint_owner = vec![usize::MAX; self.joints.len()];
        for (b, body) in self.bodies.iter().enumerate() {
            if b > 0 && body.parent >= b {
                return Err(format!("body {b} ({}): parent {} not before child", body.name, body.parent));
            }
            finite(&format!("body {}", body.name), &[body.pos.as_slice(), &body.quat, &body.ipos,
                &body.iquat, &body.inertia, &[body.mass]].concat())?;
            for (k, &j) in body.joints.iter().enumerate() {
                if j >= self.joints.len() {
                    return Err(format!("body {}: joint index {j} out of range", body.name));
                }
                if k > 0 && j != body.joints[k - 1] + 1 {
                    return Err(format!("body {}: joints must be contiguous", body.name));
                }
                joint_owner[j] = b;
            }
        }
        for (j, jt) in self.joints.iter().enumerate() {
            let ctx = format!("joint {j} ({})", jt.name);
            if joint_owner[j] != jt.body {
                return Err(format!("{ctx}: body {} does not list it", jt.body));
            }
            let (dq, dv) = match jt.kind {
                JNT_FREE => {
                    if self.bodies[jt.body].joints.len() != 1 {
                        return Err(format!("{ctx}: free joint must be the only joint of its body"));
                    }
                    (7, 6)
                }
                JNT_SLIDE | JNT_HINGE => (1, 1),
                JNT_BALL => return Err(format!("{ctx}: ball joints are not implemented")),
                k => return Err(format!("{ctx}: unknown joint kind {k}")),
            };
            if jt.qposadr != nq || jt.dofadr != nv {
                return Err(format!("{ctx}: qposadr/dofadr {}/{} not sequential (expected {nq}/{nv})",
                    jt.qposadr, jt.dofadr));
            }
            nq += dq;
            nv += dv;
            finite(&ctx, &[jt.pos.as_slice(), &jt.axis, &jt.range,
                &[jt.damping, jt.armature, jt.stiffness, jt.springref]].concat())?;
        }
        if nq != self.nq || nv != self.nv {
            return Err(format!("joints imply nq/nv {nq}/{nv}, model says {}/{}", self.nq, self.nv));
        }
        for s in &self.sites {
            if s.body >= self.bodies.len() {
                return Err(format!("site {}: body out of range", s.name));
            }
            finite(&format!("site {}", s.name), &s.pos)?;
        }
        for g in &self.geoms {
            if g.body >= self.bodies.len() {
                return Err(format!("geom {}: body out of range", g.name));
            }
            finite(&format!("geom {}", g.name), &[g.pos.as_slice(), &g.quat, &g.size].concat())?;
        }
        // couplings: scalar dependents, unique, masters written before use (topological order)
        let mut written = vec![false; self.joints.len()];
        let mut is_dep = vec![false; self.joints.len()];
        for (c, cp) in self.couplings.iter().enumerate() {
            let dep = self.joints.get(cp.dependent).ok_or(format!("coupling {c}: dependent out of range"))?;
            if !dep.is_scalar() {
                return Err(format!("coupling {c}: dependent {} is not a scalar joint", dep.name));
            }
            if is_dep[cp.dependent] {
                return Err(format!("coupling {c}: joint {} is dependent more than once", dep.name));
            }
            if let Some(mj) = cp.master {
                let mas = self.joints.get(mj).ok_or(format!("coupling {c}: master out of range"))?;
                if !mas.is_scalar() {
                    return Err(format!("coupling {c}: master {} is not a scalar joint", mas.name));
                }
                if is_dep[mj] && !written[mj] {
                    return Err(format!("coupling {c}: master {} not yet written", mas.name));
                }
                if mj == cp.dependent {
                    return Err(format!("coupling {c}: self-coupled joint {}", dep.name));
                }
            }
            finite(&format!("coupling {c}"), &cp.coef)?;
            is_dep[cp.dependent] = true;
            written[cp.dependent] = true;
        }
        // a dependent may not be the master of an earlier coupling
        for (c, cp) in self.couplings.iter().enumerate() {
            if let Some(mj) = cp.master {
                if let Some(pos) = self.couplings.iter().position(|o| o.dependent == mj) {
                    if pos > c {
                        return Err(format!("coupling {c}: master {} is written later (not topologically sorted)",
                            self.joints[mj].name));
                    }
                }
            }
        }
        if self.dynamic_joints.len() != 8 {
            return Err(format!("dynamic_joints must list 8 joints, got {}", self.dynamic_joints.len()));
        }
        for &j in &self.dynamic_joints {
            let jt = self.joints.get(j).ok_or(format!("dynamic joint {j} out of range"))?;
            if !jt.is_scalar() || is_dep[j] {
                return Err(format!("dynamic joint {} must be an independent scalar joint", jt.name));
            }
        }
        for (t, tn) in self.tendons.iter().enumerate() {
            self.validate_tendon(t, tn)?;
        }
        for (a, mu) in self.muscles.iter().enumerate() {
            let ctx = format!("muscle {a} ({})", mu.name);
            if mu.tendon >= self.tendons.len() {
                return Err(format!("{ctx}: tendon out of range"));
            }
            finite(&ctx, &[mu.gainprm.as_slice(), &mu.biasprm, &mu.dynprm, &mu.lengthrange,
                &mu.ctrlrange, &[mu.acc0]].concat())?;
        }
        Ok(())
    }

    fn validate_tendon(&self, t: usize, tn: &Tendon) -> Result<(), String> {
        let ctx = format!("tendon {t} ({})", tn.name);
        let p = &tn.path;
        if p.len() < 2 {
            return Err(format!("{ctx}: spatial path needs at least two points"));
        }
        finite(&ctx, &[tn.stiffness, tn.damping, tn.springlength[0], tn.springlength[1]])?;
        for (k, w) in p.iter().enumerate() {
            match w.kind {
                WRAP_SITE => {
                    if w.object < 0 || w.object as usize >= self.sites.len() {
                        return Err(format!("{ctx}: path[{k}] site id {} out of range", w.object));
                    }
                }
                WRAP_SPHERE | WRAP_CYLINDER => {
                    if k == 0 || k + 1 == p.len() || p[k - 1].kind != WRAP_SITE || p[k + 1].kind != WRAP_SITE {
                        return Err(format!("{ctx}: path[{k}] wrap geom must sit between two sites"));
                    }
                    let g = self.geoms.get(w.object.max(0) as usize).filter(|_| w.object >= 0)
                        .ok_or(format!("{ctx}: path[{k}] geom id {} out of range", w.object))?;
                    let want = if w.kind == WRAP_SPHERE { GEOM_SPHERE } else { GEOM_CYLINDER };
                    if g.kind != want {
                        return Err(format!("{ctx}: path[{k}] wrap kind {} on geom {} of kind {}", w.kind, g.name, g.kind));
                    }
                    let side = w.prm.round();
                    if side < -1.0 || side >= self.sites.len() as f64 {
                        return Err(format!("{ctx}: path[{k}] invalid side site {}", w.prm));
                    }
                }
                k2 => {
                    return Err(format!("{ctx}: path[{k}] wrap kind {k2} not supported (site/sphere/cylinder only; \
                        fixed tendons and pulleys are not implemented)"))
                }
            }
        }
        if p[0].kind != WRAP_SITE || p[p.len() - 1].kind != WRAP_SITE {
            return Err(format!("{ctx}: spatial path must start and end with a site"));
        }
        Ok(())
    }

    /// Write every coupled/locked joint of `q` from its master, in coupling order (exact quartic).
    pub fn project(&self, q: &mut [f64]) -> Result<(), String> {
        if q.len() != self.nq {
            return Err(format!("q length {} != nq {}", q.len(), self.nq));
        }
        for c in &self.couplings {
            let dq = self.joints[c.dependent].qposadr;
            q[dq] = match c.master {
                None => self.qpos0[dq] + c.coef[0],
                Some(m) => {
                    let mq = self.joints[m].qposadr;
                    self.qpos0[dq] + coupling_poly(&c.coef, q[mq] - self.qpos0[mq]).0
                }
            };
        }
        Ok(())
    }

    /// Chain-rule velocity projection at `q` (assumed projected): dependent dof velocity =
    /// slope * master dof velocity, locked dofs zero. Applied in coupling order, so chains are exact.
    pub fn project_velocity(&self, q: &[f64], qvel: &mut [f64]) -> Result<(), String> {
        if q.len() != self.nq || qvel.len() != self.nv {
            return Err(format!("q/qvel lengths {}/{} != nq/nv {}/{}", q.len(), qvel.len(), self.nq, self.nv));
        }
        for c in &self.couplings {
            let dd = self.joints[c.dependent].dofadr;
            qvel[dd] = match c.master {
                None => 0.0,
                Some(m) => {
                    let jm = &self.joints[m];
                    coupling_poly(&c.coef, q[jm.qposadr] - self.qpos0[jm.qposadr]).1 * qvel[jm.dofadr]
                }
            };
        }
        Ok(())
    }

    /// Residual `q[dep] - qpos0[dep] - p(q[master] - qpos0[master])` per coupling.
    pub fn coupling_residuals(&self, q: &[f64]) -> Vec<f64> {
        self.couplings.iter().map(|c| {
            let dq = self.joints[c.dependent].qposadr;
            let p = match c.master {
                None => c.coef[0],
                Some(m) => {
                    let mq = self.joints[m].qposadr;
                    coupling_poly(&c.coef, q[mq] - self.qpos0[mq]).0
                }
            };
            q[dq] - self.qpos0[dq] - p
        }).collect()
    }

    pub fn joint_id(&self, name: &str) -> Option<usize> {
        self.joints.iter().position(|j| j.name == name)
    }
    pub fn body_id(&self, name: &str) -> Option<usize> {
        self.bodies.iter().position(|b| b.name == name)
    }
    pub fn site_id(&self, name: &str) -> Option<usize> {
        self.sites.iter().position(|s| s.name == name)
    }
    pub fn tendon_id(&self, name: &str) -> Option<usize> {
        self.tendons.iter().position(|t| t.name == name)
    }
    pub fn muscle_id(&self, name: &str) -> Option<usize> {
        self.muscles.iter().position(|m| m.name == name)
    }
}
