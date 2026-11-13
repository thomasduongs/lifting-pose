//! Native finger IK from confirmed 21-landmark hands. Palm-local bone directions
//! remove camera pose and hand size; each digit solves only its own four hinges.
//! Missing hands hold their previous coordinates. No predicted hands enter this API.
use crate::{
    Model,
    kinematics::{Vec3, cross3, dot3, norm3, scl3, sub3},
};
use nalgebra::{Matrix4, Vector4};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct HandObservation {
    /// 0 = left, 1 = right, anatomical side (not screen side).
    pub side: usize,
    pub confidence: f64,
    pub points: [Vec3; 21],
}
#[derive(Clone)]
struct Spec {
    joints: [[usize; 4]; 5],
    tips: [usize; 5],
    wrist: usize,
}
pub struct HandFitter {
    model: Arc<Model>,
    specs: [Spec; 2],
    previous: [Vec<f64>; 2],
    last_t: [Option<f64>; 2],
}
fn unit(v: Vec3) -> Option<Vec3> {
    let n = norm3(&v);
    (n.is_finite() && n > 1e-5).then(|| scl3(&v, 1.0 / n))
}
fn basis(p: &[Vec3; 21]) -> Option<[Vec3; 3]> {
    let x = unit(sub3(&p[5], &p[17]))?;
    let y0 = sub3(&p[9], &p[0]);
    let y = unit(sub3(&y0, &scl3(&x, dot3(&x, &y0))))?;
    Some([x, y, cross3(&x, &y)])
}
fn directions(p: &[Vec3; 21], b: &[Vec3; 3], finger: usize) -> Option<[f64; 9]> {
    let base = 1 + finger * 4;
    let mut r = [0.0; 9];
    for i in 0..3 {
        let d = unit(sub3(&p[base + i + 1], &p[base + i]))?;
        for k in 0..3 {
            r[i * 3 + k] = dot3(&b[k], &d);
        }
    }
    Some(r)
}
impl HandFitter {
    pub fn new(model: Arc<Model>) -> Result<Self, String> {
        let mut specs = Vec::new();
        for side in ["l", "r"] {
            let jid = |stem: &str| {
                model
                    .joint_id(&format!("{stem}_{side}"))
                    .or_else(|| model.joint_id(stem))
                    .ok_or(format!("missing hand joint {stem}_{side}"))
            };
            let mut joints = [[0; 4]; 5];
            joints[0] = [
                jid("cmc_flexion")?,
                jid("cmc_abduction")?,
                jid("mp_flexion")?,
                jid("ip_flexion")?,
            ];
            for finger in 2..=5 {
                joints[finger - 1] = [
                    jid(&format!("mcp{finger}_flexion"))?,
                    jid(&format!("mcp{finger}_abduction"))?,
                    jid(&format!("pm{finger}_flexion"))?,
                    jid(&format!("md{finger}_flexion"))?,
                ];
            }
            let mut tips = [0; 5];
            for (i, name) in ["THtip", "IFtip", "MFtip", "RFtip", "LFtip"]
                .iter()
                .enumerate()
            {
                tips[i] = model
                    .site_id(&format!("{name}_{side}"))
                    .ok_or(format!("missing {name}_{side}"))?;
            }
            specs.push(Spec {
                joints,
                tips,
                wrist: jid("flexion")?,
            });
        }
        let mut out = Self {
            previous: [vec![], vec![]],
            last_t: [None; 2],
            specs: specs.try_into().unwrap_or_else(|_| unreachable!()),
            model,
        };
        out.reset();
        Ok(out)
    }
    pub fn reset(&mut self) {
        for side in 0..2 {
            self.previous[side] = self.specs[side]
                .joints
                .iter()
                .flatten()
                .map(|&j| self.model.standing_qpos[self.model.joints[j].qposadr])
                .collect();
        }
        self.last_t = [None; 2];
    }
    /// Native joint anchors / fingertip sites in MediaPipe's 21-point order.
    pub fn landmarks(&self, side: usize, q: &[f64]) -> Result<[Vec3; 21], String> {
        let spec = self.specs.get(side).ok_or("invalid hand side")?;
        let k = self.model.kinematics(q)?;
        let mut p = [[0.0; 3]; 21];
        p[0] = k.joint_anchor[spec.wrist];
        for f in 0..5 {
            for (i, j) in [0, 2, 3].into_iter().enumerate() {
                p[1 + f * 4 + i] = k.joint_anchor[spec.joints[f][j]];
            }
            p[4 + f * 4] = k.site_pos[spec.tips[f]];
        }
        Ok(p)
    }
    /// Apply retained finger state, then fit current confirmed hands independently.
    pub fn apply(
        &mut self,
        q: &mut [f64],
        observations: &[HandObservation],
        t: f64,
    ) -> Result<[bool; 2], String> {
        if q.len() != self.model.nq || !t.is_finite() {
            return Err("invalid hand fit configuration/timestamp".into());
        }
        for side in 0..2 {
            for (i, &j) in self.specs[side].joints.iter().flatten().enumerate() {
                q[self.model.joints[j].qposadr] = self.previous[side][i];
            }
        }
        let mut supported = [false; 2];
        for side in 0..2 {
            let Some(obs) = observations
                .iter()
                .filter(|o| {
                    o.side == side
                        && o.confidence.is_finite()
                        && o.confidence >= 0.85
                        && o.points.iter().flatten().all(|x| x.is_finite())
                })
                .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
            else {
                continue;
            };
            let Some(source_basis) = basis(&obs.points) else {
                continue;
            };
            let targets: Option<Vec<_>> = (0..5)
                .map(|f| directions(&obs.points, &source_basis, f))
                .collect();
            let Some(targets) = targets else {
                continue;
            };
            let native = self.landmarks(side, q)?;
            let Some(model_basis) = basis(&native) else {
                continue;
            };
            let before = q.to_vec();
            for f in 0..5 {
                let ids = self.specs[side].joints[f];
                let adr = ids.map(|j| self.model.joints[j].qposadr);
                let residual = |q: &[f64]| -> Result<[f64; 9], String> {
                    let p = self.landmarks(side, q)?;
                    let d = directions(&p, &model_basis, f).ok_or("degenerate model digit")?;
                    Ok(std::array::from_fn(|i| d[i] - targets[f][i]))
                };
                for _ in 0..6 {
                    let r = residual(q)?;
                    let cost: f64 = r.iter().map(|x| x * x).sum();
                    if cost < 1e-7 {
                        break;
                    }
                    let mut jac = [[0.0; 4]; 9];
                    for c in 0..4 {
                        let old = q[adr[c]];
                        q[adr[c]] = old + 1e-4;
                        let plus = residual(q)?;
                        q[adr[c]] = old - 1e-4;
                        let minus = residual(q)?;
                        q[adr[c]] = old;
                        for i in 0..9 {
                            jac[i][c] = (plus[i] - minus[i]) / 2e-4;
                        }
                    }
                    let mut a = Matrix4::identity() * 0.003;
                    let mut b = Vector4::zeros();
                    for i in 0..9 {
                        let j = Vector4::from_row_slice(&jac[i]);
                        a += j * j.transpose();
                        b -= j * r[i];
                    }
                    let Some(step) = a.lu().solve(&b) else {
                        break;
                    };
                    if !step.iter().all(|x| x.is_finite()) {
                        break;
                    }
                    let old = adr.map(|a| q[a]);
                    let mut accepted = false;
                    for alpha in [1.0, 0.5, 0.25] {
                        for c in 0..4 {
                            let j = &self.model.joints[ids[c]];
                            q[adr[c]] = (old[c] + alpha * step[c].clamp(-0.4, 0.4))
                                .clamp(j.range[0], j.range[1]);
                        }
                        if residual(q)?.iter().map(|x| x * x).sum::<f64>() < cost {
                            accepted = true;
                            break;
                        }
                    }
                    if !accepted {
                        for c in 0..4 {
                            q[adr[c]] = old[c];
                        }
                        break;
                    }
                }
            }
            let dt = self.last_t[side]
                .map(|old| (t - old).clamp(1.0 / 240.0, 0.1))
                .unwrap_or(0.25);
            let step = 12.0 * dt;
            let alpha = 1.0 - (-dt / 0.025).exp();
            for (i, &j) in self.specs[side].joints.iter().flatten().enumerate() {
                let a = self.model.joints[j].qposadr;
                q[a] = before[a] + (alpha * (q[a] - before[a])).clamp(-step, step);
                self.previous[side][i] = q[a];
            }
            self.last_t[side] = Some(t);
            supported[side] = true;
        }
        self.model.project(q)?;
        Ok(supported)
    }
}
