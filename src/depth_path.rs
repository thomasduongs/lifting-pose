//! Batch estimation of uncertain monocular body distance, not landmark filtering.
//! Fixed shape, image-scale evidence, and a time-based acceleration prior share
//! one objective. Arm observations and joint rotations are not averaged here.

pub struct Sample {
    pub time: f64,
    pub estimate: f64,
    pub confidence: f64,
    pub scale: Option<(usize, f64)>,
}
pub(crate) struct Factor {
    terms: [(usize, f64); 3],
    count: usize,
    target: f64,
    weight: f64,
}
impl Factor {
    pub(crate) fn new(terms: &[(usize, f64)], target: f64, weight: f64) -> Self {
        let mut values = [(0, 0.0); 3];
        values[..terms.len()].copy_from_slice(terms);
        Self {
            terms: values,
            count: terms.len(),
            target,
            weight,
        }
    }
    fn value(&self, x: &[f64]) -> f64 {
        self.terms[..self.count]
            .iter()
            .map(|&(i, a)| a * x[i])
            .sum()
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn multiply(factors: &[Factor], x: &[f64], out: &mut [f64]) {
    out.fill(0.0);
    for f in factors {
        let value = f.value(x) * f.weight;
        for &(i, a) in &f.terms[..f.count] {
            out[i] += a * value;
        }
    }
}
pub(crate) fn solve(factors: &[Factor], x: &mut [f64]) {
    let n = x.len();
    let mut diagonal = vec![0.0; n];
    let mut rhs = vec![0.0; n];
    for f in factors {
        for &(i, a) in &f.terms[..f.count] {
            diagonal[i] += f.weight * a * a;
            rhs[i] += f.weight * a * f.target;
        }
    }
    let mut ax = vec![0.0; n];
    multiply(factors, x, &mut ax);
    let mut r: Vec<_> = rhs.iter().zip(ax).map(|(b, a)| b - a).collect();
    let mut z: Vec<_> = r
        .iter()
        .zip(&diagonal)
        .map(|(r, d)| r / d.max(1e-12))
        .collect();
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let tolerance = 1e-16 * dot(&rhs, &rhs).max(1.0);
    let mut ap = vec![0.0; n];
    for _ in 0..(n * 4).clamp(100, 3000) {
        if dot(&r, &r) < tolerance {
            break;
        }
        multiply(factors, &p, &mut ap);
        let denom = dot(&p, &ap);
        if denom <= 1e-30 {
            break;
        }
        let alpha = rz / denom;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
            z[i] = r[i] / diagonal[i].max(1e-12);
        }
        let next = dot(&r, &z);
        let beta = next / rz.max(1e-30);
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = next;
    }
}

pub fn estimate(samples: &[Sample], shoulder_width: f64) -> Option<Vec<f64>> {
    estimate_impl(samples, shoulder_width, &[])
}

#[cfg(test)]
pub fn estimate_with_upper_bounds(
    samples: &[Sample],
    shoulder_width: f64,
    bounds: &[(usize, f64, f64)],
) -> Option<Vec<f64>> {
    estimate_impl(samples, shoulder_width, bounds)
}

fn estimate_impl(
    samples: &[Sample],
    shoulder_width: f64,
    bounds: &[(usize, f64, f64)],
) -> Option<Vec<f64>> {
    if samples.len() < 3 || !shoulder_width.is_finite() || shoulder_width <= 0.0 {
        return None;
    }
    if samples
        .iter()
        .any(|s| !s.time.is_finite() || !s.estimate.is_finite() || s.estimate <= 0.1)
    {
        return None;
    }
    let sigma = shoulder_width * 0.25;
    let acceleration = shoulder_width * 25.0;
    let mut x: Vec<_> = samples.iter().map(|s| s.estimate).collect();
    for _ in 0..5 {
        let mut factors = Vec::new();
        for (i, s) in samples.iter().enumerate() {
            let residual = (x[i] - s.estimate).abs();
            let robust = (sigma * 2.0 / residual.max(sigma * 2.0)).min(1.0);
            factors.push(Factor::new(
                &[(i, 1.0)],
                s.estimate,
                (s.confidence.clamp(0.0, 1.0) * robust + 1e-4) / sigma.powi(2),
            ));
            if let Some((reference, scale)) = s.scale {
                if reference < i && scale.is_finite() && (0.8..1.25).contains(&scale) {
                    let spread = (i - reference) as f64;
                    let noise = (shoulder_width * 0.08).max(s.estimate * 0.01 * spread.sqrt());
                    let error = (scale * x[i] - x[reference]).abs();
                    let robust = (noise * 2.0 / error.max(noise * 2.0)).min(1.0);
                    factors.push(Factor::new(
                        &[(i, scale), (reference, -1.0)],
                        0.0,
                        robust / noise.powi(2),
                    ));
                }
            }
        }
        for i in 1..samples.len() - 1 {
            let a = (samples[i].time - samples[i - 1].time).max(1e-3);
            let b = (samples[i + 1].time - samples[i].time).max(1e-3);
            let c = 2.0 / (a + b);
            factors.push(Factor::new(
                &[
                    (i - 1, c / a),
                    (i, -c * (1.0 / a + 1.0 / b)),
                    (i + 1, c / b),
                ],
                0.0,
                (a + b) / 2.0 / acceleration.powi(2),
            ));
        }
        for &(i, upper, confidence) in bounds {
            if i < x.len() && upper.is_finite() && upper > 0.1 && x[i] > upper {
                factors.push(Factor::new(
                    &[(i, 1.0)],
                    upper,
                    confidence.clamp(0.0, 1.0) / (shoulder_width * 0.1).powi(2),
                ));
            }
        }
        solve(&factors, &mut x);
    }
    x.iter().all(|z| z.is_finite() && *z > 0.1).then_some(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonbinding_upper_bounds_do_not_attract_body_depth() {
        let samples: Vec<_> = (0..60)
            .map(|i| Sample {
                time: i as f64 / 30.0,
                estimate: 2.0,
                confidence: 1.0,
                scale: None,
            })
            .collect();
        let base = estimate(&samples, 0.32).unwrap();
        let bounded = estimate_with_upper_bounds(&samples, 0.32, &[(30, 2.5, 1.0)]).unwrap();
        assert_eq!(base, bounded);
    }
    #[test]
    fn a_reach_bound_constrains_uncertain_depth_with_continuity() {
        let samples: Vec<_> = (0..90)
            .map(|i| Sample {
                time: i as f64 / 30.0,
                estimate: 2.0,
                confidence: if (25..65).contains(&i) { 0.0 } else { 1.0 },
                scale: None,
            })
            .collect();
        let bounded = estimate_with_upper_bounds(&samples, 0.32, &[(45, 1.8, 0.8)]).unwrap();
        assert!(bounded[45] < 1.9);
        assert!((bounded[0] - 2.0).abs() < 0.05);
        assert!((bounded[89] - 2.0).abs() < 0.05);
        assert!(bounded.windows(2).all(|p| (p[1] - p[0]).abs() < 0.04));
    }
    #[test]
    fn rejects_an_isolated_depth_outlier_without_changing_constant_distance() {
        let mut s: Vec<_> = (0..90)
            .map(|i| Sample {
                time: i as f64 / 30.0,
                estimate: 2.0,
                confidence: 1.0,
                scale: None,
            })
            .collect();
        s[45].estimate = 3.0;
        let result = estimate(&s, 0.32).unwrap();
        assert!((result[45] - 2.0).abs() < 0.08);
    }
    #[test]
    fn a_moving_body_is_not_pinned_across_an_unobserved_profile_interval() {
        let s: Vec<_> = (0..90)
            .map(|i| Sample {
                time: i as f64 / 30.0,
                estimate: 2.4 - i as f64 / 90.0 * 0.6,
                confidence: if (25..65).contains(&i) { 0.0 } else { 1.0 },
                scale: None,
            })
            .collect();
        let result = estimate(&s, 0.32).unwrap();
        assert!(
            result
                .iter()
                .zip(&s)
                .all(|(a, b)| (a - b.estimate).abs() < 1e-4)
        );
    }
    #[test]
    fn actual_image_scale_preserves_forward_back_motion_without_phase_lag() {
        let truth: Vec<_> = (0..180)
            .map(|i| 2.0 + 0.1 * (i as f64 / 30.0 * std::f64::consts::TAU * 2.0).sin())
            .collect();
        let s: Vec<_> = truth
            .iter()
            .enumerate()
            .map(|(i, &z)| Sample {
                time: i as f64 / 30.0,
                estimate: z,
                confidence: 1.0,
                scale: (i > 0).then(|| (i - 1, truth[i - 1] / z)),
            })
            .collect();
        let result = estimate(&s, 0.32).unwrap();
        let error = result[30..150]
            .iter()
            .zip(&truth[30..150])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(error < 0.01, "error {error}");
    }
}
