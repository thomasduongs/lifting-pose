//! Robust latent torso heading. Never averages observed arm coordinates.
use crate::depth_path::{Factor, solve};
pub struct Sample {
    pub time: f64,
    pub heading: f64,
    pub confidence: f64,
}
pub fn estimate(samples: &[Sample]) -> Option<Vec<f64>> {
    if samples.len() < 3
        || samples
            .iter()
            .any(|s| !s.time.is_finite() || !s.heading.is_finite() || !s.confidence.is_finite())
        || samples.windows(2).any(|w| w[1].time <= w[0].time)
    {
        return None;
    }
    let mut measurements = Vec::with_capacity(samples.len());
    let mut unwrapped = samples[0].heading;
    measurements.push(unwrapped);
    for w in samples.windows(2) {
        let d = w[1].heading - w[0].heading;
        unwrapped += d.sin().atan2(d.cos());
        measurements.push(unwrapped);
    }
    let mut x = measurements.clone();
    for _ in 0..8 {
        let mut factors = Vec::new();
        for (i, s) in samples.iter().enumerate() {
            // Foreshortened shoulders provide weaker orientation evidence.
            let sigma = (8.0 + 12.0 * (1.0 - s.heading.cos().abs())).to_radians();
            let robust = (2.0 * sigma / (x[i] - measurements[i]).abs().max(2.0 * sigma)).min(1.0);
            factors.push(Factor::new(
                &[(i, 1.0)],
                measurements[i],
                (s.confidence.clamp(0.0, 1.0) * robust + 1e-4) / sigma.powi(2),
            ));
        }
        for i in 1..samples.len() - 1 {
            let a = samples[i].time - samples[i - 1].time;
            let b = samples[i + 1].time - samples[i].time;
            // Separate takes/long gaps have no continuous acceleration evidence.
            if a > 0.5 || b > 0.5 {
                continue;
            }
            let c = 2.0 / (a + b);
            factors.push(Factor::new(
                &[
                    (i - 1, c / a),
                    (i, -c * (1.0 / a + 1.0 / b)),
                    (i + 1, c / b),
                ],
                0.0,
                (a + b) / 2.0 / 10.0_f64.powi(2),
            ));
        }
        solve(&factors, &mut x);
    }
    x.iter().all(|v| v.is_finite()).then_some(x)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuous_full_turn_crosses_angle_wrap_without_lag() {
        let truth: Vec<_> = (0..100)
            .map(|i| i as f64 * std::f64::consts::TAU / 99.0)
            .collect();
        let samples: Vec<_> = truth
            .iter()
            .enumerate()
            .map(|(i, &a)| Sample {
                time: i as f64 / 30.0,
                heading: a.sin().atan2(a.cos()),
                confidence: 1.0,
            })
            .collect();
        let fit = estimate(&samples).unwrap();
        assert!(fit.iter().zip(truth).all(|(a, b)| (a - b).abs() < 1e-5));
    }
    #[test]
    fn rejects_single_frame_profile_spike_but_keeps_fast_turn() {
        let truth: Vec<_> = (0..120)
            .map(|i| {
                std::f64::consts::FRAC_PI_2
                    + (i as f64 - 45.0).clamp(0.0, 18.0) / 18.0 * std::f64::consts::PI
            })
            .collect();
        let mut samples: Vec<_> = truth
            .iter()
            .enumerate()
            .map(|(i, &a)| Sample {
                time: i as f64 / 30.0,
                heading: a,
                confidence: 1.0,
            })
            .collect();
        samples[25].heading += 0.8;
        let fit = estimate(&samples).unwrap();
        assert!(
            (fit[25] - truth[25]).abs() < 0.3,
            "spike {}",
            fit[25] - truth[25]
        );
        assert!((fit[54] - truth[54]).abs() < 0.01);
        assert!(
            fit.iter()
                .zip(truth)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
                < 0.3
        );
    }
}
