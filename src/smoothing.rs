//! Time-based adaptive filters: suppress stationary jitter, open up during motion,
//! and reset after gaps. Filtering never changes observation confidence.
use nalgebra::{UnitQuaternion, Vector3};
type V = Vector3<f64>;
pub fn alpha(cutoff: f64, dt: f64) -> f64 {
    1.0 / (1.0 + 1.0 / (std::f64::consts::TAU * cutoff * dt))
}
#[derive(Clone, Default)]
pub struct AdaptivePoint {
    time: Option<f64>,
    value: V,
    input: V,
    velocity: V,
}
impl AdaptivePoint {
    pub fn filter(&mut self, input: V, time: f64, scale: f64, max_speed: f64, beta: f64) -> V {
        let dt = self.time.map(|t| (time - t).abs()).unwrap_or(0.0);
        if self.time.is_none() || dt > 0.5 || dt < 1e-6 {
            self.time = Some(time);
            self.value = input;
            self.input = input;
            self.velocity = V::zeros();
            return input;
        }
        let delta = input - self.value;
        let input = self.value + delta * (max_speed * dt / delta.norm().max(1e-9)).min(1.0);
        let derivative = (input - self.input) / dt;
        self.velocity += (derivative - self.velocity) * alpha(1.0, dt);
        let cutoff = 2.5 + beta * self.velocity.norm() / scale.max(1e-6);
        self.value += (input - self.value) * alpha(cutoff, dt);
        self.input = input;
        self.time = Some(time);
        self.value
    }
}
#[derive(Default)]
pub struct AdaptiveRotation {
    time: Option<f64>,
    value: UnitQuaternion<f64>,
    input: UnitQuaternion<f64>,
    speed: f64,
}
impl AdaptiveRotation {
    pub fn filter(&mut self, input: UnitQuaternion<f64>, time: f64) -> UnitQuaternion<f64> {
        let dt = self.time.map(|t| (time - t).abs()).unwrap_or(0.0);
        if self.time.is_none() || dt > 0.5 || dt < 1e-6 {
            self.time = Some(time);
            self.value = input;
            self.input = input;
            self.speed = 0.0;
            return input;
        }
        self.speed += ((self.input.inverse() * input).angle() / dt - self.speed) * alpha(1.0, dt);
        self.value = self.value.slerp(&input, alpha(2.5 + 1.5 * self.speed, dt));
        self.time = Some(time);
        self.input = input;
        self.value
    }
}

#[derive(Default)]
pub struct LandmarkSmoother {
    filters: Vec<AdaptivePoint>,
}
impl LandmarkSmoother {
    pub fn apply(&mut self, points: &mut [crate::pose::Keypoint], time: f64, width: f64) {
        self.filters
            .resize_with(points.len(), AdaptivePoint::default);
        for (filter, point) in self.filters.iter_mut().zip(points) {
            if point.score < 0.3 || point.x < 0.0 || point.y < 0.0 {
                continue;
            }
            let p = filter.filter(
                V::new(point.x as f64, point.y as f64, 0.0),
                time,
                width,
                width * 4.0,
                15.0,
            );
            point.x = p.x as f32;
            point.y = p.y as f32;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reduces_stationary_jitter_and_follows_motion_without_frame_rate_dependence() {
        let mut filter = AdaptivePoint::default();
        let mut raw = 0.0;
        let mut smooth = 0.0;
        for i in 0..120 {
            let noise = if i % 2 == 0 { 0.01 } else { -0.01 };
            let p = filter.filter(V::new(noise, 0.0, 0.0), i as f64 / 30.0, 0.35, 7.0, 1.5);
            if i > 30 {
                raw += noise * noise;
                smooth += p.x * p.x;
            }
        }
        assert!(smooth < raw * 0.2);
        for fps in [15, 30, 60] {
            let mut filter = AdaptivePoint::default();
            let mut p = V::zeros();
            for i in 0..fps {
                let time = i as f64 / fps as f64;
                p = filter.filter(V::new(time, 0.0, 0.0), time, 0.35, 7.0, 1.5);
            }
            assert!((p.x - (fps - 1) as f64 / fps as f64).abs() < 0.06);
            assert_eq!(
                filter.filter(V::repeat(2.0), 3.0, 0.35, 7.0, 1.5),
                V::repeat(2.0)
            );
        }
    }
}
