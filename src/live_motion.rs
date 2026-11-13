//! Stillness filtering for live anatomical observations. Image evidence controls
//! whether uncertain monocular depth can move a limb. Holds never restore confidence.
use crate::pose::Keypoint;
use biomech_lite::live_fit::Observation;
#[derive(Clone)]
struct Arm {
    image: [[f32; 2]; 3],
    positions: [[f64; 3]; 3],
    mask: [bool; 3],
    time: f64,
}
#[derive(Default)]
pub struct LiveMotion {
    arms: [Option<Arm>; 2],
}
impl LiveMotion {
    pub fn apply(&mut self, obs: &mut Observation, pixels: &mut [Keypoint], height: i32) {
        let band = (height.max(1) as f32 / 720.0 * 5.0).max(2.0);
        for side in 0..2 {
            let names = if side == 0 {
                ["left_shoulder", "left_elbow", "left_wrist"]
            } else {
                ["right_shoulder", "right_elbow", "right_wrist"]
            };
            let coco = if side == 0 { [5, 7, 9] } else { [6, 8, 10] };
            if coco.iter().any(|&i| i >= pixels.len()) {
                continue;
            }
            let ids: Option<Vec<_>> = names
                .iter()
                .map(|name| obs.landmarks.iter().position(|p| p.name == *name))
                .collect();
            let Some(ids) = ids else {
                self.arms[side] = None;
                continue;
            };
            let mask = std::array::from_fn(|i| {
                obs.landmarks[ids[i]].confidence >= 0.3 && pixels[coco[i]].score >= 0.55
            });
            if !mask[0] || !mask[1] {
                continue;
            }
            let image = std::array::from_fn(|i| [pixels[coco[i]].x, pixels[coco[i]].y]);
            let positions = std::array::from_fn(|i| obs.landmarks[ids[i]].position);
            let t = obs.timestamp_seconds;
            let current = Arm {
                image,
                positions,
                mask,
                time: t,
            };
            let Some(mut old) = self.arms[side].clone() else {
                self.arms[side] = Some(current);
                continue;
            };
            if t <= old.time || t - old.time > 0.5 {
                self.arms[side] = Some(current);
                continue;
            }
            // A wrist confidence dropout must not reset the shoulder/elbow anchor.
            for i in 0..3 {
                if mask[i] && !old.mask[i] {
                    old.image[i] = image[i];
                    old.positions[i] = positions[i];
                }
            }
            old.mask = mask;
            // Compare to the retained anchor, not the previous noisy sample:
            // small deliberate motion accumulates and eventually leaves the band.
            let displacement = (0..3)
                .filter(|&i| mask[i])
                .map(|i| (image[i][0] - old.image[i][0]).hypot(image[i][1] - old.image[i][1]))
                .fold(0.0f32, f32::max);
            if displacement <= band {
                for i in 0..3 {
                    if mask[i] {
                        obs.landmarks[ids[i]].position = old.positions[i];
                        pixels[coco[i]].x = old.image[i][0];
                        pixels[coco[i]].y = old.image[i][1];
                    }
                }
                // Refresh observation age without moving the stationary anchor.
                old.time = t;
                self.arms[side] = Some(old);
            } else {
                // Faster motion gets less smoothing; no long frame queues.
                let dt = (t - old.time).clamp(1.0 / 240.0, 0.1);
                let speed = f64::from(displacement) / dt;
                let tau = 0.015 + 0.065 / (1.0 + speed / 80.0);
                let alpha = 1.0 - (-dt / tau).exp();
                let mut next = current;
                for i in 0..3 {
                    if mask[i] {
                        next.positions[i] = std::array::from_fn(|k| {
                            old.positions[i][k] + alpha * (positions[i][k] - old.positions[i][k])
                        });
                        obs.landmarks[ids[i]].position = next.positions[i];
                    }
                }
                self.arms[side] = Some(next);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use biomech_lite::live_fit::Landmark;
    fn input(t: f64, offset: f32, depth: f64) -> (Observation, Vec<Keypoint>) {
        let points = [
            (5, "left_shoulder"),
            (7, "left_elbow"),
            (9, "left_wrist"),
            (6, "right_shoulder"),
            (8, "right_elbow"),
            (10, "right_wrist"),
        ];
        let mut obs = Observation {
            timestamp_seconds: t,
            landmarks: vec![],
        };
        let mut pixels = vec![
            Keypoint {
                name: "joint",
                x: 0.0,
                y: 0.0,
                score: 0.9
            };
            17
        ];
        for (i, (c, name)) in points.iter().enumerate() {
            pixels[*c].x = 100.0 + i as f32 * 20.0 + offset;
            pixels[*c].y = 200.0 + i as f32 * 20.0;
            obs.landmarks.push(Landmark {
                name: name.to_string(),
                position: [i as f64 * 0.1, 0.2, depth],
                confidence: 0.9,
            });
        }
        (obs, pixels)
    }
    #[test]
    fn stationary_image_rejects_depth_jitter_without_promoting_missing_points() {
        let mut f = LiveMotion::default();
        let (mut o, mut p) = input(0.0, 0.0, 0.0);
        f.apply(&mut o, &mut p, 720);
        for i in 1..120 {
            let (mut o, mut p) = input(
                i as f64 / 30.0,
                (i as f32).sin() * 2.0,
                (i as f64).sin() * 0.08,
            );
            // Independent depth noise changes the raw forearm direction, rather
            // than merely translating the entire skeleton.
            o.landmarks[1].position[2] *= -1.0;
            o.landmarks[4].position[2] *= -0.5;
            f.apply(&mut o, &mut p, 720);
            assert!(o.landmarks.iter().all(|l| l.position[2] == 0.0));
            assert_eq!(p[7].x, 120.0);
        }
        let (mut o, mut p) = input(4.1, 0.0, 0.1);
        o.landmarks[2].confidence = 0.0;
        p[9].score = 0.0;
        f.apply(&mut o, &mut p, 720);
        assert_eq!(o.landmarks[2].confidence, 0.0);
        assert_eq!(p[9].score, 0.0);
    }
    #[test]
    fn wrist_dropout_does_not_reset_a_still_elbow() {
        let mut f = LiveMotion::default();
        let (mut o, mut p) = input(0.0, 0.0, 0.0);
        f.apply(&mut o, &mut p, 720);
        for i in 1..20 {
            let (mut o, mut p) = input(i as f64 / 30.0, 0.0, 0.08);
            if i % 2 == 0 {
                o.landmarks[2].confidence = 0.0;
                p[9].score = 0.0;
            }
            f.apply(&mut o, &mut p, 720);
            assert_eq!(o.landmarks[1].position[2], 0.0);
        }
    }
    #[test]
    fn deliberate_motion_leaves_stillness_band_and_other_arm_remains_independent() {
        let mut f = LiveMotion::default();
        let (mut o, mut p) = input(0.0, 0.0, 0.0);
        f.apply(&mut o, &mut p, 720);
        for i in 1..15 {
            let (mut o, mut p) = input(i as f64 / 30.0, 0.0, 0.0);
            p[7].x += i as f32 * 3.0;
            o.landmarks[1].position[0] += i as f64 * 0.01;
            o.landmarks[4].confidence = 0.0;
            p[8].score = 0.0;
            f.apply(&mut o, &mut p, 720);
            if i > 2 {
                assert!(o.landmarks[1].position[0] > 0.1);
            }
            assert_eq!(o.landmarks[4].confidence, 0.0);
        }
    }
}
