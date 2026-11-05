//! Conservative image-space feedback from the learned constant-length 3D skeleton.
//! Raw neural predictions remain separate and are the only calibration inputs.
use crate::{
    angles::{self, JointAngle},
    landmarks::BONES,
    pose::{self, Bbox, Keypoint},
    twin::TwinPreview,
};
use anyhow::Result;
use opencv::{
    core::{Mat, Point, Scalar, Size},
    imgproc,
};
use serde::Serialize;

// H36M -> COCO body indices. The synthesized H36M nose is deliberately excluded.
const BODY: [(usize, usize); 12] = [
    (11, 5),
    (14, 6),
    (12, 7),
    (15, 8),
    (13, 9),
    (16, 10),
    (4, 11),
    (1, 12),
    (5, 13),
    (2, 14),
    (6, 15),
    (3, 16),
];
const MAX_MISSING: u8 = 5;

#[derive(Serialize)]
pub struct Adjustment {
    pub joint: &'static str,
    pub kind: &'static str,
    pub missing_frames: u8,
}
#[derive(Serialize)]
pub struct TrackingResult {
    pub dimension_prior_active: bool,
    /// Confidence is never increased; inferred positions receive zero confidence.
    pub keypoints: Vec<Keypoint>,
    pub angles: Vec<JointAngle>,
    pub adjustments: Vec<Adjustment>,
    pub next_crop: Option<Bbox>,
}
#[derive(Default)]
pub struct DimensionTracker {
    upper_body: bool,
    subject: Option<usize>,
    missing: [u8; 17],
}

fn in_frame(x: f32, y: f32, size: Size) -> bool {
    x.is_finite()
        && y.is_finite()
        && x >= 0.0
        && y >= 0.0
        && x < size.width as f32
        && y < size.height as f32
}
fn reliable(p: &Keypoint, threshold: f32, size: Size) -> bool {
    p.score.is_finite() && p.score > 0.0 && p.score >= threshold && in_frame(p.x, p.y, size)
}

impl DimensionTracker {
    pub fn set_upper_body(&mut self, upper: bool) {
        self.upper_body = upper;
        self.missing = [0; 17];
    }
    pub fn refine(
        &mut self,
        raw: &[Keypoint],
        twin: Option<&TwinPreview>,
        bbox: Bbox,
        size: Size,
        threshold: f32,
    ) -> TrackingResult {
        let projection = twin.and_then(|t| t.tracking_projection(size));
        let subject = twin.map(|t| t.subject_segment());
        if self.subject != subject || projection.is_none() {
            self.missing = [0; 17];
            self.subject = subject;
        }
        self.refine_projection(raw, projection, bbox, size, threshold)
    }

    fn refine_projection(
        &mut self,
        raw: &[Keypoint],
        projection: Option<[[f32; 2]; 17]>,
        bbox: Bbox,
        size: Size,
        threshold: f32,
    ) -> TrackingResult {
        let mut result = TrackingResult {
            dimension_prior_active: false,
            keypoints: raw.to_vec(),
            angles: Vec::new(),
            adjustments: Vec::new(),
            next_crop: None,
        };
        let projection = projection.filter(|p| {
            projection_agrees_with_torso(raw, p, bbox, size, threshold, self.upper_body)
        });
        if let Some(projected) = projection {
            result.dimension_prior_active = true;
            for (h, c) in BODY {
                if self.upper_body && !crate::scope::coco_upper(c) {
                    continue;
                }
                let Some(p) = raw.get(c) else {
                    continue;
                };
                let target = projected[h];
                if !in_frame(target[0], target[1], size) {
                    self.missing[c] = 0;
                    continue;
                }
                if reliable(p, threshold, size) {
                    self.missing[c] = 0;
                    // Strong image evidence wins. Only uncertain observations are blended.
                    let strong = threshold.max(0.75);
                    let amount =
                        ((strong - p.score) / (strong - threshold).max(0.01)).clamp(0.0, 0.65);
                    let distance = (p.x - target[0]).hypot(p.y - target[1]);
                    if amount > 0.01 && distance <= bbox.height * 0.2 && distance > 0.5 {
                        result.keypoints[c].x = p.x + (target[0] - p.x) * amount;
                        result.keypoints[c].y = p.y + (target[1] - p.y) * amount;
                        result.adjustments.push(Adjustment {
                            joint: p.name,
                            kind: "dimension_blend",
                            missing_frames: 0,
                        });
                    }
                } else {
                    self.missing[c] = self.missing[c].saturating_add(1);
                    if self.missing[c] <= MAX_MISSING {
                        result.keypoints[c].x = target[0];
                        result.keypoints[c].y = target[1];
                        // Never turn an inferred location into a confident observation/angle.
                        result.keypoints[c].score = 0.0;
                        result.adjustments.push(Adjustment {
                            joint: p.name,
                            kind: "inferred",
                            missing_frames: self.missing[c],
                        });
                    }
                }
            }
            result.next_crop =
                following_crop(raw, &projected, bbox, size, threshold, self.upper_body);
        } else {
            self.missing = [0; 17];
        }
        result.angles =
            angles::calculate_scoped(&result.keypoints, threshold, size, self.upper_body);
        result
    }
}

fn projection_agrees_with_torso(
    raw: &[Keypoint],
    p: &[[f32; 2]; 17],
    bbox: Bbox,
    size: Size,
    threshold: f32,
    upper_body: bool,
) -> bool {
    let mut errors = Vec::new();
    let anchors = if upper_body {
        &[(11, 5), (14, 6), (12, 7), (15, 8)][..]
    } else {
        &[(11, 5), (14, 6), (4, 11), (1, 12)][..]
    };
    for &(h, c) in anchors {
        if let Some(raw) = raw
            .get(c)
            .filter(|point| reliable(point, threshold.max(0.6), size))
        {
            errors.push((raw.x - p[h][0]).hypot(raw.y - p[h][1]));
        }
    }
    if errors.len() < if upper_body { 2 } else { 3 } || bbox.width <= 0.0 || bbox.height <= 0.0 {
        return false;
    }
    errors.sort_by(f32::total_cmp);
    // Reject a bad projection rather than forcing the observations onto it.
    errors[errors.len() / 2] <= bbox.height * 0.08
}

fn following_crop(
    raw: &[Keypoint],
    projected: &[[f32; 2]; 17],
    old: Bbox,
    size: Size,
    threshold: f32,
    upper_body: bool,
) -> Option<Bbox> {
    let mut coordinates: Vec<_> = projected
        .iter()
        .enumerate()
        .filter(|(i, _)| !upper_body || crate::scope::body_upper(*i))
        .map(|(_, p)| *p)
        .filter(|p| in_frame(p[0], p[1], size))
        .collect();
    for (index, p) in raw.iter().enumerate() {
        // Preserve observed head/hand/foot reach; detailed face points do not drive the crop.
        if (upper_body && !crate::scope::coco_upper(index))
            || (23..91).contains(&index)
            || !reliable(p, threshold.max(0.65), size)
        {
            continue;
        }
        if p.x >= old.x - old.width * 0.25
            && p.x <= old.x + old.width * 1.25
            && p.y >= old.y - old.height * 0.25
            && p.y <= old.y + old.height * 1.25
        {
            coordinates.push([p.x, p.y]);
        }
    }
    if coordinates.len() < 4 {
        return None;
    }
    let minx = coordinates
        .iter()
        .map(|p| p[0])
        .fold(f32::INFINITY, f32::min);
    let maxx = coordinates
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let miny = coordinates
        .iter()
        .map(|p| p[1])
        .fold(f32::INFINITY, f32::min);
    let maxy = coordinates
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max);
    let target_width = ((maxx - minx) * 1.3).clamp(old.width * 0.8, old.width * 1.3);
    let target_height = ((maxy - miny) * 1.25).clamp(old.height * 0.85, old.height * 1.3);
    let width = (old.width * 0.8 + target_width * 0.2).clamp(1.0, size.width as f32);
    let height = (old.height * 0.8 + target_height * 0.2).clamp(1.0, size.height as f32);
    let cx = old.x + old.width / 2.0;
    let cy = old.y + old.height / 2.0;
    let cx = cx + (((minx + maxx) / 2.0 - cx) * 0.3).clamp(-old.width * 0.08, old.width * 0.08);
    let cy = cy + (((miny + maxy) / 2.0 - cy) * 0.3).clamp(-old.height * 0.08, old.height * 0.08);
    let x = (cx - width / 2.0).clamp(0.0, size.width as f32 - width);
    let y = (cy - height / 2.0).clamp(0.0, size.height as f32 - height);
    Some(Bbox {
        x,
        y,
        width,
        height,
    })
}

pub fn draw(image: &mut Mat, result: &TrackingResult, threshold: f32) -> Result<()> {
    pose::draw(image, &result.keypoints, threshold)?;
    let inferred = |i: usize| {
        result
            .adjustments
            .iter()
            .any(|a| a.kind == "inferred" && a.joint == result.keypoints[i].name)
    };
    let point = |i: usize| {
        Point::new(
            result.keypoints[i].x.round() as i32,
            result.keypoints[i].y.round() as i32,
        )
    };
    let color = Scalar::new(160.0, 160.0, 160.0, 0.0);
    for &(a, b) in BONES {
        if a >= result.keypoints.len() || b >= result.keypoints.len() {
            continue;
        }
        let valid = |i: usize| {
            inferred(i)
                || (result.keypoints[i].score > 0.0 && result.keypoints[i].score >= threshold)
        };
        if (inferred(a) || inferred(b)) && valid(a) && valid(b) {
            imgproc::line(image, point(a), point(b), color, 1, imgproc::LINE_AA, 0)?;
        }
    }
    for (i, _) in result.keypoints.iter().enumerate() {
        if inferred(i) {
            imgproc::circle(image, point(i), 5, color, 1, imgproc::LINE_AA, 0)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Vec<Keypoint>, [[f32; 2]; 17], Bbox, Size) {
        let p = std::array::from_fn(|i| [170.0 + (i % 3) as f32 * 30.0, 100.0 + i as f32 * 10.0]);
        let mut raw = vec![
            Keypoint {
                name: "unused",
                x: 200.0,
                y: 200.0,
                score: 0.95
            };
            17
        ];
        for (h, c) in BODY {
            raw[c] = Keypoint {
                name: crate::landmarks::NAMES[c],
                x: p[h][0],
                y: p[h][1],
                score: 0.95,
            };
        }
        (
            raw,
            p,
            Bbox {
                x: 100.0,
                y: 50.0,
                width: 200.0,
                height: 350.0,
            },
            Size::new(640, 480),
        )
    }
    #[test]
    fn dimension_prior_reduces_uncertain_landmark_noise_and_preserves_strong_observations() {
        let (mut raw, p, bbox, size) = fixture();
        raw[13].x += 20.0;
        raw[13].score = 0.4;
        raw[7].x += 12.0; // Strong evidence must remain unchanged.
        let result = DimensionTracker::default().refine_projection(&raw, Some(p), bbox, size, 0.3);
        assert!(result.dimension_prior_active);
        assert!((result.keypoints[13].x - p[5][0]).abs() < (raw[13].x - p[5][0]).abs());
        assert_eq!(result.keypoints[7].x, raw[7].x);
        assert_eq!(result.keypoints[13].score, raw[13].score);
        assert_eq!(raw[13].x, p[5][0] + 20.0); // Raw evidence is untouched.
    }
    #[test]
    fn inferred_points_expire_and_never_gain_measurement_confidence() {
        let (mut raw, p, bbox, size) = fixture();
        raw[13].score = 0.0;
        raw[13].x = -1.0;
        raw[13].y = -1.0;
        let mut tracker = DimensionTracker::default();
        for frame in 1..=6 {
            let result = tracker.refine_projection(&raw, Some(p), bbox, size, 0.3);
            assert_eq!(result.keypoints[13].score, 0.0);
            if frame <= 5 {
                assert_eq!(result.keypoints[13].x, p[5][0]);
                assert!(
                    result
                        .adjustments
                        .iter()
                        .any(|a| a.kind == "inferred" && a.missing_frames == frame)
                );
            } else {
                assert_eq!(result.keypoints[13].x, -1.0);
            }
            assert!(
                result
                    .angles
                    .iter()
                    .find(|a| a.joint == "left_knee")
                    .unwrap()
                    .degrees
                    .is_none()
            );
        }
    }
    #[test]
    fn bad_projection_is_rejected_and_crop_follow_is_bounded() {
        let (raw, p, bbox, size) = fixture();
        let mut bad = p;
        for p in &mut bad {
            p[0] += 200.0;
        }
        let result =
            DimensionTracker::default().refine_projection(&raw, Some(bad), bbox, size, 0.3);
        assert!(!result.dimension_prior_active && result.next_crop.is_none());
        let crop = following_crop(&raw, &p, bbox, size, 0.3, false).unwrap();
        assert!(
            crop.x >= 0.0
                && crop.y >= 0.0
                && crop.x + crop.width <= 640.0
                && crop.y + crop.height <= 480.0
        );
        assert!(
            ((crop.x + crop.width / 2.0) - (bbox.x + bbox.width / 2.0)).abs()
                <= bbox.width * 0.08 + 1e-4
        );
        assert!(
            ((crop.y + crop.height / 2.0) - (bbox.y + bbox.height / 2.0)).abs()
                <= bbox.height * 0.08 + 1e-4
        );
    }
}
