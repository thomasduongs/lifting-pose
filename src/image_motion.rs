//! Independent image-motion evidence for detecting body landmark outliers.
//! No coordinate averaging: accept the CNN point or a validated current-frame flow point.
use crate::pose::Keypoint;
use anyhow::Result;
use opencv::{
    core::{Mat, Point2f, Size, TermCriteria, TermCriteria_Type, Vector},
    imgproc,
    prelude::*,
    video,
};

fn correspondence_roundtrip(
    index: usize,
    fb: f32,
    error: f32,
    back_error: f32,
    palm_span: f32,
) -> bool {
    fb < 1.0
        || (matches!(index, 9 | 10)
            && error < 6.0
            && back_error < 6.0
            && fb < (palm_span * 0.1).min(3.5))
}

#[derive(Default)]
pub struct ImageMotion {
    gray: Option<Mat>,
    points: Vec<Keypoint>,
    detected: Vec<Keypoint>,
    wrist_supported: [bool; 2],
    wrist_validated: [bool; 2],
    wrist_steps: [u8; 2],
    wrist_span: [f32; 2],
    hand_wrist: [Option<Keypoint>; 2],
    pub torso_scale: Option<f64>,
    pub torso_tracks: usize,
    pub torso_reference_reset: bool,
    pub torso_span: usize,
    torso_gray: Option<Mat>,
    torso_points: Vector<Point2f>,
    torso_age: usize,
}
impl ImageMotion {
    /// Validate wrist identity against independently inferred hand landmarks.
    /// A high neural body score alone cannot distinguish a cuff from a shirt.
    pub fn confirm_hands(&mut self, hands: &[crate::details::HandPose]) {
        self.wrist_supported = self.wrist_validated;
        let previous = self.hand_wrist;
        self.hand_wrist = [None; 2];
        for hand in hands.iter().filter(|h| h.confidence >= 0.65) {
            let side = if hand.side == "left" {
                0
            } else if hand.side == "right" {
                1
            } else {
                continue;
            };
            if let Some(point) = self.points.get_mut(9 + side) {
                let p = &hand.image_landmarks;
                let span = (p[5][0] - p[17][0]).hypot(p[5][1] - p[17][1]);
                let measured = Keypoint {
                    name: point.name,
                    x: p[0][0],
                    y: p[0][1],
                    score: hand.confidence,
                };
                self.hand_wrist[side] = Some(measured);
                let consistent = previous[side].is_some_and(|old| {
                    (old.x - measured.x).hypot(old.y - measured.y) <= (span * 2.0).max(40.0)
                });
                if hand.confidence >= 0.85 || consistent {
                    // Track the independently observed hand, not the body CNN's
                    // sometimes displaced wrist on a sleeve or torso.
                    *point = measured;
                    self.wrist_supported[side] = true;
                    self.wrist_steps[side] = 0;
                    self.wrist_span[side] = span
                        .max((p[0][0] - p[9][0]).hypot(p[0][1] - p[9][1]))
                        .max(8.0);
                }
            }
        }
    }
    pub fn apply(&mut self, image: &Mat, points: &mut [Keypoint]) -> Result<Vec<usize>> {
        let mut gray = Mat::default();
        imgproc::cvt_color_def(image, &mut gray, imgproc::COLOR_BGR2GRAY)?;
        let mut corrected = Vec::new();
        self.wrist_validated = [false; 2];
        self.torso_scale = None;
        self.torso_tracks = 0;
        self.torso_reference_reset = false;
        self.torso_span = 0;
        let detected = points.to_vec();
        if let Some(previous) = &self.gray {
            if previous.size()? == gray.size()? && self.points.len() >= 17 && points.len() >= 17 {
                let mut indices = Vec::new();
                let mut before = Vector::<Point2f>::new();
                for i in [5, 6, 7, 8, 9, 10] {
                    if i >= 9 && (!self.wrist_supported[i - 9] || self.wrist_steps[i - 9] >= 2) {
                        continue;
                    }
                    let a = self.points[i];
                    let b = points[i];
                    let wrist = i >= 9;
                    if a.score >= if wrist { 0.65 } else { 0.75 }
                        && b.score >= if wrist { 0.3 } else { 0.75 }
                        && a.x >= 0.0
                        && a.y >= 0.0
                        && b.x >= 0.0
                        && b.y >= 0.0
                        && (wrist || (a.x - b.x).hypot(a.y - b.y) < image.cols() as f32 * 0.15)
                    {
                        indices.push(i);
                        before.push(Point2f::new(a.x, a.y));
                    }
                }
                if !before.is_empty() {
                    let mut after = Vector::<Point2f>::new();
                    let mut status = Vector::<u8>::new();
                    let mut error = Vector::<f32>::new();
                    let criteria = TermCriteria::new(
                        TermCriteria_Type::COUNT as i32 | TermCriteria_Type::EPS as i32,
                        30,
                        0.01,
                    )?;
                    video::calc_optical_flow_pyr_lk(
                        previous,
                        &gray,
                        &before,
                        &mut after,
                        &mut status,
                        &mut error,
                        Size::new(31, 31),
                        3,
                        criteria,
                        0,
                        1e-5,
                    )?;
                    let mut backward = Vector::<Point2f>::new();
                    let mut back_status = Vector::<u8>::new();
                    let mut back_error = Vector::<f32>::new();
                    video::calc_optical_flow_pyr_lk(
                        &gray,
                        previous,
                        &after,
                        &mut backward,
                        &mut back_status,
                        &mut back_error,
                        Size::new(31, 31),
                        3,
                        criteria,
                        0,
                        1e-5,
                    )?;
                    for (j, &index) in indices.iter().enumerate() {
                        let start = before.get(j)?;
                        let end = after.get(j)?;
                        let back = backward.get(j)?;
                        let fb = (start.x - back.x).hypot(start.y - back.y);
                        let disagreement = (points[index].x - end.x).hypot(points[index].y - end.y);
                        // Gate a detector innovation, not a persistent semantic
                        // offset: this prevents tracking a shirt patch indefinitely.
                        let threshold = (image.cols() as f32 * 0.0125).max(6.0);
                        let detector_jump = self
                            .detected
                            .get(index)
                            .map(|old| (points[index].x - old.x).hypot(points[index].y - old.y))
                            .unwrap_or(0.0);
                        if status.get(j)? != 0
                            && back_status.get(j)? != 0
                            && error.get(j)? < 12.0
                            && back_error.get(j)? < 12.0
                            && correspondence_roundtrip(
                                index,
                                fb,
                                error.get(j)?,
                                back_error.get(j)?,
                                if index >= 9 {
                                    self.wrist_span[index - 9]
                                } else {
                                    0.0
                                },
                            )
                            && end.x >= 0.0
                            && end.y >= 0.0
                            && end.x < image.cols() as f32
                            && end.y < image.rows() as f32
                            && (end.x - start.x).hypot(end.y - start.y) < image.cols() as f32 * 0.15
                            && detector_jump > threshold
                            && disagreement > threshold
                        {
                            points[index].x = end.x;
                            points[index].y = end.y;
                            points[index].score =
                                points[index].score.min(self.points[index].score).min(0.9);
                            corrected.push(index);
                            if index >= 9 {
                                self.wrist_validated[index - 9] = true;
                                self.wrist_steps[index - 9] += 1;
                            }
                        }
                    }
                }
            }
        }
        self.update_torso(&gray, points)?;
        self.gray = Some(gray);
        self.points = points.to_vec();
        self.detected = detected;
        // A confirmed hand can support at most two consecutive, independently
        // validated image correspondences. Any failed step ends that chain.
        self.wrist_supported = self.wrist_validated;
        Ok(corrected)
    }
    fn update_torso(&mut self, gray: &Mat, points: &[Keypoint]) -> Result<()> {
        let mut changed_size = false;
        if let Some(reference) = &self.torso_gray {
            changed_size = reference.size()? != gray.size()?;
            if !changed_size && self.torso_points.len() >= 12 {
                let mut after = Vector::<Point2f>::new();
                let mut status = Vector::<u8>::new();
                let mut error = Vector::<f32>::new();
                let criteria = TermCriteria::new(
                    TermCriteria_Type::COUNT as i32 | TermCriteria_Type::EPS as i32,
                    30,
                    0.01,
                )?;
                video::calc_optical_flow_pyr_lk(
                    reference,
                    gray,
                    &self.torso_points,
                    &mut after,
                    &mut status,
                    &mut error,
                    Size::new(31, 31),
                    3,
                    criteria,
                    0,
                    1e-5,
                )?;
                let mut backward = Vector::<Point2f>::new();
                let mut back_status = Vector::<u8>::new();
                let mut back_error = Vector::<f32>::new();
                video::calc_optical_flow_pyr_lk(
                    gray,
                    reference,
                    &after,
                    &mut backward,
                    &mut back_status,
                    &mut back_error,
                    Size::new(31, 31),
                    3,
                    criteria,
                    0,
                    1e-5,
                )?;
                let mut pairs = Vec::new();
                for j in 0..self.torso_points.len() {
                    let a = self.torso_points.get(j)?;
                    let b = after.get(j)?;
                    let back = backward.get(j)?;
                    if status.get(j)? != 0
                        && back_status.get(j)? != 0
                        && error.get(j)? < 10.0
                        && back_error.get(j)? < 10.0
                        && (a.x - back.x).hypot(a.y - back.y) < 0.75
                    {
                        pairs.push(((a.x as f64, a.y as f64), (b.x as f64, b.y as f64)));
                    }
                }
                self.torso_tracks = pairs.len();
                self.torso_scale = vertical_scale(&pairs);
                if self.torso_scale.is_some() {
                    self.torso_span = self.torso_age + 1;
                }
            }
        }
        // Keep the last trustworthy reference across a short failed observation;
        // the next scale then covers the full gap, rather than losing movement.
        if self.torso_gray.is_none()
            || changed_size
            || self.torso_scale.is_some()
            || self.torso_age >= 10
            || self.torso_points.len() < 12
        {
            self.torso_reference_reset = self.torso_scale.is_none();
            self.torso_gray = Some(gray.try_clone()?);
            self.torso_points = torso_grid(points, gray.size()?);
            self.torso_age = 0;
        } else {
            self.torso_age += 1;
        }
        Ok(())
    }
}

fn torso_grid(points: &[Keypoint], size: Size) -> Vector<Point2f> {
    let mut grid = Vector::new();
    if points.len() < 17 || points[5].score.min(points[6].score) < 0.6 {
        return grid;
    }
    let center_x = (points[5].x + points[6].x) / 2.0;
    let top = (points[5].y + points[6].y) / 2.0 + 25.0;
    let width = (points[5].x - points[6].x)
        .abs()
        .max(size.width as f32 * 0.07);
    let bottom = if points[11].score.min(points[12].score) >= 0.6 {
        (points[11].y + points[12].y) / 2.0 - 20.0
    } else {
        size.height as f32 - 35.0
    };
    if bottom - top >= 100.0 {
        for row in 0..10 {
            for col in 0..7 {
                let x = center_x + (col as f32 / 6.0 - 0.5) * width * 0.85;
                let y = top + (bottom - top) * row as f32 / 9.0;
                if x >= 15.0
                    && x < size.width as f32 - 15.0
                    && y >= 15.0
                    && y < size.height as f32 - 15.0
                {
                    grid.push(Point2f::new(x, y));
                }
            }
        }
    }
    grid
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
/// Robust affine scale of actual shirt features along the upright torso axis.
/// Reject nonrigid/occluded patches instead of using them as depth observations.
fn vertical_scale(pairs: &[((f64, f64), (f64, f64))]) -> Option<f64> {
    use nalgebra::{Matrix4, Vector4};
    if pairs.len() < 12 {
        return None;
    }
    let center_x = pairs.iter().map(|p| p.0.0).sum::<f64>() / pairs.len() as f64;
    let center_y = pairs.iter().map(|p| p.0.1).sum::<f64>() / pairs.len() as f64;
    let min_y = pairs.iter().map(|p| p.0.1).fold(f64::INFINITY, f64::min);
    let max_y = pairs
        .iter()
        .map(|p| p.0.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_x = pairs.iter().map(|p| p.0.0).fold(f64::INFINITY, f64::min);
    let max_x = pairs
        .iter()
        .map(|p| p.0.0)
        .fold(f64::NEG_INFINITY, f64::max);
    if max_y - min_y < 100.0 || max_x - min_x < 30.0 {
        return None;
    }
    let samples: Vec<_> = pairs
        .iter()
        .map(|(a, b)| {
            let x = (a.0 - center_x) / 100.0;
            let y = (a.1 - center_y) / 100.0;
            (Vector4::new(x, y, x * y, 1.0), b.1 - a.1)
        })
        .collect();
    let mut fit = Vector4::zeros();
    for iteration in 0..5 {
        let mut a = Matrix4::identity() * 1e-6;
        let mut b = Vector4::zeros();
        for (row, target) in &samples {
            let residual = (target - row.dot(&fit)).abs();
            let weight = if iteration == 0 {
                1.0
            } else {
                (1.5 / residual.max(1.5)).min(1.0)
            };
            a += weight * row * row.transpose();
            b += weight * row * *target;
        }
        fit = a.lu().solve(&b)?;
    }
    // Upright-body vertical scale at the torso center. The x and xy terms
    // absorb perspective shear during a turn instead of inventing depth motion.
    let scale = 1.0 + fit[1] / 100.0;
    if !scale.is_finite() || !(0.8..1.25).contains(&scale) {
        return None;
    }
    let mut errors: Vec<_> = samples
        .iter()
        .map(|(row, target)| (target - row.dot(&fit)).abs())
        .collect();
    let mid = median(&mut errors);
    if mid > 1.5 || errors[errors.len() * 9 / 10] > 4.0 {
        return None;
    }
    Some(scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{self, Scalar, Size};
    #[test]
    fn small_wrist_roundtrip_error_requires_an_independent_palm_scale_and_low_residuals() {
        assert!(correspondence_roundtrip(10, 3.1, 5.1, 4.9, 80.0));
        assert!(!correspondence_roundtrip(7, 3.1, 5.1, 4.9, 80.0));
        assert!(!correspondence_roundtrip(10, 3.1, 7.0, 4.9, 80.0));
        assert!(!correspondence_roundtrip(10, 3.1, 5.1, 4.9, 20.0));
        assert!(!correspondence_roundtrip(10, 3.6, 5.1, 4.9, 80.0));
        assert!(correspondence_roundtrip(7, 0.9, 10.0, 10.0, 0.0));
    }
    fn fixture() -> (Mat, Vec<Keypoint>) {
        let mut image =
            Mat::new_rows_cols_with_default(160, 160, core::CV_8UC3, Scalar::all(0.0)).unwrap();
        for y in 0..160 {
            for x in 0..160 {
                let v = ((x / 4 * 73 + y / 4 * 137 + x / 4 * y / 4 * 19) % 255) as u8;
                *image.at_2d_mut::<core::Vec3b>(y, x).unwrap() = core::Vec3b::from([v, v, v]);
            }
        }
        let mut points = vec![
            Keypoint {
                name: "test",
                x: -1.0,
                y: -1.0,
                score: 0.0
            };
            17
        ];
        points[5] = Keypoint {
            name: "left",
            x: 50.0,
            y: 60.0,
            score: 0.99,
        };
        points[6] = Keypoint {
            name: "right",
            x: 105.0,
            y: 65.0,
            score: 0.99,
        };
        (image, points)
    }
    #[test]
    fn torso_reference_spans_a_missing_image_observation() -> Result<()> {
        let (small, mut points) = fixture();
        let mut image = Mat::default();
        imgproc::resize_def(&small, &mut image, Size::new(320, 320))?;
        for p in &mut points {
            p.x *= 2.0;
            p.y *= 2.0;
        }
        let mut state = ImageMotion::default();
        state.apply(&image, &mut points)?;
        let blank = Mat::new_rows_cols_with_default(320, 320, core::CV_8UC3, Scalar::all(80.0))?;
        let mut missing = points.clone();
        missing[5].score = 0.1;
        missing[6].score = 0.1;
        state.apply(&blank, &mut missing)?;
        assert!(state.torso_scale.is_none());
        let mut moved = Mat::default();
        let affine = Mat::from_slice_2d(&[[1.0, 0.0, 0.0], [0.0, 1.02, 3.0]])?;
        imgproc::warp_affine_def(&image, &mut moved, &affine, Size::new(320, 320))?;
        for i in [5, 6] {
            points[i].y = points[i].y * 1.02 + 3.0;
        }
        state.apply(&moved, &mut points)?;
        assert!((state.torso_scale.unwrap() - 1.02).abs() < 0.005);
        Ok(())
    }
    #[test]
    fn torso_scale_separates_depth_scale_from_image_translation_and_rejects_deformation() {
        let pairs: Vec<_> = (0..30)
            .map(|i| {
                let x = 200.0 + (i % 5) as f64 * 20.0;
                let y = 150.0 + (i / 5) as f64 * 40.0;
                ((x, y), (x + 3.0, y * 1.02 + 7.0 + 0.01 * (x - 240.0)))
            })
            .collect();
        assert!((vertical_scale(&pairs).unwrap() - 1.02).abs() < 1e-6);
        let distorted: Vec<_> = pairs
            .iter()
            .enumerate()
            .map(|(i, (a, b))| (*a, (b.0, b.1 + if i % 2 == 0 { 8.0 } else { -8.0 })))
            .collect();
        assert!(vertical_scale(&distorted).is_none());
        assert!(vertical_scale(&pairs[..6]).is_none());
    }
    #[test]
    fn validates_current_frame_motion_and_rejects_a_detector_spike() -> Result<()> {
        let (patch, mut points) = fixture();
        // Keep the tracked texture away from pyramid borders, as the recorded
        // shoulders are. Border padding is not valid correspondence evidence.
        let mut image = Mat::new_rows_cols_with_default(352, 352, core::CV_8UC3, Scalar::all(0.0))?;
        patch.copy_to(&mut Mat::roi_mut(
            &mut image,
            core::Rect::new(96, 96, 160, 160),
        )?)?;
        for p in &mut points {
            p.x += 96.0;
            p.y += 96.0;
        }
        let mut state = ImageMotion::default();
        state.apply(&image, &mut points)?;
        let mut moved = Mat::default();
        let affine = Mat::from_slice_2d(&[[1.0, 0.0, 7.0], [0.0, 1.0, 3.0]])?;
        imgproc::warp_affine_def(&image, &mut moved, &affine, Size::new(352, 352))?;
        points[5].x += 20.0;
        points[5].y += 3.0;
        points[6].x += 7.0;
        points[6].y += 3.0;
        let corrected = state.apply(&moved, &mut points)?;
        assert_eq!(corrected, vec![5]);
        assert!((points[5].x - 153.0).abs() < 0.2);
        assert!((points[5].y - 159.0).abs() < 0.2);
        assert_eq!(points[6].x, 208.0);
        assert_eq!(points[6].y, 164.0);
        Ok(())
    }
    #[test]
    fn wrist_detector_jump_uses_current_image_motion_without_inflating_confidence() -> Result<()> {
        let (patch, mut points) = fixture();
        let mut image = Mat::new_rows_cols_with_default(352, 352, core::CV_8UC3, Scalar::all(0.0))?;
        patch.copy_to(&mut Mat::roi_mut(
            &mut image,
            core::Rect::new(96, 96, 160, 160),
        )?)?;
        for p in &mut points {
            p.x += 96.0;
            p.y += 96.0;
        }
        points[9] = points[5];
        points[9].x += 25.0;
        let mut state = ImageMotion::default();
        state.apply(&image, &mut points)?;
        let mut unsupported = ImageMotion::default();
        unsupported.apply(&image, &mut points.clone())?;
        state.confirm_hands(&[crate::details::HandPose {
            side: "left".into(),
            confidence: 0.99,
            image_landmarks: [[points[5].x, points[5].y]; 21],
            world_landmarks: [[0.0; 3]; 21],
        }]);
        let mut moved = Mat::default();
        imgproc::warp_affine_def(
            &image,
            &mut moved,
            &Mat::from_slice_2d(&[[1.0, 0.0, 7.0], [0.0, 1.0, 3.0]])?,
            Size::new(352, 352),
        )?;
        for i in [5, 6, 9] {
            points[i].x += 7.0;
            points[i].y += 3.0;
        }
        points[9].x += 13.0;
        let mut unknown = points.clone();
        assert!(unsupported.apply(&moved, &mut unknown)?.is_empty());
        assert_eq!(unknown[9].x, points[9].x);
        let corrected = state.apply(&moved, &mut points)?;
        assert_eq!(corrected, vec![9]);
        assert!((points[9].x - 153.0).abs() < 0.2 && (points[9].y - 159.0).abs() < 0.2);
        assert!(points[9].score <= 0.99);
        Ok(())
    }
    #[test]
    fn hand_correspondence_expires_after_two_valid_steps_or_one_failed_step() -> Result<()> {
        let (patch, mut points) = fixture();
        let mut image = Mat::new_rows_cols_with_default(352, 352, core::CV_8UC3, Scalar::all(0.0))?;
        patch.copy_to(&mut Mat::roi_mut(
            &mut image,
            core::Rect::new(96, 96, 160, 160),
        )?)?;
        for p in &mut points {
            p.x += 96.0;
            p.y += 96.0;
        }
        points[9] = points[5];
        let original = points.clone();
        let mut state = ImageMotion::default();
        state.apply(&image, &mut points)?;
        let hand = crate::details::HandPose {
            side: "left".into(),
            confidence: 0.99,
            image_landmarks: [[points[9].x, points[9].y]; 21],
            world_landmarks: [[0.0; 3]; 21],
        };
        state.confirm_hands(&[hand.clone()]);
        for step in 1..=3 {
            let mut moved = Mat::default();
            imgproc::warp_affine_def(
                &image,
                &mut moved,
                &Mat::from_slice_2d(&[
                    [1.0, 0.0, 7.0 * step as f64],
                    [0.0, 1.0, 3.0 * step as f64],
                ])?,
                Size::new(352, 352),
            )?;
            points = original.clone();
            points[9].x += 20.0 * step as f32;
            points[9].y += 3.0 * step as f32;
            let cnn_x = points[9].x;
            let corrected = state.apply(&moved, &mut points)?;
            if step <= 2 {
                assert!(corrected.contains(&9));
                assert!((points[9].x - original[9].x - 7.0 * step as f32).abs() < 0.3);
            } else {
                assert!(!corrected.contains(&9));
                assert_eq!(points[9].x, cnn_x);
            }
            state.confirm_hands(&[]);
        }
        let mut failed = ImageMotion::default();
        failed.apply(&image, &mut original.clone())?;
        failed.confirm_hands(&[hand]);
        let blank = Mat::new_rows_cols_with_default(352, 352, core::CV_8UC3, Scalar::all(80.0))?;
        let mut wrong = original;
        wrong[9].x += 20.0;
        assert!(!failed.apply(&blank, &mut wrong)?.contains(&9));
        failed.confirm_hands(&[]);
        assert!(!failed.wrist_supported[0]);
        Ok(())
    }
    #[test]
    fn independently_seeded_wrist_can_reject_a_large_detector_teleport() -> Result<()> {
        let (patch, mut points) = fixture();
        let mut image = Mat::new_rows_cols_with_default(352, 352, core::CV_8UC3, Scalar::all(0.0))?;
        patch.copy_to(&mut Mat::roi_mut(
            &mut image,
            core::Rect::new(96, 96, 160, 160),
        )?)?;
        for p in &mut points {
            p.x += 96.0;
            p.y += 96.0;
        }
        points[9] = points[5];
        let start = points[9];
        let mut state = ImageMotion::default();
        state.apply(&image, &mut points)?;
        state.confirm_hands(&[crate::details::HandPose {
            side: "left".into(),
            confidence: 0.99,
            image_landmarks: [[start.x, start.y]; 21],
            world_landmarks: [[0.0; 3]; 21],
        }]);
        let mut moved = Mat::default();
        imgproc::warp_affine_def(
            &image,
            &mut moved,
            &Mat::from_slice_2d(&[[1.0, 0.0, 7.0], [0.0, 1.0, 3.0]])?,
            Size::new(352, 352),
        )?;
        points[9].x += 80.0;
        points[9].y += 3.0;
        assert!(state.apply(&moved, &mut points)?.contains(&9));
        assert!((points[9].x - start.x - 7.0).abs() < 0.3);
        assert!((points[9].y - start.y - 3.0).abs() < 0.3);
        Ok(())
    }
    #[test]
    fn moderate_hand_seeds_require_two_fresh_consistent_observations() {
        let (_, points) = fixture();
        let mut state = ImageMotion {
            points,
            ..ImageMotion::default()
        };
        let hand = crate::details::HandPose {
            side: "right".into(),
            confidence: 0.68,
            image_landmarks: [[80.0, 90.0]; 21],
            world_landmarks: [[0.0; 3]; 21],
        };
        state.confirm_hands(&[hand.clone()]);
        assert!(!state.wrist_supported[1]);
        state.confirm_hands(&[hand.clone()]);
        assert!(state.wrist_supported[1]);
        assert_eq!(
            (
                state.points[10].x,
                state.points[10].y,
                state.points[10].score
            ),
            (80.0, 90.0, 0.68)
        );
        state.confirm_hands(&[]);
        assert!(!state.wrist_supported[1]);
        state.confirm_hands(&[hand]);
        assert!(!state.wrist_supported[1]);
    }
    #[test]
    fn textureless_or_unobserved_points_do_not_gain_measurement_confidence() -> Result<()> {
        let image = Mat::new_rows_cols_with_default(160, 160, core::CV_8UC3, Scalar::all(80.0))?;
        let (_, mut points) = fixture();
        let mut state = ImageMotion::default();
        state.apply(&image, &mut points)?;
        points[5].x += 20.0;
        points[6].score = 0.1;
        assert!(state.apply(&image, &mut points)?.is_empty());
        assert_eq!(points[5].x, 70.0);
        assert_eq!(points[6].score, 0.1);
        Ok(())
    }
    #[test]
    #[ignore = "requires LIFTING_EVAL_DIR with cached candidate observations and source video"]
    fn evaluate_recorded_image_motion() -> Result<()> {
        use opencv::videoio;
        use std::io::Write;
        let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
        let input = std::env::var("LIFTING_EVAL_INPUT").unwrap_or("candidate".into());
        let tag = std::env::var("LIFTING_EVAL_TAG").unwrap_or("flow".into());
        anyhow::ensure!(
            [&input, &tag].iter().all(|s| s
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')),
            "Invalid diagnostic tag"
        );
        let mut capture =
            videoio::VideoCapture::from_file_def(directory.join("source.mp4").to_str().unwrap())?;
        let mut writer = std::io::BufWriter::new(std::fs::File::create(
            directory.join(format!("{tag}.jsonl")),
        )?);
        let mut state = ImageMotion::default();
        let mut corrections = Vec::new();
        for line in std::fs::read_to_string(directory.join(format!("{input}.jsonl")))?.lines() {
            let mut row: serde_json::Value = serde_json::from_str(line)?;
            let mut image = Mat::default();
            anyhow::ensure!(capture.read(&mut image)?, "Missing source frame");
            if let Some(person) = row["people"]
                .as_array_mut()
                .and_then(|people| people.first_mut())
            {
                let mut points: Vec<_> = person["keypoints"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, p)| Keypoint {
                        name: crate::landmarks::NAMES[i],
                        x: p["x"].as_f64().unwrap() as f32,
                        y: p["y"].as_f64().unwrap() as f32,
                        score: p["score"].as_f64().unwrap() as f32,
                    })
                    .collect();
                if let Some(wrists) = person["body_wrists"].as_array() {
                    for (side, i) in [9, 10].into_iter().enumerate() {
                        points[i].x = wrists[side]["x"].as_f64().unwrap() as f32;
                        points[i].y = wrists[side]["y"].as_f64().unwrap() as f32;
                        points[i].score = wrists[side]["score"].as_f64().unwrap() as f32;
                    }
                }
                let corrected = state.apply(&image, &mut points)?;
                // Keep cached torso evidence; this ablation changes arm pixels
                // only and cannot reproduce shoulder->CNN crop feedback.
                let rejected = crate::occlusion::mask_arm_overlap(&mut points, image.size()?);
                let hands: Vec<crate::details::HandPose> =
                    serde_json::from_value(person["hands"].clone())?;
                crate::details::merge_hand_landmarks(&mut points, &hands);
                state.confirm_hands(&hands);
                for &i in &corrected {
                    if i < 7 {
                        continue;
                    }
                    person["keypoints"][i]["x"] = serde_json::json!(points[i].x);
                    person["keypoints"][i]["y"] = serde_json::json!(points[i].y);
                    person["keypoints"][i]["score"] = serde_json::json!(points[i].score);
                    let node = match i {
                        7 => 12,
                        8 => 15,
                        9 => 13,
                        10 => 16,
                        _ => continue,
                    };
                    person["pose3d"]["joints"][node]["input_score"] = serde_json::json!(0.0);
                }
                for (i, node) in [(7, 12), (8, 15), (9, 13), (10, 16)] {
                    if rejected[i] {
                        person["keypoints"][i]["score"] = serde_json::json!(points[i].score);
                        person["pose3d"]["joints"][node]["input_score"] = serde_json::json!(0.0);
                    }
                }
                if !corrected.is_empty() {
                    corrections.push(serde_json::json!({"frame":row["frame"],"joints":corrected}));
                }
            } else {
                state = ImageMotion::default();
            }
            serde_json::to_writer(&mut writer, &row)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        std::fs::write(
            directory.join(format!("{tag}-corrections.json")),
            serde_json::to_vec(&corrections)?,
        )?;
        Ok(())
    }
}
