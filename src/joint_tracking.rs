//! Short-lived image correspondence and bounded, speed-adaptive joint filtering.
//! Prediction is display-only: it never acquires observation confidence.
use crate::pose::{Bbox, Keypoint};
use anyhow::Result;
use nalgebra::Vector2;
use opencv::{
    core::{Mat, Point2f, Size, TermCriteria, TermCriteria_Type, Vector},
    imgproc,
    prelude::*,
    video,
};
use serde::Serialize;
type V = Vector2<f32>;
const GAP: f64 = 0.16;
#[derive(Clone, Debug, Serialize)]
pub struct JointUpdate {
    pub joint: usize,
    pub source: &'static str,
    pub age_seconds: f64,
    pub inliers: usize,
}
#[derive(Default)]
pub struct TrackingFrame {
    pub updates: Vec<JointUpdate>,
    pub predicted: Vec<Keypoint>,
}
#[derive(Clone)]
struct State {
    raw: V,
    filtered: V,
    velocity: V,
    speed: f32,
    confidence: f32,
    measured: f64,
    last_visual: f64,
    independent_anchor: Option<f64>,
}
#[derive(Default)]
pub struct JointTracker {
    gray: Option<Mat>,
    states: Vec<Option<State>>,
    time: Option<f64>,
    bbox: Option<Bbox>,
    flow_horizon: f64,
}
fn xy(p: Keypoint) -> V {
    V::new(p.x, p.y)
}
fn valid(p: Keypoint, size: Size) -> bool {
    p.score >= 0.3
        && p.score.is_finite()
        && p.x.is_finite()
        && p.y.is_finite()
        && p.x >= 0.0
        && p.y >= 0.0
        && p.x < size.width as f32
        && p.y < size.height as f32
}
fn alpha(cutoff: f32, dt: f32) -> f32 {
    1.0 / (1.0 + 1.0 / (std::f32::consts::TAU * cutoff * dt))
}
/// A real fast movement may change instantly; the filter's error budget is pixels,
/// not a velocity cap. Smaller stationary noise receives stronger attenuation.
fn filter(old: &State, measured: V, dt: f32, scale: f32, budget: f32) -> (V, f32) {
    let speed = old.speed + alpha(3.0, dt) * ((measured - old.raw).norm() / dt / scale - old.speed);
    let gain = alpha(3.0 + 12.0 * speed, dt);
    let mut error = old.filtered + gain * (measured - old.filtered) - measured;
    if error.norm() > budget {
        error *= budget / error.norm();
    }
    (measured + error, speed)
}
fn consensus(pairs: &[(V, V)]) -> Option<(V, usize)> {
    if pairs.len() < 4 {
        return None;
    }
    let med = |mut x: Vec<f32>| {
        x.sort_by(f32::total_cmp);
        x[x.len() / 2]
    };
    let d = V::new(
        med(pairs.iter().map(|(a, b)| (b - a).x).collect()),
        med(pairs.iter().map(|(a, b)| (b - a).y).collect()),
    );
    let inliers: Vec<_> = pairs
        .iter()
        .filter(|(a, b)| (*b - *a - d).norm() < 2.5)
        .collect();
    if inliers.len() < 4 || inliers.len() * 3 < pairs.len() * 2 {
        return None;
    }
    let min = inliers
        .iter()
        .fold(V::repeat(f32::INFINITY), |v, (a, _)| v.inf(a));
    let max = inliers
        .iter()
        .fold(V::repeat(f32::NEG_INFINITY), |v, (a, _)| v.sup(a));
    if (max - min).x < 5.0 || (max - min).y < 5.0 {
        return None;
    }
    Some((
        inliers.iter().fold(V::zeros(), |v, (a, b)| v + (*b - *a)) / inliers.len() as f32,
        inliers.len(),
    ))
}
impl JointTracker {
    pub fn for_preview() -> Self {
        Self {
            flow_horizon: 0.35,
            ..Self::default()
        }
    }
    fn horizon(&self) -> f64 {
        self.flow_horizon.max(GAP)
    }
    pub fn reset(&mut self) {
        let flow_horizon = self.flow_horizon;
        *self = Self {
            flow_horizon,
            ..Self::default()
        };
    }
    /// Fuse a delayed neural update after both tracks have reached the same
    /// current image. Agreement renews identity at the neural SOURCE time;
    /// missing/rejected joints never renew themselves from optical flow alone.
    pub fn merge_current(
        &mut self,
        points: &mut Vec<Keypoint>,
        mut other: Self,
        incoming: Vec<Keypoint>,
    ) {
        let overlaps = self.bbox.zip(other.bbox).is_some_and(|(a, b)| {
            let area = ((a.x + a.width).min(b.x + b.width) - a.x.max(b.x)).max(0.0)
                * ((a.y + a.height).min(b.y + b.height) - a.y.max(b.y)).max(0.0);
            area / (a.width * a.height + b.width * b.height - area).max(1.0) > 0.15
        });
        if self.time != other.time || points.len() != incoming.len() || !overlaps {
            *self = other;
            *points = incoming;
            return;
        }
        let time = self.time.unwrap_or(0.0);
        let horizon = self.horizon();
        let scale = if incoming.len() > 6 {
            (xy(incoming[5]) - xy(incoming[6])).norm().max(60.0)
        } else {
            100.0
        };
        let tolerance = (scale * 0.06).max(6.0);
        for (i, new) in incoming.into_iter().enumerate() {
            let old_visible = points[i].score >= 0.3;
            let new_visible = new.score >= 0.3;
            let mut adopt = !old_visible && other.states[i].is_some();
            if new_visible {
                if let (Some(old), Some(next)) = (&mut self.states[i], &mut other.states[i]) {
                    let fresh_hand = matches!(i, 9 | 10)
                        && next.independent_anchor.is_some_and(|t| {
                            t > old.independent_anchor.unwrap_or(f64::NEG_INFINITY)
                        });
                    if old_visible && (xy(points[i]) - xy(new)).norm() <= tolerance && !fresh_hand {
                        // Keep the point attached to its current image patch;
                        // neural agreement supplies fresh semantic support.
                        old.measured = next.measured;
                        old.confidence = next.confidence.min(0.65);
                        old.independent_anchor = next.independent_anchor.or(old.independent_anchor);
                        points[i].score = old.confidence;
                        continue;
                    }
                    adopt = fresh_hand || !old_visible || time - old.measured > horizon;
                    next.independent_anchor = next.independent_anchor.or(old.independent_anchor);
                } else {
                    adopt = true;
                }
            }
            if adopt {
                self.states[i] = other.states[i].take();
                points[i] = new;
            }
        }
        self.bbox = other.bbox;
    }
    pub fn current_predictions(&self, points: &[Keypoint]) -> Vec<Keypoint> {
        let time = self.time.unwrap_or(0.0);
        points
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let state = self.states.get(i)?.as_ref()?;
                (p.score < 0.3
                    && time - state.last_visual <= GAP
                    && time - state.measured <= self.horizon())
                .then_some(Keypoint {
                    name: p.name,
                    x: state.raw.x,
                    y: state.raw.y,
                    score: 0.0,
                })
            })
            .collect()
    }
    pub fn apply(
        &mut self,
        image: &Mat,
        points: &mut [Keypoint],
        time: f64,
        bbox: Bbox,
        independent_wrists: [bool; 2],
        occluded: &[bool; 17],
        smooth: bool,
    ) -> Result<TrackingFrame> {
        if !time.is_finite() {
            self.reset();
            return Ok(TrackingFrame::default());
        }
        let size = image.size()?;
        let gray = if image.channels() == 1 {
            image.try_clone()?
        } else {
            let mut gray = Mat::default();
            imgproc::cvt_color_def(image, &mut gray, imgproc::COLOR_BGR2GRAY)?;
            gray
        };
        let dt = self.time.map(|t| time - t).unwrap_or(0.0);
        let same_person = self.bbox.is_none_or(|b| {
            let intersection = ((b.x + b.width).min(bbox.x + bbox.width) - b.x.max(bbox.x))
                .max(0.0)
                * ((b.y + b.height).min(bbox.y + bbox.height) - b.y.max(bbox.y)).max(0.0);
            intersection / (b.width * b.height + bbox.width * bbox.height - intersection).max(1.0)
                > 0.15
        });
        if dt <= 1e-6
            || dt > GAP
            || !same_person
            || self
                .gray
                .as_ref()
                .is_some_and(|g| g.size().ok() != Some(size))
            || self.states.len() != points.len()
        {
            self.reset();
            self.states = vec![None; points.len()];
        }
        let scale = if points.len() > 6 {
            (xy(points[5]) - xy(points[6]))
                .norm()
                .max(size.width as f32 * 0.08)
        } else {
            size.width as f32 * 0.2
        };
        let horizon = self.horizon();
        let mut flow = vec![None; points.len()];
        if let Some(previous) = &self.gray {
            let mut before = Vector::<Point2f>::new();
            let mut owners = Vec::new();
            let r = (scale * 0.04).clamp(5.0, 12.0);
            for i in (5..17.min(points.len())).chain(91..133.min(points.len())) {
                if let Some(s) = &self.states[i] {
                    if time - s.measured > horizon {
                        continue;
                    }
                    for dy in [-r, 0.0, r] {
                        for dx in [-r, 0.0, r] {
                            let p = s.raw + V::new(dx, dy);
                            if p.x >= 16.0
                                && p.y >= 16.0
                                && p.x < size.width as f32 - 16.0
                                && p.y < size.height as f32 - 16.0
                            {
                                before.push(Point2f::new(p.x, p.y));
                                owners.push(i);
                            }
                        }
                    }
                }
            }
            if !before.is_empty() {
                let mut after = Vector::<Point2f>::new();
                let mut ok = Vector::<u8>::new();
                let mut err = Vector::<f32>::new();
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
                    &mut ok,
                    &mut err,
                    Size::new(21, 21),
                    3,
                    criteria,
                    0,
                    1e-4,
                )?;
                let mut back = Vector::<Point2f>::new();
                let mut bok = Vector::<u8>::new();
                let mut berr = Vector::<f32>::new();
                video::calc_optical_flow_pyr_lk(
                    &gray,
                    previous,
                    &after,
                    &mut back,
                    &mut bok,
                    &mut berr,
                    Size::new(21, 21),
                    3,
                    criteria,
                    0,
                    1e-4,
                )?;
                let mut pairs = vec![Vec::new(); points.len()];
                for (j, &i) in owners.iter().enumerate() {
                    let a = before.get(j)?;
                    let b = after.get(j)?;
                    let c = back.get(j)?;
                    if ok.get(j)? != 0
                        && bok.get(j)? != 0
                        && err.get(j)? < 10.0
                        && berr.get(j)? < 10.0
                        && (a.x - c.x).hypot(a.y - c.y) < 1.0
                    {
                        pairs[i].push((V::new(a.x, a.y), V::new(b.x, b.y)));
                    }
                }
                for i in (5..17.min(points.len())).chain(91..133.min(points.len())) {
                    if let (Some(s), Some((d, n))) = (&self.states[i], consensus(&pairs[i])) {
                        let p = s.raw + d;
                        if p.x >= 0.0
                            && p.y >= 0.0
                            && p.x < size.width as f32
                            && p.y < size.height as f32
                            && d.norm() < scale * 0.75
                        {
                            flow[i] = Some((p, n));
                        }
                    }
                }
            }
        }
        let mut result = TrackingFrame::default();
        for (i, p) in points.iter_mut().enumerate() {
            // Dense face landmarks are intentionally excluded. Fingers use their
            // dedicated hand observations with the same bounded jitter filter.
            if !(matches!(i,0|3..=16|91..=132)) {
                continue;
            }
            let mut measured = valid(*p, size);
            let mut source = "observed";
            let mut inliers = 0;
            let mut last_measured = time;
            if let Some(old) = &self.states[i] {
                let age = time - old.measured;
                let fresh_hand = matches!(i, 9 | 10) && independent_wrists[i - 9];
                if let Some((tracked, n)) = flow[i] {
                    let threshold =
                        (scale * if matches!(i, 5 | 6) { 0.20 } else { 0.08 }).max(10.0);
                    let innovation = (xy(*p) - tracked).norm();
                    // Require agreement with a short velocity forecast as well as
                    // multiple current-frame correspondences before overriding a CNN.
                    let forecast = old.raw + old.velocity * dt as f32;
                    let lock_tolerance = if i >= 91 {
                        1.5
                    } else {
                        (scale * 0.025).clamp(2.5, 6.0)
                    };
                    let hand_supported = !matches!(i, 9 | 10)
                        || fresh_hand
                        || old.independent_anchor.is_some_and(|t| time - t <= horizon);
                    if measured
                        && !occluded.get(i).copied().unwrap_or(false)
                        && hand_supported
                        && innovation <= lock_tolerance
                    {
                        // The neural model confirms identity; current image
                        // correspondence supplies position. No coordinate averaging.
                        p.x = tracked.x;
                        p.y = tracked.y;
                        source = "image_locked";
                        inliers = n;
                    }
                    let flow_score = old.confidence.min(0.65)
                        * (-dt as f32 / (0.3 * horizon as f32 / GAP as f32)).exp();
                    if flow_score >= 0.3
                        && !occluded.get(i).copied().unwrap_or(false)
                        && !fresh_hand
                        && (i < 91 || !measured)
                        && (!matches!(i, 9 | 10)
                            || old.independent_anchor.is_some_and(|t| time - t <= horizon))
                        && age <= horizon
                        && source != "image_locked"
                        && (!measured
                            || (innovation > threshold
                                && (tracked - forecast).norm() < threshold * 2.0))
                    {
                        p.x = tracked.x;
                        p.y = tracked.y;
                        p.score = flow_score;
                        measured = p.score >= 0.3;
                        source = "image_tracked";
                        inliers = n;
                        last_measured = old.measured;
                    }
                }
                if !measured {
                    let prediction_age = time - old.last_visual;
                    if prediction_age <= GAP && age <= horizon {
                        let velocity =
                            old.velocity * (1.0 - (prediction_age / GAP) as f32).max(0.0);
                        let forecast = old.raw + velocity * dt as f32;
                        let mut predicted = *p;
                        predicted.x = forecast.x;
                        predicted.y = forecast.y;
                        predicted.score = 0.0;
                        if forecast.x >= 0.0
                            && forecast.y >= 0.0
                            && forecast.x < size.width as f32
                            && forecast.y < size.height as f32
                        {
                            result.predicted.push(predicted);
                            result.updates.push(JointUpdate {
                                joint: i,
                                source: "predicted",
                                age_seconds: age,
                                inliers: 0,
                            });
                            let mut state = old.clone();
                            state.raw = forecast;
                            state.filtered = forecast;
                            self.states[i] = Some(state);
                        } else {
                            self.states[i] = None;
                        }
                    } else {
                        self.states[i] = None;
                    }
                    // Missing image/depth remains missing, not a confident prediction.
                    p.score = 0.0;
                    continue;
                }
            }
            if !measured {
                self.states[i] = None;
                continue;
            }
            let raw = xy(*p);
            let (filtered, speed, velocity) = if let Some(old) = &self.states[i] {
                let (filtered, speed) = if smooth {
                    filter(old, raw, dt as f32, scale, if i >= 91 { 1.5 } else { 2.5 })
                } else {
                    (raw, 0.0)
                };
                let velocity = old.velocity * 0.35 + (raw - old.raw) / (dt as f32) * 0.65;
                (filtered, speed, velocity)
            } else {
                (raw, 0.0, V::zeros())
            };
            if source != "observed" || (filtered - raw).norm() > 0.1 {
                result.updates.push(JointUpdate {
                    joint: i,
                    source: if source == "observed" {
                        "smoothed"
                    } else {
                        source
                    },
                    age_seconds: time - last_measured,
                    inliers,
                });
            }
            let independent_anchor = if matches!(i, 9 | 10) && independent_wrists[i - 9] {
                Some(time)
            } else {
                self.states[i].as_ref().and_then(|s| s.independent_anchor)
            };
            self.states[i] = Some(State {
                independent_anchor,
                raw,
                filtered,
                speed,
                velocity,
                confidence: p.score,
                measured: last_measured,
                last_visual: time,
            });
            p.x = filtered.x;
            p.y = filtered.y;
        }
        self.gray = Some(gray);
        self.time = Some(time);
        self.bbox = Some(bbox);
        Ok(result)
    }
}
/// Predictions have a separate gray overlay and never masquerade as observations.
pub fn draw_predictions(image: &mut Mat, points: &[Keypoint]) -> Result<()> {
    use opencv::core::{Point, Scalar};
    for p in points {
        imgproc::circle(
            image,
            Point::new(p.x.round() as i32, p.y.round() as i32),
            5,
            Scalar::all(150.0),
            1,
            imgproc::LINE_AA,
            0,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{self, Scalar};
    #[test]
    fn adaptive_filter_reduces_stationary_noise_without_delaying_a_fast_reach() {
        let mut state = State {
            raw: V::zeros(),
            filtered: V::zeros(),
            velocity: V::zeros(),
            speed: 0.0,
            confidence: 1.0,
            measured: 0.0,
            last_visual: 0.0,
            independent_anchor: None,
        };
        let mut raw_error = 0.0;
        let mut filtered_error = 0.0;
        for n in 0..120 {
            let raw = V::new(if n % 2 == 0 { 1.0 } else { -1.0 }, 0.0);
            let (value, speed) = filter(&state, raw, 1.0 / 30.0, 200.0, 2.5);
            raw_error += raw.norm_squared();
            filtered_error += value.norm_squared();
            state.raw = raw;
            state.filtered = value;
            state.speed = speed;
        }
        assert!(filtered_error < raw_error * 0.5);
        let target = V::new(100.0, 80.0);
        let (value, _) = filter(&state, target, 1.0 / 30.0, 200.0, 2.5);
        assert!((value - target).norm() <= 2.5001);
    }
    #[test]
    fn correspondence_needs_multiple_spatially_distributed_agreeing_features() {
        let pairs: Vec<_> = [
            (0.0, 0.0),
            (10.0, 0.0),
            (0.0, 10.0),
            (10.0, 10.0),
            (5.0, 5.0),
        ]
        .into_iter()
        .map(|(x, y)| {
            let a = V::new(x, y);
            (a, a + V::new(3.0, 2.0))
        })
        .collect();
        assert!((consensus(&pairs).unwrap().0 - V::new(3.0, 2.0)).norm() < 1e-5);
        assert!(consensus(&pairs[..3]).is_none());
        let collapsed = vec![(V::zeros(), V::new(3.0, 2.0)); 5];
        assert!(consensus(&collapsed).is_none());
    }
    fn fixture() -> Result<(Mat, Vec<Keypoint>, Bbox)> {
        let mut image = Mat::new_rows_cols_with_default(256, 256, core::CV_8UC3, Scalar::all(0.0))?;
        for y in 0..256 {
            for x in 0..256 {
                let v = ((x / 3 * 73 + y / 3 * 137 + x / 3 * y / 3 * 19) % 255) as u8;
                *image.at_2d_mut::<core::Vec3b>(y, x)? = core::Vec3b::from([v, v, v]);
            }
        }
        let mut points = vec![
            Keypoint {
                name: "joint",
                x: 128.0,
                y: 128.0,
                score: 0.95
            };
            17
        ];
        points[5].x = 80.0;
        points[6].x = 175.0;
        Ok((
            image,
            points,
            Bbox {
                x: 0.0,
                y: 0.0,
                width: 256.0,
                height: 256.0,
            },
        ))
    }
    #[test]
    fn actual_image_features_reject_a_confident_joint_spike() -> Result<()> {
        let (image, mut points, bbox) = fixture()?;
        let mut tracker = JointTracker::default();
        tracker.apply(
            &image,
            &mut points,
            0.0,
            bbox,
            [false; 2],
            &[false; 17],
            false,
        )?;
        let mut moved = Mat::default();
        imgproc::warp_affine_def(
            &image,
            &mut moved,
            &Mat::from_slice_2d(&[[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]])?,
            Size::new(256, 256),
        )?;
        points[7].x = 180.0;
        points[7].y = 180.0;
        let result = tracker.apply(
            &moved,
            &mut points,
            1.0 / 30.0,
            bbox,
            [false; 2],
            &[false; 17],
            false,
        )?;
        assert!(
            result
                .updates
                .iter()
                .any(|u| u.joint == 7 && u.source == "image_tracked" && u.inliers >= 4)
        );
        assert!((xy(points[7]) - V::new(131.0, 130.0)).norm() < 1.0);
        Ok(())
    }
    #[test]
    fn agreeing_neural_measurement_renews_identity_at_the_current_image_location_without_smoothing()
    -> Result<()> {
        let (image, mut points, bbox) = fixture()?;
        let mut tracker = JointTracker::default();
        tracker.apply(
            &image,
            &mut points,
            0.0,
            bbox,
            [false; 2],
            &[false; 17],
            false,
        )?;
        let mut moved = Mat::default();
        imgproc::warp_affine_def(
            &image,
            &mut moved,
            &Mat::from_slice_2d(&[[1.0, 0.0, 3.0], [0.0, 1.0, 2.0]])?,
            Size::new(256, 256),
        )?;
        points[7].x = 133.0;
        points[7].y = 130.0;
        let result = tracker.apply(
            &moved,
            &mut points,
            1.0 / 30.0,
            bbox,
            [false; 2],
            &[false; 17],
            false,
        )?;
        assert!((points[7].x - 131.0).abs() < 0.5);
        assert!((points[7].y - 130.0).abs() < 0.5);
        assert_eq!(points[7].score, 0.95);
        assert!(
            result
                .updates
                .iter()
                .any(|u| u.joint == 7 && u.source == "image_locked")
        );
        Ok(())
    }
    #[test]
    fn explicit_arm_occlusion_blocks_flow_reacquisition_and_body_wrists_need_hand_support()
    -> Result<()> {
        let (image, mut points, bbox) = fixture()?;
        let mut tracker = JointTracker::default();
        tracker.apply(
            &image,
            &mut points,
            0.0,
            bbox,
            [false; 2],
            &[false; 17],
            false,
        )?;
        let mut occluded = [false; 17];
        occluded[7] = true;
        points[7].score = 0.0;
        points[9].x = 180.0;
        let result = tracker.apply(
            &image,
            &mut points,
            1.0 / 30.0,
            bbox,
            [false; 2],
            &occluded,
            false,
        )?;
        assert_eq!(points[7].score, 0.0);
        assert!(result.predicted.iter().any(|p| p.x == 128.0));
        assert!(
            !result
                .updates
                .iter()
                .any(|u| u.source == "image_tracked" && (u.joint == 7 || u.joint == 9))
        );
        Ok(())
    }

    #[test]
    fn predictions_expire_remain_unmeasured_and_reset_on_a_different_person() -> Result<()> {
        let (_, mut points, bbox) = fixture()?;
        let image = Mat::new_rows_cols_with_default(256, 256, core::CV_8UC3, Scalar::all(0.0))?;
        let mut tracker = JointTracker::default();
        tracker.apply(
            &image,
            &mut points,
            0.0,
            bbox,
            [false; 2],
            &[false; 17],
            true,
        )?;
        points[7].score = 0.0;
        let result = tracker.apply(
            &image,
            &mut points,
            0.05,
            bbox,
            [false; 2],
            &[false; 17],
            true,
        )?;
        assert!(
            result
                .updates
                .iter()
                .any(|u| u.joint == 7 && u.source == "predicted")
        );
        assert_eq!(points[7].score, 0.0);
        let result = tracker.apply(
            &image,
            &mut points,
            0.17,
            bbox,
            [false; 2],
            &[false; 17],
            true,
        )?;
        assert!(!result.updates.iter().any(|u| u.joint == 7));
        points[7].score = 0.95;
        points[7].x = 200.0;
        let other = Bbox {
            x: 200.0,
            y: 0.0,
            width: 56.0,
            height: 256.0,
        };
        tracker.apply(
            &image,
            &mut points,
            0.20,
            other,
            [false; 2],
            &[false; 17],
            true,
        )?;
        assert_eq!(points[7].x, 200.0);
        Ok(())
    }
}
