//! Align asynchronous neural results with the current clean camera image.
//! Only verified pixel correspondence receives tracking confidence; a delayed
//! detector result is never painted directly onto a newer frame.
use crate::{
    joint_tracking::JointTracker,
    pose::{Bbox, Keypoint},
};
use anyhow::Result;
use opencv::{core::Mat, imgproc, prelude::*};
use std::collections::VecDeque;
struct Image {
    index: u64,
    time: f64,
    gray: Mat,
}
#[derive(Default)]
pub struct PreviewTracking {
    history: VecDeque<Image>,
    tracker: JointTracker,
    pub points: Vec<Keypoint>,
    working_points: Vec<Keypoint>,
    scale: f32,
    original_size: Option<opencv::core::Size>,
    pub predicted: Vec<Keypoint>,
    bbox: Option<Bbox>,
    occluded: [bool; 17],
    neural_index: Option<u64>,
}
impl PreviewTracking {
    pub fn push(&mut self, image: &Mat, index: u64, time: f64) -> Result<()> {
        if !time.is_finite() {
            self.clear();
            return Ok(());
        }
        if self.history.back().is_some_and(|f| {
            f.index >= index || f.time >= time || self.original_size != image.size().ok()
        }) {
            *self = Self::default();
        }
        let size = image.size()?;
        self.original_size = Some(size);
        self.scale = (640.0 / size.width as f32).min(1.0);
        let mut full_gray = Mat::default();
        imgproc::cvt_color_def(image, &mut full_gray, imgproc::COLOR_BGR2GRAY)?;
        let gray = if self.scale < 1.0 {
            let mut gray = Mat::default();
            imgproc::resize(
                &full_gray,
                &mut gray,
                opencv::core::Size::new(640, (size.height as f32 * self.scale).round() as i32),
                0.0,
                0.0,
                imgproc::INTER_AREA,
            )?;
            gray
        } else {
            full_gray
        };
        self.history.push_back(Image { index, time, gray });
        while self.history.len() > 12 || self.history.front().is_some_and(|f| time - f.time > 0.4) {
            self.history.pop_front();
        }
        if let (Some(bbox), Some(frame)) = (self.bbox, self.history.back()) {
            for p in &mut self.working_points {
                p.score = 0.0;
            }
            let result = self.tracker.apply(
                &frame.gray,
                &mut self.working_points,
                frame.time,
                bbox,
                [false; 2],
                &self.occluded,
                false,
            )?;
            self.points = self
                .working_points
                .iter()
                .map(|p| self.to_original(*p))
                .collect();
            self.predicted = result
                .predicted
                .iter()
                .map(|p| self.to_original(*p))
                .collect();
        }
        Ok(())
    }
    /// Replay the short clean-image history from the detector's source frame.
    /// Future frames beyond the current camera image are never used.
    pub fn anchor(
        &mut self,
        index: u64,
        points: &[Keypoint],
        bbox: Bbox,
        wrists: [bool; 2],
        occluded: [bool; 17],
    ) -> Result<bool> {
        if self.neural_index.is_some_and(|old| index < old) {
            return Ok(false);
        }
        let Some(start) = self.history.iter().position(|f| f.index == index) else {
            return Ok(false);
        };
        let source = &self.history[start];
        let current = self.history.back().unwrap();
        // Transport, not long-range prediction: expired detections cannot renew
        // a track or resurrect a skeleton on a different person's image.
        if current.time - source.time > 0.35
            || !points
                .iter()
                .enumerate()
                .any(|(i, p)| (5..17).contains(&i) && p.score >= 0.3)
        {
            return Ok(false);
        }
        let mut tracker = JointTracker::for_preview();
        let mut tracked: Vec<_> = points
            .iter()
            .map(|p| Keypoint {
                x: p.x * self.scale,
                y: p.y * self.scale,
                ..*p
            })
            .collect();
        let bbox = Bbox {
            x: bbox.x * self.scale,
            y: bbox.y * self.scale,
            width: bbox.width * self.scale,
            height: bbox.height * self.scale,
        };
        tracker.apply(
            &source.gray,
            &mut tracked,
            source.time,
            bbox,
            wrists,
            &occluded,
            false,
        )?;
        for frame in self.history.iter().skip(start + 1) {
            for p in &mut tracked {
                p.score = 0.0;
            }
            tracker.apply(
                &frame.gray,
                &mut tracked,
                frame.time,
                bbox,
                [false; 2],
                &occluded,
                false,
            )?;
        }
        self.tracker
            .merge_current(&mut self.working_points, tracker, tracked);
        self.points = self
            .working_points
            .iter()
            .map(|p| self.to_original(*p))
            .collect();
        self.predicted = self
            .tracker
            .current_predictions(&self.working_points)
            .into_iter()
            .map(|p| self.to_original(p))
            .collect();
        self.bbox = Some(bbox);
        self.occluded = occluded;
        self.neural_index = Some(index);
        Ok(true)
    }
    fn to_original(&self, p: Keypoint) -> Keypoint {
        Keypoint {
            x: p.x / self.scale,
            y: p.y / self.scale,
            ..p
        }
    }
    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{self, Scalar, Size};
    fn fixture() -> Result<(Mat, Vec<Keypoint>, Bbox)> {
        let mut image = Mat::new_rows_cols_with_default(320, 320, core::CV_8UC3, Scalar::all(0.0))?;
        for y in 0..320 {
            for x in 0..320 {
                let v = ((x / 3 * 73 + y / 3 * 137 + x / 3 * y / 3 * 19) % 255) as u8;
                *image.at_2d_mut::<core::Vec3b>(y, x)? = core::Vec3b::from([v, v, v]);
            }
        }
        let mut points = vec![
            Keypoint {
                name: "joint",
                x: 160.0,
                y: 160.0,
                score: 0.95
            };
            133
        ];
        points[5].x = 110.0;
        points[6].x = 210.0;
        for p in &mut points[17..91] {
            p.score = 0.0;
        }
        Ok((
            image,
            points,
            Bbox {
                x: 0.0,
                y: 0.0,
                width: 320.0,
                height: 320.0,
            },
        ))
    }
    fn moved(image: &Mat, dx: f64) -> Result<Mat> {
        let mut output = Mat::default();
        imgproc::warp_affine_def(
            image,
            &mut output,
            &Mat::from_slice_2d(&[[1.0, 0.0, dx], [0.0, 1.0, 0.0]])?,
            Size::new(320, 320),
        )?;
        Ok(output)
    }
    #[test]
    fn delayed_neural_result_is_aligned_to_current_pixels_without_smoothing() -> Result<()> {
        let (image, points, bbox) = fixture()?;
        let mut live = PreviewTracking::default();
        for i in 0..=3 {
            live.push(&moved(&image, i as f64 * 3.0)?, i, i as f64 / 30.0)?;
        }
        assert!(live.anchor(0, &points, bbox, [true; 2], [false; 17])?);
        for joint in [7, 9, 91, 95] {
            assert!(
                (live.points[joint].x - 169.0).abs() < 1.0,
                "joint {joint} did not follow current pixels: {}",
                live.points[joint].x
            );
            assert!(live.points[joint].score >= 0.3);
        }
        live.push(&moved(&image, 12.0)?, 4, 4.0 / 30.0)?;
        assert!((live.points[7].x - 172.0).abs() < 1.0);
        Ok(())
    }
    #[test]
    fn partial_neural_updates_preserve_visible_fingers_without_renewing_missing_observations()
    -> Result<()> {
        let (image, points, bbox) = fixture()?;
        let mut live = PreviewTracking::default();
        live.push(&image, 0, 0.0)?;
        live.anchor(0, &points, bbox, [true; 2], [false; 17])?;
        let mut missing = points.clone();
        for p in &mut missing[91..] {
            p.score = 0.0;
        }
        for i in 1..=10 {
            live.push(&moved(&image, i as f64 * 2.0)?, i, i as f64 / 30.0)?;
            let mut observed = missing.clone();
            for p in &mut observed {
                p.x += i as f32 * 2.0;
            }
            live.anchor(i, &observed, bbox, [false; 2], [false; 17])?;
            assert!(live.points[95].score >= 0.3);
            assert!((live.points[95].x - (160.0 + i as f32 * 2.0)).abs() < 1.0);
        }
        live.push(&moved(&image, 22.0)?, 11, 11.0 / 30.0)?;
        assert_eq!(live.points[95].score, 0.0);
        assert!(!live.anchor(0, &points, bbox, [true; 2], [false; 17])?);
        Ok(())
    }
    #[test]
    fn body_then_hand_updates_from_one_camera_frame_are_both_accepted() -> Result<()> {
        let (image, points, bbox) = fixture()?;
        let mut live = PreviewTracking::default();
        live.push(&image, 0, 0.0)?;
        let mut body = points.clone();
        for p in &mut body[91..] {
            p.score = 0.0;
        }
        assert!(live.anchor(0, &body, bbox, [false; 2], [false; 17])?);
        assert_eq!(live.points[95].score, 0.0);
        assert!(live.anchor(0, &points, bbox, [true; 2], [false; 17])?);
        assert!(live.points[95].score >= 0.3);
        live.push(&image, 1, 1.0 / 30.0)?;
        live.anchor(1, &points, bbox, [true; 2], [false; 17])?;
        assert!(!live.anchor(0, &points, bbox, [true; 2], [false; 17])?);
        Ok(())
    }
    #[test]
    fn neural_spike_does_not_displace_a_verified_current_image_track() -> Result<()> {
        let (image, points, bbox) = fixture()?;
        let mut live = PreviewTracking::default();
        live.push(&image, 0, 0.0)?;
        live.anchor(0, &points, bbox, [true; 2], [false; 17])?;
        live.push(&moved(&image, 3.0)?, 1, 1.0 / 30.0)?;
        let mut wrong = points.clone();
        for p in &mut wrong {
            p.x += 3.0;
        }
        wrong[7].x += 60.0;
        live.anchor(1, &wrong, bbox, [false; 2], [false; 17])?;
        assert!((live.points[7].x - 163.0).abs() < 1.0);
        Ok(())
    }
    #[test]
    fn brief_flow_loss_predicts_from_the_last_real_image_observation_not_the_older_neural_frame()
    -> Result<()> {
        let (image, points, bbox) = fixture()?;
        let mut live = PreviewTracking::default();
        live.push(&image, 0, 0.0)?;
        live.anchor(0, &points, bbox, [true; 2], [false; 17])?;
        for i in 1..=3 {
            live.push(&moved(&image, i as f64 * 2.0)?, i, i as f64 * 0.05)?;
        }
        let blank = Mat::new_rows_cols_with_default(320, 320, core::CV_8UC3, Scalar::all(0.0))?;
        live.push(&blank, 4, 0.2)?;
        assert_eq!(live.points[7].score, 0.0);
        assert!(!live.predicted.is_empty());
        live.push(&blank, 5, 0.36)?;
        assert!(live.predicted.is_empty());
        Ok(())
    }
    #[test]
    fn scaled_tracking_returns_coordinates_in_the_original_camera_resolution() -> Result<()> {
        let (image, mut points, mut bbox) = fixture()?;
        let mut large = Mat::default();
        imgproc::resize(
            &image,
            &mut large,
            opencv::core::Size::new(1280, 1280),
            0.0,
            0.0,
            imgproc::INTER_NEAREST,
        )?;
        for p in &mut points {
            p.x *= 4.0;
            p.y *= 4.0;
        }
        bbox.width *= 4.0;
        bbox.height *= 4.0;
        let mut live = PreviewTracking::default();
        live.push(&large, 0, 0.0)?;
        live.anchor(0, &points, bbox, [true; 2], [false; 17])?;
        let mut shifted = Mat::default();
        imgproc::warp_affine_def(
            &large,
            &mut shifted,
            &Mat::from_slice_2d(&[[1.0, 0.0, 8.0], [0.0, 1.0, 0.0]])?,
            opencv::core::Size::new(1280, 1280),
        )?;
        live.push(&shifted, 1, 1.0 / 30.0)?;
        assert!((live.points[7].x - 648.0).abs() < 1.0);
        assert!((live.points[7].y - 640.0).abs() < 1.0);
        Ok(())
    }
    #[test]
    fn stale_results_and_resolution_changes_cannot_resurrect_an_old_pose() -> Result<()> {
        let (image, points, bbox) = fixture()?;
        let mut live = PreviewTracking::default();
        live.push(&image, 0, 0.0)?;
        live.push(&image, 1, 0.36)?;
        assert!(!live.anchor(0, &points, bbox, [true; 2], [false; 17])?);
        assert!(live.points.is_empty());
        live.anchor(1, &points, bbox, [true; 2], [false; 17])?;
        let small = Mat::new_rows_cols_with_default(160, 160, core::CV_8UC3, Scalar::all(0.0))?;
        live.push(&small, 2, 0.39)?;
        assert!(live.points.is_empty());
        Ok(())
    }
}

#[cfg(test)]
mod recording_evaluation {
    use super::*;
    use anyhow::ensure;
    use opencv::videoio;
    use std::{path::PathBuf, time::Instant};
    #[test]
    #[ignore = "requires LIFTING_LIVE_EVAL with source video and cached neural measurements; no camera or UI"]
    fn evaluate_delayed_live_tracking() -> Result<()> {
        let directory = PathBuf::from(std::env::var("LIFTING_LIVE_EVAL")?);
        let source = std::env::var("LIFTING_LIVE_SOURCE")?;
        let rows: Vec<serde_json::Value> = std::fs::read_to_string(directory.join("neural.jsonl"))?
            .lines()
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()?;
        let timing: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("timing.json"))?)?;
        let times: Vec<f64> = serde_json::from_value(timing["timestamps_seconds"].clone())?;
        let mut capture = videoio::VideoCapture::from_file_def(&source)?;
        let mut live = PreviewTracking::default();
        let mut runtime = Vec::new();
        let mut stale_errors = Vec::new();
        let mut live_errors = Vec::new();
        let mut visible = 0;
        let mut opportunities = 0;
        let mut last_neural = None;
        let mut output = Vec::new();
        for (index, row) in rows.iter().enumerate() {
            let mut frame = Mat::default();
            ensure!(capture.read(&mut frame)?, "Video ended early");
            let start = Instant::now();
            live.push(&frame, index as u64, times[index])?;
            if index >= 2 && (index - 2) % 3 == 0 {
                let n = index - 2;
                let person = &rows[n]["people"][0];
                let values = person["observed_keypoints"]
                    .as_array()
                    .unwrap_or_else(|| person["keypoints"].as_array().unwrap());
                let points: Vec<_> = values
                    .iter()
                    .enumerate()
                    .map(|(i, p)| Keypoint {
                        name: crate::landmarks::NAMES[i],
                        x: p["x"].as_f64().unwrap() as f32,
                        y: p["y"].as_f64().unwrap() as f32,
                        score: p["score"].as_f64().unwrap() as f32,
                    })
                    .collect();
                let b = &person["bbox"];
                let bbox = Bbox {
                    x: b["x"].as_f64().unwrap() as f32,
                    y: b["y"].as_f64().unwrap() as f32,
                    width: b["width"].as_f64().unwrap() as f32,
                    height: b["height"].as_f64().unwrap() as f32,
                };
                let wrists = ["left", "right"].map(|side| {
                    person["hands"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|h| h["side"] == side && h["confidence"].as_f64().unwrap() >= 0.85)
                });
                let occluded =
                    std::array::from_fn(|i| (7..=10).contains(&i) && points[i].score == 0.0);
                live.anchor(n as u64, &points, bbox, wrists, occluded)?;
                last_neural = Some(n);
            }
            runtime.push(start.elapsed().as_secs_f64() * 1000.0);
            if let Some(n) = last_neural {
                if times[index] >= 3.0 && times[index] <= times[times.len() - 1] - 3.0 {
                    for j in [5, 6, 7, 8, 9, 10] {
                        let current = &row["people"][0]["observed_keypoints"][j];
                        let previous = &rows[n]["people"][0]["observed_keypoints"][j];
                        if current["score"].as_f64().unwrap_or(0.0) >= 0.65
                            && previous["score"].as_f64().unwrap_or(0.0) >= 0.65
                        {
                            opportunities += 1;
                            if let Some(p) = live.points.get(j).filter(|p| p.score >= 0.3) {
                                let x = current["x"].as_f64().unwrap();
                                let y = current["y"].as_f64().unwrap();
                                stale_errors.push(
                                    (previous["x"].as_f64().unwrap() - x)
                                        .hypot(previous["y"].as_f64().unwrap() - y),
                                );
                                live_errors.push((p.x as f64 - x).hypot(p.y as f64 - y));
                                visible += 1;
                            }
                        }
                    }
                }
            }
            output.push(
                serde_json::json!({"frame":index,"points":live.points,"predicted":live.predicted}),
            );
        }
        let stats = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            serde_json::json!({"count":v.len(),"mean":v.iter().sum::<f64>()/v.len().max(1) as f64,"p95":v.get((v.len() as f64*0.95)as usize),"max":v.last()})
        };
        let metrics = serde_json::json!({"frames":rows.len(),"tracking_ms":stats(runtime),"stale_to_current_nn_pixels":stats(stale_errors),"tracked_to_current_nn_pixels":stats(live_errors),"visible":visible,"opportunities":opportunities,"note":"Cached neural updates every three frames, delivered two frames late; current-frame neural points are a comparison reference, not ground truth."});
        std::fs::write(
            directory.join("live-metrics.json"),
            serde_json::to_vec_pretty(&metrics)?,
        )?;
        std::fs::write(
            directory.join("live-tracked.json"),
            serde_json::to_vec(&output)?,
        )?;
        println!("{metrics}");
        Ok(())
    }
}
