//! OpenCV Zoo's MediaPipe Pose ONNX model: NHWC RGB [0,1], 33 landmarks
//! plus auxiliary slots, visibility/presence logits, and hip-relative world XYZ.
use crate::{
    landmarks::NAMES,
    motion3d::{self, Joint3d, Pose3d},
    pose::{Bbox, Keypoint},
};
use anyhow::{Result, ensure};
use opencv::{
    core::{self, Mat, Size, Vector},
    dnn, imgproc,
    prelude::*,
};

pub(crate) const UPPER_BODY_CROP_FACTOR: f32 = 2.0;

const COCO_FROM_MP: [usize; 23] = [
    0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 31, 29, 32, 32, 30,
];

/// Same physical landmarks as the native world-space skeleton. Do not apply
/// MotionAGFormer's synthetic chest/nose/head offsets to MediaPipe pixels.
pub(crate) fn skeleton_pixels(points: &[Keypoint]) -> [[f32; 3]; 17] {
    skeleton_pixels_for(points, true)
}
pub(crate) fn skeleton_pixels_for(points: &[Keypoint], ear_head: bool) -> [[f32; 3]; 17] {
    let point = |i: usize| [points[i].x, points[i].y, points[i].score];
    let midpoint =
        |a: [f32; 3], b: [f32; 3]| [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, a[2].min(b[2])];
    let mut result = [[0.0; 3]; 17];
    for (target, source) in [
        (1, 12),
        (2, 14),
        (3, 16),
        (4, 11),
        (5, 13),
        (6, 15),
        (9, 0),
        (11, 5),
        (12, 7),
        (13, 9),
        (14, 6),
        (15, 8),
        (16, 10),
    ] {
        result[target] = point(source);
    }
    result[0] = midpoint(point(11), point(12));
    result[8] = midpoint(point(5), point(6));
    result[7] = midpoint(result[0], result[8]);
    result[10] = if ear_head {
        midpoint(point(3), point(4))
    } else {
        midpoint(point(1), point(2))
    };
    result
}
pub fn infer(
    net: &mut dnn::Net,
    names: &Vector<String>,
    image: &Mat,
    bbox: Bbox,
    upper_body: bool,
) -> Result<(Vec<Keypoint>, Option<Pose3d>)> {
    // Retain the detector's upright orientation and pad to a square without
    // stretching. YOLOX remains responsible for closest-person selection.
    // An upper-body box needs virtual room for the unobserved lower body,
    // otherwise the landmark model mistakes the face for the torso center.
    let side = if upper_body {
        bbox.height * UPPER_BODY_CROP_FACTOR
    } else {
        bbox.width.max(bbox.height) * 1.25
    };
    let left = bbox.x + bbox.width / 2.0 - side / 2.0;
    let center_y = bbox.y
        + bbox.height
            * if upper_body {
                UPPER_BODY_CROP_FACTOR / 2.0
            } else {
                0.5
            };
    let top = center_y - side / 2.0;
    let scale = 256.0 / f64::from(side);
    let affine = Mat::from_slice_2d(&[
        [scale, 0.0, -f64::from(left) * scale],
        [0.0, scale, -f64::from(top) * scale],
    ])?;
    let mut crop = Mat::default();
    imgproc::warp_affine_def(image, &mut crop, &affine, Size::new(256, 256))?;
    let mut rgb = Mat::default();
    imgproc::cvt_color_def(&crop, &mut rgb, imgproc::COLOR_BGR2RGB)?;
    let mut float = Mat::default();
    rgb.convert_to(&mut float, core::CV_32F, 1.0 / 255.0, 0.0)?;
    let blob = float.reshape_nd(1, &[1, 256, 256, 3])?;
    net.set_input_def(&blob)?;
    let mut outputs = Vector::<Mat>::new();
    net.forward(&mut outputs, names)?;
    let mut screen = None;
    let mut world = None;
    let mut confidence = None;
    let mut heatmap = None;
    for output in outputs {
        match output.total() {
            195 => screen = Some(output.data_typed::<f32>()?.to_vec()),
            117 => world = Some(output.data_typed::<f32>()?.to_vec()),
            159744 => heatmap = Some(output.data_typed::<f32>()?.to_vec()),
            1 => confidence = Some(output.data_typed::<f32>()?[0]),
            _ => (),
        }
    }
    ensure!(
        screen.is_some() && world.is_some() && confidence.is_some(),
        "Unexpected MediaPipe Pose outputs"
    );
    let mut screen = screen.unwrap();
    let world = world.unwrap();
    let confidence = confidence.unwrap();
    ensure!(
        screen.iter().chain(&world).all(|x| x.is_finite()) && confidence.is_finite(),
        "Non-finite MediaPipe Pose output"
    );
    if let Some(heatmap) = heatmap {
        refine_from_heatmap(&mut screen, &heatmap);
    }
    decode(&screen, &world, confidence, left, top, side, upper_body)
}

/// Same-frame spatial refinement used by MediaPipe's pose graph: a confident
/// 7x7 local heatmap centroid. No temporal averaging or confidence inflation.
fn refine_from_heatmap(screen: &mut [f32], heatmap: &[f32]) {
    const SIDE: usize = 64;
    const CHANNELS: usize = 39;
    if screen.len() != CHANNELS * 5 || heatmap.len() != SIDE * SIDE * CHANNELS {
        return;
    }
    for i in 0..CHANNELS {
        let x = screen[i * 5] / 4.0;
        let y = screen[i * 5 + 1] / 4.0;
        if !x.is_finite()
            || !y.is_finite()
            || x < 0.0
            || y < 0.0
            || x >= SIDE as f32
            || y >= SIDE as f32
        {
            continue;
        }
        let (cx, cy) = (x as usize, y as usize);
        let (mut total, mut wx, mut wy, mut maximum) = (0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32);
        for row in cy.saturating_sub(3)..=(cy + 3).min(SIDE - 1) {
            for col in cx.saturating_sub(3)..=(cx + 3).min(SIDE - 1) {
                let raw = heatmap[(row * SIDE + col) * CHANNELS + i];
                if !raw.is_finite() {
                    continue;
                }
                let weight = 1.0 / (1.0 + (-raw.clamp(-80.0, 80.0)).exp());
                total += weight;
                wx += col as f32 * weight;
                wy += row as f32 * weight;
                maximum = maximum.max(weight);
            }
        }
        if maximum >= 0.5 && total > 0.0 {
            screen[i * 5] = 4.0 * wx / total;
            screen[i * 5 + 1] = 4.0 * wy / total;
        }
    }
}

fn decode(
    screen: &[f32],
    world: &[f32],
    confidence: f32,
    left: f32,
    top: f32,
    side: f32,
    upper_body: bool,
) -> Result<(Vec<Keypoint>, Option<Pose3d>)> {
    let mut points: Vec<_> = NAMES
        .iter()
        .map(|name| Keypoint {
            name,
            x: -1.0,
            y: -1.0,
            score: 0.0,
        })
        .collect();
    if confidence < 0.5 {
        return Ok((points, None));
    }
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x.clamp(-80.0, 80.0)).exp());
    let score = |i: usize| sigmoid(screen[i * 5 + 3]).min(sigmoid(screen[i * 5 + 4])) * confidence;
    let pixel = |i: usize| {
        (
            left + screen[i * 5] * side / 256.0,
            top + screen[i * 5 + 1] * side / 256.0,
        )
    };
    for (i, source) in COCO_FROM_MP.into_iter().enumerate() {
        // MediaPipe has a foot-index point, not separate big/small toes.
        if i == 18 || i == 21 {
            continue;
        }
        let (x, y) = pixel(source);
        points[i] = Keypoint {
            name: NAMES[i],
            x,
            y,
            score: score(source),
        };
    }
    for (base, wrist, thumb, index, pinky) in [(91, 15, 21, 19, 17), (112, 16, 22, 20, 18)] {
        for (target, source) in [(0, wrist), (4, thumb), (8, index), (20, pinky)] {
            let (x, y) = pixel(source);
            points[base + target] = Keypoint {
                name: NAMES[base + target],
                x,
                y,
                score: score(source),
            };
        }
    }
    if upper_body {
        crate::scope::mask_lower(&mut points);
    }
    let xyz = |i: usize| [world[i * 3], world[i * 3 + 1], world[i * 3 + 2]];
    let average = |a: [f32; 3], b: [f32; 3]| std::array::from_fn(|i| (a[i] + b[i]) / 2.0);
    let pelvis = average(xyz(23), xyz(24));
    let thorax = average(xyz(11), xyz(12));
    let mut positions = [[0.0; 3]; 17];
    positions[0] = pelvis;
    positions[7] = average(pelvis, thorax);
    positions[8] = thorax;
    positions[9] = xyz(0);
    positions[10] = average(xyz(7), xyz(8));
    for (target, source) in [
        (1, 24),
        (2, 26),
        (3, 28),
        (4, 23),
        (5, 25),
        (6, 27),
        (11, 11),
        (12, 13),
        (13, 15),
        (14, 12),
        (15, 14),
        (16, 16),
    ] {
        positions[target] = xyz(source);
    }
    let scores = skeleton_pixels(&points);
    let joints = positions
        .iter()
        .enumerate()
        .map(|(i, p)| Joint3d {
            name: motion3d::NAMES[i],
            x: p[0] - pelvis[0],
            y: p[1] - pelvis[1],
            z: p[2] - pelvis[2],
            input_score: if upper_body && i < 8 {
                0.0
            } else {
                scores[i][2]
            },
        })
        .collect::<Vec<_>>();
    let angles = motion3d::angles3d(&joints, 0.3);
    Ok((
        points,
        Some(Pose3d {
            ear_head: true,
            model: "MediaPipe-Pose",
            coordinate_space: "root_relative_camera",
            units: "estimated_meters",
            history_frames: 1,
            joints,
            angles,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn heatmap_refines_same_frame_coordinates_without_changing_confidence() {
        let mut screen = vec![0.0; 195];
        for i in 0..39 {
            screen[i * 5] = 128.0;
            screen[i * 5 + 1] = 128.0;
            screen[i * 5 + 3] = 1.2;
            screen[i * 5 + 4] = 2.3;
        }
        let mut heatmap = vec![-20.0; 64 * 64 * 39];
        heatmap[(31 * 64 + 34) * 39] = 8.0;
        refine_from_heatmap(&mut screen, &heatmap);
        assert!((screen[0] - 136.0).abs() < 0.01);
        assert!((screen[1] - 124.0).abs() < 0.01);
        assert_eq!(screen[3], 1.2);
        assert_eq!(screen[4], 2.3);
        assert_eq!(screen[5], 128.0);
        assert_eq!(screen[6], 128.0);
        screen[0] = -1.0;
        screen[1] = 300.0;
        refine_from_heatmap(&mut screen, &heatmap);
        assert_eq!(screen[0], -1.0);
        assert_eq!(screen[1], 300.0);
    }
    #[test]
    fn maps_native_landmarks_without_inventing_face_or_small_toes() {
        let mut screen = vec![0.0; 195];
        let mut world = vec![0.0; 117];
        for i in 0..39 {
            screen[i * 5] = i as f32 * 3.0 + 10.0;
            screen[i * 5 + 1] = i as f32 * 2.0 + 20.0;
            screen[i * 5 + 3] = 10.0;
            screen[i * 5 + 4] = 10.0;
            for axis in 0..3 {
                world[i * 3 + axis] = i as f32 * 0.01 * (axis + 1) as f32;
            }
        }
        let (points, pose) = decode(&screen, &world, 1.0, 100.0, 200.0, 512.0, false).unwrap();
        assert_eq!(points.len(), 133);
        assert_eq!((points[7].x, points[7].y), (198.0, 292.0));
        let skeleton = skeleton_pixels(&points);
        assert_eq!(skeleton[8][0], (points[5].x + points[6].x) / 2.0);
        assert_eq!(skeleton[8][1], (points[5].y + points[6].y) / 2.0);
        assert_eq!(skeleton[9], [points[0].x, points[0].y, points[0].score]);
        assert_eq!(skeleton[10][1], (points[3].y + points[4].y) / 2.0);
        assert!(points[17].score > 0.9 && points[19].score > 0.9);
        assert_eq!(points[18].score, 0.0);
        assert_eq!(points[21].score, 0.0);
        assert!(points[23..91].iter().all(|p| p.score == 0.0));
        assert_eq!(points[99].x, 100.0 + screen[19 * 5] * 2.0);
        let pose = pose.unwrap();
        assert!(pose.ear_head);
        assert!((pose.joints[10].x + 0.16).abs() < 1e-6);
        let legacy = skeleton_pixels_for(&points, false);
        assert_eq!(legacy[10][1], (points[1].y + points[2].y) / 2.0);
        assert_eq!(pose.joints.len(), 17);
        assert_eq!(pose.units, "estimated_meters");
        assert!(pose.joints[0].x.abs() < 1e-6);
        assert!((pose.joints[13].x + 0.085).abs() < 1e-6);
        let (upper, pose) = decode(&screen, &world, 1.0, 0.0, 0.0, 256.0, true).unwrap();
        assert!(upper[11..23].iter().all(|p| p.score == 0.0));
        assert!(
            pose.unwrap().joints[..8]
                .iter()
                .all(|p| p.input_score == 0.0)
        );
        let (missing, pose) = decode(&screen, &world, 0.1, 0.0, 0.0, 256.0, true).unwrap();
        assert!(pose.is_none());
        assert!(missing.iter().all(|p| p.score == 0.0));
    }
}
