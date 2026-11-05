use crate::pose::Keypoint;
use anyhow::Result;
use opencv::{
    core::{Mat, Point, Scalar, Size},
    imgproc,
    prelude::*,
};
use serde::Serialize;

struct Joint {
    name: &'static str,
    label: &'static str,
    points: [usize; 3], // The middle point is the angle's vertex.
    left: bool,
}

const JOINTS: &[Joint] = &[
    Joint {
        name: "left_shoulder",
        label: "L shoulder",
        points: [7, 5, 11],
        left: true,
    },
    Joint {
        name: "right_shoulder",
        label: "R shoulder",
        points: [8, 6, 12],
        left: false,
    },
    Joint {
        name: "left_elbow",
        label: "L elbow",
        points: [5, 7, 9],
        left: true,
    },
    Joint {
        name: "right_elbow",
        label: "R elbow",
        points: [6, 8, 10],
        left: false,
    },
    Joint {
        name: "left_wrist",
        label: "L wrist",
        points: [7, 9, 100],
        left: true,
    },
    Joint {
        name: "right_wrist",
        label: "R wrist",
        points: [8, 10, 121],
        left: false,
    },
    Joint {
        name: "left_hip",
        label: "L hip",
        points: [5, 11, 13],
        left: true,
    },
    Joint {
        name: "right_hip",
        label: "R hip",
        points: [6, 12, 14],
        left: false,
    },
    Joint {
        name: "left_knee",
        label: "L knee",
        points: [11, 13, 15],
        left: true,
    },
    Joint {
        name: "right_knee",
        label: "R knee",
        points: [12, 14, 16],
        left: false,
    },
    Joint {
        name: "left_ankle",
        label: "L ankle",
        points: [13, 15, 17],
        left: true,
    },
    Joint {
        name: "right_ankle",
        label: "R ankle",
        points: [14, 16, 20],
        left: false,
    },
];

#[derive(Clone, Debug, Serialize)]
pub struct JointAngle {
    pub joint: &'static str,
    /// Unsigned interior angle in the image plane (0..180 degrees).
    /// Null when landmarks are missing, unreliable, off-frame, or coincident.
    pub degrees: Option<f64>,
}

fn interior_angle(a: &Keypoint, b: &Keypoint, c: &Keypoint) -> Option<f64> {
    let u = (
        f64::from(a.x) - f64::from(b.x),
        f64::from(a.y) - f64::from(b.y),
    );
    let v = (
        f64::from(c.x) - f64::from(b.x),
        f64::from(c.y) - f64::from(b.y),
    );
    if u.0.hypot(u.1) < 1e-6 || v.0.hypot(v.1) < 1e-6 {
        return None;
    }
    let degrees = (u.0 * v.1 - u.1 * v.0)
        .abs()
        .atan2(u.0 * v.0 + u.1 * v.1)
        .to_degrees();
    degrees.is_finite().then_some(degrees)
}

pub fn calculate(points: &[Keypoint], threshold: f32, size: Size) -> Vec<JointAngle> {
    calculate_scoped(points, threshold, size, false)
}
pub fn calculate_scoped(
    points: &[Keypoint],
    threshold: f32,
    size: Size,
    upper_body: bool,
) -> Vec<JointAngle> {
    let usable = |p: &&Keypoint| {
        p.score.is_finite()
            && p.score > 0.0
            && p.score >= threshold
            && p.x.is_finite()
            && p.y.is_finite()
            && p.x >= 0.0
            && p.y >= 0.0
            && p.x < size.width as f32
            && p.y < size.height as f32
    };
    JOINTS
        .iter()
        .filter(|joint| {
            !upper_body
                || !(joint.name.contains("hip")
                    || joint.name.contains("knee")
                    || joint.name.contains("ankle"))
        })
        .map(|joint| {
            let mut indices = joint.points;
            if upper_body && joint.name == "left_shoulder" {
                indices = [7, 5, 6];
            }
            if upper_body && joint.name == "right_shoulder" {
                indices = [8, 6, 5];
            }
            let [a, b, c] = indices.map(|index| points.get(index).filter(usable));
            let degrees = match (a, b, c) {
                (Some(a), Some(b), Some(c)) => interior_angle(a, b, c),
                _ => None,
            };
            JointAngle {
                joint: joint.name,
                degrees,
            }
        })
        .collect()
}

pub fn draw(image: &mut Mat, points: &[Keypoint], angles: &[JointAngle]) -> Result<()> {
    draw_labeled(image, points, angles, "2D")
}

pub fn draw_labeled(
    image: &mut Mat,
    points: &[Keypoint],
    angles: &[JointAngle],
    space: &str,
) -> Result<()> {
    for joint in JOINTS {
        let Some(degrees) = angles
            .iter()
            .find(|a| a.joint == joint.name)
            .and_then(|a| a.degrees)
        else {
            continue;
        };
        let Some(vertex) = points.get(joint.points[1]) else {
            continue;
        };
        // OpenCV's built-in font is ASCII; use 'deg' rather than a degree symbol.
        let text = format!("{} {:.0} deg {}", joint.label, degrees, space);
        let mut baseline = 0;
        let scale = 0.42;
        let size = imgproc::get_text_size(
            &text,
            imgproc::FONT_HERSHEY_SIMPLEX,
            scale,
            1,
            &mut baseline,
        )?;
        let x = if joint.left {
            vertex.x as i32 - size.width - 8
        } else {
            vertex.x as i32 + 8
        };
        let origin = Point::new(
            x.clamp(2, (image.cols() - size.width - 2).max(2)),
            (vertex.y as i32 - 8).clamp(
                size.height + 2,
                (image.rows() - baseline - 2).max(size.height + 2),
            ),
        );
        // Dark outline keeps labels legible over clothing and skeleton lines.
        imgproc::put_text(
            image,
            &text,
            origin,
            imgproc::FONT_HERSHEY_SIMPLEX,
            scale,
            Scalar::all(0.0),
            3,
            imgproc::LINE_AA,
            false,
        )?;
        imgproc::put_text(
            image,
            &text,
            origin,
            imgproc::FONT_HERSHEY_SIMPLEX,
            scale,
            Scalar::new(0.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32, y: f32) -> Keypoint {
        Keypoint {
            name: "test",
            x,
            y,
            score: 0.9,
        }
    }

    #[test]
    fn computes_right_straight_and_acute_angles_independent_of_scale() {
        let vertex = point(10.0, 10.0);
        for (a, c, expected) in [
            (point(10.0, 5.0), point(15.0, 10.0), 90.0),
            (point(5.0, 10.0), point(20.0, 10.0), 180.0),
            (point(20.0, 10.0), point(15.0, 15.0), 45.0),
        ] {
            assert!((interior_angle(&a, &vertex, &c).unwrap() - expected).abs() < 1e-9);
        }
        assert!(interior_angle(&vertex, &vertex, &point(15.0, 10.0)).is_none());
    }

    #[test]
    fn unreliable_or_missing_landmarks_produce_null_angles() {
        let mut points = vec![point(10.0, 10.0); 17];
        points[5] = point(10.0, 5.0);
        points[7] = point(10.0, 10.0);
        points[9] = point(15.0, 10.0);
        let get = |points: &[Keypoint]| calculate(points, 0.3, Size::new(100, 100));
        let result = get(&points);
        assert_eq!(result[2].joint, "left_elbow");
        assert_eq!(result[2].degrees, Some(90.0));
        assert!(result[4].degrees.is_none()); // body-only model lacks hand landmarks
        assert!(result[10].degrees.is_none()); // body-only model lacks toes
        points[9].score = 0.1;
        assert!(get(&points)[2].degrees.is_none());
        points[9].score = 0.9;
        points[9].x = -1.0;
        assert!(get(&points)[2].degrees.is_none());
        points[9].x = f32::NAN;
        assert!(get(&points)[2].degrees.is_none());
    }

    #[test]
    fn wholebody_wrist_and_ankle_use_correct_hand_and_toe_points() {
        let mut points = vec![point(10.0, 10.0); 133];
        for (a, b, c) in [(7, 9, 100), (8, 10, 121), (13, 15, 17), (14, 16, 20)] {
            points[a] = point(10.0, 5.0);
            points[b] = point(10.0, 10.0);
            points[c] = point(15.0, 10.0);
        }
        let result = calculate(&points, 0.3, Size::new(200, 200));
        for index in [4, 5, 10, 11] {
            assert_eq!(result[index].degrees, Some(90.0));
        }
        let mut image =
            Mat::new_rows_cols_with_default(200, 200, opencv::core::CV_8UC3, Scalar::all(0.0))
                .unwrap();
        draw(&mut image, &points, &result).unwrap();
    }
}
