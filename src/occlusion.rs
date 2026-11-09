//! Reject uncertain arm landmarks lying on a reliably observed opposite arm.
//! This is observation gating, not temporal filtering or fabricated joint data.
use crate::pose::Keypoint;
use opencv::core::Size;

/// A short-lived crop/history prediction, never a measured joint position.
pub struct HandReference {
    pub side: usize,
    pub wrist: [f32; 2],
    pub palm_span: f32,
}

/// A recent independent hand track can disambiguate a moderately confident
/// body wrist copied onto the opposite arm. Only confidence is invalidated;
/// a fresh hand inference may restore the current pixel observation afterwards.
pub fn mask_recent_hand_overlap(
    points: &mut [Keypoint],
    size: Size,
    references: &[HandReference],
) -> [bool; 17] {
    let mut rejected = [false; 17];
    if points.len() < 17 || size.width <= 0 {
        return rejected;
    }
    let original = points.to_vec();
    for reference in references {
        let (wrist, opposite) = match reference.side {
            0 => (9, [6, 8, 10]),
            1 => (10, [5, 7, 9]),
            _ => continue,
        };
        let p = original[wrist];
        if !(0.3..0.8).contains(&p.score)
            || p.x < 0.0
            || p.y < 0.0
            || !p.x.is_finite()
            || !p.y.is_finite()
        {
            continue;
        }
        let contradiction = (p.x - reference.wrist[0]).hypot(p.y - reference.wrist[1]);
        if contradiction <= (reference.palm_span * 3.0).max(size.width as f32 * 0.06) {
            continue;
        }
        for pair in opposite.windows(2) {
            let a = original[pair[0]];
            let b = original[pair[1]];
            let confidence = a.score.min(b.score);
            if confidence < 0.8
                || confidence - p.score < 0.1
                || [a, b]
                    .iter()
                    .any(|q| q.x < 0.0 || q.y < 0.0 || !q.x.is_finite() || !q.y.is_finite())
            {
                continue;
            }
            let length = (b.x - a.x).hypot(b.y - a.y);
            let radius =
                (length * 0.15).clamp(size.width as f32 * 0.008, size.width as f32 * 0.025);
            if length > radius * 2.0 && distance_to_segment(p, a, b) <= radius {
                points[wrist].score = 0.0;
                rejected[wrist] = true;
                break;
            }
        }
    }
    rejected
}

fn distance_to_segment(p: Keypoint, a: Keypoint, b: Keypoint) -> f32 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let t = (((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy).max(1e-6)).clamp(0.0, 1.0);
    (p.x - a.x - t * dx).hypot(p.y - a.y - t * dy)
}

pub fn mask_arm_overlap(points: &mut [Keypoint], size: Size) -> [bool; 17] {
    let mut rejected = [false; 17];
    if points.len() < 17 || size.width <= 0 {
        return rejected;
    }
    let original = points.to_vec();
    let usable = |p: Keypoint| p.x.is_finite() && p.y.is_finite() && p.x >= 0.0 && p.y >= 0.0;
    for (weak, opposite) in [([7, 9], [6, 8, 10]), ([8, 10], [5, 7, 9])] {
        for index in weak {
            let p = original[index];
            if !usable(p) || !(0.0..0.75).contains(&p.score) {
                continue;
            }
            for pair in opposite.windows(2) {
                let a = original[pair[0]];
                let b = original[pair[1]];
                let confidence = a.score.min(b.score);
                if !usable(a) || !usable(b) || confidence < 0.8 || confidence - p.score < 0.2 {
                    continue;
                }
                let length = (b.x - a.x).hypot(b.y - a.y);
                let radius =
                    (length * 0.15).clamp(size.width as f32 * 0.008, size.width as f32 * 0.025);
                if length > radius * 2.0 && distance_to_segment(p, a, b) <= radius {
                    rejected[index] = true;
                    points[index].score = 0.0;
                    break;
                }
            }
        }
    }
    rejected
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recent_hand_rejects_forearm_alias_but_keeps_crossing_and_fresh_hand() {
        let mut p = fixture();
        p.resize(133, p[0]);
        for (i, x, y, score) in [
            (5, 778.0, 358.0, 1.0),
            (6, 726.0, 364.0, 1.0),
            (7, 720.0, 472.0, 0.900),
            (8, 725.0, 500.0, 0.563),
            (9, 574.0, 412.0, 0.997),
            (10, 660.0, 470.0, 0.779),
        ] {
            p[i] = Keypoint {
                name: "arm",
                x,
                y,
                score,
            };
        }
        let size = Size::new(1280, 720);
        assert!(!mask_arm_overlap(&mut p, size)[10]);
        let before = p.clone();
        let reference = HandReference {
            side: 1,
            wrist: [500.0, 400.0],
            palm_span: 24.0,
        };
        let rejected = mask_recent_hand_overlap(&mut p, size, &[reference]);
        assert!(rejected[10]);
        for i in 0..133 {
            assert_eq!((p[i].x, p[i].y), (before[i].x, before[i].y));
            if i != 10 {
                assert_eq!(p[i].score, before[i].score);
            }
        }
        crate::details::merge_hand_landmarks(
            &mut p,
            &[crate::details::HandPose {
                side: "right".into(),
                confidence: 0.72,
                image_landmarks: [[500.0, 400.0]; 21],
                world_landmarks: [[0.0; 3]; 21],
            }],
        );
        assert_eq!((p[10].x, p[10].y, p[10].score), (500.0, 400.0, 0.72));
        for (score, wrist) in [(0.93, [500.0, 400.0]), (0.779, [660.0, 470.0])] {
            let mut p = before.clone();
            p[10].score = score;
            assert!(
                !mask_recent_hand_overlap(
                    &mut p,
                    size,
                    &[HandReference {
                        side: 1,
                        wrist,
                        palm_span: 24.0,
                    }]
                )[10]
            );
        }
        let mut p = before;
        assert!(!mask_recent_hand_overlap(&mut p, size, &[])[10]);
    }
    fn fixture() -> Vec<Keypoint> {
        let mut p = vec![
            Keypoint {
                name: "test",
                x: -1.0,
                y: -1.0,
                score: 0.0
            };
            17
        ];
        for (i, x, y, s) in [
            (5, 100.0, 100.0, 0.95),
            (7, 200.0, 100.0, 0.95),
            (9, 300.0, 100.0, 0.95),
            (6, 110.0, 105.0, 0.95),
            (8, 202.0, 103.0, 0.5),
            (10, 298.0, 102.0, 0.4),
        ] {
            p[i] = Keypoint {
                name: "arm",
                x,
                y,
                score: s,
            };
        }
        p
    }
    #[test]
    fn weak_hidden_arm_is_unobserved_without_moving_or_changing_visible_arm() {
        let mut p = fixture();
        let before = p.clone();
        let rejected = mask_arm_overlap(&mut p, Size::new(640, 480));
        assert!(rejected[8] && rejected[10]);
        for i in [5, 7, 9] {
            assert_eq!(p[i].score, before[i].score);
        }
        for i in 0..17 {
            assert_eq!((p[i].x, p[i].y), (before[i].x, before[i].y));
        }
    }
    #[test]
    fn confident_overlap_and_crossing_arms_are_kept() {
        let mut p = fixture();
        p[8].score = 0.92;
        p[10].score = 0.93;
        assert!(
            !mask_arm_overlap(&mut p, Size::new(640, 480))
                .iter()
                .any(|x| *x)
        );
    }
    #[test]
    fn partial_occlusion_keeps_the_independently_visible_wrist() {
        let mut p = fixture();
        p[10].y = 200.0;
        let rejected = mask_arm_overlap(&mut p, Size::new(640, 480));
        assert!(rejected[8]);
        assert!(!rejected[10]);
        assert_eq!(p[10].score, 0.4);
        p[8].y = 180.0;
        p[8].score = 0.5;
        assert!(!mask_arm_overlap(&mut p, Size::new(640, 480))[8]);
    }
}
