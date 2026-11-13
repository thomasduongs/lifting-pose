//! Shared observation, confidence gating, hand fusion and skeletal fitting path.
use crate::motion3d::MotionLifter;
use crate::pose::{Bbox, PoseEstimator};
use crate::{PersonPose, angles, details, motion3d, occlusion, scope, tracking, twin};
use anyhow::Result;
use opencv::{
    core::{Mat, Size},
    prelude::*,
};

#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub full_body: bool,
    pub image_support_floor: Option<f32>,
    pub no_3d: bool,
    pub head_orientation: bool,
    pub no_dimension_tracking: bool,
    pub threshold: f32,
    pub focal_length_px: Option<f64>,
}
impl From<&crate::Args> for Options {
    fn from(args: &crate::Args) -> Self {
        Self {
            full_body: args.full_body,
            image_support_floor: None,
            no_3d: args.no_3d,
            head_orientation: args.head_orientation,
            no_dimension_tracking: args.no_dimension_tracking,
            threshold: args.threshold,
            focal_length_px: args.focal_length_px,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn track_person(
    frame: &Mat,
    estimator: &mut PoseEstimator,
    lifter: &mut Option<MotionLifter>,
    hand_model: &mut Option<details::HandModel>,
    twin_recorder: &mut twin::TwinRecorder,
    dimension_tracker: &mut tracking::DimensionTracker,
    options: Options,
    index: u64,
    time_seconds: f64,
    person_index: usize,
    bbox: Bbox,
    detection_score: Option<f32>,
) -> Result<PersonPose> {
    let mut keypoints = estimator.infer(frame, bbox)?;
    scope::mask_details(&mut keypoints);
    if !options.full_body {
        scope::mask_lower(&mut keypoints);
    }
    let mut occluded = if estimator.is_mediapipe() {
        occlusion::mask_arm_overlap(&mut keypoints, frame.size()?)
    } else {
        [false; 17]
    };
    let hand_references = hand_model
        .as_ref()
        .map(|model| model.occlusion_references(time_seconds))
        .unwrap_or_default();
    let hands = match hand_model.as_mut() {
        Some(model) => model.infer(
            frame,
            &keypoints,
            time_seconds,
            Some(bbox),
            estimator.hand_crop_wrists().as_ref(),
        )?,
        None => Vec::new(),
    };
    if estimator.is_mediapipe() {
        let recent =
            occlusion::mask_recent_hand_overlap(&mut keypoints, frame.size()?, &hand_references);
        for i in 0..17 {
            occluded[i] |= recent[i];
        }
    }
    let body_wrists = [keypoints[9], keypoints[10]];
    let predicted_hands = hand_model
        .as_ref()
        .map(|model| model.predictions(time_seconds))
        .unwrap_or_default();
    estimator.confirm_hands(&hands);
    if estimator.is_mediapipe() {
        match &hand_model {
            Some(model) => model.merge_landmarks(&mut keypoints, &hands, time_seconds),
            None => details::merge_hand_landmarks(&mut keypoints, &hands),
        }
    }
    let observed_keypoints = keypoints.clone();
    let observed_native_scores = estimator.observed_native_scores();
    let joint_frame =
        estimator.stabilize(frame, &mut keypoints, time_seconds, bbox, &hands, &occluded)?;
    let head = if options.head_orientation {
        details::head_pose(&keypoints, options.threshold)
    } else {
        None
    };
    let angles = if options.full_body {
        angles::calculate(&keypoints, options.threshold, frame.size()?)
    } else {
        angles::calculate_scoped(&keypoints, options.threshold, frame.size()?, true)
    };
    let mut pose3d = if options.no_3d {
        None
    } else if estimator.is_mediapipe() {
        estimator.take_pose3d()
    } else {
        match lifter.as_mut() {
            Some(lifter) => lifter.infer(&keypoints, bbox, frame.size()?, options.threshold)?,
            None => None,
        }
    };
    if estimator.is_mediapipe() {
        if let Some(pose) = &mut pose3d {
            for (coco, joint) in [(7, 12), (9, 13), (8, 15), (10, 16)] {
                if occluded[coco] {
                    pose.joints[joint].input_score = 0.0;
                }
            }
            let width = (keypoints[5].x - keypoints[6].x).hypot(keypoints[5].y - keypoints[6].y);
            for hand in &hands {
                let side = if hand.side == "left" { 0 } else { 1 };
                let old = body_wrists[side];
                let measured = keypoints[if side == 0 { 9 } else { 10 }];
                let wrist = [measured.x, measured.y];
                if (old.x - wrist[0]).hypot(old.y - wrist[1]) > (width * 0.3).max(40.0) {
                    // The hand detector has located the wrist elsewhere. Keep
                    // its reliable 2D constraint without trusting stale body depth.
                    pose.joints[if side == 0 { 13 } else { 16 }].input_score = 0.0;
                }
            }
            pose.angles = motion3d::angles3d(&pose.joints, options.threshold);
        }
    }
    // Live mode rejects hallucinated native depth before the global skeletal fit,
    // so an invisible limb cannot corrupt the visible limb's fitting prior.
    if let (Some(floor), Some(pose)) = (options.image_support_floor, &mut pose3d) {
        gate_image_support(pose, &keypoints, frame.size()?, floor);
    }
    let image_torso_scale = estimator.image_torso_scale();
    let image_torso_reference_reset = estimator.image_torso_reference_reset();
    let image_torso_span = estimator.image_torso_span();
    twin_recorder.set_image_torso_scale(image_torso_scale, image_torso_span);
    let twin = pose3d.as_ref().and_then(|pose| {
        twin_recorder.observe(
            index,
            time_seconds,
            pose,
            &keypoints,
            &hands,
            head.clone(),
            bbox,
            Size::new(frame.cols(), frame.rows()),
            options.threshold,
            options.focal_length_px,
        )
    });
    let tracking = if options.no_dimension_tracking || options.no_3d || estimator.is_mediapipe() {
        None
    } else {
        Some(dimension_tracker.refine(
            &keypoints,
            twin.as_ref(),
            bbox,
            frame.size()?,
            options.threshold,
        ))
    };
    Ok(PersonPose {
        person_index,
        bbox,
        detection_score,
        keypoints,
        body_wrists,
        observed_keypoints,
        observed_native_scores,
        joint_tracking: joint_frame.updates,
        predicted_joints: joint_frame.predicted,
        arm_refined: estimator.arm_refined().to_vec(),
        angles,
        pose3d,
        twin,
        tracking,
        hands,
        predicted_hands,
        image_torso_scale,
        image_torso_reference_reset,
        image_torso_span,
        head,
    })
}

pub fn image_supported(p: &crate::pose::Keypoint, size: Size, floor: f32) -> bool {
    p.score.is_finite()
        && p.score >= floor
        && p.x.is_finite()
        && p.y.is_finite()
        && p.x >= 0.0
        && p.y >= 0.0
        && p.x < size.width as f32
        && p.y < size.height as f32
}
pub fn gate_image_support(
    pose: &mut motion3d::Pose3d,
    points: &[crate::pose::Keypoint],
    size: Size,
    floor: f32,
) {
    for (coco, native) in [(5, 11), (7, 12), (9, 13), (6, 14), (8, 15), (10, 16)] {
        if !points
            .get(coco)
            .is_some_and(|p| image_supported(p, size, floor))
        {
            if let Some(joint) = pose.joints.get_mut(native) {
                joint.input_score = 0.0;
            }
        }
    }
}
