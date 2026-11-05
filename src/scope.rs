//! Active upper-body landmarks. Lower-body model slots are internal placeholders.
pub fn coco_upper(index: usize) -> bool {
    index <= 10 || index >= 23
}
pub fn body_upper(index: usize) -> bool {
    index >= 8
}
pub fn mask_lower(points: &mut [crate::pose::Keypoint]) {
    for (i, p) in points.iter_mut().enumerate() {
        if !coco_upper(i) {
            p.x = -1.0;
            p.y = -1.0;
            p.score = 0.0;
        }
    }
}
/// Keep sparse head anchors for the body model, but omit face/foot detail tracking.
pub fn mask_details(points: &mut [crate::pose::Keypoint]) {
    for point in points.iter_mut().take(91).skip(17) {
        point.x = -1.0;
        point.y = -1.0;
        point.score = 0.0;
    }
}
