use crate::detector::Detection;
use anyhow::Result;
use opencv::core::Size;

/// Reuse the last selected person crop until the next detector refresh.
/// Empty results are cached too, so an empty scene does not run detection every frame.
pub struct DetectionCache {
    interval: u64,
    last_frame: Option<u64>,
    size: Option<Size>,
    detections: Vec<Detection>,
}

impl DetectionCache {
    pub fn new(interval: u64) -> Self {
        assert!(interval > 0);
        Self {
            interval,
            last_frame: None,
            size: None,
            detections: Vec::new(),
        }
    }

    /// Returns whether detection ran on this frame. A resolution change forces a refresh.
    pub fn update(
        &mut self,
        frame: u64,
        size: Size,
        detect: impl FnOnce() -> Result<Vec<Detection>>,
    ) -> Result<bool> {
        let due = self.size != Some(size)
            || self
                .last_frame
                .is_none_or(|last| frame < last || frame.saturating_sub(last) >= self.interval);
        if due {
            // Commit only after successful inference, so a failure cannot mark a stale cache fresh.
            let detections = detect()?;
            self.detections = detections;
            self.last_frame = Some(frame);
            self.size = Some(size);
        }
        Ok(due)
    }

    pub fn detections(&self) -> &[Detection] {
        &self.detections
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::Bbox;

    fn person(x: f32) -> Detection {
        Detection {
            bbox: Bbox {
                x,
                y: 10.0,
                width: 80.0,
                height: 160.0,
            },
            score: 0.9,
        }
    }

    #[test]
    fn detects_first_frame_then_every_fifth_frame_and_reuses_selected_crop() {
        let mut cache = DetectionCache::new(5);
        let size = Size::new(640, 480);
        let mut calls = Vec::new();
        for frame in 0..12 {
            let ran = cache
                .update(frame, size, || {
                    calls.push(frame);
                    Ok(vec![person(frame as f32)])
                })
                .unwrap();
            assert_eq!(ran, frame % 5 == 0);
            assert_eq!(cache.detections()[0].bbox.x, (frame / 5 * 5) as f32);
        }
        assert_eq!(calls, vec![0, 5, 10]);
    }

    #[test]
    fn refresh_clears_lost_person_and_caches_empty_results_until_reacquisition() {
        let mut cache = DetectionCache::new(2);
        let size = Size::new(640, 480);
        cache.update(0, size, || Ok(vec![person(10.0)])).unwrap();
        cache
            .update(1, size, || panic!("Unexpected detector call"))
            .unwrap();
        cache.update(2, size, || Ok(Vec::new())).unwrap();
        assert!(cache.detections().is_empty());
        cache
            .update(3, size, || panic!("Empty result should be cached"))
            .unwrap();
        cache.update(4, size, || Ok(vec![person(20.0)])).unwrap();
        assert_eq!(cache.detections()[0].bbox.x, 20.0);
    }

    #[test]
    fn resolution_changes_force_detection_and_interval_one_detects_each_frame() {
        let mut cache = DetectionCache::new(5);
        cache
            .update(0, Size::new(640, 480), || Ok(vec![person(10.0)]))
            .unwrap();
        assert!(
            cache
                .update(1, Size::new(1280, 720), || Ok(vec![person(20.0)]))
                .unwrap()
        );
        assert_eq!(cache.detections()[0].bbox.x, 20.0);
        let mut cache = DetectionCache::new(1);
        for frame in 0..3 {
            assert!(
                cache
                    .update(frame, Size::new(640, 480), || Ok(Vec::new()))
                    .unwrap()
            );
        }
    }
}
