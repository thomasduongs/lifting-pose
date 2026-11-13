//! Native live split view: anatomical model (LEFT half) fitted from the live camera
//! pose (RIGHT half). Camera-driven constrained kinematics only: nothing here is a
//! measurement of muscle activation or force. Nothing is recorded or transmitted.

use crate::biomech_estimate::{Estimate, EstimateMode, Estimator, Sample};
use crate::biomech_live_capture::{
    CaptureState, Config as CaptureConfig, LiveCapture, PoseFrame, VideoFrame,
};
use anyhow::{Context, Result, anyhow, bail};
use biomech_lite::live_fit::{FitResult, LiveFitter};
use biomech_lite::model::Model;
use biomech_viewer::camera::Camera;
use biomech_viewer::render::{Options as LayerOptions, Renderer};
use biomech_viewer::scene::VisualModel;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use opencv::{core::Mat, imgcodecs, prelude::*};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Config {
    pub capture: CaptureConfig,
    pub model: PathBuf,
    pub visual: PathBuf,
    pub max_frames: Option<u64>,
    pub snapshot: Option<PathBuf>,
}

const HEADER: usize = 60;
const FOOTER: usize = 40;
const STALE_SECONDS: f64 = 0.5;
const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(90);
const BG: u32 = 0x10141c;
const PANEL: u32 = 0x1a202b;
const TEXT: u32 = 0xdfe6f0;
const DIM: u32 = 0x8b97a8;
const GOOD: u32 = 0x4fd18b;
const WARN: u32 = 0xf2b84b;
const BAD: u32 = 0xe5534b;

// ---------------------------------------------------------------- layout

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x as f32
            && y >= self.y as f32
            && x < (self.x + self.w) as f32
            && y < (self.y + self.h) as f32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub header: Rect,
    pub footer: Rect,
    pub left: Rect,
    pub right: Rect,
}

/// Header and footer span the full width; the content rectangle between them is split
/// EXACTLY 50/50 (any odd leftover column is left blank, never given to one side).
pub fn split_layout(width: usize, height: usize) -> Layout {
    let content_h = height.saturating_sub(HEADER + FOOTER);
    let half = width / 2;
    Layout {
        header: Rect {
            x: 0,
            y: 0,
            w: width,
            h: HEADER.min(height),
        },
        footer: Rect {
            x: 0,
            y: height.saturating_sub(FOOTER),
            w: width,
            h: FOOTER.min(height),
        },
        left: Rect {
            x: 0,
            y: HEADER,
            w: half,
            h: content_h,
        },
        right: Rect {
            x: half,
            y: HEADER,
            w: half,
            h: content_h,
        },
    }
}

/// Aspect-preserving fit of `src` into `dst`, centred (no crop, no mirror).
pub fn letterbox(src_w: usize, src_h: usize, dst: Rect) -> Rect {
    if src_w == 0 || src_h == 0 || dst.w == 0 || dst.h == 0 {
        return Rect {
            x: dst.x,
            y: dst.y,
            w: 0,
            h: 0,
        };
    }
    let scale = (dst.w as f64 / src_w as f64).min(dst.h as f64 / src_h as f64);
    let w = ((src_w as f64 * scale).round() as usize).clamp(1, dst.w);
    let h = ((src_h as f64 * scale).round() as usize).clamp(1, dst.h);
    Rect {
        x: dst.x + (dst.w - w) / 2,
        y: dst.y + (dst.h - h) / 2,
        w,
        h,
    }
}

// ---------------------------------------------------------------- status

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PoseStatus {
    /// Waiting for the first usable pose.
    Waiting,
    Tracking,
    /// Pose older than the staleness limit: the model is holding its last fit.
    Held {
        age: f64,
    },
    /// No person / no pose at all while the camera runs.
    Lost,
    Paused,
    Error,
    /// Fit ran but no arm had usable observations: model is holding its prior.
    NoArms,
}

pub fn pose_status(
    pose_age: Option<f64>,
    have_pose: bool,
    ever_posed: bool,
    paused: bool,
    error: bool,
) -> PoseStatus {
    if error {
        return PoseStatus::Error;
    }
    if paused {
        return PoseStatus::Paused;
    }
    match (have_pose, pose_age) {
        (true, Some(age)) if age > STALE_SECONDS => PoseStatus::Held { age },
        (true, Some(_)) => PoseStatus::Tracking,
        _ if ever_posed => PoseStatus::Lost,
        _ => PoseStatus::Waiting,
    }
}

impl PoseStatus {
    fn label(&self) -> String {
        match self {
            PoseStatus::Waiting => "WAITING FOR POSE".into(),
            PoseStatus::Tracking => "TRACKING".into(),
            PoseStatus::Held { age } => format!("HELD - pose {age:.1}s old"),
            PoseStatus::Lost => "LOST - no person".into(),
            PoseStatus::Paused => "PAUSED - fit frozen".into(),
            PoseStatus::Error => "ERROR".into(),
            PoseStatus::NoArms => "HELD - no arm supported".into(),
        }
    }
    fn color(&self) -> u32 {
        match self {
            PoseStatus::Tracking => GOOD,
            PoseStatus::Waiting | PoseStatus::Paused => WARN,
            _ => BAD,
        }
    }
}

// ---------------------------------------------------------------- drawing

struct Canvas<'a> {
    buf: &'a mut [u32],
    w: usize,
    h: usize,
}

impl Canvas<'_> {
    fn rect(&mut self, x: usize, y: usize, w: usize, h: usize, c: u32) {
        for yy in y..(y + h).min(self.h) {
            let a = yy * self.w + x.min(self.w);
            let b = yy * self.w + (x + w).min(self.w);
            self.buf[a..b].fill(c);
        }
    }
    fn outline(&mut self, r: Rect, c: u32) {
        if r.w == 0 || r.h == 0 {
            return;
        }
        self.rect(r.x, r.y, r.w, 1, c);
        self.rect(r.x, r.y + r.h - 1, r.w, 1, c);
        self.rect(r.x, r.y, 1, r.h, c);
        self.rect(r.x + r.w - 1, r.y, 1, r.h, c);
    }
    fn text(&mut self, x: usize, y: usize, s: &str, scale: usize, c: u32) {
        for (i, ch) in s.chars().enumerate() {
            let code = if (ch as usize) < 128 {
                ch as usize
            } else {
                b'?' as usize
            };
            let glyph = font8x8::legacy::BASIC_LEGACY[code];
            for (r, bits) in glyph.iter().enumerate() {
                for col in 0..8 {
                    if bits >> col & 1 == 1 {
                        self.rect(x + (i * 8 + col) * scale, y + r * scale, scale, scale, c);
                    }
                }
            }
        }
    }
    fn line(&mut self, a: (f32, f32), b: (f32, f32), c: u32) {
        let steps = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) as usize).clamp(1, 4000);
        for i in 0..=steps {
            let t = i as f32 / steps as f32;
            let (x, y) = (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t);
            if x >= 1.0 && y >= 1.0 {
                self.rect(x as usize - 1, y as usize - 1, 2, 2, c);
            }
        }
    }
}

/// Nearest-neighbour blit of an unmirrored RGB frame into its letterboxed rectangle.
fn blit_video(canvas: &mut Canvas, video: &VideoFrame, dst: Rect) {
    if dst.w == 0 || dst.h == 0 || video.rgb.len() < video.width * video.height {
        return;
    }
    for dy in 0..dst.h {
        let sy = dy * video.height / dst.h;
        let row = &video.rgb[sy * video.width..(sy + 1) * video.width];
        let out = (dst.y + dy) * canvas.w + dst.x;
        if out + dst.w > canvas.buf.len() {
            return;
        }
        for dx in 0..dst.w {
            canvas.buf[out + dx] = row[dx * video.width / dst.w];
        }
    }
}

const CHAINS: [(&str, [&str; 3]); 2] = [
    ("left", ["left_shoulder", "left_elbow", "left_wrist"]),
    ("right", ["right_shoulder", "right_elbow", "right_wrist"]),
];

/// Shared tracking output: body and confirmed hands, plus gray display-only hands. Pose and video may differ in age, so it is labelled.
fn draw_overlay(canvas: &mut Canvas, pose: &PoseFrame, video: &VideoFrame, dst: Rect) {
    let sx = dst.w as f32 / video.width.max(1) as f32;
    let sy = dst.h as f32 / video.height.max(1) as f32;
    for (_, names) in CHAINS {
        let pts: Vec<Option<(f32, f32)>> = names
            .iter()
            .map(|n| {
                pose.points
                    .iter()
                    .find(|p| p.0 == *n && p.3 > 0.3 && p.1.is_finite() && p.2.is_finite())
                    .map(|p| (dst.x as f32 + p.1 * sx, dst.y as f32 + p.2 * sy))
                    .filter(|p| dst.contains(p.0, p.1))
            })
            .collect();
        for w in pts.windows(2) {
            if let (Some(a), Some(b)) = (w[0], w[1]) {
                canvas.line(a, b, 0x3ddcff);
            }
        }
        for p in pts.iter().flatten() {
            canvas.rect(p.0 as usize - 3, p.1 as usize - 3, 6, 6, 0xffd34d);
        }
    }
    // Predictions remain a separate gray overlay and never drive the fit.
    for (hands, color) in [(&pose.predicted_hands, 0x7d858f), (&pose.hands, 0x52ef9a)] {
        for hand in hands {
            if hand.confidence < 0.3 {
                continue;
            }
            for i in 1..21 {
                let a = hand.image_landmarks[crate::details::HAND_PARENT[i]];
                let b = hand.image_landmarks[i];
                let a = (dst.x as f32 + a[0] * sx, dst.y as f32 + a[1] * sy);
                let b = (dst.x as f32 + b[0] * sx, dst.y as f32 + b[1] * sy);
                if dst.contains(a.0, a.1) && dst.contains(b.0, b.1) {
                    canvas.line(a, b, color);
                }
            }
        }
    }
}

pub struct Hud {
    pub status: PoseStatus,
    pub footer: String,
    pub notice: Option<String>,
    pub fit_note: Option<String>,
    pub muscle_lines: Vec<String>,
    pub buttons: Vec<(Rect, String, bool)>,
    pub overlay: bool,
    pub camera_title: String,
}

/// Compose the whole window: header + toolbar, model pixels left, camera right, footer.
pub fn compose_frame(
    buf: &mut [u32],
    w: usize,
    h: usize,
    model_pixels: Option<(&[u32], usize, usize)>,
    state: &CaptureState,
    hud: &Hud,
) {
    if buf.len() < w * h {
        return;
    }
    buf[..w * h].fill(BG);
    let lay = split_layout(w, h);
    let mut c = Canvas {
        buf: &mut buf[..w * h],
        w,
        h,
    };
    c.rect(
        lay.header.x,
        lay.header.y,
        lay.header.w,
        lay.header.h,
        PANEL,
    );
    c.rect(
        lay.footer.x,
        lay.footer.y,
        lay.footer.w,
        lay.footer.h,
        PANEL,
    );
    c.text(14, 8, "LIVE ANATOMICAL TWIN", 2, TEXT);
    c.text(14, 34, "tracked pose + estimated muscle activity", 1, DIM);
    for (r, label, on) in &hud.buttons {
        c.rect(r.x, r.y, r.w, r.h, if *on { 0x2d4f7c } else { 0x2a313e });
        c.outline(*r, if *on { 0x6aa7ee } else { 0x465063 });
        c.text(r.x + 6, r.y + (r.h.saturating_sub(8)) / 2, label, 1, TEXT);
    }

    // Left: model pixels, cropped to the pane if the renderer is a different size.
    if let Some((px, pw, ph)) = model_pixels {
        for y in 0..lay.left.h.min(ph) {
            let src = &px[y * pw..y * pw + pw.min(lay.left.w)];
            let o = (lay.left.y + y) * w + lay.left.x;
            c.buf[o..o + src.len()].copy_from_slice(src);
        }
    }
    // Right: letterboxed camera, never mirrored.
    c.rect(lay.right.x, lay.right.y, lay.right.w, lay.right.h, 0x07090c);
    if let Some(video) = &state.video {
        let dst = letterbox(video.width, video.height, lay.right);
        blit_video(&mut c, video, dst);
        if hud.overlay {
            if let Some(pose) = &state.pose {
                let age = video
                    .captured
                    .saturating_duration_since(pose.captured)
                    .as_secs_f64()
                    * 1000.0;
                let msg = if age <= STALE_SECONDS * 1000.0 {
                    draw_overlay(&mut c, pose, video, dst);
                    format!("2D overlay: estimated, pose {:.0}ms behind frame", age)
                } else {
                    format!("2D overlay held: pose {:.0}ms behind frame", age)
                };
                c.text(
                    lay.right.x + 8,
                    lay.right.y + lay.right.h.saturating_sub(16),
                    &msg,
                    1,
                    0xffd34d,
                );
            }
        }
    } else {
        let (msg, col) = match &state.error {
            Some(e) => (format!("CAMERA ERROR: {e}"), BAD),
            None => (state.status.clone(), WARN),
        };
        let cols = (lay.right.w / 8).saturating_sub(4).max(8);
        for (i, line) in wrap_text(&msg, cols)
            .iter()
            .take(lay.right.h / 12 / 2)
            .enumerate()
        {
            c.text(
                lay.right.x + 16,
                lay.right.y + lay.right.h / 3 + i * 12,
                line,
                1,
                col,
            );
        }
    }
    c.outline(lay.left, 0x303847);
    c.outline(lay.right, 0x303847);

    // Pane titles.
    for (r, t) in [
        (lay.left, "Model - live pose fit"),
        (lay.right, hud.camera_title.as_str()),
    ] {
        c.rect(r.x + 1, r.y + 1, r.w.saturating_sub(2), 20, 0x000000);
        c.text(r.x + 8, r.y + 6, t, 1, TEXT);
    }
    // Conspicuous status banner on the model side.
    let col = hud.status.color();
    if !matches!(hud.status, PoseStatus::Tracking) {
        c.rect(
            lay.left.x + 1,
            lay.left.y + 22,
            lay.left.w.saturating_sub(2),
            26,
            col,
        );
        c.text(
            lay.left.x + 10,
            lay.left.y + 26,
            &truncate(&hud.status.label(), lay.left.w / 16 - 2),
            2,
            0x101010,
        );
    } else {
        c.rect(lay.left.x + 8, lay.left.y + 28, 10, 10, col);
        c.text(lay.left.x + 24, lay.left.y + 29, "TRACKING", 1, col);
    }
    let mut ty = lay.left.y + 52;
    if let Some(err) = &state.error {
        for line in wrap_text(
            &format!("ERROR: {err}"),
            (lay.left.w / 8).saturating_sub(2).max(8),
        )
        .iter()
        .take(4)
        {
            c.text(lay.left.x + 8, ty, line, 1, BAD);
            ty += 12;
        }
    }
    if let Some(f) = &hud.fit_note {
        for line in wrap_text(f, (lay.left.w / 8).saturating_sub(2).max(8))
            .iter()
            .take(3)
        {
            c.text(lay.left.x + 8, ty, line, 1, DIM);
            ty += 12;
        }
    }
    if let Some(n) = &hud.notice {
        c.text(
            lay.left.x + 8,
            ty,
            &truncate(n, lay.left.w / 8 - 2),
            1,
            WARN,
        );
    }
    if !hud.muscle_lines.is_empty() {
        let max_lines = (lay.left.h / 3 / 12).max(1).min(hud.muscle_lines.len());
        let panel_h = max_lines * 12 + 12;
        let y = lay.left.y + lay.left.h.saturating_sub(panel_h + 6);
        c.rect(
            lay.left.x + 6,
            y,
            lay.left.w.saturating_sub(12),
            panel_h,
            0x101822,
        );
        for (i, line) in hud.muscle_lines.iter().take(max_lines).enumerate() {
            if line.starts_with("Activation scale:") {
                let step = lay.left.w.saturating_sub(24) / 5;
                for j in 0..5 {
                    let x = lay.left.x + 12 + j * step;
                    let color = biomech_viewer::render::muscle_activity_color(Some(j as f64 / 4.0));
                    c.rect(x, y + 7 + i * 12, 10.min(step), 7, color);
                    if step >= 50 {
                        c.text(x + 14, y + 6 + i * 12, &format!("{}%", j * 25), 1, color);
                    }
                }
                continue;
            }
            c.text(
                lay.left.x + 12,
                y + 6 + i * 12,
                &truncate(line, (lay.left.w / 8).saturating_sub(4)),
                1,
                if i == 0 { 0xf0c777 } else { TEXT },
            );
        }
    }
    c.text(
        14,
        lay.footer.y + 16,
        &truncate(&hud.footer, w / 8 - 2),
        1,
        DIM,
    );
}

fn wrap_text(s: &str, cols: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        let mut word = word.to_string();
        while word.chars().count() > cols {
            let head: String = word.chars().take(cols).collect();
            word = word.chars().skip(cols).collect();
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            out.push(head);
        }
        if !cur.is_empty() && cur.chars().count() + 1 + word.chars().count() > cols {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(&word);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max.saturating_sub(1)).collect::<String>() + "~"
    }
}

// ---------------------------------------------------------------- session

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Act {
    Front,
    Side,
    Back,
    Frame,
    Reset,
    Pause,
    Bones,
    Muscles,
    Ligaments,
    Overlay,
    Estimates,
}

struct Session {
    model: Arc<Model>,
    visual: VisualModel,
    fitter: LiveFitter,
    hand_fitter: biomech_lite::hand_fit::HandFitter,
    supported_hands: [bool; 2],
    estimator: Estimator,
    estimates_enabled: bool,
    estimate: Option<Arc<Estimate>>,
    estimate_error: Option<String>,
    estimate_arms: [bool; 2],
    last_source_time: f64,
    estimate_colors: Option<Vec<Option<f64>>>,
    renderer: Renderer,
    camera: Camera,
    layers: LayerOptions,
    qpos: Vec<f64>,
    fit: Option<FitResult>,
    paused: bool,
    overlay: bool,
    upper_framing: bool,
    dirty: bool,
    last_pose_index: Option<u64>,
    subject_segment: Option<usize>,
    ever_posed: bool,
    poses_fitted: u64,
    fit_ms: f64,
    notice: Option<String>,
    file_input: bool,
    last_consumed: Instant,
    // camera rate tracking
    last_video_index: Option<u64>,
    video_times: std::collections::VecDeque<Instant>,
}

impl Session {
    fn new(model_path: &Path, visual_path: &Path, left: Rect) -> Result<Self> {
        let model = Model::load(model_path)
            .map_err(|e| anyhow!("Cannot load biomech model {}: {e}", model_path.display()))?;
        let visual = VisualModel::load(visual_path, &model)
            .map_err(|e| anyhow!("Cannot load biomech visuals {}: {e}", visual_path.display()))?;
        let model = Arc::new(model);
        let fitter = LiveFitter::new(model.clone())
            .map_err(|e| anyhow!("Cannot create live fitter: {e}"))?;
        let hand_fitter =
            biomech_lite::hand_fit::HandFitter::new(model.clone()).map_err(|e| anyhow!(e))?;
        let estimator = Estimator::start(model.clone()).map_err(|e| anyhow!(e))?;
        let mut renderer = Renderer::new(left.w.max(1), left.h.max(1));
        renderer.threads = 4;
        let qpos = model.standing_qpos.clone();
        let mut s = Session {
            model,
            visual,
            fitter,
            hand_fitter,
            supported_hands: [false; 2],
            estimator,
            estimates_enabled: true,
            estimate: None,
            estimate_error: None,
            estimate_arms: [false; 2],
            last_source_time: 0.0,
            estimate_colors: None,
            renderer,
            camera: Camera::front(),
            layers: LayerOptions::default(),
            qpos,
            fit: None,
            paused: false,
            overlay: true,
            upper_framing: false,
            dirty: true,
            last_pose_index: None,
            subject_segment: None,
            ever_posed: false,
            poses_fitted: 0,
            fit_ms: 0.0,
            notice: None,
            file_input: false,
            last_consumed: Instant::now(),
            last_video_index: None,
            video_times: Default::default(),
        };
        s.frame_camera(left);
        Ok(s)
    }

    /// Only called on start / F / reset - never per frame.
    fn frame_camera(&mut self, left: Rect) {
        self.camera.aspect = left.w.max(1) as f64 / left.h.max(1) as f64;
        let standing = self.model.standing_qpos.clone();
        match self.model.kinematics(&standing) {
            Ok(kin) => {
                self.camera
                    .fit(&self.model, &kin, &self.visual, self.upper_framing);
                // Future live poses can reach beyond the standing mesh bounds.
                self.camera.distance *= 1.2;
                self.dirty = true;
            }
            Err(e) => self.notice = Some(format!("framing failed: {e}")),
        }
    }

    fn reset(&mut self, left: Rect) {
        self.fitter.reset();
        self.hand_fitter.reset();
        self.estimator.reset();
        self.estimate = None;
        self.estimate_error = None;
        self.estimate_arms = [false; 2];
        self.supported_hands = [false; 2];
        self.qpos = self.model.standing_qpos.clone();
        self.fit = None;
        self.last_pose_index = None;
        self.frame_camera(left);
        self.dirty = true;
    }

    fn track_video(&mut self, state: &CaptureState) {
        if let Some(v) = &state.video {
            if self.last_video_index != Some(v.index) {
                self.last_video_index = Some(v.index);
                self.video_times.push_back(Instant::now());
            }
        }
        while self
            .video_times
            .front()
            .is_some_and(|t| t.elapsed() > Duration::from_secs(2))
        {
            self.video_times.pop_front();
        }
    }

    fn camera_fps(&self) -> f64 {
        match (self.video_times.front(), self.video_times.len()) {
            (Some(first), n) if n >= 2 => {
                let span = first.elapsed().as_secs_f64();
                if span > 0.0 {
                    (n - 1) as f64 / span
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }

    /// Fit the newest distinct pose once. Returns true if a pose was consumed.
    fn follow(&mut self, state: &CaptureState) -> bool {
        self.track_video(state);
        let Some(pose) = &state.pose else {
            if self.estimate_arms != [false; 2] {
                self.estimator.reset();
            }
            self.estimate_arms = [false; 2];
            return false;
        };
        if self.last_pose_index == Some(pose.index) {
            return false;
        }
        self.last_pose_index = Some(pose.index);
        self.last_source_time = pose.observation.timestamp_seconds;
        let reacquired = self.last_consumed.elapsed().as_secs_f64() > 1.0;
        self.last_consumed = Instant::now();
        if !self.file_input && pose.captured.elapsed().as_secs_f64() > STALE_SECONDS {
            return false; // never fit a late webcam pose; the preview stays fresh
        }
        self.ever_posed = true;
        if self.paused {
            return false;
        }
        if pose.subject_segment != self.subject_segment {
            // Detector crop changes can split the recording fitter's segment.
            // They are not a reason to reset both live limbs during continuous tracking.
            if reacquired {
                self.fitter.reset();
                self.hand_fitter.reset();
                self.estimator.reset();
                self.estimate = None;
            }
            self.subject_segment = pose.subject_segment;
        }
        let t = Instant::now();
        match self.fitter.update(&pose.observation) {
            Ok(mut r) if r.qpos.len() == self.model.nq && r.qpos.iter().all(|x| x.is_finite()) => {
                let hands: Vec<_> = pose
                    .hands
                    .iter()
                    .filter_map(|h| {
                        let side = match h.side.as_str() {
                            "left" => 0,
                            "right" => 1,
                            _ => return None,
                        };
                        Some(biomech_lite::hand_fit::HandObservation {
                            side,
                            confidence: h.confidence as f64,
                            points: h.world_landmarks.map(|p| p.map(f64::from)),
                        })
                    })
                    .collect();
                self.notice = None;
                match self.hand_fitter.apply(
                    &mut r.qpos,
                    &hands,
                    pose.observation.timestamp_seconds,
                ) {
                    Ok(supported) => self.supported_hands = supported,
                    Err(e) => {
                        self.supported_hands = [false; 2];
                        self.notice = Some(format!("hand fit: {e}"));
                    }
                }
                self.estimate_arms = std::array::from_fn(|side| {
                    r.supported_arms[side]
                        && ["shoulder", "elbow", "wrist"].iter().all(|part| {
                            let name =
                                format!("{}_{part}", if side == 0 { "left" } else { "right" });
                            pose.observation.landmarks.iter().any(|p| {
                                p.name == name
                                    && p.confidence >= 0.3
                                    && p.position.iter().all(|v| v.is_finite())
                            })
                        })
                });
                if self.estimates_enabled {
                    self.estimator.submit(Sample {
                        index: pose.index,
                        captured: pose.captured,
                        timestamp_seconds: pose.observation.timestamp_seconds,
                        qpos: r.qpos.clone(),
                        supported_arms: self.estimate_arms,
                    });
                }
                self.qpos = r.qpos.clone();
                self.fit = Some(r);
                self.dirty = true;
            }
            Ok(r) => {
                self.notice = Some(format!(
                    "fit rejected: bad qpos (len {}) - holding",
                    r.qpos.len()
                ))
            }
            Err(e) => self.notice = Some(format!("fit error: {e} - holding")),
        }
        self.fit_ms = t.elapsed().as_secs_f64() * 1000.0;
        self.poses_fitted += 1;
        true
    }

    fn estimate_age(&self, estimate: &Estimate) -> f64 {
        if self.file_input {
            (self.last_source_time - estimate.sample.timestamp_seconds).max(0.0)
        } else {
            estimate.sample.captured.elapsed().as_secs_f64()
        }
    }

    fn estimate_ready(&self) -> bool {
        self.estimates_enabled
            && !self.paused
            && self.estimate_error.is_none()
            && self.estimate.as_ref().is_some_and(|e| {
                self.estimate_age(e) <= 0.5
                    && (0..2).all(|i| !e.sample.supported_arms[i] || self.estimate_arms[i])
                    && e.sample.supported_arms.iter().any(|&v| v)
                    && self.estimate_arms.iter().any(|&v| v)
            })
    }

    fn estimate_feasible(e: &Estimate) -> bool {
        e.recruitment.residual_norm_nm.is_finite()
            && e.recruitment.relative_residual.is_finite()
            && (e.recruitment.residual_norm_nm <= 1.0 || e.recruitment.relative_residual <= 0.1)
    }

    fn poll_estimate(&mut self) {
        let state = self.estimator.latest();
        self.estimate = state.estimate;
        self.estimate_error = state.error;
        let colors = if !self.estimates_enabled {
            None
        } else if self.estimate_ready()
            && self
                .estimate
                .as_ref()
                .is_some_and(|e| Self::estimate_feasible(e))
        {
            self.estimate.as_ref().map(|e| {
                e.recruitment
                    .activation
                    .iter()
                    .zip(&e.recruitment.supported_muscles)
                    .map(|(&a, &supported)| supported.then_some(a))
                    .collect::<Vec<_>>()
            })
        } else {
            Some(vec![None; self.model.muscles.len()])
        };
        if colors != self.estimate_colors {
            if let Err(e) = self.renderer.set_muscle_activity(colors.as_deref()) {
                self.notice = Some(format!("muscle display: {e}"));
            }
            self.estimate_colors = colors;
            self.dirty = true;
        }
    }

    fn muscle_lines(&self) -> Vec<String> {
        if !self.estimates_enabled {
            return vec!["Muscle estimation OFF (E) | anatomical paths only".into()];
        }
        let mut lines = vec!["ESTIMATED MUSCLE ACTIVATION | unloaded arms | E toggle".into()];
        let unavailable = if self.paused {
            Some("paused; estimates unavailable".to_string())
        } else if let Some(error) = &self.estimate_error {
            Some(format!("unavailable: {error}"))
        } else if !self.estimate_ready() {
            Some("waiting for a fresh supported arm pose; gray = unknown".into())
        } else {
            None
        };
        if let Some(reason) = unavailable {
            lines.push(reason);
            return lines;
        }
        let e = self.estimate.as_ref().unwrap();
        let r = &e.recruitment;
        let mode = match e.mode {
            EstimateMode::QuasiStaticWarmup => "motion warming; cold arms static",
            EstimateMode::Dynamic => "motion estimate",
        };
        lines.push(format!(
            "{mode} | source {:.0}ms | solve {:.0}ms",
            self.estimate_age(e) * 1000.0,
            e.computation_ms
        ));
        lines.push(format!(
            "Supported arms L:{} R:{}",
            if e.sample.supported_arms[0] {
                "yes"
            } else {
                "unknown"
            },
            if e.sample.supported_arms[1] {
                "yes"
            } else {
                "unknown"
            }
        ));
        if let Some(warning) = e.warnings.first() {
            lines.push(warning.clone());
        }
        lines.push(format!(
            "unexplained torque {:.2} Nm | optimizer {}",
            r.residual_norm_nm,
            if r.converged {
                "converged"
            } else {
                "iteration limit"
            }
        ));
        if !Self::estimate_feasible(e) {
            lines.push("Demand cannot be matched; gray = unavailable".into());
        } else {
            lines.push("Gray: unknown | estimated activation scale:".into());
            lines.push("Activation scale: 0% 25% 50% 75% 100%".into());
            let mut ids: Vec<usize> = (0..r.activation.len())
                .filter(|&i| r.supported_muscles[i])
                .collect();
            ids.sort_by(|&a, &b| r.activation[b].total_cmp(&r.activation[a]));
            for i in ids.into_iter().take(4) {
                lines.push(format!(
                    "{}: {:.0}% | {:.1} N estimated tension",
                    self.model.muscles[i].name,
                    r.activation[i] * 100.0,
                    r.tension_n[i]
                ));
            }
        }
        lines.push("Reference anatomy; fixed torso; model estimate, not EMG".into());
        lines
    }

    fn render(&mut self, left: Rect) {
        if left.w == 0 || left.h == 0 {
            return;
        }
        self.renderer.resize(left.w, left.h);
        self.camera.aspect = left.w as f64 / left.h as f64;
        match self.model.kinematics(&self.qpos) {
            Ok(kin) => {
                if let Err(e) = self.renderer.render(
                    &self.model,
                    &kin,
                    &self.visual,
                    &self.camera,
                    &self.layers,
                ) {
                    self.notice = Some(format!("render error: {e}"));
                }
            }
            Err(e) => self.notice = Some(format!("kinematics error: {e}")),
        }
        self.dirty = false;
    }

    fn act(&mut self, a: Act, left: Rect) {
        match a {
            Act::Front => self.camera.preset("front"),
            Act::Side => self.camera.preset("side"),
            Act::Back => self.camera.preset("back"),
            Act::Frame => {
                self.upper_framing = !self.upper_framing;
                self.frame_camera(left);
            }
            Act::Reset => self.reset(left),
            Act::Pause => self.paused = !self.paused,
            Act::Bones => self.layers.bones = !self.layers.bones,
            Act::Muscles => self.layers.muscles = !self.layers.muscles,
            Act::Ligaments => self.layers.ligaments = !self.layers.ligaments,
            Act::Overlay => self.overlay = !self.overlay,
            Act::Estimates => {
                self.estimates_enabled = !self.estimates_enabled;
                self.estimator.reset();
                self.estimate = None;
                self.estimate_error = None;
            }
        }
        self.dirty = true;
    }

    fn status(&self, state: &CaptureState) -> PoseStatus {
        let age = state
            .pose
            .as_ref()
            .map(|p| p.captured.elapsed().as_secs_f64());
        // A finished file source necessarily ages; report it as held, not tracking.
        let st = pose_status(
            age,
            state.pose.is_some(),
            self.ever_posed,
            self.paused,
            state.error.is_some(),
        );
        match (&st, &self.fit) {
            (PoseStatus::Tracking, Some(f)) if !f.supported_arms[0] && !f.supported_arms[1] => {
                PoseStatus::NoArms
            }
            (PoseStatus::Tracking, None) => PoseStatus::Waiting,
            _ => st,
        }
    }

    fn footer(&self, state: &CaptureState) -> String {
        let infer = state.pose.as_ref().map(|p| p.inference_ms);
        let age = state
            .pose
            .as_ref()
            .map(|p| p.captured.elapsed().as_secs_f64() * 1000.0);
        let sup = self
            .fit
            .as_ref()
            .map(|f| {
                format!(
                    "arms L:{} R:{}",
                    if f.supported_arms[0] { "fit" } else { "held" },
                    if f.supported_arms[1] { "fit" } else { "held" }
                )
            })
            .unwrap_or_else(|| "arms --".into());
        let pipeline = state
            .pose
            .as_ref()
            .map(|p| {
                format!(
                    "hands {} (rig L:{} R:{}) | refined {} | inferred depth {}",
                    p.hands.len(),
                    if self.supported_hands[0] {
                        "fit"
                    } else {
                        "held"
                    },
                    if self.supported_hands[1] {
                        "fit"
                    } else {
                        "held"
                    },
                    p.refined_joints,
                    p.image_constrained
                )
            })
            .unwrap_or_default();
        format!(
            "cam {:.0}fps | pose age {} | infer {} | fit {:.1}ms | render {:.0}ms | {} | {} | Space pause R reset 1/2/3 view F frame B/M/G layers P png Esc quit",
            self.camera_fps(),
            age.map(|a| format!("{a:.0}ms"))
                .unwrap_or_else(|| "--".into()),
            infer
                .map(|a| format!("{a:.0}ms"))
                .unwrap_or_else(|| "--".into()),
            self.fit_ms,
            self.renderer.stats().millis,
            sup,
            format!("{} | {}", state.status, pipeline),
        )
    }

    fn buttons(&self, w: usize) -> Vec<(Rect, String, bool, Act)> {
        let items: [(&str, bool, Act); 11] = [
            ("FRONT", false, Act::Front),
            ("SIDE", false, Act::Side),
            ("BACK", false, Act::Back),
            ("FRAME", self.upper_framing, Act::Frame),
            ("RESET", false, Act::Reset),
            ("PAUSE", self.paused, Act::Pause),
            ("BONES", self.layers.bones, Act::Bones),
            ("MUSCLE", self.layers.muscles, Act::Muscles),
            ("EST", self.estimates_enabled, Act::Estimates),
            ("LIGAM", self.layers.ligaments, Act::Ligaments),
            ("2D", self.overlay, Act::Overlay),
        ];
        let mut x = 440usize;
        let mut out = Vec::new();
        for (label, on, act) in items {
            let bw = label.len() * 8 + 12;
            if x + bw > w {
                break;
            }
            out.push((
                Rect {
                    x,
                    y: 18,
                    w: bw,
                    h: 24,
                },
                label.to_string(),
                on,
                act,
            ));
            x += bw + 6;
        }
        out
    }

    fn compose(
        &mut self,
        buf: &mut [u32],
        w: usize,
        h: usize,
        state: &CaptureState,
    ) -> Vec<(Rect, Act)> {
        self.poll_estimate();
        let lay = split_layout(w, h);
        if self.dirty || self.renderer.width() != lay.left.w || self.renderer.height() != lay.left.h
        {
            self.render(lay.left);
        }
        let buttons = self.buttons(w);
        let hud = Hud {
            status: self.status(state),
            footer: self.footer(state),
            notice: self.notice.clone(),
            muscle_lines: self.muscle_lines(),
            fit_note: self.fit.as_ref().map(|f| match f.rmse_m {
                Some(r) => format!("fit: {} (rmse {:.1}cm)", f.status, r * 100.0),
                None => format!("fit: {}", f.status),
            }),
            buttons: buttons.iter().map(|b| (b.0, b.1.clone(), b.2)).collect(),
            overlay: self.overlay,
            camera_title: if self.file_input {
                "Camera - file input".into()
            } else {
                "Camera - live".into()
            },
        };
        let px = self.renderer.pixels();
        let (rw, rh) = (self.renderer.width(), self.renderer.height());
        compose_frame(buf, w, h, Some((px, rw, rh)), state, &hud);
        buttons.into_iter().map(|b| (b.0, b.3)).collect()
    }
}

// ---------------------------------------------------------------- png

fn write_png(buf: &[u32], w: usize, h: usize, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Cannot create {}", parent.display()))?;
    }
    let bgr: Vec<u8> = buf[..w * h]
        .iter()
        .flat_map(|p| {
            [
                (p & 0xff) as u8,
                ((p >> 8) & 0xff) as u8,
                ((p >> 16) & 0xff) as u8,
            ]
        })
        .collect();
    let flat = Mat::from_slice(&bgr)?;
    let img = flat.reshape(3, h as i32)?;
    let ok = imgcodecs::imwrite_def(path.to_str().ok_or_else(|| anyhow!("Non-UTF8 path"))?, &img)?;
    if !ok {
        bail!("Could not encode PNG at {}", path.display());
    }
    Ok(())
}

/// `~/Pictures/Biomech Viewer/biomech_live_NNN.png`, first unused number (no overwrite).
fn next_picture_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|h| !h.is_empty())
        .ok_or_else(|| anyhow!("HOME is not set; use --live-snapshot PATH instead"))?;
    let dir = PathBuf::from(home).join("Pictures").join("Biomech Viewer");
    for i in 1..10000 {
        let p = dir.join(format!("biomech_live_{i:03}.png"));
        if !p.exists() {
            return Ok(p);
        }
    }
    bail!("Too many screenshots in {}", dir.display())
}

// ---------------------------------------------------------------- run

pub fn run(config: Config) -> Result<()> {
    let headless = config.snapshot.is_some();
    if headless && config.capture.input.is_none() {
        bail!("--live-snapshot requires --input; it never opens the camera");
    }
    let (w, h) = (1440usize, 900usize);
    let mut session = Session::new(&config.model, &config.visual, split_layout(w, h).left)?;
    session.file_input = config.capture.input.is_some();
    let capture = LiveCapture::start(config.capture).map_err(|e| anyhow!(e))?;
    if headless {
        return run_snapshot(session, capture, config.snapshot.unwrap(), w, h);
    }
    let mut window = Window::new(
        "LIVE ANATOMICAL TWIN",
        w,
        h,
        WindowOptions {
            resize: true,
            ..WindowOptions::default()
        },
    )
    .map_err(|e| anyhow!("Cannot open window: {e}"))?;
    window.set_target_fps(60);
    let mut buf = vec![BG; w * h];
    let (mut cw, mut ch) = (w, h);
    let mut buttons: Vec<(Rect, Act)> = Vec::new();
    let mut drag: Option<(f32, f32)> = None;
    let mut was_down = false;
    let started = Instant::now();
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let (nw, nh) = window.get_size();
        if (nw, nh) != (cw, ch) && nw > 0 && nh > 0 {
            (cw, ch) = (nw, nh);
            buf = vec![BG; cw * ch];
            session.dirty = true;
        }
        let lay = split_layout(cw, ch);
        let state = capture.latest();
        session.follow(&state);

        // Keyboard.
        let keys = [
            (Key::Space, Act::Pause),
            (Key::R, Act::Reset),
            (Key::Key1, Act::Front),
            (Key::Key2, Act::Side),
            (Key::Key3, Act::Back),
            (Key::F, Act::Frame),
            (Key::B, Act::Bones),
            (Key::M, Act::Muscles),
            (Key::G, Act::Ligaments),
            (Key::O, Act::Overlay),
            (Key::E, Act::Estimates),
        ];
        for (k, a) in keys {
            if window.is_key_pressed(k, KeyRepeat::No) {
                session.act(a, lay.left);
            }
        }
        // Mouse: buttons anywhere in header; orbit/zoom confined to the left viewport.
        if let Some((mx, my)) = window.get_mouse_pos(MouseMode::Discard) {
            let down = window.get_mouse_down(MouseButton::Left);
            if down && !was_down {
                if let Some((_, a)) = buttons.iter().find(|(r, _)| r.contains(mx, my)) {
                    session.act(*a, lay.left);
                } else if lay.left.contains(mx, my) {
                    drag = Some((mx, my));
                }
            }
            if down {
                if let Some((px, py)) = drag {
                    session.camera.orbit((mx - px) as f64, (my - py) as f64);
                    drag = Some((mx, my));
                    session.dirty = true;
                }
            } else {
                drag = None;
            }
            was_down = down;
            if lay.left.contains(mx, my) {
                if let Some((_, sy)) = window.get_scroll_wheel() {
                    if sy != 0.0 {
                        session.camera.zoom(sy as f64);
                        session.dirty = true;
                    }
                }
            }
        }

        buttons = session.compose(&mut buf, cw, ch, &state);
        if window.is_key_pressed(Key::P, KeyRepeat::No) {
            session.notice = Some(
                match next_picture_path().and_then(|p| write_png(&buf, cw, ch, &p).map(|_| p)) {
                    Ok(p) => format!("saved {}", p.display()),
                    Err(e) => format!("PNG save failed: {e}"),
                },
            );
        }
        window
            .update_with_buffer(&buf, cw, ch)
            .map_err(|e| anyhow!("Window update failed: {e}"))?;

        if let Some(max) = config.max_frames {
            let drained = state.ended && session.last_consumed.elapsed() > Duration::from_secs(3);
            if session.poses_fitted >= max
                || drained
                || (started.elapsed() > SNAPSHOT_TIMEOUT * 2 && session.poses_fitted == 0)
            {
                break;
            }
        }
    }
    drop(capture); // stops workers and releases the camera
    Ok(())
}

fn run_snapshot(
    mut session: Session,
    capture: LiveCapture,
    path: PathBuf,
    w: usize,
    h: usize,
) -> Result<()> {
    let start = Instant::now();
    let wanted = config_frames(&session);
    let mut state = loop {
        let state = capture.latest();
        if let Some(err) = &state.error {
            bail!("Live snapshot failed: {err}");
        }
        session.follow(&state);
        if state.pose.is_some()
            && session.poses_fitted >= wanted
            && session
                .fit
                .as_ref()
                .is_some_and(|f| f.supported_arms[0] || f.supported_arms[1])
        {
            break state;
        }
        if start.elapsed() > SNAPSHOT_TIMEOUT {
            bail!(
                "Live snapshot timed out after {}s with no usable pose and a supported arm (status: {})",
                SNAPSHOT_TIMEOUT.as_secs(),
                state.status
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    // Wait for the estimate belonging to the frozen file pose, without fitting
    // newer frames or opening a camera. Force and geometry share a source sample.
    let estimate_wait = Instant::now();
    loop {
        session.poll_estimate();
        if session
            .estimate
            .as_ref()
            .is_some_and(|e| Some(e.sample.index) == session.last_pose_index)
        {
            break;
        }
        if let Some(error) = &session.estimate_error {
            bail!("Muscle estimate failed: {error}");
        }
        if !session.estimate_arms.iter().any(|&v| v)
            || estimate_wait.elapsed() > Duration::from_secs(5)
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    // A file snapshot pairs the fit with its exact source frame; live preview
    // continues to use the freshest independent camera frame.
    if let Some(pose) = &state.pose {
        state.video = Some(pose.source_video.clone());
    }
    let mut buf = vec![BG; w * h];
    session.compose(&mut buf, w, h, &state);
    write_png(&buf, w, h, &path)?;
    eprintln!(
        "Wrote live snapshot {} (fit {:.1}ms, render {:.0}ms)",
        path.display(),
        session.fit_ms,
        session.renderer.stats().millis
    );
    if let Some(e) = &session.estimate {
        eprintln!(
            "Muscle estimate: source frame {}, mode {:?}, residual {:.3} Nm, converged {}, solve {:.1}ms",
            e.sample.index,
            e.mode,
            e.recruitment.residual_norm_nm,
            e.recruitment.converged,
            e.computation_ms
        );
    }
    drop(capture);
    Ok(())
}

fn config_frames(_session: &Session) -> u64 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video(w: usize, h: usize, color: u32) -> Arc<VideoFrame> {
        Arc::new(VideoFrame {
            index: 0,
            captured: Instant::now(),
            width: w,
            height: h,
            rgb: vec![color; w * h],
        })
    }

    #[test]
    #[ignore = "requires existing recording fixture and ONNX models; no camera or window"]
    fn tracked_file_drives_dynamic_muscle_estimates() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let capture = LiveCapture::start(CaptureConfig {
            camera: 0,
            input: Some(root.join("output/live-biomech/test-source.mp4")),
            pose_model: root.join("models/mediapipe-pose.onnx"),
            detector_model: root.join("models/yolox-s.onnx"),
            no_detection: false,
            arm_model: Some(root.join("models/rtmpose-m.onnx")),
            hand_model: Some(root.join("models/handpose.onnx")),
            motion_model: root.join("models/motionagformer-xs.onnx"),
            options: crate::tracking_pipeline::Options {
                full_body: false,
                image_support_floor: Some(0.55),
                no_3d: false,
                head_orientation: false,
                no_dimension_tracking: false,
                threshold: 0.3,
                focal_length_px: None,
            },
            stabilization: true,
            smoothing: true,
            upright_torso: true,
            acceleration: true,
            bbox: None,
            detection_interval: 15,
            detection_threshold: 0.4,
            nms_threshold: 0.45,
            opencv_threads: 4,
        })
        .unwrap();
        let mut session = Session::new(
            &root.join("assets/biomech-lite/model.json"),
            &root.join("assets/biomech-lite/visual.json"),
            split_layout(1440, 900).left,
        )
        .unwrap();
        session.file_input = true;
        let start = Instant::now();
        let mut seen = None;
        let (mut fits, mut estimates, mut dynamic, mut feasible) = (0, 0, 0, 0);
        let mut solve_ms = Vec::new();
        while start.elapsed() < Duration::from_secs(18) {
            let state = capture.latest();
            assert!(state.error.is_none(), "{:?}", state.error);
            if session.follow(&state) {
                fits += 1;
            }
            let before = session.qpos.clone();
            session.poll_estimate();
            assert_eq!(session.qpos, before);
            if let Some(e) = &session.estimate {
                if seen != Some(e.sample.index) {
                    seen = Some(e.sample.index);
                    estimates += 1;
                    dynamic += usize::from(e.mode == EstimateMode::Dynamic);
                    feasible += usize::from(Session::estimate_feasible(e));
                    solve_ms.push(e.computation_ms);
                    assert_eq!(e.recruitment.activation.len(), session.model.muscles.len());
                    assert!(
                        e.recruitment
                            .activation
                            .iter()
                            .all(|a| a.is_finite() && (0.0..=1.0).contains(a))
                    );
                    assert!(e.recruitment.tension_n.iter().all(|a| a.is_finite()));
                    assert!(e.sample.index <= session.last_pose_index.unwrap());
                }
            }
            if fits >= 35 && estimates >= 10 && dynamic >= 3 {
                break;
            }
            if state.ended && session.last_consumed.elapsed() > Duration::from_secs(2) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        eprintln!(
            "tracking-to-muscle: {fits} fitted frames, {estimates} source-tagged estimates, {dynamic} dynamic, {feasible} feasible; solve ms {solve_ms:?}"
        );
        assert!(
            estimates >= 5 && dynamic >= 1 && feasible >= 1,
            "incomplete live recruitment integration"
        );
    }

    fn synthetic_estimate(model: &Arc<Model>, captured: Instant) -> Arc<Estimate> {
        let mut recruiter = biomech_lite::inverse::Recruiter::new(model.clone()).unwrap();
        let observed =
            std::array::from_fn(|k| model.joints[model.dynamic_joints[k]].name.ends_with("_l"));
        let recruitment = recruiter
            .solve(&model.standing_qpos, &[0.0; 8], &[0.0; 8], observed)
            .unwrap();
        Arc::new(Estimate {
            sample: Sample {
                index: 0,
                captured,
                timestamp_seconds: 0.2,
                qpos: model.standing_qpos.clone(),
                supported_arms: [true, false],
            },
            recruitment,
            mode: EstimateMode::QuasiStaticWarmup,
            computation_ms: 1.0,
            warnings: vec![],
        })
    }

    #[test]
    fn stale_or_unsupported_estimates_are_not_displayed_as_activity() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut session = Session::new(
            &root.join("assets/biomech-lite/model.json"),
            &root.join("assets/biomech-lite/visual.json"),
            split_layout(1440, 900).left,
        )
        .unwrap();
        session.estimate_arms = [true, false];
        session.estimate = Some(synthetic_estimate(
            &session.model,
            Instant::now() - Duration::from_secs(1),
        ));
        assert!(!session.estimate_ready());
        session.file_input = true;
        session.last_source_time = 0.2;
        assert!(session.estimate_ready());
        session.estimate_arms = [true, true];
        assert!(
            session.estimate_ready(),
            "healthy subset should survive opposite-arm failure"
        );
        session.estimate_arms = [false, true];
        assert!(!session.estimate_ready());
        session.estimate_arms = [true, false];
        session.last_source_time = 0.8;
        assert!(!session.estimate_ready());
        session.last_source_time = 0.2;
        session.paused = true;
        assert!(!session.estimate_ready());
        session.paused = false;
        session.estimate_error = Some("invalid derivative".into());
        assert!(!session.estimate_ready());
        session.estimate_error = None;
        let mut invalid = (*session.estimate.as_ref().unwrap().as_ref()).clone();
        invalid.recruitment.residual_norm_nm = 100.0;
        invalid.recruitment.relative_residual = 1.0;
        assert!(!Session::estimate_feasible(&invalid));
    }

    #[test]
    fn muscle_estimation_never_moves_the_prescribed_skeleton() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut session = Session::new(
            &root.join("assets/biomech-lite/model.json"),
            &root.join("assets/biomech-lite/visual.json"),
            split_layout(1440, 900).left,
        )
        .unwrap();
        session.file_input = true;
        session.last_source_time = 0.2;
        session.estimate_arms = [true, false];
        let q = session.qpos.clone();
        session.estimator.submit(Sample {
            index: 123,
            captured: Instant::now(),
            timestamp_seconds: 0.2,
            qpos: q.clone(),
            supported_arms: [true, false],
        });
        let started = Instant::now();
        loop {
            session.poll_estimate();
            assert!(
                session.estimate_error.is_none(),
                "{:?}",
                session.estimate_error
            );
            if session
                .estimate
                .as_ref()
                .is_some_and(|e| e.sample.index == 123)
            {
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(2),
                "estimate worker timed out"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(session.qpos, q);
        assert!(session.estimate_ready());
        assert!(
            session
                .muscle_lines()
                .iter()
                .any(|s| s.contains("unloaded arms"))
        );
        assert_eq!(
            session.estimate_colors.as_ref().unwrap().len(),
            session.model.muscles.len()
        );
        session.act(Act::Estimates, split_layout(1440, 900).left);
        session.poll_estimate();
        assert_eq!(session.qpos, q);
        assert!(session.estimate_colors.is_none());
    }

    #[test]
    fn crop_segment_changes_do_not_reset_the_visible_arm() {
        use biomech_lite::live_fit::{Landmark, Observation};
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let model = root.join("assets/biomech-lite/model.json");
        let visual = root.join("assets/biomech-lite/visual.json");
        let left = split_layout(1440, 900).left;
        let mut stable = Session::new(&model, &visual, left).unwrap();
        let mut changing_crop = Session::new(&model, &visual, left).unwrap();
        let points = [
            ("left_shoulder", [0.2, 0.0, 0.0]),
            ("left_elbow", [0.4, 0.2, 0.0]),
            ("left_wrist", [0.4, 0.1, -0.2]),
        ];
        for i in 0..12 {
            let make_state = |segment| CaptureState {
                video: None,
                pose: Some(Arc::new(PoseFrame {
                    source_video: video(640, 480, 0),
                    index: i,
                    captured: Instant::now(),
                    observation: Observation {
                        timestamp_seconds: i as f64 / 30.0,
                        landmarks: points
                            .iter()
                            .map(|(name, position)| Landmark {
                                name: name.to_string(),
                                position: *position,
                                confidence: 0.9,
                            })
                            .collect(),
                    },
                    points: vec![],
                    inference_ms: 1.0,
                    hands: vec![],
                    predicted_hands: vec![],
                    image_constrained: 0,
                    subject_segment: Some(segment),
                    refined_joints: 0,
                })),
                ..Default::default()
            };
            assert!(stable.follow(&make_state(0)));
            assert!(changing_crop.follow(&make_state(i as usize)));
            assert_eq!(
                stable.qpos, changing_crop.qpos,
                "crop change reset fit at frame {i}"
            );
            assert_eq!(
                changing_crop.fit.as_ref().unwrap().supported_arms,
                [true, false]
            );
        }
    }

    #[test]
    fn halves_are_exactly_equal() {
        for (w, h) in [(1280, 800), (1440, 900), (1001, 700), (640, 480)] {
            let l = split_layout(w, h);
            assert_eq!(l.left.w, l.right.w, "{w}x{h}");
            assert_eq!(l.left.h, l.right.h);
            assert_eq!(l.left.y, HEADER);
            assert_eq!(l.right.x, l.left.x + l.left.w);
            assert_eq!(l.left.h + HEADER + FOOTER, h);
            assert_eq!(l.header.h, HEADER);
            assert_eq!(l.footer.h, FOOTER);
        }
    }

    #[test]
    fn letterbox_keeps_aspect_and_centres() {
        let dst = Rect {
            x: 720,
            y: 60,
            w: 720,
            h: 800,
        };
        let r = letterbox(640, 480, dst);
        assert_eq!(r.w, 720);
        assert_eq!(r.h, 540);
        assert_eq!(r.x, 720);
        assert_eq!(r.y, 60 + (800 - 540) / 2);
        let tall = letterbox(480, 640, dst);
        assert!(tall.h <= 800 && tall.w < 720);
        assert_eq!(letterbox(0, 10, dst).w, 0);
    }

    #[test]
    fn stale_pose_reports_held_and_lost() {
        assert_eq!(
            pose_status(Some(0.1), true, true, false, false),
            PoseStatus::Tracking
        );
        assert!(matches!(
            pose_status(Some(0.9), true, true, false, false),
            PoseStatus::Held { .. }
        ));
        assert_eq!(
            pose_status(None, false, true, false, false),
            PoseStatus::Lost
        );
        assert_eq!(
            pose_status(None, false, false, false, false),
            PoseStatus::Waiting
        );
        assert_eq!(
            pose_status(Some(0.1), true, true, true, false),
            PoseStatus::Paused
        );
        assert_eq!(
            pose_status(Some(0.1), true, true, false, true),
            PoseStatus::Error
        );
    }

    #[test]
    fn compose_has_model_left_and_camera_right() {
        let (w, h) = (800, 500);
        let lay = split_layout(w, h);
        let model = vec![0x123456u32; lay.left.w * lay.left.h];
        let state = CaptureState {
            video: Some(video(64, 48, 0xff0000)),
            status: "ok".into(),
            ..Default::default()
        };
        let hud = Hud {
            status: PoseStatus::Tracking,
            footer: "f".into(),
            notice: None,
            fit_note: None,
            muscle_lines: vec![],
            buttons: vec![],
            overlay: false,
            camera_title: "Camera - live".into(),
        };
        let mut buf = vec![0u32; w * h];
        compose_frame(
            &mut buf,
            w,
            h,
            Some((&model, lay.left.w, lay.left.h)),
            &state,
            &hud,
        );
        let at = |x: usize, y: usize| buf[y * w + x];
        assert_eq!(at(lay.left.x + 100, lay.left.y + 200), 0x123456);
        let cam = letterbox(64, 48, lay.right);
        assert_eq!(
            at(cam.x + cam.w / 2, cam.y + cam.h / 2),
            0xff0000,
            "camera is on the right, unmirrored"
        );
        assert_ne!(
            at(lay.left.x + 100, lay.left.y + 200),
            at(cam.x + cam.w / 2, cam.y + cam.h / 2)
        );
    }

    #[test]
    fn video_is_not_mirrored() {
        let mut rgb = vec![0u32; 4 * 2];
        rgb[0] = 0xff0000; // top-left red
        let v = VideoFrame {
            index: 0,
            captured: Instant::now(),
            width: 4,
            height: 2,
            rgb,
        };
        let mut buf = vec![0u32; 8 * 4];
        let mut c = Canvas {
            buf: &mut buf,
            w: 8,
            h: 4,
        };
        blit_video(
            &mut c,
            &v,
            Rect {
                x: 0,
                y: 0,
                w: 8,
                h: 4,
            },
        );
        assert_eq!(buf[0], 0xff0000);
        assert_eq!(buf[7], 0);
    }

    fn assets() -> Option<(PathBuf, PathBuf)> {
        let d = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/biomech-lite");
        (d.join("model.json").is_file() && d.join("visual.json").is_file())
            .then(|| (d.join("model.json"), d.join("visual.json")))
    }

    #[test]
    fn session_renders_nonblank_model_beside_camera() {
        let Some((m, v)) = assets() else { return };
        let (w, h) = (800, 500);
        let lay = split_layout(w, h);
        let mut s = Session::new(&m, &v, lay.left).expect("session");
        let state = CaptureState {
            video: Some(video(64, 48, 0x00ff00)),
            status: "test".into(),
            ..Default::default()
        };
        let mut buf = vec![0u32; w * h];
        s.compose(&mut buf, w, h, &state);
        let distinct: std::collections::HashSet<u32> = (lay.left.y..lay.left.y + lay.left.h)
            .step_by(7)
            .flat_map(|y| (0..lay.left.w).step_by(7).map(move |x| (x, y)))
            .map(|(x, y)| buf[y * w + x])
            .collect();
        assert!(
            distinct.len() > 8,
            "model pane should contain shaded anatomy"
        );
        let cam = letterbox(64, 48, lay.right);
        assert_eq!(buf[(cam.y + cam.h / 2) * w + cam.x + cam.w / 2], 0x00ff00);
    }

    #[test]
    fn png_save_creates_parents_and_file() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("a/b/c.png");
        write_png(&vec![0x336699; 16 * 8], 16, 8, &p).unwrap();
        assert!(p.is_file());
    }
}

#[cfg(test)]
mod wrap_tests {
    use super::wrap_text;
    #[test]
    fn wraps_long_text_within_columns() {
        let lines = wrap_text(
            "Cannot open camera 0 after 25s. Check permission in System Settings",
            20,
        );
        assert!(lines.len() > 2 && lines.iter().all(|l| l.chars().count() <= 20));
    }
}
