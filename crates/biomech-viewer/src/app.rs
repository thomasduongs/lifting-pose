//! Viewer application state, UI layout, input handling and software-composited overlay.
//!
//! Everything here is window-free: the native window (main.rs) only forwards input and
//! blits [`App::compose`], and the headless `--snapshot` path uses the very same code.

use crate::camera::Camera;
use crate::render::{Options, Renderer};
use crate::scene::VisualModel;
use crate::session::{DEFAULT_EXCITATION, Motion, SimCommand, SimSnapshot, Simulation};
use biomech_lite::Model;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub const HEADER_H: usize = 60;
pub const SIDEBAR_W: usize = 280;
pub const TIMELINE_H: usize = 40;
pub const DEFAULT_SIZE: (usize, usize) = (1280, 860);
pub const MIN_SIZE: (usize, usize) = (640, 480);
pub const SPEEDS: [f64; 4] = [0.25, 0.5, 1.0, 2.0];

const BG: u32 = 0x0b1220;
const PANEL: u32 = 0x111a2e;
const PANEL_EDGE: u32 = 0x24324f;
const BTN: u32 = 0x1c2a47;
const BTN_ON: u32 = 0x2f6f9f;
const BTN_OFF: u32 = 0x151d30;
const TEXT: u32 = 0xe8edf7;
const DIM: u32 = 0x93a1bd;
const WARN: u32 = 0xff8a7a;
const CYAN: u32 = 0x4fd8e8;
const AMBER: u32 = 0xf2b84b;
const MUSCLE_RED: u32 = 0xc4545c;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Recording,
    Simulation,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Action {
    TogglePlay,
    Reset,
    ToggleLoop,
    ToggleMode,
    SetMode(Mode),
    Speed(f64),
    SpeedStep(i32),
    StepFrame(i32),
    PrevMuscle,
    NextMuscle,
    ExcitationDelta(f64),
    View(u8), // 0 front, 1 side, 2 back
    Frame,
    ToggleUpper,
    ToggleBones,
    ToggleMuscles,
    ToggleLigaments,
    ToggleJoints,
    ToggleXray,
    Screenshot,
    Scrub(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    pub fn contains(&self, px: f32, py: f32) -> bool {
        px >= self.x as f32 && py >= self.y as f32 && px < (self.x + self.w) as f32 && py < (self.y + self.h) as f32
    }
}

#[derive(Debug, Clone)]
pub struct Button {
    pub rect: Rect,
    pub label: String,
    pub action: Action,
    pub active: bool,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Layout {
    pub width: usize,
    pub height: usize,
    pub viewport: Rect,
    pub timeline: Rect,
    pub scrubber: Rect,
}

pub fn layout(width: usize, height: usize) -> Layout {
    let vw = width.saturating_sub(SIDEBAR_W).max(16);
    let vh = height.saturating_sub(HEADER_H + TIMELINE_H).max(16);
    let timeline = Rect { x: SIDEBAR_W, y: HEADER_H + vh, w: vw, h: TIMELINE_H };
    let scrubber = Rect { x: timeline.x + 12, y: timeline.y + 22, w: vw.saturating_sub(24).max(8), h: 12 };
    Layout { width, height, viewport: Rect { x: SIDEBAR_W, y: HEADER_H, w: vw, h: vh }, timeline, scrubber }
}

pub struct App {
    pub model: Arc<Model>,
    pub visual: VisualModel,
    pub motion: Motion,
    pub renderer: Renderer,
    pub camera: Camera,
    pub options: Options,
    pub mode: Mode,
    pub playing: bool,
    pub looping: bool,
    pub upper: bool,
    pub speed: f64,
    /// Recording time, seconds after the first recorded stamp.
    pub time: f64,
    pub muscle: usize,
    pub excitation: f64,
    pub status: String,
    pub render_ms: f64,
    sim: Option<Simulation>,
    pub snapshot: Option<SimSnapshot>,
    sim_rate: Option<f64>,
    sim_mark: Option<(Instant, f64)>,
    size: (usize, usize),
    frame: Vec<u32>,
    drag: Option<(f32, f32, bool, bool)>, // last x, y, started in viewport, is_pan
    scrubbing: bool,
    last_pointer: (bool, bool),
    /// 3D pixels are stale (pose, camera, layers or size changed).
    dirty: bool,
    /// World AABB corners of every observed/unobserved geom over all recorded poses.
    envelope: Vec<([f64; 3], bool)>,
}

impl App {
    pub fn new(model: Model, visual: VisualModel, motion: Motion, mode: Mode, size: (usize, usize)) -> Result<Self, String> {
        let model = Arc::new(model);
        let size = (size.0.max(MIN_SIZE.0), size.1.max(MIN_SIZE.1));
        let lay = layout(size.0, size.1);
        let muscle = model.muscle_id("BIClong").unwrap_or(0);
        let mut app = App {
            renderer: Renderer::new(lay.viewport.w, lay.viewport.h),
            camera: Camera::front(),
            options: Options::default(),
            model,
            visual,
            motion,
            mode: Mode::Recording,
            playing: true,
            looping: true,
            upper: false,
            speed: 1.0,
            time: 0.0,
            muscle,
            excitation: DEFAULT_EXCITATION,
            status: String::new(),
            render_ms: 0.0,
            sim: None,
            snapshot: None,
            sim_rate: None,
            sim_mark: None,
            size,
            frame: vec![BG; size.0 * size.1],
            drag: None,
            scrubbing: false,
            last_pointer: (false, false),
            dirty: true,
            envelope: Vec::new(),
        };
        app.options.selected_muscle = Some(muscle);
        if mode == Mode::Simulation {
            app.set_mode(Mode::Simulation);
            app.playing = false; // simulation starts paused until play
        }
        app.envelope = motion_envelope(&app.model, &app.visual, &app.motion)?;
        app.fit_camera()?;
        Ok(app)
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn resize(&mut self, w: usize, h: usize) {
        let s = (w.max(MIN_SIZE.0), h.max(MIN_SIZE.1));
        if s != self.size {
            self.size = s;
            self.dirty = true;
            let l = layout(s.0, s.1);
            self.camera.aspect = l.viewport.w as f64 / l.viewport.h as f64;
            let l = layout(s.0, s.1);
            self.renderer.resize(l.viewport.w, l.viewport.h);
            self.frame = vec![BG; s.0 * s.1];
        }
    }

    // ------------------------------------------------------------------ state

    pub fn duration(&self) -> f64 {
        self.motion.duration()
    }

    /// Full qpos shown at the current state of the active mode.
    pub fn current_qpos(&self) -> Result<Vec<f64>, String> {
        match self.mode {
            Mode::Recording => self.motion.sample(&self.model, self.time),
            Mode::Simulation => match &self.snapshot {
                Some(s) => Ok(s.qpos.clone()),
                None => Ok(self.model.standing_qpos.clone()),
            },
        }
    }

    /// Frame the camera. Recording uses the stable envelope of every recorded pose (no
    /// per-time zoom); simulation frames the current pose.
    pub fn fit_camera(&mut self) -> Result<(), String> {
        let lay = layout(self.size.0, self.size.1);
        self.camera.aspect = lay.viewport.w as f64 / lay.viewport.h as f64;
        if self.mode == Mode::Recording && !self.envelope.is_empty() {
            fit_points(&mut self.camera, &self.envelope, self.upper);
            return Ok(());
        }
        let q = self.current_qpos()?;
        let kin = self.model.kinematics(&q)?;
        self.camera.fit(&self.model, &kin, &self.visual, self.upper);
        Ok(())
    }

    /// (unactuated tendon routes, those with passive stiffness).
    fn ligament_counts(&self) -> (usize, usize) {
        let used: std::collections::HashSet<usize> = self.model.muscles.iter().map(|m| m.tendon).collect();
        let lig: Vec<_> = self.model.tendons.iter().enumerate().filter(|(i, _)| !used.contains(i)).map(|(_, t)| t).collect();
        (lig.len(), lig.iter().filter(|t| t.stiffness > 0.0).count())
    }

    fn ensure_sim(&mut self) {
        if self.sim.is_none() {
            match Simulation::spawn(Arc::clone(&self.model)) {
                Ok(s) => {
                    let _ = s.send(SimCommand::Muscle(self.muscle));
                    let _ = s.send(SimCommand::Excitation(self.excitation));
                    self.sim = Some(s);
                }
                Err(e) => self.status = format!("simulation unavailable: {e}"),
            }
        }
    }

    fn sim_send(&mut self, c: SimCommand) {
        if let Some(s) = &self.sim {
            if let Err(e) = s.send(c) {
                self.status = e;
            }
        }
    }

    pub fn set_mode(&mut self, mode: Mode) {
        if mode == self.mode && (mode == Mode::Recording || self.sim.is_some()) {
            return;
        }
        // switching pauses the mode being left
        self.playing = false;
        if self.mode == Mode::Simulation {
            self.sim_send(SimCommand::Pause(true));
        }
        self.mode = mode;
        self.sim_mark = None;
        if mode == Mode::Simulation {
            self.ensure_sim();
            self.sim_send(SimCommand::Pause(true));
        }
        let _ = self.fit_camera();
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Apply a UI action. Errors go to `status`.
    pub fn apply(&mut self, action: Action) {
        self.dirty = true;
        match action {
            Action::TogglePlay => {
                self.playing = !self.playing;
                match self.mode {
                    Mode::Recording => {
                        if self.playing && !self.looping && self.time >= self.duration() {
                            self.time = 0.0;
                        }
                    }
                    Mode::Simulation => {
                        let p = self.playing;
                        self.sim_mark = None;
                        self.sim_send(SimCommand::Pause(!p));
                    }
                }
            }
            Action::Reset => match self.mode {
                Mode::Recording => self.time = 0.0,
                Mode::Simulation => {
                    self.playing = false;
                    self.sim_rate = None;
                    self.sim_mark = None;
                    self.sim_send(SimCommand::Pause(true));
                    self.sim_send(SimCommand::Reset);
                }
            },
            Action::ToggleLoop => self.looping = !self.looping,
            Action::ToggleMode => {
                let m = if self.mode == Mode::Recording { Mode::Simulation } else { Mode::Recording };
                self.set_mode(m);
            }
            Action::SetMode(m) => self.set_mode(m),
            Action::Speed(s) => {
                if SPEEDS.contains(&s) {
                    self.speed = s;
                }
            }
            Action::SpeedStep(d) => {
                let i = SPEEDS.iter().position(|&s| s == self.speed).unwrap_or(2) as i32;
                self.speed = SPEEDS[(i + d).clamp(0, SPEEDS.len() as i32 - 1) as usize];
            }
            Action::StepFrame(d) => {
                if self.mode == Mode::Recording {
                    self.playing = false;
                    let cur = self.motion.frame_at(self.time);
                    // standing exactly on a frame -> move to neighbour; between frames -> nearest in direction
                    let on = (self.motion.frame_time(cur) - self.time).abs() < 1e-9;
                    let target = if d > 0 {
                        cur + 1
                    } else if on {
                        cur.saturating_sub(1)
                    } else {
                        cur
                    };
                    let target = target.min(self.motion.len() - 1);
                    self.time = self.motion.frame_time(target);
                }
            }
            Action::PrevMuscle => self.select_muscle(-1),
            Action::NextMuscle => self.select_muscle(1),
            Action::ExcitationDelta(d) => {
                self.excitation = (((self.excitation + d) * 100.0).round() / 100.0).clamp(0.0, 1.0);
                if self.mode == Mode::Simulation || self.sim.is_some() {
                    let e = self.excitation;
                    self.sim_send(SimCommand::Excitation(e));
                }
            }
            Action::View(v) => {
                self.camera.preset(["front", "side", "back"][v as usize % 3]);
                let _ = self.fit_camera();
            }
            Action::Frame => {
                let _ = self.fit_camera();
            }
            Action::ToggleUpper => {
                self.upper = !self.upper;
                let _ = self.fit_camera();
            }
            Action::ToggleBones => self.options.bones = !self.options.bones,
            Action::ToggleMuscles => self.options.muscles = !self.options.muscles,
            Action::ToggleLigaments => self.options.ligaments = !self.options.ligaments,
            Action::ToggleJoints => self.options.joints = !self.options.joints,
            Action::ToggleXray => self.options.xray = !self.options.xray,
            Action::Screenshot => match self.save_png_auto() {
                Ok(p) => self.status = format!("saved {}", p.display()),
                Err(e) => self.status = format!("screenshot failed: {e}"),
            },
            Action::Scrub(f) => {
                if self.mode == Mode::Recording {
                    self.time = f.clamp(0.0, 1.0) * self.duration();
                }
            }
        }
    }

    fn select_muscle(&mut self, d: i32) {
        let n = self.model.muscles.len() as i32;
        if n == 0 {
            return;
        }
        self.muscle = (self.muscle as i32 + d).rem_euclid(n) as usize;
        self.options.selected_muscle = Some(self.muscle);
        if self.sim.is_some() {
            let m = self.muscle;
            self.sim_send(SimCommand::Muscle(m));
            self.playing = false; // selection resets simulation, which then waits for play
            self.sim_send(SimCommand::Pause(true));
            self.sim_mark = None;
            self.sim_rate = None;
        }
    }

    /// Advance by `dt` wall seconds and collect simulation snapshots.
    pub fn update(&mut self, dt: f64) {
        match self.mode {
            Mode::Recording => {
                if self.playing && dt.is_finite() && dt > 0.0 {
                    self.time += dt * self.speed;
                    self.dirty = true;
                    let d = self.duration();
                    if self.time >= d {
                        if self.looping && d > 0.0 {
                            self.time %= d;
                        } else {
                            self.time = d;
                            self.playing = false;
                        }
                    }
                }
            }
            Mode::Simulation => {}
        }
        if let Some(sim) = self.sim.as_mut() {
            if let Some(s) = sim.latest() {
                if let Some(e) = &s.error {
                    self.status = format!("simulation error: {e}");
                    self.playing = false;
                }
                let now = Instant::now();
                if self.playing {
                    if let Some((t0, s0)) = self.sim_mark {
                        let w = now.duration_since(t0).as_secs_f64();
                        if w > 0.2 {
                            let r = (s.time_seconds - s0) / w;
                            self.sim_rate = Some(match self.sim_rate {
                                Some(p) => 0.5 * p + 0.5 * r,
                                None => r,
                            });
                            self.sim_mark = Some((now, s.time_seconds));
                        }
                    } else {
                        self.sim_mark = Some((now, s.time_seconds));
                    }
                }
                self.snapshot = Some(s);
                self.dirty = true;
            }
        }
    }

    // ------------------------------------------------------------------ buttons

    pub fn buttons(&self) -> Vec<Button> {
        let rec = self.mode == Mode::Recording;
        let mut out = Vec::new();
        let mut y = HEADER_H + 10;
        let x0 = 10usize;
        let full = SIDEBAR_W - 20;
        let row = |out: &mut Vec<Button>, y: &mut usize, items: Vec<(String, Action, bool, bool)>| {
            let n = items.len();
            let gap = 4;
            let w = (full - gap * (n - 1)) / n;
            for (i, (label, action, active, enabled)) in items.into_iter().enumerate() {
                out.push(Button { rect: Rect { x: x0 + i * (w + gap), y: *y, w, h: 26 }, label, action, active, enabled });
            }
            *y += 30;
        };
        row(&mut out, &mut y, vec![
            ("Recording".into(), Action::SetMode(Mode::Recording), rec, true),
            ("Simulation".into(), Action::SetMode(Mode::Simulation), !rec, true),
        ]);
        y += 6;
        row(&mut out, &mut y, vec![
            (if self.playing { "Pause" } else { "Play" }.into(), Action::TogglePlay, self.playing, true),
            ("Reset".into(), Action::Reset, false, true),
            ("Loop".into(), Action::ToggleLoop, self.looping, rec),
        ]);
        row(&mut out, &mut y, vec![
            ("< Frame".into(), Action::StepFrame(-1), false, rec),
            ("Frame >".into(), Action::StepFrame(1), false, rec),
        ]);
        row(
            &mut out,
            &mut y,
            SPEEDS.iter().map(|&s| (format!("{s}x"), Action::Speed(s), rec && self.speed == s, rec)).collect(),
        );
        y += 6;
        row(&mut out, &mut y, vec![
            ("< Muscle".into(), Action::PrevMuscle, false, true),
            ("Muscle >".into(), Action::NextMuscle, false, true),
        ]);
        row(&mut out, &mut y, vec![
            ("Excit -".into(), Action::ExcitationDelta(-0.05), false, true),
            ("Excit +".into(), Action::ExcitationDelta(0.05), false, true),
        ]);
        y += 6;
        row(&mut out, &mut y, vec![
            ("Front".into(), Action::View(0), false, true),
            ("Side".into(), Action::View(1), false, true),
            ("Back".into(), Action::View(2), false, true),
        ]);
        row(&mut out, &mut y, vec![
            ("Frame all".into(), Action::Frame, false, true),
            ("Upper body".into(), Action::ToggleUpper, self.upper, true),
        ]);
        y += 6;
        let o = &self.options;
        row(&mut out, &mut y, vec![
            ("Bones".into(), Action::ToggleBones, o.bones, true),
            ("Muscles".into(), Action::ToggleMuscles, o.muscles, true),
        ]);
        row(&mut out, &mut y, vec![
            ("Ligaments".into(), Action::ToggleLigaments, o.ligaments, true),
            ("Joints".into(), Action::ToggleJoints, o.joints, true),
        ]);
        row(&mut out, &mut y, vec![
            ("X-ray paths".into(), Action::ToggleXray, o.xray, true),
            ("Save PNG".into(), Action::Screenshot, false, true),
        ]);
        out
    }

    /// y coordinate below the last sidebar button (info text starts here).
    fn info_top(&self) -> usize {
        self.buttons().iter().map(|b| b.rect.y + b.rect.h).max().unwrap_or(HEADER_H) + 12
    }

    // ------------------------------------------------------------------ pointer

    /// Forward pointer state. Returns true if a redraw is useful.
    pub fn pointer(&mut self, x: f32, y: f32, left: bool, right: bool, scroll: f32) {
        let lay = layout(self.size.0, self.size.1);
        let vp = lay.viewport;
        if left || right || scroll != 0.0 {
            self.dirty = true;
        }
        if scroll != 0.0 && vp.contains(x, y) {
            self.camera.zoom(scroll as f64);
        }
        let pressed_left = left && !self.last_pointer.0;
        let pressed_right = right && !self.last_pointer.1;
        if pressed_left {
            if let Some(b) = self.buttons().into_iter().find(|b| b.enabled && b.rect.contains(x, y)) {
                self.apply(b.action);
            } else if self.mode == Mode::Recording && hit_scrub(&lay, x, y) {
                self.scrubbing = true;
            }
        }
        if self.scrubbing {
            if left {
                let f = ((x - lay.scrubber.x as f32) / lay.scrubber.w as f32) as f64;
                self.apply(Action::Scrub(f));
            } else {
                self.scrubbing = false;
            }
        }
        if (pressed_left || pressed_right) && vp.contains(x, y) {
            self.drag = Some((x, y, true, pressed_right));
        }
        if !left && !right {
            self.drag = None;
        }
        if let Some((lx, ly, true, pan)) = self.drag {
            let (dx, dy) = ((x - lx) as f64, (y - ly) as f64);
            if dx != 0.0 || dy != 0.0 {
                if pan {
                    self.camera.pan(dx, dy);
                } else {
                    self.camera.orbit(dx, dy);
                }
                self.drag = Some((x, y, true, pan));
            }
        }
        self.last_pointer = (left, right);
    }

    // ------------------------------------------------------------------ drawing

    /// Render the 3D viewport with the shared renderer and compose the full UI frame.
    pub fn compose(&mut self) -> Result<&[u32], String> {
        let lay = layout(self.size.0, self.size.1);
        if self.dirty {
            let q = self.current_qpos()?;
            let kin = self.model.kinematics(&q)?;
            let t0 = Instant::now();
            self.renderer.render(&self.model, &kin, &self.visual, &self.camera, &self.options)?;
            self.render_ms = t0.elapsed().as_secs_f64() * 1000.0;
            self.dirty = false;
        }
        let (w, h) = self.size;
        self.frame.iter_mut().for_each(|p| *p = BG);
        let vp = lay.viewport;
        let px = self.renderer.pixels();
        for row in 0..vp.h {
            let src = &px[row * vp.w..(row + 1) * vp.w];
            let dst = (vp.y + row) * w + vp.x;
            self.frame[dst..dst + vp.w].copy_from_slice(src);
        }
        self.draw_ui(&lay);
        let _ = h;
        Ok(&self.frame)
    }

    fn draw_ui(&mut self, lay: &Layout) {
        let (w, h) = self.size;
        let rec = self.mode == Mode::Recording;
        let buttons = self.buttons();
        let dur = self.duration();
        let (routes, springs) = self.ligament_counts();
        let info_top = self.info_top();
        let mut f = Canvas { buf: &mut self.frame, w, h };
        // header
        f.rect(0, 0, w, HEADER_H, PANEL);
        f.rect(0, HEADER_H - 1, w, 1, PANEL_EDGE);
        f.text(14, 10, "MyoSim full-body reference", 2, TEXT);
        f.text(14, 34, "Native Rust viewer | bones: sourced MyoSim meshes | muscles: MTU paths, not volumetric bellies", 1, DIM);
        f.text(14, 46, "Kinematics, tendon wrapping and muscle dynamics: biomech-lite (MuJoCo 3.17 port)", 1, DIM);
        let mode_txt = if rec { "RECORDING REPLAY" } else { "LIVE SIMULATION" };
        let mw = mode_txt.len() * 16;
        f.text(w.saturating_sub(mw + 14), 10, mode_txt, 2, if rec { CYAN } else { AMBER });
        // sidebar
        f.rect(0, HEADER_H, SIDEBAR_W, h - HEADER_H, PANEL);
        f.rect(SIDEBAR_W - 1, HEADER_H, 1, h - HEADER_H, PANEL_EDGE);
        for b in &buttons {
            let col = if !b.enabled { BTN_OFF } else if b.active { BTN_ON } else { BTN };
            f.rect(b.rect.x, b.rect.y, b.rect.w, b.rect.h, col);
            f.outline(b.rect.x, b.rect.y, b.rect.w, b.rect.h, PANEL_EDGE);
            let tw = b.label.len() * 8;
            f.text(b.rect.x + b.rect.w.saturating_sub(tw) / 2, b.rect.y + 9, &b.label, 1, if b.enabled { TEXT } else { DIM });
        }
        let mname = &self.model.muscles[self.muscle.min(self.model.muscles.len() - 1)].name;
        let mut y = info_top;
        let line = |f: &mut Canvas, y: &mut usize, s: String, c: u32| {
            f.text(12, *y, &s, 1, c);
            *y += 12;
        };
        if rec {
            line(&mut f, &mut y, format!("Mode: recording, {}", if self.playing { "playing" } else { "paused" }), TEXT);
            line(&mut f, &mut y, format!("Time: {:.3} / {:.3} s", self.time, dur), TEXT);
            line(&mut f, &mut y, format!("Speed: {}x  Loop: {}", self.speed, if self.looping { "on" } else { "off" }), TEXT);
            let fi = self.motion.frame_at(self.time);
            line(&mut f, &mut y, format!("Frame {}/{} (source idx {})", fi + 1, self.motion.len(), self.motion.frame_indices[fi]), DIM);
        } else {
            line(&mut f, &mut y, format!("Mode: simulation, {}", if self.playing { "running" } else { "paused" }), TEXT);
            let st = self.snapshot.as_ref().map(|s| s.time_seconds).unwrap_or(0.0);
            line(&mut f, &mut y, format!("Sim clock: {st:.3} s"), TEXT);
            match (self.playing, self.sim_rate) {
                (true, Some(r)) => line(&mut f, &mut y, format!("Rate: {r:.2}x real time"), if r < 0.98 { AMBER } else { TEXT }),
                _ => line(&mut f, &mut y, "Rate: -".into(), DIM),
            }
            let a = self.snapshot.as_ref().and_then(|s| s.activation.get(self.muscle).copied()).unwrap_or(0.0);
            line(&mut f, &mut y, format!("Activation: {a:.3}"), TEXT);
        }
        line(&mut f, &mut y, format!("Muscle: {mname}"), CYAN);
        if rec {
            line(&mut f, &mut y, "(highlight only; no force shown)".into(), DIM);
        } else {
            line(&mut f, &mut y, format!("Excitation: {:.2}", self.excitation), TEXT);
        }
        y += 6;
        line(&mut f, &mut y, "Legend".into(), TEXT);
        for (c, s) in [(0xe9e4d2, "bone (source mesh)"), (MUSCLE_RED, "muscle-tendon path"), (AMBER, "ligament / passive path"), (CYAN, "selected muscle")] {
            f.rect(12, y, 8, 8, c);
            f.text(26, y, s, 1, DIM);
            y += 12;
        }
        y += 6;
        line(&mut f, &mut y, "Lower body: held reference".into(), DIM);
        line(&mut f, &mut y, "anatomy, not observed.".into(), DIM);
        if !rec {
            line(&mut f, &mut y, "Torso fixed; 8 dynamic arm".into(), DIM);
            line(&mut f, &mut y, "coords, others at standing.".into(), DIM);
        }
        line(&mut f, &mut y, format!("Ligaments: {routes} routes, {springs} springs"), DIM);
        line(&mut f, &mut y, format!("Render {:.0} ms", self.render_ms), DIM);
        y += 6;
        if !self.status.is_empty() {
            let bad = self.status.contains("error") || self.status.contains("fail") || self.status.contains("unavail");
            for chunk in wrap(&self.status, (SIDEBAR_W - 24) / 8) {
                line(&mut f, &mut y, chunk, if bad { WARN } else { CYAN });
            }
        }
        y += 6;
        for s in [
            "Keys: Space play  Home reset",
            "Arrows step  L loop  Tab mode",
            "1/2/3 view F frame U upper",
            "B M G J layers  X x-ray",
            "[ ] muscle  +/- excitation",
            ", . speed  P save PNG",
            "Drag orbit, right-drag pan,",
            "wheel zoom.",
        ] {
            if y + 12 < h {
                line(&mut f, &mut y, s.into(), DIM);
            }
        }
        // timeline
        let t = lay.timeline;
        f.rect(t.x, t.y, t.w, t.h, PANEL);
        f.rect(t.x, t.y, t.w, 1, PANEL_EDGE);
        let s = lay.scrubber;
        if rec {
            f.rect(s.x, s.y, s.w, s.h, BTN);
            let frac = if dur > 0.0 { (self.time / dur).clamp(0.0, 1.0) } else { 0.0 };
            f.rect(s.x, s.y, (frac * s.w as f64) as usize, s.h, BTN_ON);
            f.rect(s.x + (frac * (s.w - 3) as f64) as usize, s.y - 2, 3, s.h + 4, TEXT);
            f.text(t.x + 12, t.y + 6, &format!("{:.2}s / {:.2}s  (irregular recorded clock, interpolated)", self.time, dur), 1, TEXT);
        } else {
            let st = self.snapshot.as_ref().map(|s| s.time_seconds).unwrap_or(0.0);
            f.text(t.x + 12, t.y + 6, &format!("Simulation clock {st:.3} s (dt 1 ms, actual Rust dynamics; scrubber disabled)"), 1, TEXT);
        }
    }

    // ------------------------------------------------------------------ png

    pub fn save_png(&self, path: &Path) -> Result<(), String> {
        let (w, h) = self.size;
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        }
        let file = File::create(path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        let mut enc = png::Encoder::new(BufWriter::new(file), w as u32, h as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header().map_err(|e| e.to_string())?;
        let mut data = Vec::with_capacity(w * h * 3);
        for p in &self.frame {
            data.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, *p as u8]);
        }
        wr.write_image_data(&data).map_err(|e| e.to_string())
    }

    /// Writes `biomech_viewer_NNN.png` into `$HOME/Pictures/Biomech Viewer` (or `%USERPROFILE%`)
    /// without overwriting existing files. The cwd is not used: a Finder-launched app has cwd `/`.
    pub fn save_png_auto(&mut self) -> Result<PathBuf, String> {
        let dir = screenshot_dir()?;
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        self.compose()?;
        for i in 1..10000 {
            let p = dir.join(format!("biomech_viewer_{i:03}.png"));
            if !p.exists() {
                self.save_png(&p)?;
                return Ok(p);
            }
        }
        Err(format!("too many screenshots in {}", dir.display()))
    }
}

/// `$HOME/Pictures/Biomech Viewer`, falling back to `%USERPROFILE%`. Errors clearly if neither is set.
fn screenshot_dir() -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|h| !h.is_empty())
        .ok_or_else(|| "cannot save screenshot: HOME is not set (use --snapshot PATH to choose a file)".to_string())?;
    Ok(PathBuf::from(home).join("Pictures").join("Biomech Viewer"))
}

fn hit_scrub(lay: &Layout, x: f32, y: f32) -> bool {
    let s = lay.scrubber;
    x >= (s.x as f32 - 4.0) && x < (s.x + s.w) as f32 + 4.0 && y >= s.y as f32 - 6.0 && y < (s.y + s.h) as f32 + 6.0
}

fn wrap(s: &str, cols: usize) -> Vec<String> {
    let cols = cols.max(8);
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in s.split_whitespace() {
        if !cur.is_empty() && cur.len() + 1 + word.len() > cols {
            out.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(word);
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

struct Canvas<'a> {
    buf: &'a mut [u32],
    w: usize,
    h: usize,
}

impl Canvas<'_> {
    fn rect(&mut self, x: usize, y: usize, w: usize, h: usize, c: u32) {
        for yy in y..(y + h).min(self.h) {
            let (a, b) = (yy * self.w + x.min(self.w), yy * self.w + (x + w).min(self.w));
            self.buf[a..b].fill(c);
        }
    }

    fn outline(&mut self, x: usize, y: usize, w: usize, h: usize, c: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.rect(x, y, w, 1, c);
        self.rect(x, y + h - 1, w, 1, c);
        self.rect(x, y, 1, h, c);
        self.rect(x + w - 1, y, 1, h, c);
    }

    fn text(&mut self, x: usize, y: usize, s: &str, scale: usize, c: u32) {
        for (i, ch) in s.chars().enumerate() {
            let code = if (ch as usize) < 128 { ch as usize } else { b'?' as usize };
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
}

/// World-space AABB corners of every visual geom at every recorded pose (flag = unobserved).
pub fn motion_envelope(model: &Model, visual: &VisualModel, motion: &Motion) -> Result<Vec<([f64; 3], bool)>, String> {
    let boxes: Vec<([f64; 3], [f64; 3])> = visual
        .meshes
        .iter()
        .map(|m| {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for v in &m.vertices {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
            (lo, hi)
        })
        .collect();
    let mut out = Vec::with_capacity(motion.len() * visual.geoms.len() * 8);
    for q in &motion.qpos {
        let mut q = q.clone();
        model.project(&mut q)?;
        let kin = model.kinematics(&q)?;
        for vg in &visual.geoms {
            let (lo, hi) = boxes[vg.mesh];
            if !lo.iter().chain(&hi).all(|x| x.is_finite()) {
                continue;
            }
            for c in 0..8 {
                let local = [
                    if c & 1 == 0 { lo[0] } else { hi[0] },
                    if c & 2 == 0 { lo[1] } else { hi[1] },
                    if c & 4 == 0 { lo[2] } else { hi[2] },
                ];
                let r = biomech_lite::kinematics::mat_vec(&kin.geom_mat[vg.geom], &local);
                let p = biomech_lite::kinematics::add3(&kin.geom_pos[vg.geom], &r);
                out.push((p, vg.unobserved));
            }
        }
    }
    Ok(out)
}

/// Perspective-exact framing of a world point set for the camera's current direction/aspect.
pub fn fit_points(cam: &mut Camera, pts: &[([f64; 3], bool)], upper: bool) {
    use biomech_lite::kinematics::{add3, dot3, scl3, sub3};
    let has_obs = pts.iter().any(|p| !p.1);
    let sel: Vec<&[f64; 3]> = pts.iter().filter(|p| !upper || !has_obs || !p.1).map(|p| &p.0).collect();
    if sel.is_empty() {
        return;
    }
    let (right, up, forward) = cam.basis();
    let mut ext = [[f64::INFINITY, f64::NEG_INFINITY]; 3];
    for p in &sel {
        for (k, a) in [&right, &up, &forward].into_iter().enumerate() {
            let s = dot3(p, a);
            ext[k][0] = ext[k][0].min(s);
            ext[k][1] = ext[k][1].max(s);
        }
    }
    let mid = [0.5 * (ext[0][0] + ext[0][1]), 0.5 * (ext[1][0] + ext[1][1]), 0.5 * (ext[2][0] + ext[2][1])];
    let target = add3(&add3(&scl3(&right, mid[0]), &scl3(&up, mid[1])), &scl3(&forward, mid[2]));
    let t = (0.5 * cam.fov_y).tan();
    let aspect = if cam.aspect.is_finite() && cam.aspect > 0.0 { cam.aspect } else { 1.0 };
    let margin = 1.06;
    let mut dist = 0.0f64;
    for p in &sel {
        let d = sub3(p, &target);
        let (x, y, f) = (dot3(&d, &right), dot3(&d, &up), dot3(&d, &forward));
        let need = (y.abs() / t).max(x.abs() / (t * aspect)) * margin - f;
        dist = dist.max(need);
    }
    if dist.is_finite() && dist > 0.0 && target.iter().all(|x| x.is_finite()) {
        cam.target = target;
        cam.distance = dist.clamp(crate::camera::MIN_DISTANCE, crate::camera::MAX_DISTANCE);
    }
}
