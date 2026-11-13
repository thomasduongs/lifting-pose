//! Software 3D renderer: z-buffered (reciprocal-depth) triangles with flat diffuse lighting,
//! near-plane clipping, depth-tested thick path lines (optional x-ray), a ground grid and
//! joint anchors. One renderer serves both the native window and headless PNG snapshots.
//!
//! Bones are the sourced MyoSim meshes posed by biomech-lite FK. Muscles are drawn as
//! musculotendon (MTU) paths from `Model::tendon_path`, not volumetric bellies; wrapped
//! segments are drawn as chords between tangent points (physics keeps exact arc length).

use std::time::Instant;

use biomech_lite::kinematics::{cross3, dot3, sub3};
use biomech_lite::{Kinematics, Model};

use crate::camera::{Camera, View};
use crate::scene::VisualModel;

/// Colours (0x00RRGGBB) shared with the UI legend.
pub mod palette {
    pub const BACKGROUND_TOP: u32 = 0x16223a;
    pub const BACKGROUND: u32 = 0x0e1628;
    pub const BACKGROUND_BOTTOM: u32 = 0x090f1c;
    pub const BONE: u32 = 0xeee4cc;
    pub const BONE_UNOBSERVED: u32 = 0x8f97a8;
    pub const MUSCLE: u32 = 0xc4504f;
    pub const LIGAMENT: u32 = 0xe0a63c;
    pub const SELECTED: u32 = 0x35e0f0;
    pub const JOINT: u32 = 0x6fe08a;
    pub const JOINT_OUTLINE: u32 = 0x0b2a16;
    pub const GRID: u32 = 0x24324c;
    pub const GRID_MAJOR: u32 = 0x33486a;
}

/// Layer switches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Options {
    pub bones: bool,
    pub muscles: bool,
    pub ligaments: bool,
    pub joints: bool,
    /// Draw paths through bones (occluded parts dimmed) instead of depth-testing them.
    pub xray: bool,
    /// Muscle index (into `Model::muscles`) to highlight.
    pub selected_muscle: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        Options { bones: true, muscles: true, ligaments: true, joints: false, xray: false, selected_muscle: None }
    }
}

/// Counters and timing of the last [`Renderer::render`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RenderStats {
    pub triangles_submitted: usize,
    pub triangles_rasterized: usize,
    pub triangles_culled: usize,
    pub triangles_clipped: usize,
    pub muscle_paths: usize,
    pub ligament_paths: usize,
    pub joints: usize,
    pub threads: usize,
    /// Wall time of the whole frame and of the bone pass (ms).
    pub millis: f64,
    pub bone_millis: f64,
}

/// Screen-space triangle ready for rasterization: (x, y, 1/z) per vertex.
#[derive(Debug, Clone, Copy)]
struct ScreenTri {
    p: [[f32; 3]; 3],
    color: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Depth test (strictly nearer) and depth write.
    Solid,
    /// Depth test with a small bias towards the eye, no depth write.
    Overlay,
    /// Always drawn; occluded pixels blended with what is behind.
    Xray,
}

/// Relative 1/z bias letting paths that run on a bone surface win the depth test.
const LINE_DEPTH_BIAS: f32 = 0.004;
const XRAY_OCCLUDED_ALPHA: u32 = 110; // of 255

pub struct Renderer {
    width: usize,
    height: usize,
    color: Vec<u32>,
    /// 1/z per pixel, 0 = empty (infinitely far).
    depth: Vec<f32>,
    background: Vec<u32>,
    cam: Vec<[f64; 3]>,
    scr: Vec<[f64; 3]>,
    tris: Vec<ScreenTri>,
    stats: RenderStats,
    /// Cull triangles facing away (meshes are closed bone surfaces). Default true.
    pub cull_backfaces: bool,
    /// Raster threads for the bone pass (0 = available parallelism, capped at 8).
    pub threads: usize,
    /// Per-muscle estimated activation in [0, 1]; `None` entry = unknown. `None` = uniform muscle color.
    muscle_activity: Option<Vec<Option<f64>>>,
}

/// Estimated activation: blue -> green -> yellow -> orange -> red.
const ACTIVITY_LOW: u32 = 0x3288ff;
const ACTIVITY_HIGH: u32 = 0xff394f;
const ACTIVITY_STOPS: [u32; 5] = [ACTIVITY_LOW, 0x39d98a, 0xffdf4d, 0xff922e, ACTIVITY_HIGH];
const ACTIVITY_UNKNOWN: u32 = 0x3a3f4b;

/// Color for one muscle's estimated activation (`None` = unknown). Values are clamped to [0, 1].
pub fn muscle_activity_color(activity: Option<f64>) -> u32 {
    match activity {
        None => ACTIVITY_UNKNOWN,
        Some(a) => {
            let position = a.clamp(0.0, 1.0) * 4.0;
            let band = (position.floor() as usize).min(3);
            lerp_color(ACTIVITY_STOPS[band], ACTIVITY_STOPS[band + 1], position - band as f64)
        },
    }
}

#[inline]
fn shade(color: u32, s: f64) -> u32 {
    let s = s.clamp(0.0, 1.0);
    let c = |shift: u32| ((((color >> shift) & 0xff) as f64 * s).round() as u32).min(255) << shift;
    c(16) | c(8) | c(0)
}

#[inline]
fn blend(dst: u32, src: u32, alpha: u32) -> u32 {
    let inv = 255 - alpha;
    let ch = |shift: u32| ((((src >> shift) & 0xff) * alpha + ((dst >> shift) & 0xff) * inv + 127) / 255) << shift;
    ch(16) | ch(8) | ch(0)
}

#[inline]
fn lerp_color(a: u32, b: u32, t: f64) -> u32 {
    blend(a, b, (t.clamp(0.0, 1.0) * 255.0).round() as u32)
}

#[inline]
fn plot(color: &mut [u32], depth: &mut [f32], i: usize, z: f32, c: u32, mode: Mode) {
    match mode {
        Mode::Solid => {
            if z > depth[i] {
                depth[i] = z;
                color[i] = c;
            }
        }
        Mode::Overlay => {
            if z * (1.0 + LINE_DEPTH_BIAS) >= depth[i] {
                color[i] = c;
            }
        }
        Mode::Xray => {
            color[i] = if z * (1.0 + LINE_DEPTH_BIAS) >= depth[i] { c } else { blend(color[i], c, XRAY_OCCLUDED_ALPHA) };
        }
    }
}

/// Rasterize one screen triangle into the band of rows `[y0, y0 + rows)`; buffers are
/// band-local. Pixel centres at +0.5; edges inclusive. Returns whether any pixel was covered.
fn raster_tri(color: &mut [u32], depth: &mut [f32], width: usize, y0: usize, rows: usize,
              p: &[[f64; 3]; 3], c: u32, mode: Mode) -> bool {
    let (mut a, mut b, d) = (p[0], p[1], p[2]);
    let mut area = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0]);
    if !(area.abs() > 1e-12) {
        return false;
    }
    if area < 0.0 {
        std::mem::swap(&mut a, &mut b);
        area = -area;
    }
    let minx = a[0].min(b[0]).min(d[0]).floor().max(0.0);
    let maxx = a[0].max(b[0]).max(d[0]).ceil().min(width as f64);
    let miny = a[1].min(b[1]).min(d[1]).floor().max(y0 as f64);
    let maxy = a[1].max(b[1]).max(d[1]).ceil().min((y0 + rows) as f64);
    if !(minx < maxx && miny < maxy) {
        return false;
    }
    let (minx, maxx, miny, maxy) = (minx as usize, maxx as usize, miny as usize, maxy as usize);
    // weights: w0 opposite a (edge b->d), w1 opposite b (edge d->a), w2 opposite d (edge a->b)
    let edge = |s: &[f64; 3], e: &[f64; 3], x: f64, y: f64| (e[0] - s[0]) * (y - s[1]) - (e[1] - s[1]) * (x - s[0]);
    let inv = 1.0 / area;
    let (dx0, dx1, dx2) = (-(d[1] - b[1]), -(a[1] - d[1]), -(b[1] - a[1]));
    let mut any = false;
    for y in miny..maxy {
        let py = y as f64 + 0.5;
        let px = minx as f64 + 0.5;
        let mut w0 = edge(&b, &d, px, py);
        let mut w1 = edge(&d, &a, px, py);
        let mut w2 = edge(&a, &b, px, py);
        let row = (y - y0) * width;
        for x in minx..maxx {
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let z = ((w0 * a[2] + w1 * b[2] + w2 * d[2]) * inv) as f32;
                plot(color, depth, row + x, z, c, mode);
                any = true;
            }
            w0 += dx0;
            w1 += dx1;
            w2 += dx2;
        }
    }
    any
}

/// Filled screen disc at (cx, cy) with constant 1/z.
fn raster_disc(color: &mut [u32], depth: &mut [f32], width: usize, height: usize,
               cx: f64, cy: f64, iz: f64, r: f64, c: u32, mode: Mode) {
    let minx = (cx - r).floor().max(0.0);
    let maxx = (cx + r).ceil().min(width as f64);
    let miny = (cy - r).floor().max(0.0);
    let maxy = (cy + r).ceil().min(height as f64);
    if !(minx < maxx && miny < maxy) {
        return;
    }
    let r2 = r * r;
    for y in miny as usize..maxy as usize {
        let dy = y as f64 + 0.5 - cy;
        for x in minx as usize..maxx as usize {
            let dx = x as f64 + 0.5 - cx;
            if dx * dx + dy * dy <= r2 {
                plot(color, depth, y * width + x, iz as f32, c, mode);
            }
        }
    }
}

/// Clip a camera-space polygon against z >= near (Sutherland-Hodgman, one plane).
fn clip_near(input: &[[f64; 3]], near: f64, out: &mut Vec<[f64; 3]>) {
    out.clear();
    let n = input.len();
    for i in 0..n {
        let a = input[i];
        let b = input[(i + 1) % n];
        let (ina, inb) = (a[2] >= near, b[2] >= near);
        if ina {
            out.push(a);
        }
        if ina != inb {
            let t = (near - a[2]) / (b[2] - a[2]);
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), near]);
        }
    }
}

/// Clip a camera-space segment to z >= near.
fn clip_segment(a: [f64; 3], b: [f64; 3], near: f64) -> Option<([f64; 3], [f64; 3])> {
    match (a[2] >= near, b[2] >= near) {
        (true, true) => Some((a, b)),
        (false, false) => None,
        (ina, _) => {
            let t = (near - a[2]) / (b[2] - a[2]);
            let m = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), near];
            Some(if ina { (a, m) } else { (m, b) })
        }
    }
}

/// Camera-space flat lighting: key light from upper left, fill from the eye.
#[inline]
fn lambert(n: &[f64; 3]) -> f64 {
    const KEY: [f64; 3] = [-0.42, 0.58, -0.70];
    let len = dot3(n, n).sqrt();
    if !(len > 0.0) {
        return 0.5;
    }
    let key = (dot3(n, &KEY) / (len * dot3(&KEY, &KEY).sqrt())).max(0.0);
    let fill = (-n[2] / len).max(0.0);
    0.20 + 0.62 * key + 0.24 * fill
}

impl Renderer {
    pub fn new(width: usize, height: usize) -> Self {
        let mut r = Renderer {
            width: 0,
            height: 0,
            color: Vec::new(),
            depth: Vec::new(),
            background: Vec::new(),
            cam: Vec::new(),
            scr: Vec::new(),
            tris: Vec::new(),
            stats: RenderStats::default(),
            cull_backfaces: true,
            threads: 0,
            muscle_activity: None,
        };
        r.resize(width, height);
        r
    }

    /// Sets per-muscle activation for coloring (`None` restores the uniform muscle color).
    /// Each `Some` value must be finite and in [0, 1]; the length is checked against the model in `render`.
    pub fn set_muscle_activity(&mut self, activity: Option<&[Option<f64>]>) -> Result<(), String> {
        match activity {
            None => self.muscle_activity = None,
            Some(values) => {
                for (i, v) in values.iter().enumerate() {
                    if let Some(a) = v {
                        if !a.is_finite() || !(0.0..=1.0).contains(a) {
                            return Err(format!("muscle activity {i} = {a} must be finite in [0, 1]"));
                        }
                    }
                }
                self.muscle_activity = Some(values.to_vec());
            }
        }
        Ok(())
    }

    pub fn resize(&mut self, width: usize, height: usize) {
        if width == self.width && height == self.height && !self.color.is_empty() {
            return;
        }
        self.width = width;
        self.height = height;
        let n = width * height;
        self.color = vec![palette::BACKGROUND; n];
        self.depth = vec![0.0; n];
        self.background = Vec::with_capacity(n);
        for y in 0..height {
            let t = if height > 1 { y as f64 / (height - 1) as f64 } else { 0.5 };
            let c = if t < 0.55 {
                lerp_color(palette::BACKGROUND_TOP, palette::BACKGROUND, t / 0.55)
            } else {
                lerp_color(palette::BACKGROUND, palette::BACKGROUND_BOTTOM, (t - 0.55) / 0.45)
            };
            self.background.extend(std::iter::repeat_n(c, width));
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    /// Frame buffer, 0x00RRGGBB, row-major `width * height`.
    pub fn pixels(&self) -> &[u32] {
        &self.color
    }
    /// Reciprocal depth buffer (0 = nothing drawn).
    pub fn depth(&self) -> &[f32] {
        &self.depth
    }
    pub fn stats(&self) -> RenderStats {
        self.stats
    }
    /// Background colour of row `y` (vertical gradient).
    pub fn background_at(&self, y: usize) -> u32 {
        self.background.get(y * self.width).copied().unwrap_or(palette::BACKGROUND)
    }

    /// Reset colour to the background gradient and depth to empty.
    pub fn clear(&mut self) {
        self.color.copy_from_slice(&self.background);
        self.depth.fill(0.0);
    }

    /// Depth-tested, depth-writing world triangle (no culling, flat colour). For overlays/tests.
    pub fn draw_triangle(&mut self, view: &View, world: [[f64; 3]; 3], color: u32) {
        let cam = world.map(|p| view.to_camera(&p));
        self.raster_camera_polygon(view, &cam, color, Mode::Solid);
    }

    /// Thick world line. Depth-tested unless `xray` (occluded part blended).
    pub fn draw_line(&mut self, view: &View, a: [f64; 3], b: [f64; 3], width_px: f64, color: u32, xray: bool) {
        let mode = if xray { Mode::Xray } else { Mode::Overlay };
        self.segment(view, view.to_camera(&a), view.to_camera(&b), width_px, color, mode, false);
    }

    fn raster_camera_polygon(&mut self, view: &View, cam: &[[f64; 3]; 3], color: u32, mode: Mode) -> bool {
        let (w, h) = (self.width, self.height);
        let inside = cam.iter().filter(|c| c[2] >= view.near).count();
        if inside == 3 {
            let p = cam.map(|c| view.project_camera(&c));
            return raster_tri(&mut self.color, &mut self.depth, w, 0, h, &p, color, mode);
        }
        if inside == 0 {
            return false;
        }
        let mut poly = Vec::with_capacity(4);
        clip_near(cam, view.near, &mut poly);
        let mut any = false;
        for k in 1..poly.len().saturating_sub(1) {
            let p = [poly[0], poly[k], poly[k + 1]].map(|c| view.project_camera(&c));
            any |= raster_tri(&mut self.color, &mut self.depth, w, 0, h, &p, color, mode);
        }
        any
    }

    /// Camera-space thick segment as a screen quad (+ round caps).
    fn segment(&mut self, view: &View, a: [f64; 3], b: [f64; 3], width_px: f64, color: u32, mode: Mode, caps: bool) {
        let Some((a, b)) = clip_segment(a, b, view.near) else { return };
        let (pa, pb) = (view.project_camera(&a), view.project_camera(&b));
        let (w, h) = (self.width, self.height);
        let half = 0.5 * width_px.max(1.0);
        let (dx, dy) = (pb[0] - pa[0], pb[1] - pa[1]);
        let len = (dx * dx + dy * dy).sqrt();
        if len > 1e-6 {
            let (nx, ny) = (-dy / len * half, dx / len * half);
            let q = [
                [pa[0] + nx, pa[1] + ny, pa[2]],
                [pa[0] - nx, pa[1] - ny, pa[2]],
                [pb[0] - nx, pb[1] - ny, pb[2]],
                [pb[0] + nx, pb[1] + ny, pb[2]],
            ];
            raster_tri(&mut self.color, &mut self.depth, w, 0, h, &[q[0], q[1], q[2]], color, mode);
            raster_tri(&mut self.color, &mut self.depth, w, 0, h, &[q[0], q[2], q[3]], color, mode);
        }
        if caps && mode != Mode::Xray && half >= 1.0 {
            raster_disc(&mut self.color, &mut self.depth, w, h, pa[0], pa[1], pa[2], half, color, mode);
            raster_disc(&mut self.color, &mut self.depth, w, h, pb[0], pb[1], pb[2], half, color, mode);
        }
    }

    fn polyline(&mut self, view: &View, pts: &[[f64; 3]], width_px: f64, color: u32, mode: Mode) {
        for s in pts.windows(2) {
            let (a, b) = (view.to_camera(&s[0]), view.to_camera(&s[1]));
            self.segment(view, a, b, width_px, color, mode, true);
        }
    }

    /// Render one frame of the posed model.
    pub fn render(&mut self, model: &Model, kin: &Kinematics, visual: &VisualModel,
                  camera: &Camera, options: &Options) -> Result<(), String> {
        let t0 = Instant::now();
        self.stats = RenderStats::default();
        if let Some(m) = options.selected_muscle {
            if m >= model.muscles.len() {
                return Err(format!("selected muscle {m} out of range ({} muscles)", model.muscles.len()));
            }
        }
        if kin.geom_pos.len() != model.geoms.len() || kin.site_pos.len() != model.sites.len()
            || kin.joint_anchor.len() != model.joints.len()
        {
            return Err("kinematics do not match the model".into());
        }
        visual.check_kinematics(kin)?;
        if let Some(activity) = &self.muscle_activity {
            if activity.len() != model.muscles.len() {
                return Err(format!(
                    "muscle activity has {} entries, model has {} muscles",
                    activity.len(),
                    model.muscles.len()
                ));
            }
        }
        if self.width == 0 || self.height == 0 {
            return Ok(());
        }
        self.clear();
        let view = camera.view(self.width, self.height);
        let scale = (self.height as f64 / 760.0).clamp(0.5, 3.0);

        if options.bones {
            let tb = Instant::now();
            self.setup_bones(&view, kin, visual);
            self.raster_bones();
            self.stats.bone_millis = tb.elapsed().as_secs_f64() * 1e3;
        }

        self.draw_grid(&view, model, kin);

        let mode = if options.xray { Mode::Xray } else { Mode::Overlay };
        if options.ligaments || options.muscles {
            let mut is_muscle_tendon = vec![false; model.tendons.len()];
            for m in &model.muscles {
                is_muscle_tendon[m.tendon] = true;
            }
            if options.ligaments {
                for t in (0..model.tendons.len()).filter(|&t| !is_muscle_tendon[t]) {
                    let path = model.tendon_path(kin, t)?;
                    self.polyline(&view, &path.points, 1.6 * scale, palette::LIGAMENT, mode);
                    self.stats.ligament_paths += 1;
                }
            }
            if options.muscles {
                for (i, m) in model.muscles.iter().enumerate() {
                    if Some(i) == options.selected_muscle {
                        continue;
                    }
                    let path = model.tendon_path(kin, m.tendon)?;
                    let color = match &self.muscle_activity {
                        Some(activity) => muscle_activity_color(activity[i]),
                        None => palette::MUSCLE,
                    };
                    self.polyline(&view, &path.points, 1.7 * scale, color, mode);
                    self.stats.muscle_paths += 1;
                }
                if let Some(i) = options.selected_muscle {
                    let path = model.tendon_path(kin, model.muscles[i].tendon)?;
                    // the highlight is always visible: occluded parts are blended through
                    self.polyline(&view, &path.points, 4.0 * scale, palette::SELECTED, Mode::Xray);
                    self.polyline(&view, &path.points, 4.0 * scale, palette::SELECTED, Mode::Overlay);
                    self.stats.muscle_paths += 1;
                }
            }
        }

        if options.joints {
            let (w, h) = (self.width, self.height);
            for (j, jt) in model.joints.iter().enumerate() {
                if !jt.is_scalar() {
                    continue;
                }
                if let Some(p) = view.project(&kin.joint_anchor[j]) {
                    raster_disc(&mut self.color, &mut self.depth, w, h, p[0], p[1], p[2], 4.0 * scale, palette::JOINT_OUTLINE, mode);
                    raster_disc(&mut self.color, &mut self.depth, w, h, p[0], p[1], p[2], 2.6 * scale, palette::JOINT, mode);
                    self.stats.joints += 1;
                }
            }
        }
        self.stats.millis = t0.elapsed().as_secs_f64() * 1e3;
        Ok(())
    }

    /// Transform, cull, light and clip every bone triangle into `self.tris`.
    fn setup_bones(&mut self, view: &View, kin: &Kinematics, visual: &VisualModel) {
        self.tris.clear();
        self.tris.reserve(visual.triangle_count);
        let near = view.near;
        let basis = [view.right, view.up, view.forward];
        let mut poly: Vec<[f64; 3]> = Vec::with_capacity(4);
        for vg in &visual.geoms {
            let mesh = &visual.meshes[vg.mesh];
            let r = &kin.geom_mat[vg.geom];
            // camera-from-local: M = B * R (R row-major), t = B * (p - eye)
            let mut m = [[0.0; 3]; 3];
            for (i, bi) in basis.iter().enumerate() {
                for j in 0..3 {
                    m[i][j] = bi[0] * r[j] + bi[1] * r[3 + j] + bi[2] * r[6 + j];
                }
            }
            let d = sub3(&kin.geom_pos[vg.geom], &view.eye);
            let t = basis.map(|bi| dot3(&bi, &d));
            self.cam.clear();
            self.scr.clear();
            for v in &mesh.vertices {
                let c = [
                    m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2] + t[0],
                    m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2] + t[1],
                    m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2] + t[2],
                ];
                self.cam.push(c);
                self.scr.push(if c[2] >= near { view.project_camera(&c) } else { [0.0; 3] });
            }
            let base = if vg.unobserved { palette::BONE_UNOBSERVED } else { palette::BONE };
            for f in &mesh.faces {
                let (i0, i1, i2) = (f[0] as usize, f[1] as usize, f[2] as usize);
                let (c0, c1, c2) = (self.cam[i0], self.cam[i1], self.cam[i2]);
                self.stats.triangles_submitted += 1;
                let mut n = cross3(&sub3(&c1, &c0), &sub3(&c2, &c0));
                n = [n[0] * mesh.winding, n[1] * mesh.winding, n[2] * mesh.winding];
                if dot3(&n, &c0) >= 0.0 {
                    if self.cull_backfaces {
                        self.stats.triangles_culled += 1;
                        continue;
                    }
                    n = [-n[0], -n[1], -n[2]];
                }
                let col = shade(base, lambert(&n));
                let inside = (c0[2] >= near) as u8 + (c1[2] >= near) as u8 + (c2[2] >= near) as u8;
                if inside == 3 {
                    let p = [self.scr[i0], self.scr[i1], self.scr[i2]];
                    // reject triangles fully off screen early
                    let w = self.width as f64;
                    let h = self.height as f64;
                    let (lx, hx) = (p[0][0].min(p[1][0]).min(p[2][0]), p[0][0].max(p[1][0]).max(p[2][0]));
                    let (ly, hy) = (p[0][1].min(p[1][1]).min(p[2][1]), p[0][1].max(p[1][1]).max(p[2][1]));
                    if hx < 0.0 || hy < 0.0 || lx > w || ly > h {
                        continue;
                    }
                    self.tris.push(ScreenTri { p: p.map(|q| q.map(|x| x as f32)), color: col });
                } else if inside > 0 {
                    self.stats.triangles_clipped += 1;
                    clip_near(&[c0, c1, c2], near, &mut poly);
                    for k in 1..poly.len().saturating_sub(1) {
                        let p = [poly[0], poly[k], poly[k + 1]].map(|c| view.project_camera(&c));
                        self.tris.push(ScreenTri { p: p.map(|q| q.map(|x| x as f32)), color: col });
                    }
                }
            }
        }
    }

    /// Rasterize `self.tris` with horizontal bands in parallel (identical result to serial:
    /// each pixel belongs to exactly one band and triangles are applied in order).
    fn raster_bones(&mut self) {
        let (w, h) = (self.width, self.height);
        let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let threads = if self.threads == 0 { avail.min(8) } else { self.threads }.clamp(1, h.max(1));
        let rows = h.div_ceil(threads);
        let tris = &self.tris;
        let raster_band = |color: &mut [u32], depth: &mut [f32], y0: usize| {
            let nrows = color.len() / w;
            let (fy0, fy1) = (y0 as f32, (y0 + nrows) as f32);
            for t in tris {
                let ly = t.p[0][1].min(t.p[1][1]).min(t.p[2][1]);
                let hy = t.p[0][1].max(t.p[1][1]).max(t.p[2][1]);
                if hy < fy0 || ly > fy1 {
                    continue;
                }
                let p = t.p.map(|q| q.map(|x| x as f64));
                raster_tri(color, depth, w, y0, nrows, &p, t.color, Mode::Solid);
            }
        };
        if threads == 1 {
            raster_band(&mut self.color, &mut self.depth, 0);
        } else {
            std::thread::scope(|s| {
                for (k, (c, d)) in self.color.chunks_mut(rows * w).zip(self.depth.chunks_mut(rows * w)).enumerate() {
                    let f = &raster_band;
                    s.spawn(move || f(c, d, k * rows));
                }
            });
        }
        // front-facing, on-screen triangles handed to the rasterizer
        self.stats.triangles_rasterized = self.tris.len();
        self.stats.threads = threads;
    }

    /// Subtle depth-tested floor grid (0.25 m, major every 1 m) under the subject.
    fn draw_grid(&mut self, view: &View, model: &Model, kin: &Kinematics) {
        let floor_z = model.geoms.first().filter(|g| g.kind == 0).map(|_| kin.geom_pos[0][2]).unwrap_or(0.0);
        let root = kin.body_pos.get(1).copied().unwrap_or([0.0; 3]);
        let (cx, cy) = ((root[0] * 4.0).round() / 4.0, (root[1] * 4.0).round() / 4.0);
        let n = 8i32; // +-2 m
        let s = 0.25;
        for i in -n..=n {
            let major = i % 4 == 0;
            let c = if major { palette::GRID_MAJOR } else { palette::GRID };
            let o = i as f64 * s;
            let e = n as f64 * s;
            let lines = [
                ([cx + o, cy - e, floor_z], [cx + o, cy + e, floor_z]),
                ([cx - e, cy + o, floor_z], [cx + e, cy + o, floor_z]),
            ];
            for (a, b) in lines {
                let (ca, cb) = (view.to_camera(&a), view.to_camera(&b));
                self.segment(view, ca, cb, 1.0, c, Mode::Overlay, false);
            }
        }
    }
}

#[cfg(test)]
mod activity_tests {
    use super::*;

    #[test]
    fn activity_color_endpoints_midpoint_and_unknown() {
        assert_eq!(muscle_activity_color(Some(0.0)), ACTIVITY_LOW);
        assert_eq!(muscle_activity_color(Some(1.0)), ACTIVITY_HIGH);
        assert_eq!(muscle_activity_color(None), ACTIVITY_UNKNOWN);
        for (i, expected) in ACTIVITY_STOPS.iter().enumerate() {
            assert_eq!(muscle_activity_color(Some(i as f64 / 4.0)), *expected);
        }
        assert_eq!(muscle_activity_color(Some(0.375)), lerp_color(ACTIVITY_STOPS[1], ACTIVITY_STOPS[2], 0.5));
        assert_eq!(muscle_activity_color(Some(-0.1)), ACTIVITY_LOW);
        assert_eq!(muscle_activity_color(Some(1.1)), ACTIVITY_HIGH);
    }

    #[test]
    fn setter_rejects_invalid_activity_and_accepts_none() {
        let mut r = Renderer::new(16, 16);
        assert!(r.set_muscle_activity(Some(&[Some(0.0), None, Some(1.0)])).is_ok());
        assert!(r.set_muscle_activity(Some(&[Some(1.5)])).is_err());
        assert!(r.set_muscle_activity(Some(&[Some(-0.1)])).is_err());
        assert!(r.set_muscle_activity(Some(&[Some(f64::NAN)])).is_err());
        assert!(r.set_muscle_activity(Some(&[Some(f64::INFINITY)])).is_err());
        assert!(r.set_muscle_activity(None).is_ok());
    }
}
