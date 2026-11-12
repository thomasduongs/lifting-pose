//! Orbit camera (native z-up; the subject faces -Y) and the perspective view used by the
//! renderer. Camera space: x right, y up, z forward (depth); screen y grows downwards.

use std::f64::consts::{FRAC_PI_2, PI};

use biomech_lite::kinematics::{cross3, dot3, normalize3, scl3, sub3, add3};
use biomech_lite::{Kinematics, Model};

use crate::scene::VisualModel;

/// Radians of orbit per mouse pixel.
pub const ORBIT_RADIANS_PER_PIXEL: f64 = 0.008;
/// Zoom factor exponent per wheel unit (`distance *= exp(-ZOOM_RATE * delta)`).
pub const ZOOM_RATE: f64 = 0.12;
pub const MIN_DISTANCE: f64 = 0.1;
pub const MAX_DISTANCE: f64 = 30.0;
/// Elevation limit (radians), keeps the up vector well defined.
pub const MAX_ELEVATION: f64 = 89.0 * PI / 180.0;
/// Near clipping plane (metres in front of the eye).
pub const NEAR: f64 = 0.02;
/// Reference viewport height used to scale pan pixels when the real one is unknown.
pub const REFERENCE_HEIGHT: f64 = 760.0;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Orbit centre (world).
    pub target: [f64; 3],
    /// Eye distance from `target` (m).
    pub distance: f64,
    /// Eye direction around +Z: eye = target + d (cos el cos az, cos el sin az, sin el).
    pub azimuth: f64,
    pub elevation: f64,
    /// Vertical field of view (radians).
    pub fov_y: f64,
    /// Viewport width / height, used by [`Camera::fit`].
    pub aspect: f64,
}

/// Perspective view of a camera on a `width x height` viewport.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct View {
    pub eye: [f64; 3],
    pub right: [f64; 3],
    pub up: [f64; 3],
    pub forward: [f64; 3],
    /// Focal length in pixels.
    pub focal: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: usize,
    pub height: usize,
    pub near: f64,
}

impl View {
    /// World point -> camera coordinates (x right, y up, z depth).
    #[inline]
    pub fn to_camera(&self, p: &[f64; 3]) -> [f64; 3] {
        let d = sub3(p, &self.eye);
        [dot3(&d, &self.right), dot3(&d, &self.up), dot3(&d, &self.forward)]
    }

    /// Camera point (z >= near) -> (screen x, screen y, 1/z).
    #[inline]
    pub fn project_camera(&self, c: &[f64; 3]) -> [f64; 3] {
        let iz = 1.0 / c[2];
        [self.cx + self.focal * c[0] * iz, self.cy - self.focal * c[1] * iz, iz]
    }

    /// World point -> (screen x, screen y, 1/z), `None` behind the near plane.
    pub fn project(&self, p: &[f64; 3]) -> Option<[f64; 3]> {
        let c = self.to_camera(p);
        (c[2] >= self.near).then(|| self.project_camera(&c))
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::front()
    }
}

impl Camera {
    /// Default front view: the eye is on -Y looking at the subject's face.
    pub fn front() -> Self {
        Camera {
            target: [0.0, 0.0, 0.95],
            distance: 3.4,
            azimuth: -FRAC_PI_2,
            elevation: 6.0 * PI / 180.0,
            fov_y: 35.0 * PI / 180.0,
            aspect: 1000.0 / 760.0,
        }
    }

    /// `front`, `side` (subject's right, eye on -X) or `back`; keeps target and distance.
    pub fn try_preset(&mut self, name: &str) -> Result<(), String> {
        self.azimuth = match name.to_ascii_lowercase().as_str() {
            "front" => -FRAC_PI_2,
            "side" | "right" => PI,
            "left" => 0.0,
            "back" => FRAC_PI_2,
            other => return Err(format!("unknown camera preset {other:?} (front|side|back)")),
        };
        self.elevation = 6.0 * PI / 180.0;
        Ok(())
    }

    /// Like [`Camera::try_preset`]; unknown names leave the camera unchanged.
    pub fn preset(&mut self, name: &str) {
        let _ = self.try_preset(name);
    }

    pub fn eye(&self) -> [f64; 3] {
        let (se, ce) = self.elevation.sin_cos();
        let (sa, ca) = self.azimuth.sin_cos();
        add3(&self.target, &scl3(&[ce * ca, ce * sa, se], self.distance))
    }

    /// Orthonormal (right, up, forward) basis.
    pub fn basis(&self) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let (se, ce) = self.elevation.sin_cos();
        let (sa, ca) = self.azimuth.sin_cos();
        let forward = [-ce * ca, -ce * sa, -se];
        // right = forward x Z, horizontal and well defined while |elevation| < 90 deg
        let mut right = cross3(&forward, &[0.0, 0.0, 1.0]);
        if normalize3(&mut right) < 1e-9 {
            right = [-sa, ca, 0.0];
        }
        let up = cross3(&right, &forward);
        (right, up, forward)
    }

    pub fn view(&self, width: usize, height: usize) -> View {
        let (right, up, forward) = self.basis();
        let h = height.max(1) as f64;
        View {
            eye: self.eye(),
            right,
            up,
            forward,
            focal: 0.5 * h / (0.5 * self.fov_y).tan(),
            cx: 0.5 * width as f64,
            cy: 0.5 * h,
            width,
            height,
            near: NEAR,
        }
    }

    /// Frame the bones (all, or only the observed upper body) for the current direction.
    pub fn fit(&mut self, _model: &Model, kin: &Kinematics, visual: &VisualModel, upper: bool) {
        let has_observed = visual.geoms.iter().any(|g| !g.unobserved);
        let pick = |g: &crate::scene::VisualGeom| !upper || !has_observed || !g.unobserved;
        let Some((lo, hi)) = visual.world_bounds(kin, pick) else { return };
        let centre = scl3(&add3(&lo, &hi), 0.5);
        let (right, up, forward) = self.basis();
        let mut ext = [[f64::INFINITY, f64::NEG_INFINITY]; 3];
        for vg in visual.geoms.iter().filter(|g| pick(g)) {
            for i in 0..visual.meshes[vg.mesh].vertices.len() {
                let d = sub3(&visual.world_vertex(kin, vg, i), &centre);
                for (k, axis) in [&right, &up, &forward].into_iter().enumerate() {
                    let s = dot3(&d, axis);
                    ext[k][0] = ext[k][0].min(s);
                    ext[k][1] = ext[k][1].max(s);
                }
            }
        }
        if !ext.iter().all(|e| e[0].is_finite() && e[1].is_finite()) {
            return;
        }
        let mid = |k: usize| 0.5 * (ext[k][0] + ext[k][1]);
        let half = |k: usize| 0.5 * (ext[k][1] - ext[k][0]);
        let target = add3(&centre, &add3(&scl3(&right, mid(0)), &add3(&scl3(&up, mid(1)), &scl3(&forward, mid(2)))));
        let t = (0.5 * self.fov_y).tan();
        let aspect = if self.aspect.is_finite() && self.aspect > 0.0 { self.aspect } else { 1.0 };
        let margin = 1.08;
        let dist = (half(1) / t).max(half(0) / (t * aspect)) * margin + half(2);
        if target.iter().all(|x| x.is_finite()) && dist.is_finite() {
            self.target = target;
            self.distance = dist.clamp(MIN_DISTANCE, MAX_DISTANCE);
        }
    }

    /// Orbit by mouse-drag pixels (drag right turns the subject to the right).
    pub fn orbit(&mut self, dx: f64, dy: f64) {
        if !(dx.is_finite() && dy.is_finite()) {
            return;
        }
        self.azimuth = (self.azimuth - dx * ORBIT_RADIANS_PER_PIXEL).rem_euclid(2.0 * PI);
        self.elevation = (self.elevation + dy * ORBIT_RADIANS_PER_PIXEL).clamp(-MAX_ELEVATION, MAX_ELEVATION);
    }

    /// Pan by mouse-drag pixels: the scene follows the cursor.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        if !(dx.is_finite() && dy.is_finite()) {
            return;
        }
        let (right, up, _) = self.basis();
        let s = self.distance * 2.0 * (0.5 * self.fov_y).tan() / REFERENCE_HEIGHT;
        let delta = add3(&scl3(&right, -dx * s), &scl3(&up, dy * s));
        let t = add3(&self.target, &delta);
        if t.iter().all(|x| x.is_finite() && x.abs() < 1e3) {
            self.target = t;
        }
    }

    /// Zoom by wheel units (positive = closer).
    pub fn zoom(&mut self, delta: f64) {
        if !delta.is_finite() {
            return;
        }
        self.distance = (self.distance * (-ZOOM_RATE * delta.clamp(-50.0, 50.0)).exp())
            .clamp(MIN_DISTANCE, MAX_DISTANCE);
    }
}
