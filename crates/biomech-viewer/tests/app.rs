use biomech_lite::Model;
use biomech_viewer::app::{Action, App, Mode, layout, SIDEBAR_W, HEADER_H, TIMELINE_H};
use biomech_viewer::scene::VisualModel;
use biomech_viewer::session::Motion;
use std::path::PathBuf;

fn assets() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/biomech-lite")
}

fn app(mode: Mode) -> App {
    let model = Model::load(assets().join("model.json")).unwrap();
    let visual = VisualModel::load(assets().join("visual.json"), &model).unwrap();
    let motion = Motion::load(assets().join("motion.json"), &model).unwrap();
    App::new(model, visual, motion, mode, (1280, 860)).unwrap()
}

#[test]
fn layout_matches_contract() {
    let l = layout(1280, 860);
    assert_eq!((l.viewport.x, l.viewport.y, l.viewport.w, l.viewport.h), (SIDEBAR_W, HEADER_H, 1000, 760));
    assert_eq!(l.timeline.h, TIMELINE_H);
    assert_eq!(l.viewport.y + l.viewport.h, l.timeline.y);
}

#[test]
fn recording_plays_loops_and_steps() {
    let mut a = app(Mode::Recording);
    assert!(a.is_playing());
    let d = a.duration();
    a.update(0.5);
    assert!((a.time - 0.5).abs() < 1e-12);
    a.apply(Action::Speed(2.0));
    a.update(0.5);
    assert!((a.time - 1.5).abs() < 1e-12);
    a.time = d - 0.1;
    a.update(0.3);
    assert!(a.is_playing() && a.time < 0.51, "loop wraps, t={}", a.time);
    a.apply(Action::ToggleLoop);
    a.time = d - 0.01;
    a.update(1.0);
    assert_eq!(a.time, d);
    assert!(!a.is_playing(), "stops at last frame without loop");
    a.apply(Action::StepFrame(-1));
    assert!(a.time < d);
    a.apply(Action::Scrub(0.5));
    assert!((a.time - 0.5 * d).abs() < 1e-12);
}

#[test]
fn simulation_starts_paused_and_mode_switch_pauses() {
    let mut a = app(Mode::Simulation);
    assert_eq!(a.mode, Mode::Simulation);
    assert!(!a.is_playing());
    a.apply(Action::TogglePlay);
    assert!(a.is_playing());
    a.apply(Action::ToggleMode);
    assert_eq!(a.mode, Mode::Recording);
    assert!(!a.is_playing());
    // excitation is bounded
    for _ in 0..40 {
        a.apply(Action::ExcitationDelta(0.05));
    }
    assert_eq!(a.excitation, 1.0);
    for _ in 0..40 {
        a.apply(Action::ExcitationDelta(-0.05));
    }
    assert_eq!(a.excitation, 0.0);
}

#[test]
fn clicking_buttons_and_composing_works() {
    let mut a = app(Mode::Recording);
    let play = a.buttons().into_iter().find(|b| b.action == Action::TogglePlay).unwrap();
    let (x, y) = ((play.rect.x + 2) as f32, (play.rect.y + 2) as f32);
    a.pointer(x, y, true, false, 0.0);
    a.pointer(x, y, false, false, 0.0);
    assert!(!a.is_playing());
    // layers can be toggled via buttons
    let before = a.options.xray;
    a.apply(Action::ToggleXray);
    assert_ne!(before, a.options.xray);
    let px = a.compose().unwrap().to_vec();
    assert_eq!(px.len(), 1280 * 860);
    // viewport is not blank
    let l = layout(1280, 860);
    let bg = px[(l.viewport.y + 2) * 1280 + l.viewport.x + 2];
    assert!(px.iter().any(|&p| p != bg));
    // resize keeps things consistent
    a.resize(800, 600);
    assert_eq!(a.compose().unwrap().len(), 800 * 600);
}

fn all_vertices_in_viewport(a: &App, qpos: &[f64], upper: bool) {
    let mut q = qpos.to_vec();
    a.model.project(&mut q).unwrap();
    let kin = a.model.kinematics(&q).unwrap();
    let (w, h) = a.size();
    let l = layout(w, h);
    let view = a.camera.view(l.viewport.w, l.viewport.h);
    let has_obs = a.visual.geoms.iter().any(|g| !g.unobserved);
    for vg in a.visual.geoms.iter().filter(|g| !upper || !has_obs || !g.unobserved) {
        for i in 0..a.visual.meshes[vg.mesh].vertices.len() {
            let wv = a.visual.world_vertex(&kin, vg, i);
            let p = view.project(&wv).expect("in front of camera");
            assert!(
                p[0] >= 0.0 && p[0] <= l.viewport.w as f64 && p[1] >= 0.0 && p[1] <= l.viewport.h as f64,
                "vertex outside viewport at ({:.1},{:.1})",
                p[0],
                p[1]
            );
        }
    }
}

#[test]
fn envelope_framing_keeps_all_poses_in_view() {
    let mut a = app(Mode::Recording);
    let n = a.motion.len();
    for upper in [false, true] {
        a.upper = upper;
        for v in 0..3u8 {
            a.apply(Action::View(v));
            for k in [4usize, n / 2, n - 1] {
                let q = a.motion.qpos[k].clone();
                all_vertices_in_viewport(&a, &q, upper);
            }
        }
    }
}

#[test]
fn framing_is_independent_of_time() {
    let mut a = app(Mode::Recording);
    a.time = 0.0;
    a.apply(Action::Frame);
    let c0 = a.camera;
    a.time = a.duration() * 0.7;
    a.apply(Action::Frame);
    assert_eq!(c0, a.camera);
}

#[test]
fn save_png_creates_parent_dirs() {
    let mut a = app(Mode::Recording);
    a.compose().unwrap();
    let root = std::env::temp_dir().join(format!("bv_test_{}", std::process::id()));
    let p = root.join("nested").join("deeper").join("x.png");
    a.save_png(&p).unwrap();
    assert!(p.exists());
    let _ = std::fs::remove_dir_all(&root);
}
