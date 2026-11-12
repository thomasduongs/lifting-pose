//! Renderer tests: sourced mesh attachment vs the native reference, real-model front/side
//! renders, z-buffer occlusion, near-plane clipping and camera robustness.
//!
//! Rendered frames are written as PNG to `$CARGO_TARGET_TMPDIR/render-png` (override with
//! `BIOMECH_RENDER_PNG_DIR`) for visual inspection.
//! `cargo test --release --test render -- --ignored --nocapture` prints full-size timings.

use std::path::PathBuf;
use std::sync::OnceLock;

use biomech_lite::Model;
use biomech_viewer::camera::{Camera, MAX_ELEVATION, MAX_DISTANCE, MIN_DISTANCE};
use biomech_viewer::render::{palette, Options, Renderer};
use biomech_viewer::scene::{VisualModel, GEOM_MESH};

fn assets() -> PathBuf {
    if let Ok(p) = std::env::var("BIOMECH_ASSETS") {
        return PathBuf::from(p);
    }
    let by_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/biomech-lite");
    if by_manifest.join("model.json").exists() {
        return by_manifest;
    }
    // this file lives in <root>/crates/biomech-viewer/tests/
    std::path::Path::new(file!()).parent().unwrap().join("../../../assets/biomech-lite")
}

fn loaded() -> &'static (Model, VisualModel) {
    static CELL: OnceLock<(Model, VisualModel)> = OnceLock::new();
    CELL.get_or_init(|| {
        let model = Model::load(assets().join("model.json")).expect("model.json");
        let visual = VisualModel::load(assets().join("visual.json"), &model).expect("visual.json");
        (model, visual)
    })
}

fn standing_kin(model: &Model) -> biomech_lite::Kinematics {
    let mut q = model.standing_qpos.clone();
    model.project(&mut q).unwrap();
    model.kinematics(&q).unwrap()
}

fn save_png(name: &str, r: &Renderer) {
    let dir = std::env::var("BIOMECH_RENDER_PNG_DIR")
        .unwrap_or_else(|_| format!("{}/render-png", env!("CARGO_TARGET_TMPDIR")));
    std::fs::create_dir_all(&dir).unwrap();
    let file = std::fs::File::create(PathBuf::from(dir).join(name)).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), r.width() as u32, r.height() as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let data: Vec<u8> = r.pixels().iter().flat_map(|p| [(p >> 16) as u8, (p >> 8) as u8, *p as u8]).collect();
    enc.write_header().unwrap().write_image_data(&data).unwrap();
}

fn count_color(r: &Renderer, c: u32) -> usize {
    r.pixels().iter().filter(|&&p| p == c).count()
}

fn non_background(r: &Renderer) -> usize {
    let w = r.width();
    r.pixels().iter().enumerate().filter(|(i, p)| **p != r.background_at(i / w)).count()
}

fn is_bone_like(p: u32) -> bool {
    // shaded ivory: red >= green >= blue, warm and not dark
    let (r, g, b) = ((p >> 16) & 0xff, (p >> 8) & 0xff, p & 0xff);
    r >= g && g >= b && r > 60 && r - b > 8
}

// ------------------------------------------------------------------ sourced geometry

#[test]
fn visual_asset_matches_model_and_native_reference() {
    let (model, visual) = loaded();
    assert_eq!(visual.meshes.len(), 104, "all native meshes kept");
    let mesh_geoms = model.geoms.iter().filter(|g| g.kind == GEOM_MESH).count();
    assert_eq!(visual.geoms.len(), mesh_geoms);
    assert_eq!(visual.geoms.len(), 102);
    let all_faces: usize = visual.meshes.iter().map(|m| m.faces.len()).sum();
    assert!(all_faces > 320_000, "source scale, got {all_faces}");
    assert!(visual.triangle_count > 280_000, "instanced geom triangles {}", visual.triangle_count);

    // independent attachment check: native engine world positions at the reference qpos
    let kin = model.kinematics(&visual.reference_qpos).unwrap();
    assert!(visual.reference_samples.len() >= 3 * 102);
    let mut worst: f64 = 0.0;
    for s in &visual.reference_samples {
        let vg = visual.geom_by_native(s.geom).unwrap();
        let p = visual.world_vertex(&kin, vg, s.vertex);
        let err = (0..3).map(|k| (p[k] - s.world_position[k]).abs()).fold(0.0, f64::max);
        worst = worst.max(err);
    }
    assert!(worst < 1e-6, "worst attachment error {worst} m");
}

#[test]
fn visual_asset_rejects_mismatched_source_hash() {
    let (model, _) = loaded();
    let text = r#"{"schema_version":1,"source_model_sha256":"deadbeef","meshes":[],"geoms":[],
        "reference":{"qpos":[],"samples":[]}}"#;
    let err = VisualModel::from_json_str(text, model).unwrap_err();
    assert!(err.contains("does not match"), "{err}");
}

// ------------------------------------------------------------------ real-model renders

#[test]
fn front_and_side_renders_show_bones_and_paths() {
    let (model, visual) = loaded();
    let kin = standing_kin(model);
    let mut r = Renderer::new(500, 380);
    let mut cam = Camera::front();
    cam.aspect = 500.0 / 380.0;
    cam.fit(model, &kin, visual, false);
    let opts = Options { joints: true, selected_muscle: model.muscle_id("BIClong"), ..Options::default() };

    r.render(model, &kin, visual, &cam, &opts).unwrap();
    save_png("front.png", &r);
    let front = r.pixels().to_vec();
    let n = (r.width() * r.height()) as f64;
    let covered = non_background(&r) as f64 / n;
    let bone = r.pixels().iter().filter(|&&p| is_bone_like(p)).count() as f64 / n;
    assert!(covered > 0.05 && covered < 0.9, "front coverage {covered}");
    assert!(bone > 0.02, "front bone pixels {bone}");
    assert!(count_color(&r, palette::MUSCLE) > 100, "muscle paths visible");
    assert!(count_color(&r, palette::SELECTED) > 20, "selected path visible");
    assert!(count_color(&r, palette::JOINT) > 20, "joint anchors visible");
    let st = r.stats();
    assert_eq!(st.triangles_submitted, visual.triangle_count, "no anatomy dropped");
    assert!(st.triangles_culled > 0 && st.triangles_rasterized > 0);
    assert_eq!(st.muscle_paths, model.muscles.len());
    assert_eq!(st.ligament_paths, model.tendons.len() - model.muscles.len());

    // the front view sees the face: the head's front is nearer than its back, so the
    // skull occupies the upper middle of the frame
    let (w, h) = (r.width(), r.height());
    let head_rows = (0..h / 4).flat_map(|y| (w * 2 / 5..w * 3 / 5).map(move |x| (x, y)));
    assert!(head_rows.filter(|&(x, y)| is_bone_like(r.pixels()[y * w + x])).count() > 50, "skull at top centre");

    cam.preset("side");
    r.render(model, &kin, visual, &cam, &opts).unwrap();
    save_png("side.png", &r);
    let covered_side = non_background(&r) as f64 / n;
    assert!(covered_side > 0.03, "side coverage {covered_side}");
    let differ = r.pixels().iter().zip(&front).filter(|(a, b)| a != b).count();
    assert!(differ as f64 > 0.02 * n, "side view differs from front");

    cam.preset("back");
    cam.fit(model, &kin, visual, true);
    r.render(model, &kin, visual, &cam, &Options { xray: true, ..Options::default() }).unwrap();
    save_png("back_upper_xray.png", &r);
    assert!(non_background(&r) as f64 / n > 0.05);

    // layers off: only background + grid
    let none = Options { bones: false, muscles: false, ligaments: false, joints: false, xray: false, selected_muscle: None };
    r.render(model, &kin, visual, &cam, &none).unwrap();
    assert_eq!(r.pixels().iter().filter(|&&p| is_bone_like(p) && p != palette::LIGAMENT).count(), 0);
    assert!(r.render(model, &kin, visual, &cam, &Options { selected_muscle: Some(10_000), ..Options::default() }).is_err());
}

#[test]
fn render_is_deterministic_across_thread_counts() {
    let (model, visual) = loaded();
    let kin = standing_kin(model);
    let cam = Camera::front();
    let mut a = Renderer::new(320, 240);
    a.threads = 1;
    let mut b = Renderer::new(320, 240);
    b.threads = 5;
    a.render(model, &kin, visual, &cam, &Options::default()).unwrap();
    b.render(model, &kin, visual, &cam, &Options::default()).unwrap();
    assert!(a.pixels() == b.pixels());
}

#[test]
fn camera_inside_the_body_is_safe() {
    let (model, visual) = loaded();
    let kin = standing_kin(model);
    let mut cam = Camera::front();
    cam.distance = MIN_DISTANCE;
    // put the near plane exactly through bone vertices, so their faces straddle it
    let (_, _, forward) = cam.basis();
    for (k, vg) in visual.geoms.iter().enumerate().step_by(17) {
        let v = visual.world_vertex(&kin, vg, visual.meshes[vg.mesh].vertices.len() / 2);
        let eye = [0, 1, 2].map(|i| v[i] - forward[i] * biomech_viewer::camera::NEAR);
        let off = [0, 1, 2].map(|i| cam.eye()[i] - cam.target[i]);
        cam.target = [0, 1, 2].map(|i| eye[i] - off[i]);
        let mut r = Renderer::new(200, 150);
        r.cull_backfaces = k % 2 == 0;
        r.render(model, &kin, visual, &cam, &Options::default()).unwrap();
        assert!(r.cull_backfaces || r.stats().triangles_clipped > 0, "geom {}", vg.geom);
        assert!(r.depth().iter().all(|d| d.is_finite() && *d >= 0.0));
        assert!(r.pixels().iter().all(|p| *p <= 0xffffff));
    }
}

#[test]
#[ignore = "timing report; run in release"]
fn full_resolution_timing() {
    let (model, visual) = loaded();
    let kin = standing_kin(model);
    let mut r = Renderer::new(1000, 760);
    let cam = Camera::front();
    r.render(model, &kin, visual, &cam, &Options::default()).unwrap();
    let mut best = f64::INFINITY;
    let mut bones = f64::INFINITY;
    for _ in 0..10 {
        r.render(model, &kin, visual, &cam, &Options::default()).unwrap();
        best = best.min(r.stats().millis);
        bones = bones.min(r.stats().bone_millis);
    }
    eprintln!("1000x760: frame {best:.1} ms (bones {bones:.1} ms), {:?}", r.stats());
}

// ------------------------------------------------------------------ synthetic raster tests

/// Eye at (0,-1,0) looking down +Y (front preset), so camera depth = world y + 1.
fn origin_view(w: usize, h: usize) -> biomech_viewer::camera::View {
    let mut cam = Camera::front();
    cam.target = [0.0, 0.0, 0.0];
    cam.elevation = 0.0;
    cam.distance = 1.0;
    let mut v = cam.view(w, h);
    v.eye = [0.0, -1.0, 0.0];
    v
}

#[test]
fn z_buffer_occludes_regardless_of_draw_order() {
    let v = origin_view(100, 100);
    let near_tri = [[-0.2, 1.0, -0.2], [0.2, 1.0, -0.2], [0.0, 1.0, 0.2]];
    let far_tri = [[-2.0, 3.0, -2.0], [2.0, 3.0, -2.0], [0.0, 3.0, 2.0]];
    for near_first in [true, false] {
        let mut r = Renderer::new(100, 100);
        r.clear();
        if near_first {
            r.draw_triangle(&v, near_tri, 0xff0000);
            r.draw_triangle(&v, far_tri, 0x00ff00);
        } else {
            r.draw_triangle(&v, far_tri, 0x00ff00);
            r.draw_triangle(&v, near_tri, 0xff0000);
        }
        assert_eq!(r.pixels()[50 * 100 + 50], 0xff0000, "near triangle wins (near_first={near_first})");
        assert_eq!(r.pixels()[95 * 100 + 10], 0x00ff00, "far triangle visible outside");
        // reciprocal depth: centre stores 1/(2 m)
        assert!((r.depth()[50 * 100 + 50] - 0.5).abs() < 1e-4);
    }
}

#[test]
fn interpenetrating_triangles_use_perspective_depth() {
    // a slanted triangle crossing a flat one: each wins on its own half
    let v = origin_view(100, 100);
    let mut r = Renderer::new(100, 100);
    r.clear();
    r.draw_triangle(&v, [[-1.0, 2.0, -1.0], [1.0, 2.0, -1.0], [0.0, 2.0, 1.0]], 0x0000ff);
    r.draw_triangle(&v, [[-1.0, 1.0, -1.0], [1.0, 3.0, -1.0], [0.0, 2.0, 1.0]], 0xff0000);
    let row = 60 * 100;
    assert_eq!(r.pixels()[row + 40], 0xff0000, "left: slanted triangle nearer");
    assert_eq!(r.pixels()[row + 60], 0x0000ff, "right: flat triangle nearer");
}

#[test]
fn lines_are_depth_tested_and_xray_shows_through() {
    let v = origin_view(100, 100);
    let wall = [[-2.0, 2.0, -2.0], [2.0, 2.0, -2.0], [0.0, 2.0, 2.0]];
    let mut r = Renderer::new(100, 100);
    r.clear();
    r.draw_triangle(&v, wall, 0x404040);
    r.draw_line(&v, [-0.5, 3.0, 0.0], [0.5, 3.0, 0.0], 3.0, 0xffff00, false);
    assert_eq!(r.pixels()[50 * 100 + 50], 0x404040, "line behind wall hidden");
    r.draw_line(&v, [-0.5, 3.0, 0.0], [0.5, 3.0, 0.0], 3.0, 0xffff00, true);
    let p = r.pixels()[50 * 100 + 50];
    assert!(p != 0x404040 && p != 0xffff00, "xray blends occluded line: {p:06x}");
    r.draw_line(&v, [-0.5, 1.5, 0.1], [0.5, 1.5, 0.1], 3.0, 0x00ffff, false);
    let y = (50.0 - v.focal * 0.1 / 2.5).floor() as usize;
    assert_eq!(r.pixels()[y * 100 + 50], 0x00ffff, "line in front visible");
}

#[test]
fn near_plane_clipping_is_safe() {
    let v = origin_view(100, 100);
    let mut r = Renderer::new(100, 100);
    r.clear();
    // fully behind the eye: nothing drawn
    r.draw_triangle(&v, [[-1.0, -2.0, -1.0], [1.0, -2.0, -1.0], [0.0, -2.0, 1.0]], 0xff0000);
    assert_eq!(count_color(&r, 0xff0000), 0);
    // straddling the eye plane: drawn, finite, covering the near half of the screen
    r.draw_triangle(&v, [[-1.0, -1.0, -0.2], [1.0, -1.0, -0.2], [0.0, 3.0, -0.2]], 0xff0000);
    assert!(count_color(&r, 0xff0000) > 500);
    assert!(r.depth().iter().all(|d| d.is_finite()));
    // degenerate and huge inputs do not panic
    r.draw_triangle(&v, [[0.0, 1.0, 0.0]; 3], 0x00ff00);
    r.draw_triangle(&v, [[-1e7, 1.0, -1e7], [1e7, 1.0, -1e7], [0.0, 1.0, 1e7]], 0x00ff00);
    r.draw_line(&v, [0.0, -5.0, 0.0], [0.0, 5.0, 0.0], 2.0, 0x0000ff, false);
    r.draw_line(&v, [0.0, -5.0, 0.0], [0.0, -4.0, 0.0], 2.0, 0x0000ff, true);
}

// ------------------------------------------------------------------ camera

#[test]
fn camera_presets_orbit_pan_zoom_stay_finite() {
    let mut c = Camera::front();
    let target = c.target;
    assert!(c.eye()[1] < target[1], "front eye on -Y (subject faces -Y)");
    c.preset("back");
    assert!(c.eye()[1] > target[1]);
    c.preset("side");
    assert!((c.eye()[0] - target[0]).abs() > 0.9 * c.distance);
    assert!(c.try_preset("bogus").is_err());
    let before = c;
    c.preset("bogus");
    assert_eq!(c, before);

    for i in 0..2000 {
        let s = (i as f64 * 0.37).sin() * 400.0;
        c.orbit(s, -s * 1.3);
        c.pan(s * 0.1, s * 0.05);
        c.zoom(if i % 2 == 0 { 7.0 } else { -3.0 });
    }
    c.orbit(f64::NAN, f64::INFINITY);
    c.pan(f64::NAN, 1.0);
    c.zoom(f64::INFINITY);
    c.zoom(1e300);
    assert!(c.target.iter().chain(c.eye().iter()).all(|x| x.is_finite()));
    assert!(c.elevation.abs() <= MAX_ELEVATION && (MIN_DISTANCE..=MAX_DISTANCE).contains(&c.distance));
    let (r, u, f) = c.basis();
    let d = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    assert!(d(r, u).abs() < 1e-9 && d(r, f).abs() < 1e-9 && d(u, f).abs() < 1e-9);
    assert!((d(r, r) - 1.0).abs() < 1e-9 && (d(u, u) - 1.0).abs() < 1e-9);

    // zoom in moves closer; projection of the target lands on the centre
    let mut z = Camera::front();
    let d0 = z.distance;
    z.zoom(1.0);
    assert!(z.distance < d0);
    let v = z.view(640, 480);
    let p = v.project(&z.target).unwrap();
    assert!((p[0] - 320.0).abs() < 1e-9 && (p[1] - 240.0).abs() < 1e-9);
}

#[test]
fn fit_frames_whole_and_upper_body() {
    let (model, visual) = loaded();
    let kin = standing_kin(model);
    let mut full = Camera::front();
    full.fit(model, &kin, visual, false);
    let mut upper = Camera::front();
    upper.fit(model, &kin, visual, true);
    assert!(upper.distance < full.distance, "upper body framed closer");
    assert!(upper.target[2] > full.target[2], "upper body framing is higher");
    // every bone vertex of the full fit projects inside the viewport
    let v = full.view(1000, 760);
    for vg in &visual.geoms {
        for i in (0..visual.meshes[vg.mesh].vertices.len()).step_by(7) {
            let p = v.project(&visual.world_vertex(&kin, vg, i)).unwrap();
            assert!((-1.0..=1001.0).contains(&p[0]) && (-1.0..=761.0).contains(&p[1]), "{p:?}");
        }
    }
}
