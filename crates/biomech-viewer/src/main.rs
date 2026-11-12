//! Native biomech viewer entry point (window + CLI). All logic lives in `biomech_viewer::app`.

use biomech_lite::Model;
use biomech_viewer::app::{Action, App, DEFAULT_SIZE, Mode};
use biomech_viewer::scene::VisualModel;
use biomech_viewer::session::Motion;
use minifb::{Key, KeyRepeat, MouseButton, MouseMode, Window, WindowOptions};
use std::path::{Path, PathBuf};
use std::time::Instant;

const HELP: &str = "biomech-viewer: native viewer for the MyoSim full-body reference model

usage: biomech-viewer [options]
  --model PATH       model.json   (default assets/biomech-lite/model.json)
  --visual PATH      visual.json  (default assets/biomech-lite/visual.json)
  --motion PATH      motion.json  (default assets/biomech-lite/motion.json)
  --simulation       start in live arm-simulation mode (paused until Space)
  --snapshot PATH    headless: render one full-UI PNG and exit (no window)
  --time SEC         recording time for --snapshot (default 0)
  --view NAME        front | side | back (default front)
  --upper-body       frame the upper body
  --frames N         open the window, render N frames, then exit (smoke test)
  -h, --help         this help

controls:
  Space play/pause      Home reset          L loop          Tab recording/simulation
  Left/Right step frame , . slower/faster   1/2/3 front/side/back
  F frame all           U upper/full body   B/M/G/J bones/muscles/ligaments/joints
  X x-ray paths         [ ] select muscle   + - excitation  P save PNG
  Mouse: drag orbit, right-drag pan, wheel zoom, click sidebar buttons / timeline.
  Esc quits.";

struct Cli {
    model: Option<PathBuf>,
    visual: Option<PathBuf>,
    motion: Option<PathBuf>,
    simulation: bool,
    snapshot: Option<PathBuf>,
    time: f64,
    view: u8,
    upper: bool,
    frames: Option<u64>,
}

fn parse() -> Result<Option<Cli>, String> {
    let mut cli = Cli { model: None, visual: None, motion: None, simulation: false, snapshot: None, time: 0.0, view: 0, upper: false, frames: None };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "-h" | "--help" => return Ok(None),
            "--model" => cli.model = Some(val("--model")?.into()),
            "--visual" => cli.visual = Some(val("--visual")?.into()),
            "--motion" => cli.motion = Some(val("--motion")?.into()),
            "--simulation" => cli.simulation = true,
            "--snapshot" => cli.snapshot = Some(val("--snapshot")?.into()),
            "--time" => {
                cli.time = val("--time")?.parse().map_err(|_| "--time needs a number".to_string())?;
                if !cli.time.is_finite() || cli.time < 0.0 {
                    return Err("--time must be a finite non-negative number".into());
                }
            }
            "--view" => {
                cli.view = match val("--view")?.as_str() {
                    "front" => 0,
                    "side" => 1,
                    "back" => 2,
                    o => return Err(format!("--view must be front, side or back, got {o}")),
                }
            }
            "--upper-body" => cli.upper = true,
            "--frames" => cli.frames = Some(val("--frames")?.parse().map_err(|_| "--frames needs an integer".to_string())?),
            o => return Err(format!("unknown argument {o} (try --help)")),
        }
    }
    Ok(Some(cli))
}

/// Explicit path, else ./assets/biomech-lite/NAME, else the app bundle's
/// Contents/Resources/assets/biomech-lite (exe in Contents/MacOS), else the project root next to this crate.
fn resolve(explicit: &Option<PathBuf>, name: &str) -> PathBuf {
    if let Some(p) = explicit {
        return p.clone();
    }
    let local = Path::new("assets/biomech-lite").join(name);
    if local.exists() {
        return local;
    }
    if let Some(bundled) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|d| d.join("../Resources/assets/biomech-lite").join(name)))
        .filter(|p| p.exists())
    {
        return bundled;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/biomech-lite").join(name)
}

fn build(cli: &Cli) -> Result<App, String> {
    let model = Model::load(resolve(&cli.model, "model.json"))?;
    let visual = VisualModel::load(resolve(&cli.visual, "visual.json"), &model)?;
    let motion = Motion::load(resolve(&cli.motion, "motion.json"), &model)?;
    let mode = if cli.simulation { Mode::Simulation } else { Mode::Recording };
    let mut app = App::new(model, visual, motion, mode, DEFAULT_SIZE)?;
    app.upper = cli.upper;
    app.apply(Action::View(cli.view));
    Ok(app)
}

fn run() -> Result<(), String> {
    let Some(cli) = parse()? else {
        println!("{HELP}");
        return Ok(());
    };
    let mut app = build(&cli)?;
    if let Some(path) = &cli.snapshot {
        app.time = cli.time.min(app.duration());
        app.playing = false;
        app.compose()?;
        app.save_png(path)?;
        println!("wrote {} ({}x{}, viewport render {:.0} ms)", path.display(), app.size().0, app.size().1, app.render_ms);
        return Ok(());
    }
    window_loop(&mut app, cli.frames)
}

fn key_actions(window: &Window) -> Vec<Action> {
    let mut v = Vec::new();
    let table: &[(Key, Action)] = &[
        (Key::Space, Action::TogglePlay),
        (Key::Home, Action::Reset),
        (Key::L, Action::ToggleLoop),
        (Key::Tab, Action::ToggleMode),
        (Key::Left, Action::StepFrame(-1)),
        (Key::Right, Action::StepFrame(1)),
        (Key::Comma, Action::SpeedStep(-1)),
        (Key::Period, Action::SpeedStep(1)),
        (Key::Key1, Action::View(0)),
        (Key::Key2, Action::View(1)),
        (Key::Key3, Action::View(2)),
        (Key::F, Action::Frame),
        (Key::U, Action::ToggleUpper),
        (Key::B, Action::ToggleBones),
        (Key::M, Action::ToggleMuscles),
        (Key::G, Action::ToggleLigaments),
        (Key::J, Action::ToggleJoints),
        (Key::X, Action::ToggleXray),
        (Key::LeftBracket, Action::PrevMuscle),
        (Key::RightBracket, Action::NextMuscle),
        (Key::Equal, Action::ExcitationDelta(0.05)),
        (Key::NumPadPlus, Action::ExcitationDelta(0.05)),
        (Key::Minus, Action::ExcitationDelta(-0.05)),
        (Key::NumPadMinus, Action::ExcitationDelta(-0.05)),
        (Key::P, Action::Screenshot),
    ];
    for (k, a) in table {
        let repeat = matches!(a, Action::StepFrame(_) | Action::ExcitationDelta(_));
        if window.is_key_pressed(*k, if repeat { KeyRepeat::Yes } else { KeyRepeat::No }) {
            v.push(*a);
        }
    }
    v
}

fn window_loop(app: &mut App, frames: Option<u64>) -> Result<(), String> {
    let (w, h) = app.size();
    let opts = WindowOptions { resize: true, ..WindowOptions::default() };
    let mut window = Window::new("MyoSim biomech viewer (Rust)", w, h, opts).map_err(|e| format!("cannot open window: {e}"))?;
    window.set_target_fps(60);
    let mut last = Instant::now();
    let mut n = 0u64;
    while window.is_open() && !window.is_key_down(Key::Escape) {
        let now = Instant::now();
        let dt = now.duration_since(last).as_secs_f64().min(0.1);
        last = now;
        let (ww, wh) = window.get_size();
        app.resize(ww, wh);
        for a in key_actions(&window) {
            app.apply(a);
        }
        if let Some((x, y)) = window.get_mouse_pos(MouseMode::Clamp) {
            let scroll = window.get_scroll_wheel().map(|s| s.1).unwrap_or(0.0);
            app.pointer(x, y, window.get_mouse_down(MouseButton::Left), window.get_mouse_down(MouseButton::Right), scroll);
        }
        app.update(dt);
        let (fw, fh) = app.size();
        let buf = app.compose()?;
        window.update_with_buffer(buf, fw, fh).map_err(|e| format!("window update failed: {e}"))?;
        n += 1;
        if frames.is_some_and(|f| n >= f) {
            break;
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("biomech-viewer: {e}");
        std::process::exit(1);
    }
}
