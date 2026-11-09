//! Local evaluation of the official VideoPose3D COCO checkpoint exported to ONNX.
//! Kept outside the production path until recording-specific validation passes.
use anyhow::{Result, ensure};
use opencv::{
    core::{Mat, Vector},
    dnn,
    prelude::*,
};
use std::path::Path;
const WINDOW: usize = 243;
struct VideoPose3d {
    net: dnn::Net,
    names: Vector<String>,
}
impl VideoPose3d {
    fn new(path: &Path) -> Result<Self> {
        let mut net = opencv::opencv_branch_5! {{dnn::read_net_from_onnx_def(path)}else{dnn::read_net_from_onnx(path)}}?;
        net.set_preferable_backend(dnn::DNN_BACKEND_OPENCV)?;
        opencv::not_opencv_branch_5! {net.set_preferable_target(dnn::DNN_TARGET_CPU)?;}
        let names = net.get_unconnected_out_layers_names()?;
        Ok(Self { net, names })
    }
    fn infer(&mut self, input: &[f32]) -> Result<[[f32; 3]; 17]> {
        ensure!(
            input.len() == WINDOW * 17 * 2 && input.iter().all(|v| v.is_finite()),
            "Invalid VideoPose3D input"
        );
        let data = Mat::from_slice(input)?;
        let blob = data.reshape_nd(1, &[1, WINDOW as i32, 17, 2])?;
        self.net.set_input_def(&blob)?;
        let mut outputs = Vector::<Mat>::new();
        self.net.forward(&mut outputs, &self.names)?;
        ensure!(outputs.len() == 1, "Unexpected output count");
        let output = outputs.get(0)?;
        ensure!(
            output.mat_size().as_ref() == [1, 1, 17, 3],
            "Wrong VideoPose3D output shape"
        );
        let values = output.data_typed::<f32>()?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "Non-finite VideoPose3D prediction"
        );
        Ok(std::array::from_fn(|i| {
            [
                values[i * 3] - values[0],
                values[i * 3 + 1] - values[1],
                values[i * 3 + 2] - values[2],
            ]
        }))
    }
}
fn sample_window(times: &[f64], frames: &[Vec<f32>], time: f64, hz: f64) -> Result<Vec<f32>> {
    ensure!(
        !times.is_empty()
            && times.len() == frames.len()
            && frames.iter().all(|f| f.len() == 34)
            && hz > 0.0,
        "Invalid temporal sequence"
    );
    let mut input = Vec::with_capacity(WINDOW * 34);
    for offset in 0..WINDOW {
        let at = time + (offset as f64 - (WINDOW / 2) as f64) / hz;
        let end = times.partition_point(|t| *t <= at);
        let a = end.saturating_sub(1);
        let b = end.min(times.len() - 1);
        let u = if a == b {
            0.0
        } else {
            ((at - times[a]) / (times[b] - times[a])).clamp(0.0, 1.0)
        } as f32;
        input.extend((0..34).map(|j| frames[a][j] * (1.0 - u) + frames[b][j] * u));
    }
    Ok(input)
}
#[test]
fn temporal_window_preserves_target_time_and_uses_real_intervals() -> Result<()> {
    let times = [0.0, 0.2, 0.5];
    let frames = vec![vec![0.0; 34], vec![2.0; 34], vec![5.0; 34]];
    let window = sample_window(&times, &frames, 0.2, 50.0)?;
    assert_eq!(&window[121 * 34..122 * 34], frames[1].as_slice());
    assert!((window[120 * 34] - 1.8).abs() < 1e-6);
    assert!((window[122 * 34] - 2.2).abs() < 1e-6);
    assert_eq!(window[0], 0.0);
    assert_eq!(*window.last().unwrap(), 5.0);
    Ok(())
}
#[test]
#[ignore = "requires prepared local recording and VideoPose3D ONNX export"]
fn evaluate_recorded_videopose3d() -> Result<()> {
    use crate::{
        details::{HandPose, HeadPose},
        motion3d::{Joint3d, NAMES, Pose3d},
        pose::{Bbox, Keypoint},
        twin::{RecordingTiming, TwinRecorder},
    };
    use opencv::core::Size;
    use std::io::Write;
    let directory = std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?);
    let prepared: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("videopose3d/input2d.json"))?)?;
    let times: Vec<f64> = serde_json::from_value(prepared["times"].clone())?;
    let frames: Vec<Vec<f32>> = serde_json::from_value(prepared["frames"].clone())?;
    let native: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("verifiedarms-twin.json"))?)?;
    let body_width = native["subjects"][0]["person"]["shoulder_width"]
        .as_f64()
        .unwrap() as f32;
    let mut model = VideoPose3d::new(&directory.join("videopose3d/videopose3d.onnx"))?;
    let pixel_scores = std::env::var_os("LIFTING_EVAL_VP_PIXEL_SCORES").is_some();
    let tag = if pixel_scores { "vp3dpixel" } else { "vp3d" };
    let mut recorder = TwinRecorder::default();
    recorder.set_upright_torso(true);
    let mut writer = std::io::BufWriter::new(std::fs::File::create(
        directory.join(format!("{tag}.jsonl")),
    )?);
    for line in std::fs::read_to_string(directory.join("verifiedarms.jsonl"))?.lines() {
        let mut row: serde_json::Value = serde_json::from_str(line)?;
        let frame = row["frame"].as_u64().unwrap();
        let size = Size::new(
            row["width"].as_i64().unwrap() as i32,
            row["height"].as_i64().unwrap() as i32,
        );
        let input = sample_window(
            &times,
            &frames,
            times[frame as usize],
            prepared["sample_hz"].as_f64().unwrap(),
        )?;
        let xyz = model.infer(&input)?;
        let center = std::array::from_fn::<_, 3, _>(|i| (xyz[11][i] + xyz[14][i]) / 2.0);
        let width = ((xyz[11][0] - xyz[14][0]).powi(2)
            + (xyz[11][1] - xyz[14][1]).powi(2)
            + (xyz[11][2] - xyz[14][2]).powi(2))
        .sqrt();
        let scale = body_width / width.max(1e-6);
        for person in row["people"].as_array_mut().unwrap() {
            let raw = &person["pose3d"]["joints"];
            let points: Vec<_> = person["keypoints"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, k)| Keypoint {
                    name: crate::landmarks::NAMES[i],
                    x: k["x"].as_f64().unwrap() as f32,
                    y: k["y"].as_f64().unwrap() as f32,
                    score: k["score"].as_f64().unwrap() as f32,
                })
                .collect();
            let scores = crate::mediapipe::skeleton_pixels(&points);
            let anchor = [
                raw[8]["x"].as_f64().unwrap() as f32,
                raw[8]["y"].as_f64().unwrap() as f32,
                raw[8]["z"].as_f64().unwrap() as f32,
            ];
            let joints: Vec<_> = (0..17)
                .map(|i| {
                    let coord = if [8, 9, 10].contains(&i) {
                        [
                            raw[i]["x"].as_f64().unwrap() as f32,
                            raw[i]["y"].as_f64().unwrap() as f32,
                            raw[i]["z"].as_f64().unwrap() as f32,
                        ]
                    } else {
                        std::array::from_fn(|axis| {
                            (xyz[i][axis] - center[axis]) * scale + anchor[axis]
                        })
                    };
                    Joint3d {
                        name: NAMES[i],
                        x: coord[0],
                        y: coord[1],
                        z: coord[2],
                        input_score: if pixel_scores {
                            scores[i][2]
                        } else {
                            raw[i]["input_score"].as_f64().unwrap() as f32
                        },
                    }
                })
                .collect();
            let pose = Pose3d {
                ear_head: true,
                model: "VideoPose3D-NativeScale-Diagnostic",
                coordinate_space: "root_relative_camera",
                units: "estimated_meters",
                history_frames: WINDOW,
                angles: crate::motion3d::angles3d(&joints, 0.3),
                joints,
            };
            let b = &person["bbox"];
            let bbox = Bbox {
                x: b["x"].as_f64().unwrap() as f32,
                y: b["y"].as_f64().unwrap() as f32,
                width: b["width"].as_f64().unwrap() as f32,
                height: b["height"].as_f64().unwrap() as f32,
            };
            let hands: Vec<HandPose> = serde_json::from_value(person["hands"].clone())?;
            let head: Option<HeadPose> = serde_json::from_value(person["head"].clone())?;
            recorder.set_image_torso_scale(
                person["image_torso_scale"].as_f64(),
                person["image_torso_span"].as_u64().unwrap_or(1) as usize,
            );
            recorder.observe(
                frame,
                times[frame as usize],
                &pose,
                &points,
                &hands,
                head,
                bbox,
                size,
                0.3,
                None,
            );
            person["pose3d"] = serde_json::to_value(pose)?;
        }
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    recorder.set_timing(RecordingTiming {
        fps: native["recording"]["fps"].as_f64().unwrap(),
        frame_count: times.len() as u64,
        timestamps_seconds: times,
        source_video: Some("source.mp4".into()),
        trim_start_seconds: 3.0,
        trim_end_seconds: 3.0,
        upper_body: true,
        assume_upright_torso: true,
    });
    recorder.save(&directory.join(format!("{tag}-twin.json")))?;
    Ok(())
}
#[test]
#[ignore = "requires local VideoPose3D export and reference tensors"]
fn videopose3d_matches_export_reference() -> Result<()> {
    let directory =
        std::path::PathBuf::from(std::env::var("LIFTING_EVAL_DIR")?).join("videopose3d");
    let read = |file: &str| -> Result<Vec<f32>> {
        let bytes = std::fs::read(directory.join(file))?;
        Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect())
    };
    let input = read("videopose3d.input.f32")?;
    let expected = read("videopose3d.expected.f32")?;
    ensure!(expected.len() == 51, "Invalid reference shape");
    let mut model = VideoPose3d::new(&directory.join("videopose3d.onnx"))?;
    for (i, joint) in model.infer(&input)?.iter().enumerate() {
        for (axis, value) in joint.iter().enumerate() {
            assert!(
                (value - (expected[i * 3 + axis] - expected[axis])).abs() < 1e-4,
                "Joint {i} axis {axis}"
            );
        }
    }
    Ok(())
}
