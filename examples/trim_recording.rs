//! Trim a previously captured session without rerunning neural inference.
#[allow(dead_code)]
#[path = "../src/recording.rs"]
mod recording;
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let source = std::path::PathBuf::from(
        args.next()
            .ok_or_else(|| anyhow::anyhow!("Usage: trim_recording <source.mp4> [output.mp4]"))?,
    );
    let output = args
        .next()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| source.parent().unwrap().join("recording.mp4"));
    let capture = recording::existing_timing(&source)?
        .ok_or_else(|| anyhow::anyhow!("source.mp4 needs adjacent capture.json"))?;
    recording::export_trimmed(&capture, &output, 3.0, 3.0)
}
