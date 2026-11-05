//! One reusable recording file. Intermediates live in a temporary directory.
use anyhow::{Context, Result, ensure};
use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tempfile::{NamedTempFile, TempDir};
use zip::{ZipArchive, ZipWriter, write::SimpleFileOptions};

pub const DEFAULT_SAVE: &str = "output/recording.twin";
pub const DEFAULT_REPLAY: &str = "output/replay.mp4";
const ASSETS: [&str; 6] = [
    "source.mp4",
    "capture.json",
    "recording.mp4",
    "tracked.mp4",
    "twin.json",
    "poses.jsonl",
];

pub fn is_bundle(path: &Path) -> Result<bool> {
    let mut signature = [0; 4];
    let mut file =
        File::open(path).with_context(|| format!("Cannot open save {}", path.display()))?;
    Ok(file.read(&mut signature)? == 4 && signature == *b"PK\x03\x04")
}

pub fn save(directory: &Path, destination: &Path) -> Result<()> {
    for name in ["source.mp4", "capture.json", "twin.json"] {
        ensure!(
            directory.join(name).is_file(),
            "Recording is incomplete: missing {name}"
        );
    }
    ensure!(
        !destination.is_dir(),
        "--record takes a .twin file, not a session folder. Use --record by itself for output/recording.twin."
    );
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let mut staging = NamedTempFile::new_in(parent)?;
    {
        let mut archive = ZipWriter::new(staging.as_file_mut());
        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(true);
        archive.start_file("save-format.json", options)?;
        archive.write_all(b"{\"format\":\"lifting-pose\",\"version\":1}")?;
        for name in ASSETS {
            let path = directory.join(name);
            if !path.is_file() {
                continue;
            }
            archive.start_file(name, options)?;
            if name == "capture.json" {
                let mut metadata: serde_json::Value = serde_json::from_reader(File::open(path)?)?;
                metadata["source"] = serde_json::Value::String("source.mp4".into());
                serde_json::to_writer(&mut archive, &metadata)?;
            } else {
                std::io::copy(&mut File::open(path)?, &mut archive)?;
            }
        }
        archive.finish()?;
    }
    staging.as_file().sync_all()?;
    staging
        .persist(destination)
        .with_context(|| format!("Cannot replace recording save {}", destination.display()))?;
    Ok(())
}

/// Replay only needs the twin JSON, so videos are not extracted or duplicated.
pub fn open_for_replay(path: &Path) -> Result<(TempDir, PathBuf)> {
    let mut archive = ZipArchive::new(File::open(path)?)?;
    let format: serde_json::Value = serde_json::from_reader(
        archive
            .by_name("save-format.json")
            .context("Not a lifting-pose save")?,
    )?;
    ensure!(
        format["format"] == "lifting-pose" && format["version"] == 1,
        "Unsupported recording save format"
    );
    let directory = tempfile::tempdir()?;
    let twin = directory.path().join("twin.json");
    std::io::copy(
        &mut archive.by_name("twin.json")?,
        &mut File::create(&twin)?,
    )?;
    Ok((directory, twin))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn workspace() -> TempDir {
        let work = tempfile::tempdir().unwrap();
        std::fs::write(work.path().join("source.mp4"), b"video").unwrap();
        std::fs::write(
            work.path().join("capture.json"),
            br#"{"source":"/temporary/source.mp4","fps":30,"timestamps_seconds":[0]}"#,
        )
        .unwrap();
        std::fs::write(work.path().join("twin.json"), b"first").unwrap();
        work
    }
    #[test]
    fn one_save_is_replaced_and_opens_without_extracting_videos() {
        let output = tempfile::tempdir().unwrap();
        let target = output.path().join("recording.twin");
        let work = workspace();
        save(work.path(), &target).unwrap();
        assert!(is_bundle(&target).unwrap());
        std::fs::write(work.path().join("twin.json"), b"second").unwrap();
        save(work.path(), &target).unwrap();
        assert_eq!(std::fs::read_dir(output.path()).unwrap().count(), 1);
        let (temporary, twin) = open_for_replay(&target).unwrap();
        assert_eq!(std::fs::read(twin).unwrap(), b"second");
        assert_eq!(std::fs::read_dir(temporary.path()).unwrap().count(), 1);
        let mut archive = ZipArchive::new(File::open(target).unwrap()).unwrap();
        let metadata: serde_json::Value =
            serde_json::from_reader(archive.by_name("capture.json").unwrap()).unwrap();
        assert_eq!(metadata["source"], "source.mp4");
    }
    #[test]
    fn failed_replacement_preserves_the_previous_save() {
        let output = tempfile::tempdir().unwrap();
        let target = output.path().join("recording.twin");
        let work = workspace();
        save(work.path(), &target).unwrap();
        let before = std::fs::read(&target).unwrap();
        std::fs::remove_file(work.path().join("source.mp4")).unwrap();
        assert!(save(work.path(), &target).is_err());
        assert_eq!(std::fs::read(target).unwrap(), before);
    }
}
