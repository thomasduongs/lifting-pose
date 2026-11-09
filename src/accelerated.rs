//! Native macOS inference with identical input/output tensors, per-model warm-up
//! selection and the existing OpenCV backend as a portable fallback.
use anyhow::{Context, Result};
use opencv::{
    core::{Mat, ToInputArray, Vector},
    dnn,
    prelude::*,
};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};
static ENABLED: AtomicBool = AtomicBool::new(true);
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}
#[derive(Default)]
pub struct Inference {
    #[cfg(target_os = "macos")]
    session: Option<ort::session::Session>,
    #[cfg(target_os = "macos")]
    select_pending: bool,
    model_path: Option<PathBuf>,
}
fn opencv_forward(
    net: &mut dnn::Net,
    blob: &impl ToInputArray,
    names: &Vector<String>,
) -> Result<Vector<Mat>> {
    net.set_input_def(blob)?;
    let mut outputs = Vector::<Mat>::new();
    net.forward(&mut outputs, names)?;
    Ok(outputs)
}
#[cfg(target_os = "macos")]
fn native_forward(
    session: &mut ort::session::Session,
    blob: &impl MatTraitConst,
    names: &Vector<String>,
) -> Result<Vector<Mat>> {
    let shape: Vec<i64> = blob.mat_size().iter().map(|&n| i64::from(n)).collect();
    let tensor = ort::value::Tensor::from_array((
        shape,
        blob.data_typed::<f32>()?.to_vec().into_boxed_slice(),
    ))?;
    let outputs = session.run(ort::inputs![tensor])?;
    let mut result = Vector::<Mat>::new();
    for name in names.iter() {
        let output = outputs
            .get(&name)
            .with_context(|| format!("Missing native output {name}"))?;
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        anyhow::ensure!(
            data.iter().all(|v| v.is_finite()),
            "Non-finite native output {name}"
        );
        let shape: Vec<i32> = shape
            .iter()
            .map(|&n| i32::try_from(n))
            .collect::<std::result::Result<_, _>>()?;
        let values = Mat::from_slice(data)?;
        result.push(values.reshape_nd(1, &shape)?.try_clone()?);
    }
    Ok(result)
}
impl Inference {
    pub fn enable(&mut self, path: &Path) -> Result<()> {
        self.enable_format(path, false)
    }
    pub fn enable_crop(&mut self, path: &Path) -> Result<()> {
        self.enable_format(path, true)
    }
    fn enable_format(&mut self, path: &Path, legacy: bool) -> Result<()> {
        if !ENABLED.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.model_path = Some(path.to_path_buf());
        #[cfg(target_os = "macos")]
        {
            use ort::{
                execution_providers::coreml::{
                    CoreMLComputeUnits, CoreMLExecutionProvider, CoreMLModelFormat,
                },
                session::Session,
            };
            use std::hash::{Hash, Hasher};
            ort::init()
                .with_name("lifting-pose")
                .with_telemetry(false)
                .commit()?;
            ort::environment::get_environment()?.set_log_level(ort::logging::LogLevel::Error);
            let mut key = std::collections::hash_map::DefaultHasher::new();
            (if legacy {
                "ort122-NeuralNetwork-All-fp32"
            } else {
                "ort122-MLProgram-All-fp32"
            })
            .hash(&mut key);
            std::fs::read(path)?.hash(&mut key);
            let id = format!("{:016x}", key.finish());
            let temporary = std::env::temp_dir().join("lifting-pose-native").join(&id);
            let persistent = std::env::var_os("HOME")
                .map(PathBuf::from)
                .map(|root| root.join("Library/Caches/lifting-pose/coreml").join(&id));
            let mut cache = persistent.unwrap_or_else(|| temporary.clone());
            if let Some(parent) = cache.parent() {
                std::fs::create_dir_all(parent)?;
            }
            if temporary.exists()
                && !cache.exists()
                && temporary != cache
                && std::fs::rename(&temporary, &cache).is_err()
            {
                cache = temporary;
            }
            std::fs::create_dir_all(&cache)?;
            eprintln!(
                "Preparing native model {} (first compilation may take several minutes)...",
                path.display()
            );
            let ep = CoreMLExecutionProvider::default()
                .with_model_cache_dir(cache.to_string_lossy())
                .with_model_format(if legacy {
                    CoreMLModelFormat::NeuralNetwork
                } else {
                    CoreMLModelFormat::MLProgram
                })
                .with_compute_units(CoreMLComputeUnits::All)
                .with_low_precision_accumulation_on_gpu(false)
                .build()
                .error_on_failure();
            self.session = Some(
                Session::builder()?
                    .with_intra_threads(1)?
                    .with_inter_threads(1)?
                    .with_execution_providers([ep])?
                    .commit_from_file(path)?,
            );
            self.select_pending = true;
        }
        #[cfg(not(target_os = "macos"))]
        let _ = legacy;
        Ok(())
    }
    #[cfg(test)]
    pub fn force_native(&mut self) {
        #[cfg(target_os = "macos")]
        {
            self.select_pending = false;
        }
    }
    #[cfg(test)]
    pub fn active(&self) -> bool {
        #[cfg(target_os = "macos")]
        {
            self.session.is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
    pub fn forward(
        &mut self,
        net: &mut dnn::Net,
        blob: &(impl MatTraitConst + ToInputArray),
        names: &Vector<String>,
    ) -> Result<Vector<Mat>> {
        #[cfg(target_os = "macos")]
        if let Some(session) = &mut self.session {
            let selected = (|| -> Result<(bool, Vector<Mat>)> {
                if self.select_pending {
                    self.select_pending = false;
                    // Discard first-call graph allocation/specialization costs.
                    native_forward(session, blob, names)?;
                    opencv_forward(net, blob, names)?;
                    let mut native_ms = [0.0f64; 3];
                    let mut cpu_ms = [0.0f64; 3];
                    let mut native_output = Vector::new();
                    let mut cpu_output = Vector::new();
                    for i in 0..3 {
                        let started = std::time::Instant::now();
                        native_output = native_forward(session, blob, names)?;
                        native_ms[i] = started.elapsed().as_secs_f64() * 1000.0;
                        let started = std::time::Instant::now();
                        cpu_output = opencv_forward(net, blob, names)?;
                        cpu_ms[i] = started.elapsed().as_secs_f64() * 1000.0;
                    }
                    native_ms.sort_by(f64::total_cmp);
                    cpu_ms.sort_by(f64::total_cmp);
                    let use_native = native_ms[1] < cpu_ms[1] * 0.9;
                    eprintln!(
                        "{}: {} selected (native {:.1} ms, OpenCV {:.1} ms)",
                        self.model_path
                            .as_ref()
                            .map(|p| p.display().to_string())
                            .unwrap_or_default(),
                        if use_native { "native" } else { "OpenCV" },
                        native_ms[1],
                        cpu_ms[1]
                    );
                    Ok((
                        use_native,
                        if use_native {
                            native_output
                        } else {
                            cpu_output
                        },
                    ))
                } else {
                    Ok((true, native_forward(session, blob, names)?))
                }
            })();
            match selected {
                Ok((true, outputs)) => return Ok(outputs),
                Ok((false, outputs)) => {
                    self.session = None;
                    return Ok(outputs);
                }
                Err(error) => {
                    eprintln!("Native inference failed: {error:#}; using OpenCV");
                    self.session = None;
                }
            }
        }
        opencv_forward(net, blob, names)
    }
}
