use std::collections::HashMap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{JoinHandle, ThreadId};

use astro_frames::port::{is_flat_frame, load_for_analysis};
use astro_gpu::port::{Gpu, PreferredGpu, preferred_gpu};
use astro_measurements::port::{
    Candidate, Measurement, MeasurementConfig, measure_cancellable, snr,
};
use astro_stars::port::{DetectionConfig, StarDetector};
use rayon::prelude::*;

use crate::cache::{AnalysisCache, save as save_cache};
use crate::catalog::{AnalysisResult, AnalysisStatus, normalize_filter};

pub struct Batch {
    pub results: Vec<AnalysisResult>,
    pub flat_count: usize,
    pub flat_files: Vec<PathBuf>,
    pub gpu: Option<String>,
    pub gpu_count: usize,
    pub cache_hits: usize,
    pub cache_path: Option<PathBuf>,
    pub cache_warning: Option<String>,
    pub gpu_error: Option<String>,
}

static GPU: OnceLock<Result<Gpu, String>> = OnceLock::new();

pub struct AnalysisJob {
    cancelled: Arc<AtomicBool>,
    worker: Option<JoinHandle<Result<Batch, String>>>,
}

impl AnalysisJob {
    pub fn spawn(
        project: PathBuf,
        files: Vec<PathBuf>,
        workers: usize,
        use_gpu: bool,
        progress: impl Fn(usize, usize) + Send + Sync + 'static,
    ) -> Result<Self, String> {
        let cancelled = Arc::new(AtomicBool::new(false));
        let task_cancelled = cancelled.clone();
        let (started_send, started_receive) = std::sync::mpsc::channel();
        let worker = std::thread::Builder::new()
            .name("fits-analysis-manager".into())
            .spawn(move || {
                if started_send.send(()).is_err() {
                    return Err("analysis manager stopped".into());
                }
                let result = analyze_files(
                    &project,
                    &project,
                    &files,
                    workers,
                    use_gpu,
                    &task_cancelled,
                    progress,
                );
                if task_cancelled.load(Ordering::Relaxed) {
                    return Err("analysis cancelled".into());
                }
                result
            })
            .map_err(|error| error.to_string())?;
        if started_receive.recv().is_err() {
            return Err("analysis manager failed to start".into());
        }
        Ok(Self {
            cancelled,
            worker: Some(worker),
        })
    }

    pub fn cancel_token(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }

    pub fn join(mut self) -> Result<Batch, String> {
        let Some(worker) = self.worker.take() else {
            return Err("analysis manager stopped".into());
        };
        worker
            .join()
            .map_err(|_| "analysis worker panicked".to_owned())?
    }
}

impl Drop for AnalysisJob {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
        let Some(worker) = self.worker.take() else {
            return;
        };
        let _ = std::thread::Builder::new()
            .name("fits-analysis-reaper".into())
            .spawn(move || {
                let _ = worker.join();
            });
    }
}

fn looks_like_flat_name(path: &Path) -> bool {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| {
            stem.chars()
                .filter(|character| character.is_ascii_alphanumeric())
                .flat_map(char::to_uppercase)
                .collect::<String>()
                .contains("FLAT")
        })
}

pub fn analyze_files(
    project: &Path,
    source: &Path,
    files: &[PathBuf],
    workers: usize,
    use_gpu: bool,
    cancelled: &AtomicBool,
    progress: impl Fn(usize, usize) + Sync,
) -> Result<Batch, String> {
    let mut cache = AnalysisCache::load(project, if use_gpu { "gpu" } else { "cpu" });
    let mut cache_warning = cache.take_warning();
    let mut entries = Vec::with_capacity(files.len());
    let mut flat_files = Vec::new();
    for file in files {
        let probe = cache.probe(source, file);
        let fingerprint = probe.fingerprint;
        if probe.result.is_some() && !looks_like_flat_name(file) {
            entries.push((file.clone(), fingerprint, false, probe.result));
        } else if is_flat_frame(file) {
            entries.push((file.clone(), fingerprint, true, None));
            flat_files.push(file.clone());
        } else {
            entries.push((file.clone(), fingerprint, false, None));
        }
    }
    let flat_count = flat_files.len();
    let cached_results: Vec<_> = entries
        .iter()
        .enumerate()
        .filter_map(|(index, (_, _, _, result))| {
            result.as_ref().map(|result| (index, result.clone()))
        })
        .collect();
    let cache_hits = cached_results.len();
    let cached_indices: HashSet<_> = cached_results.iter().map(|(index, _)| *index).collect();
    let light_count = entries.iter().filter(|(_, _, flat, _)| !flat).count();
    let missing: Vec<_> = entries
        .iter()
        .enumerate()
        .filter_map(|(index, (_, _, flat, result))| {
            (!flat && result.is_none() && !cached_indices.contains(&index)).then_some(index)
        })
        .collect();
    let gpu = (!missing.is_empty() && use_gpu)
        .then(|| GPU.get_or_init(Gpu::new).as_ref().ok())
        .flatten();
    if missing.is_empty() {
        progress(cache_hits, light_count);
    }
    let workers = effective_workers(workers, missing.len());
    let missing_files: Vec<_> = missing
        .iter()
        .map(|&index| entries[index].0.clone())
        .collect();
    let computed =
        analyze_files_with_jobs(&missing_files, workers, gpu, cancelled, |current, _| {
            progress(cache_hits + current, light_count)
        })?;
    let mut results = vec![None; entries.len()];
    for (index, result) in cached_results {
        results[index] = Some(result);
    }
    let gpu_count = computed.gpu_count;
    for (index, result) in missing.into_iter().zip(computed.results) {
        results[index] = Some(result);
    }
    let cache_path = if !cancelled.load(Ordering::Relaxed) {
        let cache_entries = entries
            .iter()
            .zip(&results)
            .filter_map(|((_, fingerprint, flat, _), result)| {
                if *flat {
                    return None;
                }
                let fingerprint = fingerprint.clone()?;
                let result = result.clone()?;
                Some((fingerprint, result))
            })
            .collect();
        match save_cache(project, cache.engine(), cache_entries) {
            Ok(path) => Some(path),
            Err(error) => {
                cache_warning = Some(format!("写入分析缓存失败: {error}"));
                None
            }
        }
    } else {
        None
    };
    let results: Vec<_> = results.into_iter().flatten().collect();
    Ok(Batch {
        results,
        flat_count,
        flat_files,
        gpu: computed.gpu,
        gpu_count,
        cache_hits,
        cache_path,
        cache_warning,
        gpu_error: computed.gpu_error,
    })
}

fn analyze_files_with_jobs(
    files: &[PathBuf],
    workers: usize,
    gpu: Option<&Gpu>,
    cancelled: &AtomicBool,
    progress: impl Fn(usize, usize) + Sync,
) -> Result<Batch, String> {
    if files.is_empty() {
        return Ok(Batch {
            results: Vec::new(),
            flat_count: 0,
            flat_files: Vec::new(),
            gpu: gpu.map(|device| device.name().to_owned()),
            gpu_count: 0,
            cache_hits: 0,
            cache_path: None,
            cache_warning: None,
            gpu_error: None,
        });
    }
    let mut shared_detector = StarDetector::new();
    if let Some(baseline) = files.get(files.len() / 2)
        && let Ok(frame) = load_for_analysis(baseline)
    {
        shared_detector.prepare_background(
            &frame.pixels,
            frame.width,
            frame.height,
            &DetectionConfig::default(),
        );
    }
    let shared_detector = Arc::new(shared_detector);
    let detectors = Mutex::new(HashMap::<ThreadId, StarDetector>::new());
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("fits-analysis-{index}"))
        .build()
        .map_err(|error| error.to_string())?;
    let completed = Mutex::new(0);
    let gpu_count = AtomicUsize::new(0);
    let gpu_failures = Mutex::new(Vec::new());
    let slots = pool.install(|| {
        files
            .par_iter()
            .map(|file| {
                let (result, gpu_used, gpu_error) =
                    analyze_file(file, gpu, &detectors, &shared_detector, || {
                        cancelled.load(Ordering::Relaxed)
                    })?;
                if gpu_used {
                    gpu_count.fetch_add(1, Ordering::Relaxed);
                }
                if let Some(error) = gpu_error
                    && let Ok(mut failures) = gpu_failures.lock()
                {
                    failures.push(error);
                }
                if !cancelled.load(Ordering::Relaxed)
                    && let Ok(mut count) = completed.lock()
                {
                    *count += 1;
                    progress(*count, files.len());
                }
                Some(result)
            })
            .collect::<Vec<_>>()
    });
    Ok(Batch {
        results: slots.into_iter().flatten().collect(),
        flat_count: 0,
        flat_files: Vec::new(),
        gpu: gpu.map(|device| device.name().to_owned()),
        gpu_count: gpu_count.into_inner(),
        cache_hits: 0,
        cache_path: None,
        cache_warning: None,
        gpu_error: {
            let failures = gpu_failures.into_inner().unwrap_or_default();
            failures.into_iter().next()
        },
    })
}

fn effective_workers(requested: usize, file_count: usize) -> usize {
    requested.max(1).min(file_count.max(1))
}

fn analyze_file(
    file: &Path,
    gpu: Option<&Gpu>,
    detectors: &Mutex<HashMap<ThreadId, StarDetector>>,
    shared_detector: &StarDetector,
    cancelled: impl Fn() -> bool + Sync,
) -> Option<(AnalysisResult, bool, Option<String>)> {
    if cancelled() {
        return None;
    }
    let frame = match load_for_analysis(file) {
        Ok(frame) => frame,
        Err(error) => {
            return Some((
                AnalysisResult::failed(file, AnalysisStatus::ReadFailed, error),
                false,
                None,
            ));
        }
    };
    if cancelled() {
        return None;
    }
    let filter = match frame.metadata.analysis_filter() {
        Ok(filter) => filter,
        Err(error) => {
            return Some((
                AnalysisResult::failed(file, AnalysisStatus::InvalidFilter, error),
                false,
                None,
            ));
        }
    };
    let stars = match detectors.lock() {
        Ok(mut detectors) => {
            let detector = detectors
                .entry(std::thread::current().id())
                .or_insert_with(|| (*shared_detector).clone());
            detector.detect_cancellable(
                &frame.pixels,
                frame.width,
                frame.height,
                &DetectionConfig::default(),
                &cancelled,
            )
        }
        Err(_) => {
            return Some((
                AnalysisResult::failed(file, AnalysisStatus::NoStars, "星点检测器状态不可用"),
                false,
                None,
            ));
        }
    };
    if cancelled() {
        return None;
    }

    let candidates: Vec<_> = stars
        .iter()
        .map(|star| Candidate {
            x: star.x,
            y: star.y,
            peak: star.peak,
        })
        .collect();
    let config = MeasurementConfig::default();
    let mut gpu_error = None;
    let from_gpu = gpu.and_then(|device| {
        match device.measure_cancellable(
            &frame.pixels,
            frame.width,
            frame.height,
            &candidates,
            &config,
            &cancelled,
        ) {
            Ok(measurement) => measurement,
            Err(error) => {
                gpu_error = Some(error);
                None
            }
        }
    });
    let used_gpu = from_gpu.is_some();
    let measurement = from_gpu.or_else(|| {
        measure_cancellable(
            &frame.pixels,
            frame.width,
            frame.height,
            &candidates,
            &config,
            &cancelled,
        )
    });
    let Measurement {
        fwhm,
        eccentricity,
        used_stars,
    } = measurement.unwrap_or_default();
    let status = if stars.is_empty() || used_stars == 0 {
        AnalysisStatus::NoStars
    } else {
        AnalysisStatus::Success
    };
    let snr = snr(&frame.pixels);
    let mean_brightness = frame.pixels.iter().sum::<f64>() / frame.pixels.len() as f64;
    if !snr.is_finite() || !mean_brightness.is_finite() {
        return Some((
            AnalysisResult::failed(
                file,
                AnalysisStatus::InvalidPixels,
                "SNR 或平均亮度非有限值",
            ),
            used_gpu,
            gpu_error,
        ));
    }
    let error = (status == AnalysisStatus::NoStars).then(|| "未检出可测量的星点".to_owned());

    (!cancelled()).then(|| {
        (
            AnalysisResult {
                filename: file.to_path_buf(),
                status,
                error,
                filter_type: normalize_filter(filter),
                eccentricity,
                fwhm,
                snr,
                star_count: stars.len(),
                mean_brightness,
            },
            used_gpu,
            gpu_error,
        )
    })
}

pub fn available_workers() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

pub fn auto_device() -> PreferredGpu {
    preferred_gpu()
}

pub fn default_workers() -> usize {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let available_gib = system.available_memory() as f64 / (1024.0_f64).powi(3);
    let logical = available_workers();
    let physical = sysinfo::System::physical_core_count().unwrap_or(logical);
    recommended_workers(logical, physical, available_gib)
}

fn recommended_workers(logical: usize, physical: usize, available_gib: f64) -> usize {
    let logical = logical.max(1);
    let physical = physical.clamp(1, logical);
    let by_cpu = physical.saturating_sub(1).max(1);
    let by_memory = (available_gib * 0.8).floor().max(1.0) as usize;
    by_cpu.min(by_memory).min(logical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn synthetic_fits(project: &Path, name: &str) -> PathBuf {
        let path = project.join(name);
        let side = 160_u32;
        let mut bytes = Vec::new();
        for (keyword, value) in [
            ("SIMPLE", "                    T"),
            ("BITPIX", "                   16"),
            ("NAXIS", "                    2"),
            ("NAXIS1", &format!("{side:>20}")),
            ("NAXIS2", &format!("{side:>20}")),
            ("FILTER", "'Ha'"),
        ] {
            let card = format!("{keyword:<8}= {value}{}", " ".repeat(80));
            bytes.extend_from_slice(&card.as_bytes()[..80]);
        }
        let end = format!("{:<80}", "END");
        bytes.extend_from_slice(end.as_bytes());
        while !bytes.len().is_multiple_of(2880) {
            bytes.push(b' ');
        }
        for y in 0..side {
            for x in 0..side {
                let dx = x as f64 - 80.3;
                let dy = y as f64 - 79.6;
                let level = 1000.0
                    + f64::from((x * 13 + y * 17) % 5)
                    + 1800.0 * (-(dx * dx + dy * dy) / 3.645).exp();
                bytes.extend_from_slice(&(level as i16).to_be_bytes());
            }
        }
        while !bytes.len().is_multiple_of(2880) {
            bytes.push(0);
        }
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn preserves_input_order_and_reports_progress_once_per_file() {
        let files = [PathBuf::from("absent_b.fit"), PathBuf::from("absent_a.fit")];
        let progress = Mutex::new(Vec::new());
        let root = tempfile::tempdir().unwrap();
        let batch = analyze_files(
            root.path(),
            root.path(),
            &files,
            2,
            false,
            &AtomicBool::new(false),
            |current, total| {
                progress.lock().unwrap().push((current, total));
            },
        )
        .unwrap();
        let ordered = batch
            .results
            .into_iter()
            .map(|result| result.filename)
            .collect::<Vec<_>>();
        assert_eq!(ordered, files);
        assert_eq!(progress.into_inner().unwrap(), [(1, 2), (2, 2)]);
    }

    #[test]
    fn cancellation_stops_progress_and_skips_remaining_files() {
        let files = vec![PathBuf::from("absent.fit"); 8];
        let cancelled = AtomicBool::new(false);
        let root = tempfile::tempdir().unwrap();
        let batch = analyze_files(
            root.path(),
            root.path(),
            &files,
            1,
            false,
            &cancelled,
            |current, _| {
                assert_eq!(current, 1);
                cancelled.store(true, Ordering::Relaxed);
            },
        )
        .unwrap();
        assert_eq!(batch.results.len(), 1);
    }

    #[test]
    fn worker_recommendation_reserves_one_core_for_the_interface() {
        assert_eq!(recommended_workers(4, 4, 16.0), 3);
        assert_eq!(recommended_workers(8, 4, 16.0), 3);
        assert_eq!(recommended_workers(1, 1, 16.0), 1);
        assert_eq!(recommended_workers(4, 4, 2.5), 2);
    }

    #[test]
    fn gpu_requests_keep_cpu_worker_count() {
        assert_eq!(effective_workers(6, 12), 6);
        assert_eq!(effective_workers(6, 2), 2);
        assert_eq!(effective_workers(0, 2), 1);
    }

    #[test]
    fn empty_work_does_not_panic() {
        let project = tempfile::tempdir().unwrap();
        let batch = analyze_files(
            project.path(),
            project.path(),
            &[],
            4,
            false,
            &AtomicBool::new(false),
            |_, _| {},
        )
        .unwrap();
        assert!(batch.results.is_empty());
    }

    #[test]
    fn repeated_analysis_reuses_cache_and_changed_files_are_recomputed() {
        let project = tempfile::tempdir().unwrap();
        for name in ["b.fits", "a.fits"] {
            synthetic_fits(project.path(), name);
        }
        let files = [project.path().join("b.fits"), project.path().join("a.fits")];
        let cancelled = AtomicBool::new(false);
        let first = analyze_files(
            project.path(),
            project.path(),
            &files,
            2,
            false,
            &cancelled,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(first.cache_hits, 0);
        assert_eq!(
            first.cache_path,
            Some(project.path().join(crate::cache::CACHE_FILE_NAME))
        );
        assert_eq!(
            first
                .results
                .iter()
                .map(|result| &result.filename)
                .collect::<Vec<_>>(),
            files.iter().collect::<Vec<_>>()
        );

        let second = analyze_files(
            project.path(),
            project.path(),
            &files,
            2,
            false,
            &cancelled,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(second.cache_hits, 2);

        fs::write(project.path().join("a.fits"), b"changed").unwrap();
        let third = analyze_files(
            project.path(),
            project.path(),
            &files,
            2,
            false,
            &cancelled,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(third.cache_hits, 1);
    }

    #[test]
    fn unreadable_images_are_not_persisted_in_the_cache() {
        let project = tempfile::tempdir().unwrap();
        let file = project.path().join("broken.fits");
        fs::write(&file, b"not a FITS file").unwrap();
        let cancelled = AtomicBool::new(false);
        analyze_files(
            project.path(),
            project.path(),
            &[file],
            1,
            false,
            &cancelled,
            |_, _| {},
        )
        .unwrap();
        let cache: serde_json::Value = serde_json::from_slice(
            &fs::read(project.path().join(crate::cache::CACHE_FILE_NAME)).unwrap(),
        )
        .unwrap();
        assert_eq!(cache["engines"]["cpu"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn cpu_cache_is_not_reused_for_gpu_measurements() {
        let project = tempfile::tempdir().unwrap();
        let file = synthetic_fits(project.path(), "frame.fits");
        let files = [file];
        let cancelled = AtomicBool::new(false);
        analyze_files(
            project.path(),
            project.path(),
            &files,
            1,
            false,
            &cancelled,
            |_, _| {},
        )
        .unwrap();
        let result = analyze_files(
            project.path(),
            project.path(),
            &files,
            1,
            true,
            &cancelled,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(result.cache_hits, 0);
        assert_eq!(result.results.len(), 1);
    }

    #[test]
    fn valid_cache_skips_fits_heads_and_computation() {
        let project = tempfile::tempdir().unwrap();
        let file = synthetic_fits(project.path(), "frame.fits");
        let cancelled = AtomicBool::new(false);
        analyze_files(
            project.path(),
            project.path(),
            std::slice::from_ref(&file),
            1,
            false,
            &cancelled,
            |_, _| {},
        )
        .unwrap();

        fs::write(&file, b"not a FITS header").unwrap();
        let metadata = file.metadata().unwrap();
        let cache_path = project.path().join(crate::cache::CACHE_FILE_NAME);
        let mut document: serde_json::Value =
            serde_json::from_slice(&fs::read(&cache_path).unwrap()).unwrap();
        let entries = document["engines"]["cpu"].as_array_mut().unwrap();
        entries[0]["size"] = metadata.len().into();
        entries[0]["modified_ns"] = u64::try_from(
            metadata
                .modified()
                .unwrap()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        )
        .unwrap()
        .into();
        fs::write(&cache_path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

        let batch = analyze_files(
            project.path(),
            project.path(),
            &[file],
            1,
            false,
            &cancelled,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(batch.cache_hits, 1);
        assert_eq!(batch.results.len(), 1);
    }

    #[test]
    fn flat_frames_are_removed_before_analysis() {
        let project = tempfile::tempdir().unwrap();
        let light = project.path().join("light.fits");
        let flat = project.path().join("flat_001.fits");
        synthetic_fits(project.path(), "light.fits");
        fs::copy(&light, &flat).unwrap();
        let batch = analyze_files(
            project.path(),
            project.path(),
            &[light.clone(), flat.clone()],
            1,
            false,
            &AtomicBool::new(false),
            |_, _| {},
        )
        .unwrap();
        assert_eq!(batch.flat_count, 1);
        assert_eq!(batch.flat_files, [flat]);
        assert_eq!(
            batch
                .results
                .iter()
                .map(|result| &result.filename)
                .collect::<Vec<_>>(),
            [&light]
        );
    }

    #[test]
    fn cache_records_remain_bound_to_original_light_files_after_a_flat() {
        let project = tempfile::tempdir().unwrap();
        let flat = project.path().join("flat.fits");
        let first = project.path().join("first.fits");
        let second = project.path().join("second.fits");
        synthetic_fits(project.path(), "first.fits");
        fs::copy(&first, &flat).unwrap();
        fs::copy(&first, &second).unwrap();
        let files = [flat, first.clone(), second.clone()];
        let cancelled = AtomicBool::new(false);
        let batch = analyze_files(
            project.path(),
            project.path(),
            &files,
            1,
            false,
            &cancelled,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(batch.flat_count, 1);

        let cache: serde_json::Value = serde_json::from_slice(
            &fs::read(project.path().join(crate::cache::CACHE_FILE_NAME)).unwrap(),
        )
        .unwrap();
        let paths: Vec<_> = cache["engines"]["cpu"]
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| entry["path"].as_str().unwrap())
            .collect();
        assert_eq!(paths, ["first.fits", "second.fits"]);

        let repeated = analyze_files(
            project.path(),
            project.path(),
            &[first, second],
            1,
            false,
            &cancelled,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(repeated.cache_hits, 2);
    }
}
