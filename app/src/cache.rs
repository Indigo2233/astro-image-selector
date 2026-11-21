use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use crate::catalog::AnalysisResult;

pub const CACHE_FILE_NAME: &str = ".astro_selector_analysis_cache.json";

const CACHE_SCHEMA: &str = "astro-selector-analysis";
const CACHE_VERSION: u32 = 5;

#[derive(Debug, Deserialize, Serialize)]
struct CacheDocument {
    schema: String,
    version: u32,
    analysis_version: u32,
    engines: HashMap<String, Vec<CacheEntry>>,
}

#[derive(Debug, Deserialize)]
struct LegacyCacheDocument {
    schema: String,
    version: u32,
    analysis_version: u32,
    engine: String,
    entries: Vec<CacheEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CacheEntry {
    path: String,
    size: u64,
    modified_ns: u64,
    filter_type: String,
    status: crate::catalog::AnalysisStatus,
    error: Option<String>,
    eccentricity: f64,
    fwhm: f64,
    snr: f64,
    star_count: usize,
    mean_brightness: f64,
}

impl CacheEntry {
    fn matches(&self, fingerprint: &FileFingerprint) -> bool {
        self.path == fingerprint.key
            && self.size == fingerprint.size
            && self.modified_ns == fingerprint.modified_ns
    }

    fn result(&self, filename: &Path) -> AnalysisResult {
        AnalysisResult {
            filename: filename.to_path_buf(),
            status: self.status,
            error: self.error.clone(),
            filter_type: self.filter_type.clone(),
            eccentricity: self.eccentricity,
            fwhm: self.fwhm,
            snr: self.snr,
            star_count: self.star_count,
            mean_brightness: self.mean_brightness,
        }
    }

    fn from_result(fingerprint: FileFingerprint, result: &AnalysisResult) -> Self {
        Self {
            path: fingerprint.key,
            size: fingerprint.size,
            modified_ns: fingerprint.modified_ns,
            filter_type: result.filter_type.clone(),
            status: result.status,
            error: result.error.clone(),
            eccentricity: result.eccentricity,
            fwhm: result.fwhm,
            snr: result.snr,
            star_count: result.star_count,
            mean_brightness: result.mean_brightness,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FileFingerprint {
    key: String,
    size: u64,
    modified_ns: u64,
}

impl FileFingerprint {
    pub fn read(source: &Path, filename: &Path) -> Option<Self> {
        let relative = filename.strip_prefix(source).ok()?;
        let metadata = filename.metadata().ok()?;
        let modified = metadata.modified().ok()?;
        let duration = modified.duration_since(UNIX_EPOCH).ok()?;
        Some(Self {
            key: relative.to_string_lossy().replace('\\', "/"),
            size: metadata.len(),
            modified_ns: u64::try_from(duration.as_nanos()).ok()?,
        })
    }
}

pub struct CacheProbe {
    pub fingerprint: Option<FileFingerprint>,
    pub result: Option<AnalysisResult>,
}

pub struct AnalysisCache {
    engines: HashMap<String, HashMap<String, CacheEntry>>,
    engine: String,
    warning: Option<String>,
}

impl AnalysisCache {
    pub fn load(project: &Path, engine: &str) -> Self {
        let engine = engine.to_owned();
        let path = project.join(CACHE_FILE_NAME);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Self {
                    engines: HashMap::new(),
                    engine,
                    warning: None,
                };
            }
            Err(error) => {
                return Self {
                    engines: HashMap::new(),
                    engine,
                    warning: Some(format!("读取分析缓存失败: {error}")),
                };
            }
        };
        let (schema, version, analysis_version, engines) =
            match serde_json::from_slice::<CacheDocument>(&bytes) {
                Ok(document) if document.version == CACHE_VERSION => (
                    document.schema,
                    document.version,
                    document.analysis_version,
                    document.engines,
                ),
                _ => match serde_json::from_slice::<LegacyCacheDocument>(&bytes) {
                    Ok(document) => (
                        document.schema,
                        document.version,
                        document.analysis_version,
                        HashMap::from([(document.engine.clone(), document.entries)]),
                    ),
                    Err(error) => {
                        return Self {
                            engines: HashMap::new(),
                            engine,
                            warning: Some(format!("分析缓存无效，将重新生成: {error}")),
                        };
                    }
                },
            };
        if schema != CACHE_SCHEMA
            || (version != CACHE_VERSION && version != CACHE_VERSION - 1)
            || analysis_version != crate::ANALYSIS_VERSION
        {
            return Self {
                engines: HashMap::new(),
                engine,
                warning: Some("分析缓存版本不兼容，将重新生成".into()),
            };
        }
        let engines = engines
            .into_iter()
            .map(|(name, entries)| {
                let entries = entries
                    .into_iter()
                    .filter_map(|entry| match entry.status {
                        crate::catalog::AnalysisStatus::Success => {
                            Some((entry.path.clone(), entry))
                        }
                        _ => None,
                    })
                    .collect();
                (name, entries)
            })
            .collect();
        Self {
            engines,
            engine,
            warning: None,
        }
    }

    pub fn probe(&self, source: &Path, filename: &Path) -> CacheProbe {
        let Some(fingerprint) = FileFingerprint::read(source, filename) else {
            return CacheProbe {
                fingerprint: None,
                result: None,
            };
        };
        let result = self
            .engines
            .get(&self.engine)
            .and_then(|entries| entries.get(&fingerprint.key))
            .filter(|entry| entry.matches(&fingerprint))
            .map(|entry| entry.result(filename));
        CacheProbe {
            fingerprint: Some(fingerprint),
            result,
        }
    }

    pub fn take_warning(&mut self) -> Option<String> {
        self.warning.take()
    }

    pub fn engine(&self) -> &str {
        &self.engine
    }
}

pub fn save(
    project: &Path,
    engine: &str,
    entries: Vec<(FileFingerprint, AnalysisResult)>,
) -> Result<PathBuf, String> {
    let mut entries: Vec<_> = entries
        .into_iter()
        .filter(|(_, result)| {
            result.status.is_success()
                && !flat_path(&result.filename)
                && result.filter_type != "Unknown"
                && result.eccentricity.is_finite()
                && result.fwhm.is_finite()
                && result.snr.is_finite()
                && result.mean_brightness.is_finite()
        })
        .map(|(fingerprint, result)| CacheEntry::from_result(fingerprint, &result))
        .collect();
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    let mut engines = load_engines(project);
    engines.insert(engine.to_owned(), entries);
    for engine_entries in engines.values_mut() {
        engine_entries.retain(|entry| entry.status.is_success());
        engine_entries.sort_by(|a, b| a.path.cmp(&b.path));
    }
    let document = CacheDocument {
        schema: CACHE_SCHEMA.into(),
        version: CACHE_VERSION,
        analysis_version: crate::ANALYSIS_VERSION,
        engines,
    };
    let bytes = serde_json::to_vec_pretty(&document).map_err(|error| error.to_string())?;
    let destination = project.join(CACHE_FILE_NAME);
    let file_name = format!("{CACHE_FILE_NAME}.{}.tmp", std::process::id());
    let temporary_path = project.join(file_name);
    let mut temporary = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary_path)
        .map_err(|error| error.to_string())?;
    temporary
        .write_all(&bytes)
        .and_then(|()| temporary.sync_all())
        .map_err(|error| error.to_string())?;
    drop(temporary);
    replace_cache(project, &temporary_path, &destination).map(|()| destination)
}

fn replace_cache(project: &Path, source: &Path, destination: &Path) -> Result<(), String> {
    let previous = project.join(format!("{CACHE_FILE_NAME}.previous"));
    let had_previous = destination.exists();
    if had_previous {
        let _ = fs::remove_file(&previous);
        fs::copy(destination, &previous).map_err(|error| error.to_string())?;
    }
    let result = fs::rename(source, destination).map_err(|error| error.to_string());
    match result {
        Ok(()) => {
            if had_previous {
                let _ = fs::remove_file(&previous);
            }
            Ok(())
        }
        Err(error) => {
            if had_previous {
                let _ = fs::copy(&previous, destination);
                let _ = fs::remove_file(&previous);
            }
            Err(error)
        }
    }
}

fn load_engines(project: &Path) -> HashMap<String, Vec<CacheEntry>> {
    let Ok(bytes) = fs::read(project.join(CACHE_FILE_NAME)) else {
        return HashMap::new();
    };
    if let Ok(document) = serde_json::from_slice::<CacheDocument>(&bytes)
        && document.schema == CACHE_SCHEMA
        && document.version == CACHE_VERSION
        && document.analysis_version == crate::ANALYSIS_VERSION
    {
        return document.engines;
    }
    if let Ok(document) = serde_json::from_slice::<LegacyCacheDocument>(&bytes)
        && document.schema == CACHE_SCHEMA
        && document.version == CACHE_VERSION - 1
        && document.analysis_version == crate::ANALYSIS_VERSION
    {
        return HashMap::from([(document.engine, document.entries)]);
    }
    HashMap::new()
}

fn flat_path(path: &Path) -> bool {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| {
            let normalized: String = stem
                .chars()
                .filter(|character| character.is_ascii_alphanumeric())
                .flat_map(char::to_uppercase)
                .collect();
            normalized.contains("FLAT")
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn matching_cache_entry_rebinds_the_current_absolute_path() {
        let project = tempdir();
        let file = project.path().join("session").join("frame.fits");
        fs::create_dir(file.parent().unwrap()).unwrap();
        fs::write(&file, b"FITS").unwrap();
        let mut result =
            AnalysisResult::failed(&file, crate::catalog::AnalysisStatus::Success, "unused");
        result.filter_type = "Ha".into();
        result.fwhm = 4.25;
        let fingerprint = FileFingerprint::read(project.path(), &file).unwrap();
        save(project.path(), "cpu", vec![(fingerprint, result)]).unwrap();

        let cache = AnalysisCache::load(project.path(), "cpu");
        let hit = cache.probe(project.path(), &file).result.unwrap();
        assert_eq!(hit.filename, file);
        assert_eq!(hit.filter_type, "Ha");
        assert_eq!(hit.fwhm, 4.25);
    }

    #[test]
    fn changed_file_size_invalidates_the_entry() {
        let project = tempdir();
        let file = project.path().join("frame.fits");
        fs::write(&file, b"short").unwrap();
        let fingerprint = FileFingerprint::read(project.path(), &file).unwrap();
        let result =
            AnalysisResult::failed(&file, crate::catalog::AnalysisStatus::Success, "unused");
        save(project.path(), "cpu", vec![(fingerprint, result)]).unwrap();
        fs::write(&file, b"a longer file").unwrap();

        let cache = AnalysisCache::load(project.path(), "cpu");
        assert!(cache.probe(project.path(), &file).result.is_none());
    }

    #[test]
    fn malformed_cache_is_ignored_and_can_be_replaced() {
        let project = tempdir();
        fs::write(project.path().join(CACHE_FILE_NAME), b"not json").unwrap();
        let mut cache = AnalysisCache::load(project.path(), "cpu");
        assert!(cache.take_warning().is_some());
        assert!(
            AnalysisCache::load(project.path(), "cpu")
                .probe(project.path(), &project.path().join("missing.fits"))
                .result
                .is_none()
        );
    }

    #[test]
    fn cache_only_restores_successful_results() {
        let project = tempdir();
        let file = project.path().join("failed.fits");
        fs::write(&file, b"FITS").unwrap();
        let fingerprint = FileFingerprint::read(project.path(), &file).unwrap();
        save(
            project.path(),
            "cpu",
            vec![(
                fingerprint,
                AnalysisResult::failed(
                    &file,
                    crate::catalog::AnalysisStatus::ReadFailed,
                    "unreadable",
                ),
            )],
        )
        .unwrap();

        let cache = AnalysisCache::load(project.path(), "cpu");
        assert!(cache.probe(project.path(), &file).result.is_none());
    }

    #[test]
    fn switching_engines_preserves_both_cache_partitions() {
        let project = tempdir();
        let file = project.path().join("frame.fits");
        fs::write(&file, b"FITS").unwrap();
        let fingerprint = FileFingerprint::read(project.path(), &file).unwrap();
        let mut result =
            AnalysisResult::failed(&file, crate::catalog::AnalysisStatus::Success, "unused");
        result.filter_type = "Ha".into();
        result.fwhm = 4.0;
        save(project.path(), "cpu", vec![(fingerprint.clone(), result)]).unwrap();
        let mut gpu_result =
            AnalysisResult::failed(&file, crate::catalog::AnalysisStatus::Success, "unused");
        gpu_result.filter_type = "Ha".into();
        gpu_result.fwhm = 6.0;
        save(project.path(), "gpu", vec![(fingerprint, gpu_result)]).unwrap();

        assert!(
            AnalysisCache::load(project.path(), "cpu")
                .probe(project.path(), &file)
                .result
                .is_some()
        );
        assert!(
            AnalysisCache::load(project.path(), "gpu")
                .probe(project.path(), &file)
                .result
                .is_some()
        );
        let cpu = AnalysisCache::load(project.path(), "cpu")
            .probe(project.path(), &file)
            .result
            .unwrap();
        let gpu = AnalysisCache::load(project.path(), "gpu")
            .probe(project.path(), &file)
            .result
            .unwrap();
        assert_eq!(cpu.fwhm, 4.0);
        assert_eq!(gpu.fwhm, 6.0);
    }

    #[test]
    fn previous_single_engine_document_is_migrated() {
        let project = tempdir();
        let file = project.path().join("frame.fits");
        fs::write(&file, b"FITS").unwrap();
        let fingerprint = FileFingerprint::read(project.path(), &file).unwrap();
        let mut result =
            AnalysisResult::failed(&file, crate::catalog::AnalysisStatus::Success, "unused");
        result.filter_type = "Ha".into();
        let entry = CacheEntry::from_result(fingerprint, &result);
        let legacy = json_document(&entry);
        fs::write(project.path().join(CACHE_FILE_NAME), legacy).unwrap();

        assert!(
            AnalysisCache::load(project.path(), "cpu")
                .probe(project.path(), &file)
                .result
                .is_some()
        );
    }

    fn json_document(entry: &CacheEntry) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "schema": CACHE_SCHEMA,
            "version": CACHE_VERSION - 1,
            "analysis_version": crate::ANALYSIS_VERSION,
            "engine": "cpu",
            "entries": [entry],
        }))
        .unwrap()
    }

    #[test]
    fn unreadable_replacement_keeps_prior_cache() {
        let project = tempdir();
        let destination = project.path().join(CACHE_FILE_NAME);
        fs::write(&destination, b"valid-cache").unwrap();
        let missing_source = project.path().join("missing.tmp");
        assert!(replace_cache(project.path(), &missing_source, &destination).is_err());
        assert_eq!(fs::read(destination).unwrap(), b"valid-cache");
        assert!(
            !project
                .path()
                .join(format!("{CACHE_FILE_NAME}.previous"))
                .exists()
        );
    }
}
