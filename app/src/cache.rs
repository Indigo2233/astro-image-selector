use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

use crate::catalog::AnalysisResult;

pub const CACHE_FILE_NAME: &str = ".astro_selector_analysis_cache.json";

const CACHE_SCHEMA: &str = "astro-selector-analysis";
const CACHE_VERSION: u32 = 2;

#[derive(Debug, Deserialize, Serialize)]
struct CacheDocument {
    schema: String,
    version: u32,
    entries: Vec<CacheEntry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CacheEntry {
    path: String,
    size: u64,
    modified_ns: u64,
    filter_type: String,
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
    pub fn read(project: &Path, filename: &Path) -> Option<Self> {
        let relative = filename.strip_prefix(project).ok()?;
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
    entries: HashMap<String, CacheEntry>,
    warning: Option<String>,
}

impl AnalysisCache {
    pub fn load(project: &Path) -> Self {
        let path = project.join(CACHE_FILE_NAME);
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Self {
                    entries: HashMap::new(),
                    warning: None,
                };
            }
            Err(error) => {
                return Self {
                    entries: HashMap::new(),
                    warning: Some(format!("读取分析缓存失败: {error}")),
                };
            }
        };
        let document: CacheDocument = match serde_json::from_slice(&bytes) {
            Ok(document) => document,
            Err(error) => {
                return Self {
                    entries: HashMap::new(),
                    warning: Some(format!("分析缓存无效，将重新生成: {error}")),
                };
            }
        };
        if document.schema != CACHE_SCHEMA || document.version != CACHE_VERSION {
            return Self {
                entries: HashMap::new(),
                warning: Some("分析缓存版本不兼容，将重新生成".into()),
            };
        }
        let entries = document
            .entries
            .into_iter()
            .map(|entry| (entry.path.clone(), entry))
            .collect();
        Self {
            entries,
            warning: None,
        }
    }

    pub fn probe(&self, project: &Path, filename: &Path) -> CacheProbe {
        let Some(fingerprint) = FileFingerprint::read(project, filename) else {
            return CacheProbe {
                fingerprint: None,
                result: None,
            };
        };
        let result = self
            .entries
            .get(&fingerprint.key)
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
}

pub fn save(
    project: &Path,
    entries: Vec<(FileFingerprint, AnalysisResult)>,
) -> Result<PathBuf, String> {
    let mut entries: Vec<_> = entries
        .into_iter()
        .filter(|(_, result)| {
            result.filter_type != "Unknown"
                && !flat_path(&result.filename)
                && result.eccentricity.is_finite()
                && result.fwhm.is_finite()
                && result.snr.is_finite()
                && result.mean_brightness.is_finite()
        })
        .map(|(fingerprint, result)| CacheEntry::from_result(fingerprint, &result))
        .collect();
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    let document = CacheDocument {
        schema: CACHE_SCHEMA.into(),
        version: CACHE_VERSION,
        entries,
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
    if destination.exists() {
        fs::remove_file(&destination).map_err(|error| error.to_string())?;
    }
    match fs::rename(&temporary_path, &destination) {
        Ok(()) => Ok(destination),
        Err(error) => {
            let _ = fs::remove_file(&temporary_path);
            Err(error.to_string())
        }
    }
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
        let mut result = AnalysisResult::unreadable(&file);
        result.filter_type = "Ha".into();
        result.fwhm = 4.25;
        let fingerprint = FileFingerprint::read(project.path(), &file).unwrap();
        save(project.path(), vec![(fingerprint, result)]).unwrap();

        let cache = AnalysisCache::load(project.path());
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
        save(
            project.path(),
            vec![(fingerprint, AnalysisResult::unreadable(&file))],
        )
        .unwrap();
        fs::write(&file, b"a longer file").unwrap();

        let cache = AnalysisCache::load(project.path());
        assert!(cache.probe(project.path(), &file).result.is_none());
    }

    #[test]
    fn malformed_cache_is_ignored_and_can_be_replaced() {
        let project = tempdir();
        fs::write(project.path().join(CACHE_FILE_NAME), b"not json").unwrap();
        let mut cache = AnalysisCache::load(project.path());
        assert!(cache.take_warning().is_some());
        assert!(
            AnalysisCache::load(project.path())
                .probe(project.path(), &project.path().join("missing.fits"))
                .result
                .is_none()
        );
    }
}
