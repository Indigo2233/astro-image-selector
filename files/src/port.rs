//! FITS discovery and project file operations.

#![forbid(unsafe_code)]

use std::ffi::OsString;
use std::fs::{self, File, FileTimes, Metadata};
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

pub fn scan_fits_files(directory: String) -> Result<Vec<String>, String> {
    Ok(scan(Path::new(&directory))
        .into_iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect())
}

pub fn scan(directory: &Path) -> Vec<PathBuf> {
    scan_cancellable(directory, &AtomicBool::new(false))
}

pub fn scan_cancellable(directory: &Path, cancelled: &AtomicBool) -> Vec<PathBuf> {
    let mut files = walk(directory, cancelled, true);
    files.retain(|file| {
        file.file_name().is_some_and(|name| {
            let bytes = name.as_encoded_bytes();
            [b".fit".as_slice(), b".fits"].iter().any(|suffix| {
                bytes
                    .get(bytes.len().saturating_sub(suffix.len())..)
                    .is_some_and(|end| end.eq_ignore_ascii_case(suffix))
            })
        })
    });
    files.sort_unstable_by(|a, b| a.as_os_str().cmp(b.as_os_str()));
    files
}

fn walk(root: &Path, cancelled: &AtomicBool, skip_rejected: bool) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    let mut index = 0;

    while let Some(dir) = dirs.get(index).cloned() {
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
        index += 1;
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if cancelled.load(Ordering::Relaxed) {
                break;
            }
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if is_link(&metadata) {
                if !path.is_dir() {
                    files.push(path);
                }
            } else if metadata.is_dir() {
                if skip_rejected
                    && dir == root
                    && path
                        .file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.eq_ignore_ascii_case("rejected"))
                {
                    continue;
                }
                dirs.push(path);
            } else {
                files.push(path);
            }
        }
    }
    files
}

fn walk_with_directories(root: &Path) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let mut files = Vec::new();
    let mut dirs = vec![root.to_path_buf()];
    let mut index = 0;

    while let Some(dir) = dirs.get(index) {
        index += 1;
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(metadata) = entry.metadata() else {
                continue;
            };
            if is_link(&metadata) {
                if !path.is_dir() {
                    files.push(path);
                }
            } else if metadata.is_dir() {
                dirs.push(path);
            } else {
                files.push(path);
            }
        }
    }
    (files, dirs)
}

#[cfg(windows)]
fn is_link(metadata: &Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const REPARSE_POINT: u32 = 0x400;
    metadata.file_attributes() & REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn is_link(metadata: &Metadata) -> bool {
    metadata.file_type().is_symlink()
}

fn invalid_path() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "path must stay in the project directory",
    )
}

fn strict_relative(path: &Path) -> io::Result<&Path> {
    if path.as_os_str().is_empty()
        || !path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
    {
        return Err(invalid_path());
    }
    Ok(path)
}

fn checked_directory(root: &Path, path: &Path, create: bool) -> io::Result<PathBuf> {
    let mut directory = root.to_path_buf();
    for part in path.components() {
        let Component::Normal(name) = part else {
            return Err(invalid_path());
        };
        directory.push(name);
        if create {
            match fs::create_dir(&directory) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error),
            }
        }
        let metadata = fs::symlink_metadata(&directory)?;
        if !metadata.is_dir() || is_link(&metadata) {
            return Err(invalid_path());
        }
    }
    Ok(directory)
}

pub struct FileManager {
    project_directory: PathBuf,
}

impl FileManager {
    pub fn new(directory: impl AsRef<Path>) -> Self {
        Self {
            project_directory: directory.as_ref().to_path_buf(),
        }
    }

    pub fn ensure_rejected_directory(&self) -> io::Result<PathBuf> {
        checked_directory(
            &self.project_directory.canonicalize()?,
            Path::new("rejected"),
            true,
        )
    }

    pub fn reject(&self, filename: &Path) -> io::Result<PathBuf> {
        let root = self.project_directory.canonicalize()?;
        let source = root.join(filename).canonicalize()?;
        let relative = source.strip_prefix(&root).map_err(|_| invalid_path())?;
        let relative = strict_relative(relative)?;
        let name = relative.file_name().ok_or_else(invalid_path)?;
        if !source.is_file() {
            return Err(invalid_path());
        }
        let rejected = self.ensure_rejected_directory()?;
        let parent = relative.parent().ok_or_else(invalid_path)?;
        let destination_directory = checked_directory(&rejected, parent, true)?;
        let destination = destination_directory.join(name);
        let stem = relative.file_stem().ok_or_else(invalid_path)?;
        let extension = relative.extension();
        let mut candidate = destination;
        for index in 1.. {
            match move_file(&source, &candidate) {
                Ok(()) => return Ok(candidate),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                    let mut name = OsString::from(stem);
                    name.push(format!("_{index}"));
                    if let Some(extension) = extension {
                        name.push(".");
                        name.push(extension);
                    }
                    candidate = destination_directory.join(name);
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("the numbering iterator does not terminate")
    }

    pub fn restore(&self, filename: &Path) -> io::Result<PathBuf> {
        let filename = strict_relative(filename)?;
        let root = self.project_directory.canonicalize()?;
        let rejected = checked_directory(&root, Path::new("rejected"), false)?;
        let parent = filename.parent().ok_or_else(invalid_path)?;
        let name = filename.file_name().ok_or_else(invalid_path)?;
        let source = checked_directory(&rejected, parent, false)?.join(name);
        let metadata = fs::symlink_metadata(&source)?;
        if !metadata.is_file() || is_link(&metadata) {
            return Err(invalid_path());
        }
        let destination = checked_directory(&root, parent, true)?.join(name);
        move_file(&source, &destination)?;
        Ok(destination)
    }

    pub fn rejected_files(&self) -> Vec<PathBuf> {
        let Some(directory) = self.existing_rejected_directory() else {
            return Vec::new();
        };
        let mut files: Vec<_> = walk_with_directories(&directory)
            .0
            .into_iter()
            .filter(|path| {
                path.is_file()
                    && path.extension().is_some_and(|extension| {
                        ["fit", "fits"]
                            .iter()
                            .any(|expected| extension.eq_ignore_ascii_case(expected))
                    })
            })
            .filter_map(|path| path.strip_prefix(&directory).ok().map(Path::to_path_buf))
            .collect();
        files.sort_unstable_by(|a, b| a.as_os_str().cmp(b.as_os_str()));
        files
    }

    pub fn clean_rejected_directory(&self) -> usize {
        let Some(directory) = self.existing_rejected_directory() else {
            return 0;
        };
        let (files, directories) = walk_with_directories(&directory);
        let removed = files
            .into_iter()
            .filter(|file| file.is_file() && fs::remove_file(file).is_ok())
            .count();
        for directory in directories.into_iter().skip(1).rev() {
            let _ = fs::remove_dir(directory);
        }
        removed
    }

    fn existing_rejected_directory(&self) -> Option<PathBuf> {
        let root = self.project_directory.canonicalize().ok()?;
        checked_directory(&root, Path::new("rejected"), false).ok()
    }
}

fn move_file(source: &Path, destination: &Path) -> io::Result<()> {
    if fs::symlink_metadata(destination).is_ok() {
        return Err(io::ErrorKind::AlreadyExists.into());
    }
    match fs::hard_link(source, destination) {
        Ok(()) => {}
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => return Err(error),
        Err(_) => {
            let mut input = File::open(source)?;
            let metadata = input.metadata()?;
            let directory = destination.parent().ok_or_else(invalid_path)?;
            let mut output = tempfile::NamedTempFile::new_in(directory)?;
            copy_file(&mut input, output.as_file_mut(), &metadata)?;
            output
                .persist_noclobber(destination)
                .map_err(|error| error.error)?;
        }
    }
    if let Err(error) = fs::remove_file(source) {
        let _ = fs::remove_file(destination);
        return Err(error);
    }
    Ok(())
}

fn copy_file(source: &mut File, destination: &mut File, metadata: &Metadata) -> io::Result<()> {
    io::copy(source, destination)?;
    let mut times = FileTimes::new();
    if let Ok(accessed) = metadata.accessed() {
        times = times.set_accessed(accessed);
    }
    if let Ok(modified) = metadata.modified() {
        times = times.set_modified(modified);
    }
    destination.set_times(times)?;
    destination.set_permissions(metadata.permissions())
}

#[cfg(test)]
mod scan_tests {
    use super::*;

    #[test]
    fn cancelled_scan_returns_no_files() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("frame.fits"), []).unwrap();
        let cancelled = AtomicBool::new(true);

        assert!(scan_cancellable(root.path(), &cancelled).is_empty());
    }

    #[test]
    fn active_scan_still_finds_nested_fits_files() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("session");
        fs::create_dir(&nested).unwrap();
        fs::write(nested.join("frame.FITS"), []).unwrap();
        fs::write(root.path().join("notes.txt"), []).unwrap();
        let cancelled = AtomicBool::new(false);

        assert_eq!(
            scan_cancellable(root.path(), &cancelled),
            [nested.join("frame.FITS")]
        );
    }

    #[test]
    fn scan_skips_rejected_subtree_at_the_project_root() {
        let root = tempfile::tempdir().unwrap();
        let rejected = root.path().join("Rejected");
        let nested = root.path().join("session");
        let nested_rejected = nested.join("rejected");
        fs::create_dir(&rejected).unwrap();
        fs::create_dir(&nested).unwrap();
        fs::create_dir(&nested_rejected).unwrap();
        fs::write(root.path().join("light.fits"), []).unwrap();
        fs::write(rejected.join("old.fits"), []).unwrap();
        fs::write(nested_rejected.join("frame.fits"), []).unwrap();

        assert_eq!(
            scan(root.path()),
            [
                root.path().join("light.fits"),
                nested_rejected.join("frame.fits")
            ]
        );
    }

    #[test]
    fn restore_reports_destination_conflicts() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("frame.fits"), b"rejected").unwrap();
        let manager = FileManager::new(root.path());
        manager.reject(Path::new("frame.fits")).unwrap();
        fs::write(root.path().join("frame.fits"), b"current").unwrap();

        let error = manager.restore(Path::new("frame.fits")).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(
            fs::read(root.path().join("rejected/frame.fits")).unwrap(),
            b"rejected"
        );
        assert_eq!(
            fs::read(root.path().join("frame.fits")).unwrap(),
            b"current"
        );
    }

    #[test]
    fn restore_and_reject_round_trip_preserves_nested_paths() {
        let root = tempfile::tempdir().unwrap();
        let nested = root.path().join("session");
        fs::create_dir(&nested).unwrap();
        fs::write(nested.join("frame.fits"), b"pixels").unwrap();
        let manager = FileManager::new(root.path());

        manager.reject(Path::new("session/frame.fits")).unwrap();
        assert_eq!(
            manager.rejected_files(),
            [PathBuf::from("session/frame.fits")]
        );
        let restored = manager.restore(Path::new("session/frame.fits")).unwrap();
        assert_eq!(fs::read(restored).unwrap(), b"pixels");
        assert_eq!(fs::read(nested.join("frame.fits")).unwrap(), b"pixels");
    }
}
