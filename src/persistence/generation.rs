//! Crash-oriented publication of multi-file index generations.

use super::error::{PersistenceError, PersistenceResult};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static GENERATION_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const CURRENT_FILE: &str = "CURRENT";

/// The result of a generation publication after the staging directory is
/// renamed into place.
#[derive(Debug)]
pub enum PublicationOutcome {
    /// `CURRENT` was replaced and its containing directory was synced.
    Committed(PathBuf),
    /// `CURRENT` was replaced, but syncing the root failed. The generation is
    /// visible to readers, but its power-loss durability is unknown.
    DurabilityUncertain {
        /// The generation directory that may now be current.
        published_dir: PathBuf,
        /// The error encountered while completing the durability barrier.
        error: PersistenceError,
    },
}

/// A new, unpublished generation being assembled beneath a persistence root.
///
/// Files must be written below [`Self::path`]. Dropping a writer intentionally
/// leaves its staging directory for a later cleanup pass rather than deleting
/// data that may be useful for recovery diagnostics.
pub struct GenerationWriter {
    root: PathBuf,
    generations_dir: PathBuf,
    staging_dir: PathBuf,
    generation_id: String,
}

impl GenerationWriter {
    /// Create a unique hidden staging directory below `root`.
    pub fn create(root: impl AsRef<Path>) -> PersistenceResult<Self> {
        let root = root.as_ref().to_path_buf();
        let generations_dir = root.join("generations");
        std::fs::create_dir_all(&generations_dir)?;
        let generation_id = unique_generation_id()?;
        let staging_dir = generations_dir.join(format!(".staging-{generation_id}"));
        std::fs::create_dir(&staging_dir)?;
        Ok(Self {
            root,
            generations_dir,
            staging_dir,
            generation_id,
        })
    }

    /// Return the generation ID that will become current on publication.
    pub fn generation_id(&self) -> &str {
        &self.generation_id
    }

    /// Return the staging directory for a format-specific writer.
    pub fn directory(&self) -> &Path {
        &self.staging_dir
    }

    /// Resolve a safe relative component path inside the staging directory.
    pub fn path(&self, relative: impl AsRef<Path>) -> PersistenceResult<PathBuf> {
        let relative = relative.as_ref();
        validate_relative_path(relative)?;
        reject_symlink_components(&self.staging_dir, relative)?;
        let path = self.staging_dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(path)
    }

    /// Sync one completed component file before publishing the generation.
    pub fn sync_file(&self, relative: impl AsRef<Path>) -> PersistenceResult<()> {
        let path = self.path(relative)?;
        std::fs::File::open(path)?.sync_all()?;
        Ok(())
    }

    /// Sync the staging directory after all component files have been synced.
    pub fn sync(&self) -> PersistenceResult<()> {
        sync_tree(&self.staging_dir)
    }

    /// Publish this generation by atomically replacing the root `CURRENT`
    /// pointer. The old pointer remains intact if any pre-publication step
    /// fails.
    pub fn publish(self) -> PersistenceResult<PathBuf> {
        match self.publish_with_outcome()? {
            PublicationOutcome::Committed(path) => Ok(path),
            PublicationOutcome::DurabilityUncertain { error, .. } => Err(error),
        }
    }

    /// Validate and publish this generation, preserving whether a failure
    /// happened before or after the `CURRENT` pointer was replaced.
    pub fn publish_with_outcome(self) -> PersistenceResult<PublicationOutcome> {
        sync_tree(&self.staging_dir)?;
        let published_dir = self.generations_dir.join(&self.generation_id);
        std::fs::rename(&self.staging_dir, &published_dir)?;
        sync_directory(&self.generations_dir)?;
        match write_current(&self.root, &self.generation_id) {
            Ok(()) => Ok(PublicationOutcome::Committed(published_dir)),
            Err(CurrentWriteError::BeforeCommit(error)) => Err(error),
            Err(CurrentWriteError::AfterCommit(error)) => {
                Ok(PublicationOutcome::DurabilityUncertain {
                    published_dir,
                    error,
                })
            }
        }
    }
}

/// Resolve the directory named by the root `CURRENT` pointer.
pub fn open_current(root: impl AsRef<Path>) -> PersistenceResult<PathBuf> {
    let root = root.as_ref();
    let current = std::fs::read_to_string(root.join(CURRENT_FILE))?;
    let generation_id = current.trim();
    validate_generation_id(generation_id)?;
    let path = root.join("generations").join(generation_id);
    let metadata = std::fs::symlink_metadata(&path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(PersistenceError::Format(format!(
            "current generation {} is not a directory",
            path.display()
        )));
    }
    Ok(path)
}

fn unique_generation_id() -> PersistenceResult<String> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| PersistenceError::InvalidState(error.to_string()))?
        .as_nanos();
    let sequence = GENERATION_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    Ok(format!("{nanos}-{}-{sequence}", std::process::id()))
}

fn validate_generation_id(id: &str) -> PersistenceResult<()> {
    if id.is_empty() || id == "." || id == ".." || id.contains('/') || id.contains('\\') {
        return Err(PersistenceError::Format(
            "CURRENT must contain one safe generation basename".into(),
        ));
    }
    Ok(())
}

fn validate_relative_path(path: &Path) -> PersistenceResult<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(PersistenceError::Format(
            "generation component path must be relative and traversal-free".into(),
        ));
    }
    Ok(())
}

enum CurrentWriteError {
    BeforeCommit(PersistenceError),
    AfterCommit(PersistenceError),
}

fn write_current(root: &Path, generation_id: &str) -> Result<(), CurrentWriteError> {
    let current = root.join(CURRENT_FILE);
    let tmp = root.join(format!(
        ".CURRENT.tmp.{}.{}",
        std::process::id(),
        GENERATION_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    {
        let file = std::fs::File::create(&tmp)
            .map_err(PersistenceError::from)
            .map_err(CurrentWriteError::BeforeCommit)?;
        use std::io::Write;
        let mut writer = std::io::BufWriter::new(file);
        writer
            .write_all(generation_id.as_bytes())
            .map_err(PersistenceError::from)
            .map_err(CurrentWriteError::BeforeCommit)?;
        writer
            .write_all(b"\n")
            .map_err(PersistenceError::from)
            .map_err(CurrentWriteError::BeforeCommit)?;
        writer
            .flush()
            .map_err(PersistenceError::from)
            .map_err(CurrentWriteError::BeforeCommit)?;
        writer
            .get_ref()
            .sync_all()
            .map_err(PersistenceError::from)
            .map_err(CurrentWriteError::BeforeCommit)?;
    }
    let result = std::fs::rename(&tmp, &current).map_err(PersistenceError::from);
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result.map_err(CurrentWriteError::BeforeCommit)?;
    #[cfg(test)]
    if let Ok(mut target) = FAIL_ROOT_SYNC.lock() {
        if target.as_deref() == Some(root) {
            target.take();
            return Err(CurrentWriteError::AfterCommit(PersistenceError::Io(
                std::io::Error::other("injected root sync failure"),
            )));
        }
    }
    sync_directory(root).map_err(CurrentWriteError::AfterCommit)
}

fn sync_directory(path: &Path) -> PersistenceResult<()> {
    #[cfg(unix)]
    {
        std::fs::File::open(path)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn sync_tree(path: &Path) -> PersistenceResult<()> {
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        let metadata = std::fs::symlink_metadata(&entry_path)?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            return Err(PersistenceError::Format(format!(
                "generation contains symlink {}",
                entry_path.display()
            )));
        }
        if file_type.is_dir() {
            sync_tree(&entry_path)?;
        } else if file_type.is_file() {
            std::fs::File::open(&entry_path)?.sync_all()?;
        } else {
            return Err(PersistenceError::Format(format!(
                "generation contains unsupported entry {}",
                entry_path.display()
            )));
        }
    }
    sync_directory(path)
}

fn reject_symlink_components(staging_dir: &Path, relative: &Path) -> PersistenceResult<()> {
    let mut current = staging_dir.to_path_buf();
    for component in relative.components() {
        current.push(component.as_os_str());
        if let Ok(metadata) = std::fs::symlink_metadata(&current) {
            if metadata.file_type().is_symlink() {
                return Err(PersistenceError::Format(
                    "generation component path must not traverse symlinks".into(),
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
static FAIL_ROOT_SYNC: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

#[cfg(test)]
mod tests {
    use super::{open_current, GenerationWriter, PublicationOutcome, FAIL_ROOT_SYNC};
    use tempfile::tempdir;

    #[test]
    fn publishes_and_resolves_generation() {
        let root = tempdir().unwrap();
        let writer = GenerationWriter::create(root.path()).unwrap();
        let component = writer.path("manifest.json").unwrap();
        std::fs::write(component, b"{}").unwrap();
        writer.sync_file("manifest.json").unwrap();
        let published = writer.publish().unwrap();
        assert!(published.join("manifest.json").is_file());
        assert_eq!(open_current(root.path()).unwrap(), published);
    }

    #[test]
    fn rejects_pointer_traversal() {
        let root = tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("generations")).unwrap();
        std::fs::write(root.path().join("CURRENT"), "../escape\n").unwrap();
        assert!(open_current(root.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_current_generation() {
        let root = tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("generations")).unwrap();
        std::fs::write(root.path().join("CURRENT"), "linked\n").unwrap();
        std::os::unix::fs::symlink("/tmp", root.path().join("generations/linked")).unwrap();
        assert!(open_current(root.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_generation_entries_before_publication() {
        let root = tempdir().unwrap();
        let writer = GenerationWriter::create(root.path()).unwrap();
        std::os::unix::fs::symlink("/etc/passwd", writer.directory().join("manifest")).unwrap();
        let error = writer.publish().unwrap_err();
        assert!(error.to_string().contains("symlink"));
        assert!(!root.path().join("CURRENT").exists());
    }

    #[test]
    fn reports_durability_uncertainty_after_pointer_replacement() {
        let root = tempdir().unwrap();
        let writer = GenerationWriter::create(root.path()).unwrap();
        std::fs::write(writer.path("manifest.json").unwrap(), b"{}").unwrap();
        *FAIL_ROOT_SYNC.lock().unwrap() = Some(root.path().to_path_buf());
        let outcome = writer.publish_with_outcome().unwrap();
        match outcome {
            PublicationOutcome::DurabilityUncertain { published_dir, .. } => {
                assert!(published_dir.join("manifest.json").is_file());
                assert_eq!(open_current(root.path()).unwrap(), published_dir);
            }
            PublicationOutcome::Committed(_) => panic!("injected sync should be uncertain"),
        }
    }
}
