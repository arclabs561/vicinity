//! Crash-oriented publication of multi-file index generations.

use super::error::{PersistenceError, PersistenceResult};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static GENERATION_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const CURRENT_FILE: &str = "CURRENT";

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

    /// Resolve a safe relative component path inside the staging directory.
    pub fn path(&self, relative: impl AsRef<Path>) -> PersistenceResult<PathBuf> {
        let relative = relative.as_ref();
        validate_relative_path(relative)?;
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
        sync_directory(&self.staging_dir)
    }

    /// Publish this generation by atomically replacing the root `CURRENT`
    /// pointer. The old pointer remains intact if any pre-publication step
    /// fails.
    pub fn publish(self) -> PersistenceResult<PathBuf> {
        self.sync()?;
        let published_dir = self.generations_dir.join(&self.generation_id);
        std::fs::rename(&self.staging_dir, &published_dir)?;
        sync_directory(&self.generations_dir)?;
        write_current(&self.root, &self.generation_id)?;
        Ok(published_dir)
    }
}

/// Resolve the directory named by the root `CURRENT` pointer.
pub fn open_current(root: impl AsRef<Path>) -> PersistenceResult<PathBuf> {
    let root = root.as_ref();
    let current = std::fs::read_to_string(root.join(CURRENT_FILE))?;
    let generation_id = current.trim();
    validate_generation_id(generation_id)?;
    let path = root.join("generations").join(generation_id);
    if !path.is_dir() {
        return Err(PersistenceError::NotFound(format!(
            "current generation {}",
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

fn write_current(root: &Path, generation_id: &str) -> PersistenceResult<()> {
    let current = root.join(CURRENT_FILE);
    let tmp = root.join(format!(
        ".CURRENT.tmp.{}.{}",
        std::process::id(),
        GENERATION_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    {
        let file = std::fs::File::create(&tmp)?;
        use std::io::Write;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(generation_id.as_bytes())?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    let result = std::fs::rename(&tmp, &current);
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result?;
    sync_directory(root)
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

#[cfg(test)]
mod tests {
    use super::{open_current, GenerationWriter};
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
}
