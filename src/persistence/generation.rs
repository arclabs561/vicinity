//! Crash-oriented publication of multi-file index generations.

use super::error::{PersistenceError, PersistenceResult};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static GENERATION_SEQUENCE: AtomicU64 = AtomicU64::new(0);
const CURRENT_FILE: &str = "CURRENT";
const PUBLISH_LOCK_FILE: &str = ".PUBLISH.lock";
const INVENTORY_FILE: &str = "GENERATION.json";

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
    expected_current: Option<String>,
}

impl GenerationWriter {
    /// Create a unique hidden staging directory below `root`.
    pub fn create(root: impl AsRef<Path>) -> PersistenceResult<Self> {
        let root = root.as_ref().to_path_buf();
        let generations_dir = root.join("generations");
        std::fs::create_dir_all(&generations_dir)?;
        let expected_current = match std::fs::read_to_string(root.join(CURRENT_FILE)) {
            Ok(value) => Some(value.trim().to_string()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let generation_id = unique_generation_id()?;
        let staging_dir = generations_dir.join(format!(".staging-{generation_id}"));
        std::fs::create_dir(&staging_dir)?;
        Ok(Self {
            root,
            generations_dir,
            staging_dir,
            generation_id,
            expected_current,
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
        let _lock = PublicationLock::acquire(&self.root)?;
        verify_expected_current(&self.root, self.expected_current.as_deref())?;
        #[cfg(test)]
        if let Some(error) = take_injected_failure(&self.root, FaultPoint::BeforeStagingSync) {
            return Err(error);
        }
        write_inventory(&self.staging_dir)?;
        sync_tree(&self.staging_dir)?;
        let published_dir = self.generations_dir.join(&self.generation_id);
        std::fs::rename(&self.staging_dir, &published_dir)?;
        #[cfg(test)]
        if let Some(error) = take_injected_failure(&self.root, FaultPoint::AfterGenerationRename) {
            return Err(error);
        }
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

fn verify_expected_current(root: &Path, expected: Option<&str>) -> PersistenceResult<()> {
    let current = match std::fs::read_to_string(root.join(CURRENT_FILE)) {
        Ok(value) => Some(value.trim().to_string()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    if current.as_deref() != expected {
        return Err(PersistenceError::LockFailed {
            resource: root.join(CURRENT_FILE).display().to_string(),
            reason: "CURRENT changed while this generation was being built".into(),
        });
    }
    Ok(())
}

struct PublicationLock {
    file: std::fs::File,
}

impl PublicationLock {
    fn acquire(root: &Path) -> PersistenceResult<Self> {
        let path = root.join(PUBLISH_LOCK_FILE);
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        use fs4::fs_std::FileExt;
        file.try_lock_exclusive().map_err(|error| {
            if error.kind() == std::io::ErrorKind::WouldBlock {
                PersistenceError::LockFailed {
                    resource: path.display().to_string(),
                    reason: "another generation publisher holds the advisory lock".into(),
                }
            } else {
                PersistenceError::Io(error)
            }
        })?;
        Ok(Self { file })
    }
}

impl Drop for PublicationLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
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

/// A kernel-held shared lease protecting a generation from future cleanup.
///
/// The lease is advisory and only has meaning for cleanup code that acquires
/// the matching exclusive lock. Existing callers may continue using
/// [`open_current`] when they do not perform generation retention.
pub struct GenerationLease {
    generation: PathBuf,
    file: std::fs::File,
}

impl GenerationLease {
    /// Return the immutable generation directory protected by this lease.
    pub fn path(&self) -> &Path {
        &self.generation
    }
}

impl Drop for GenerationLease {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

/// Resolve `CURRENT` and acquire a shared lease before returning its path.
///
/// The pointer is read again after locking so a publication racing this open
/// causes a retry instead of returning a lease for a generation that is no
/// longer current.
pub fn open_current_pinned(root: impl AsRef<Path>) -> PersistenceResult<GenerationLease> {
    let root = root.as_ref();
    let leases = root.join("leases");
    std::fs::create_dir_all(&leases)?;
    for _ in 0..3 {
        let current = std::fs::read_to_string(root.join(CURRENT_FILE))?;
        let generation_id = current.trim().to_owned();
        validate_generation_id(&generation_id)?;
        let generation = root.join("generations").join(&generation_id);
        let metadata = std::fs::symlink_metadata(&generation)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(PersistenceError::Format(format!(
                "current generation {} is not a directory",
                generation.display()
            )));
        }
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(leases.join(format!("{generation_id}.lock")))?;
        file.try_lock_shared()
            .map_err(|error| PersistenceError::LockFailed {
                resource: generation.display().to_string(),
                reason: error.to_string(),
            })?;
        let reread = std::fs::read_to_string(root.join(CURRENT_FILE))?;
        if reread.trim() == generation_id {
            return Ok(GenerationLease { generation, file });
        }
        let _ = file.unlock();
    }
    Err(PersistenceError::LockFailed {
        resource: root.join(CURRENT_FILE).display().to_string(),
        reason: "CURRENT changed while acquiring a generation lease".into(),
    })
}

/// Verify every file in a generation against its `GENERATION.json` inventory.
///
/// The inventory is intentionally not required by [`open_current`], so files
/// written by older versions remain loadable. Callers that need integrity
/// verification should use this strict API; a missing or malformed inventory,
/// an unexpected file, a missing file, or a checksum/length mismatch fails.
pub fn verify_generation(generation: impl AsRef<Path>) -> PersistenceResult<()> {
    let generation = generation.as_ref();
    let inventory_path = generation.join(INVENTORY_FILE);
    let bytes = std::fs::read(&inventory_path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            PersistenceError::Format(format!(
                "generation inventory {} is missing",
                inventory_path.display()
            ))
        } else {
            error.into()
        }
    })?;
    let inventory: GenerationInventory = serde_json::from_slice(&bytes).map_err(|error| {
        PersistenceError::Format(format!("invalid generation inventory: {error}"))
    })?;
    if inventory.version != 1 {
        return Err(PersistenceError::Format(format!(
            "unsupported generation inventory version {}",
            inventory.version
        )));
    }
    if inventory
        .files
        .windows(2)
        .any(|pair| pair[0].path >= pair[1].path)
    {
        return Err(PersistenceError::Format(
            "generation inventory paths must be strictly sorted and unique".into(),
        ));
    }
    for file in &inventory.files {
        validate_relative_path(Path::new(&file.path))?;
        if file.path == INVENTORY_FILE {
            return Err(PersistenceError::Format(
                "GENERATION.json cannot inventory itself".into(),
            ));
        }
    }

    let mut actual = Vec::new();
    collect_files(generation, generation, &mut actual)?;
    actual.sort_by(|a, b| a.path.cmp(&b.path));
    for file in &actual {
        let Some(expected) = inventory
            .files
            .binary_search_by(|candidate| candidate.path.cmp(&file.path))
            .ok()
            .map(|index| &inventory.files[index])
        else {
            return Err(PersistenceError::Format(format!(
                "generation file {} is absent from GENERATION.json inventory",
                file.path
            )));
        };
        if expected.length != file.length {
            return Err(PersistenceError::Format(format!(
                "generation file {} length mismatch: expected {}, got {}",
                file.path, expected.length, file.length
            )));
        }
        if expected.crc32 != file.crc32 {
            return Err(PersistenceError::ChecksumMismatch {
                expected: expected.crc32,
                actual: file.crc32,
            });
        }
    }
    if inventory.files.len() != actual.len() {
        return Err(PersistenceError::Format(
            "GENERATION.json lists a file that is missing from the generation".into(),
        ));
    }
    Ok(())
}

/// Resolve and strictly verify the generation named by `CURRENT`.
pub fn verify_current(root: impl AsRef<Path>) -> PersistenceResult<PathBuf> {
    let generation = open_current(root)?;
    verify_generation(&generation)?;
    Ok(generation)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationFile {
    path: String,
    length: u64,
    crc32: u32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerationInventory {
    version: u32,
    files: Vec<GenerationFile>,
}

fn write_inventory(generation: &Path) -> PersistenceResult<()> {
    let inventory_path = generation.join(INVENTORY_FILE);
    if inventory_path.exists() {
        return Err(PersistenceError::Format(
            "GENERATION.json is reserved for the generation inventory".into(),
        ));
    }
    let mut files = Vec::new();
    collect_files(generation, generation, &mut files)?;
    files.sort_by(|a, b| a.path.cmp(&b.path));
    let inventory = GenerationInventory { version: 1, files };
    let bytes = serde_json::to_vec_pretty(&inventory).map_err(|error| {
        PersistenceError::Serialization(format!("generation inventory: {error}"))
    })?;
    std::fs::write(inventory_path, bytes)?;
    Ok(())
}

fn collect_files(
    root: &Path,
    path: &Path,
    files: &mut Vec<GenerationFile>,
) -> PersistenceResult<()> {
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
            collect_files(root, &entry_path, files)?;
        } else if file_type.is_file() {
            let relative = entry_path.strip_prefix(root).map_err(|error| {
                PersistenceError::InvalidState(format!("generation path: {error}"))
            })?;
            let relative = relative.to_str().ok_or_else(|| {
                PersistenceError::Format("generation paths must be valid UTF-8".into())
            })?;
            if relative == INVENTORY_FILE {
                continue;
            }
            let (length, crc32) = checksum_file(&entry_path)?;
            files.push(GenerationFile {
                path: relative.replace(std::path::MAIN_SEPARATOR, "/"),
                length,
                crc32,
            });
        } else {
            return Err(PersistenceError::Format(format!(
                "generation contains unsupported entry {}",
                entry_path.display()
            )));
        }
    }
    Ok(())
}

fn checksum_file(path: &Path) -> PersistenceResult<(u64, u32)> {
    let mut crc = u32::MAX;
    let mut length = 0;
    let mut bytes = [0_u8; 64 * 1024];
    let mut file = std::fs::File::open(path)?;
    loop {
        let read = file.read(&mut bytes)?;
        if read == 0 {
            break;
        }
        length += read as u64;
        for &byte in &bytes[..read] {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0xEDB8_8320 & (!((crc & 1).wrapping_sub(1))));
            }
        }
    }
    Ok((length, !crc))
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
        || path.to_string_lossy().contains('\\')
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
    #[cfg(test)]
    if let Some(error) = take_injected_failure(root, FaultPoint::AfterCurrentTempSync) {
        let _ = std::fs::remove_file(&tmp);
        return Err(CurrentWriteError::BeforeCommit(error));
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FaultPoint {
    BeforeStagingSync,
    AfterGenerationRename,
    AfterCurrentTempSync,
}

#[cfg(test)]
static INJECTED_FAILURES: std::sync::Mutex<Vec<(PathBuf, FaultPoint)>> =
    std::sync::Mutex::new(Vec::new());

#[cfg(test)]
fn take_injected_failure(root: &Path, point: FaultPoint) -> Option<PersistenceError> {
    let mut failures = INJECTED_FAILURES.lock().unwrap();
    failures
        .iter()
        .position(|(failure_root, failure_point)| failure_root == root && *failure_point == point)
        .map(|index| {
            failures.remove(index);
            PersistenceError::Io(std::io::Error::other(format!(
                "injected publication failure at {point:?}"
            )))
        })
}

#[cfg(test)]
mod tests {
    use super::{
        open_current, open_current_pinned, verify_current, verify_generation, FaultPoint,
        GenerationWriter, PublicationLock, PublicationOutcome, CURRENT_FILE, FAIL_ROOT_SYNC,
        INJECTED_FAILURES, PUBLISH_LOCK_FILE,
    };
    use crate::persistence::PersistenceError;
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
        assert_eq!(verify_current(root.path()).unwrap(), published);
    }

    #[test]
    fn pinned_current_holds_shared_lease_and_exposes_generation_path() {
        let root = tempdir().unwrap();
        let published = publish_test_generation(root.path(), b"leased");
        let lease = open_current_pinned(root.path()).unwrap();
        assert_eq!(lease.path(), published.as_path());
        assert!(root
            .path()
            .join("leases")
            .join(format!(
                "{}.lock",
                published.file_name().unwrap().to_string_lossy()
            ))
            .is_file());
        let second = open_current_pinned(root.path()).unwrap();
        assert_eq!(second.path(), published.as_path());
    }

    fn publish_test_generation(root: &std::path::Path, bytes: &[u8]) -> std::path::PathBuf {
        let writer = GenerationWriter::create(root).unwrap();
        std::fs::write(writer.path("manifest.json").unwrap(), bytes).unwrap();
        writer.publish().unwrap()
    }

    #[test]
    fn inventory_is_deterministic_and_detects_tampering() {
        let root = tempdir().unwrap();
        let writer = GenerationWriter::create(root.path()).unwrap();
        std::fs::write(writer.path("z.bin").unwrap(), b"z").unwrap();
        std::fs::write(writer.path("a.bin").unwrap(), b"a").unwrap();
        let published = writer.publish().unwrap();
        let inventory = std::fs::read_to_string(published.join("GENERATION.json")).unwrap();
        assert!(inventory.find("a.bin").unwrap() < inventory.find("z.bin").unwrap());
        verify_generation(&published).unwrap();
        std::fs::write(published.join("a.bin"), b"x").unwrap();
        assert!(matches!(
            verify_generation(&published),
            Err(PersistenceError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn strict_verification_rejects_legacy_generation_without_inventory() {
        let root = tempdir().unwrap();
        let generation = root.path().join("generations/legacy");
        std::fs::create_dir_all(&generation).unwrap();
        std::fs::write(generation.join("manifest"), b"legacy").unwrap();
        std::fs::write(root.path().join(CURRENT_FILE), "legacy\n").unwrap();
        assert!(open_current(root.path()).is_ok());
        assert!(verify_current(root.path()).is_err());
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

    fn publish_baseline(root: &std::path::Path) -> std::path::PathBuf {
        let writer = GenerationWriter::create(root).unwrap();
        std::fs::write(writer.path("manifest.json").unwrap(), b"baseline").unwrap();
        writer.publish().unwrap()
    }

    fn arm_failure(root: &std::path::Path, point: FaultPoint) {
        INJECTED_FAILURES
            .lock()
            .unwrap()
            .push((root.to_path_buf(), point));
    }

    #[test]
    fn prepublication_failure_keeps_current_and_staging_recoverable() {
        let root = tempdir().unwrap();
        let baseline = publish_baseline(root.path());
        let writer = GenerationWriter::create(root.path()).unwrap();
        let staging = writer.directory().to_path_buf();
        std::fs::write(writer.path("manifest.json").unwrap(), b"candidate").unwrap();
        arm_failure(root.path(), FaultPoint::BeforeStagingSync);

        let error = writer.publish().unwrap_err();
        assert!(error.to_string().contains("BeforeStagingSync"));
        assert_eq!(open_current(root.path()).unwrap(), baseline);
        assert!(
            staging.is_dir(),
            "pre-publication failure must retain staging"
        );
        assert_eq!(
            std::fs::read_dir(root.path().join("generations"))
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_type().unwrap().is_dir())
                .count(),
            2
        );
    }

    #[test]
    fn generation_rename_failure_leaves_old_current_and_complete_new_directory() {
        let root = tempdir().unwrap();
        let baseline = publish_baseline(root.path());
        let writer = GenerationWriter::create(root.path()).unwrap();
        let generation_id = writer.generation_id().to_owned();
        std::fs::write(writer.path("manifest.json").unwrap(), b"candidate").unwrap();
        arm_failure(root.path(), FaultPoint::AfterGenerationRename);

        let error = writer.publish().unwrap_err();
        assert!(error.to_string().contains("AfterGenerationRename"));
        assert_eq!(open_current(root.path()).unwrap(), baseline);
        let published = root.path().join("generations").join(generation_id);
        assert!(published.join("manifest.json").is_file());
        assert!(!published.join("manifest.json").read_link().is_ok());
    }

    #[test]
    fn current_temp_sync_failure_keeps_old_pointer_and_removes_temp_file() {
        let root = tempdir().unwrap();
        let baseline = publish_baseline(root.path());
        let writer = GenerationWriter::create(root.path()).unwrap();
        let generation_id = writer.generation_id().to_owned();
        std::fs::write(writer.path("manifest.json").unwrap(), b"candidate").unwrap();
        arm_failure(root.path(), FaultPoint::AfterCurrentTempSync);

        let error = writer.publish().unwrap_err();
        assert!(error.to_string().contains("AfterCurrentTempSync"));
        assert_eq!(open_current(root.path()).unwrap(), baseline);
        assert!(root.path().join("generations").join(generation_id).is_dir());
        assert!(!root
            .path()
            .read_dir()
            .unwrap()
            .filter_map(Result::ok)
            .any(|entry| entry
                .file_name()
                .to_string_lossy()
                .starts_with(".CURRENT.tmp.")));
    }

    #[test]
    fn permanent_lock_path_releases_without_deletion() {
        let root = tempdir().unwrap();
        let lock = PublicationLock::acquire(root.path()).unwrap();
        let lock_path = root.path().join(PUBLISH_LOCK_FILE);
        assert!(lock_path.is_file());
        drop(lock);
        let reacquired = PublicationLock::acquire(root.path()).unwrap();
        assert!(lock_path.is_file());
        drop(reacquired);
        assert!(!root.path().join(CURRENT_FILE).exists());
    }

    #[test]
    fn rejects_stale_publisher_after_newer_generation_commits() {
        let root = tempdir().unwrap();
        let stale = GenerationWriter::create(root.path()).unwrap();
        let current = GenerationWriter::create(root.path()).unwrap();
        std::fs::write(current.path("manifest.json").unwrap(), b"new").unwrap();
        current.publish().unwrap();
        std::fs::write(stale.path("manifest.json").unwrap(), b"stale").unwrap();
        let error = stale.publish().unwrap_err();
        assert!(error.to_string().contains("CURRENT changed"));
    }
}
