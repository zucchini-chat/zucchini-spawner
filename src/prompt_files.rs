//! A resident reads its appended system prompt (our instructions and the
//! chat's user system prompt) from a private file, not from its argv, where
//! another agent's `pkill -f <word>` could match it. One file per spawn,
//! deleted when that spawn's reader exits; boot clears what a crashed spawner
//! left behind.

use std::io;
use std::path::{Path, PathBuf};

fn dir() -> PathBuf {
    crate::zucchini_spawner_dir().join("prompts")
}

/// A written prompt file, deleted on drop.
pub struct PromptFile(PathBuf);

impl PromptFile {
    /// A new file for one spawn of `chat_id`, so a reader that outlives its
    /// teardown can't delete the next spawn's file.
    pub fn write(chat_id: &str, contents: &str) -> io::Result<Self> {
        write_in(&dir(), chat_id, contents)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for PromptFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn write_in(dir: &Path, chat_id: &str, contents: &str) -> io::Result<PromptFile> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(format!("{chat_id}-{}.txt", uuid::Uuid::now_v7()));
    crate::atomic::atomic_write_private(&path, contents.as_bytes())?;
    Ok(PromptFile(path))
}

/// Call only while holding the single-instance lock, before any agent spawns.
pub fn clear() {
    let _ = std::fs::remove_dir_all(dir());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_spawn_gets_its_own_private_file_until_dropped() {
        let dir = std::env::temp_dir().join(format!("zucchini-prompts-{}", uuid::Uuid::now_v7()));
        let first = write_in(&dir, "chat", "be terse").unwrap();
        let second = write_in(&dir, "chat", "be terse").unwrap();
        assert_ne!(first.path(), second.path());
        assert_eq!(std::fs::read_to_string(first.path()).unwrap(), "be terse");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(first.path())
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let path = first.path().to_path_buf();
        drop(first);
        assert!(!path.exists());
        assert!(second.path().exists());
        drop(second);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
