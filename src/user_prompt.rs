//! The per-chat user system prompt (`chats.system_prompt_files` +
//! `chats.system_prompt`): files and free text appended to the agent's system
//! prompt after the spawner's own instructions.

use std::io::Read;
use std::path::{Path, PathBuf};

/// Unix: the whole command is one `bash -lic` argument and Linux caps a single
/// argument at 128 KB.
const UNIX_MAX_BYTES: usize = 64 * 1024;
/// CreateProcess caps the command line at 32,767 chars; the rest is the
/// spawner's own flags and instructions
/// (`claude::tests::a_full_user_prompt_fits_the_windows_command_line`).
pub(crate) const WINDOWS_MAX_BYTES: usize = 20 * 1024;

/// Cap on what a rendered prompt adds to the command, escaping included
/// (`command_bytes`), and on each file read for it.
pub const MAX_BYTES: usize = if cfg!(windows) {
    WINDOWS_MAX_BYTES
} else {
    UNIX_MAX_BYTES
};

/// Cap on the text an agent sets (`system-prompt set`), the clients' cap on
/// what the user types: its ciphertext must stay under the backend's limit.
pub const MAX_TEXT_BYTES: usize = 8000;

/// Both fields as the user typed them. A spawn knob: the rendered prompt is
/// baked into the process, so a change respawns the resident.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct UserSystemPrompt {
    /// Comma-separated paths.
    pub files: Option<String>,
    pub text: Option<String>,
    /// A column is set but didn't decrypt: nothing can be spawned with this
    /// prompt, since leaving the user's instructions out is not an option.
    pub undecryptable: bool,
}

impl UserSystemPrompt {
    /// The block to append to the system prompt: each file under its path as
    /// typed, then the text. `None` when nothing is set. Byte-stable for
    /// unchanged fields and files (see `AgentAdapter::turn_time_line`). `Err`
    /// is the reason to show in the chat.
    pub fn render(&self, project_path: Option<&str>) -> Result<Option<String>, String> {
        if self.undecryptable {
            return Err("can't decrypt the chat's system prompt".to_string());
        }
        let project = project_path.map(Path::new);
        let home = crate::home_dir_opt();
        let mut blocks = Vec::new();
        for entry in self.files.as_deref().unwrap_or_default().split(',') {
            let entry = entry.trim();
            if entry.is_empty() {
                continue;
            }
            let content = read_file(entry, project, home.as_deref())
                .map_err(|reason| format!("system prompt file `{entry}`: {reason}"))?;
            blocks.push(format!(
                "<file path=\"{entry}\">\n{}\n</file>",
                content.trim_end()
            ));
        }
        if let Some(text) = self
            .text
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty())
        {
            blocks.push(text.to_string());
        }
        if blocks.is_empty() {
            return Ok(None);
        }
        let rendered = format!(
            "The user set these instructions for this chat:\n\n{}",
            blocks.join("\n\n")
        );
        let size = command_bytes(&rendered);
        if size > MAX_BYTES {
            return Err(format!(
                "system prompt with its files takes {} KB, the limit is {} KB",
                size.div_ceil(1024),
                MAX_BYTES / 1024
            ));
        }
        Ok(Some(rendered))
    }
}

/// What `s` occupies in the command.
fn command_bytes(s: &str) -> usize {
    if cfg!(windows) {
        windows_arg_bytes(s)
    } else {
        // `shell_escape` turns each `'` into four bytes.
        s.len() + 3 * s.matches('\'').count()
    }
}

/// Upper bound on `s` as part of a Windows command line: argv quoting
/// backslash-escapes each `"` and doubles the backslashes in front of one.
pub(crate) fn windows_arg_bytes(s: &str) -> usize {
    s.len() + s.matches(['"', '\\']).count()
}

fn read_file(entry: &str, project: Option<&Path>, home: Option<&Path>) -> Result<String, String> {
    let io = |e: std::io::Error| e.to_string();
    let path = resolve(entry, project, home)?.canonicalize().map_err(io)?;
    let file = open_nonblocking(&path).map_err(io)?;
    if !file.metadata().map_err(io)?.is_file() {
        return Err("not a regular file".to_string());
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(io)?;
    if bytes.len() > MAX_BYTES {
        return Err(format!("over the {} KB limit", MAX_BYTES / 1024));
    }
    String::from_utf8(bytes).map_err(|_| "not UTF-8 text".to_string())
}

/// Opening a FIFO for reading would otherwise block until it gets a writer.
fn open_nonblocking(path: &Path) -> std::io::Result<std::fs::File> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::custom_flags(&mut options, libc::O_NONBLOCK);
    options.open(path)
}

/// `~/` is the home dir; a relative path is relative to the project.
fn resolve(entry: &str, project: Option<&Path>, home: Option<&Path>) -> Result<PathBuf, String> {
    if let Some(rest) = entry.strip_prefix("~/") {
        return home
            .map(|h| h.join(rest))
            .ok_or_else(|| "no home directory".to_string());
    }
    let path = Path::new(entry);
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    project
        .map(|p| p.join(path))
        .ok_or_else(|| "the chat has no project folder".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A project dir and a sibling "outside" dir under one temp root.
    struct Dirs {
        root: PathBuf,
        project: String,
    }

    impl Dirs {
        fn new() -> Self {
            let root =
                std::env::temp_dir().join(format!("zucchini-user-prompt-{}", uuid::Uuid::now_v7()));
            std::fs::create_dir_all(root.join("project/docs")).unwrap();
            std::fs::create_dir_all(root.join("outside")).unwrap();
            let project = root.join("project").to_string_lossy().into_owned();
            Self { root, project }
        }

        fn write(&self, rel: &str, content: impl AsRef<[u8]>) -> String {
            let path = self.root.join(rel);
            std::fs::write(&path, content).unwrap();
            path.to_string_lossy().into_owned()
        }

        fn render(&self, files: &str, text: &str) -> Result<String, String> {
            let prompt = UserSystemPrompt {
                files: Some(files.to_string()),
                text: Some(text.to_string()),
                undecryptable: false,
            };
            prompt
                .render(Some(&self.project))
                .map(Option::unwrap_or_default)
        }
    }

    impl Drop for Dirs {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn files_then_text_each_file_under_its_path() {
        let d = Dirs::new();
        d.write("project/rules.md", "be terse\n");
        let abs = d.write("outside/style.md", "no emojis");
        let rendered = d
            .render(
                &format!(" rules.md, ,{abs},docs/../rules.md"),
                "  Reply in French.\n",
            )
            .unwrap();
        assert_eq!(
            rendered,
            format!(
                "The user set these instructions for this chat:\n\n\
                 <file path=\"rules.md\">\nbe terse\n</file>\n\n\
                 <file path=\"{abs}\">\nno emojis\n</file>\n\n\
                 <file path=\"docs/../rules.md\">\nbe terse\n</file>\n\n\
                 Reply in French."
            )
        );
        assert_eq!(
            d.render("rules.md", "").unwrap(),
            "The user set these instructions for this chat:\n\n\
             <file path=\"rules.md\">\nbe terse\n</file>"
        );
    }

    #[test]
    fn nothing_set_renders_nothing() {
        let d = Dirs::new();
        assert_eq!(UserSystemPrompt::default().render(None), Ok(None));
        assert_eq!(d.render(" , ", "  \n").unwrap(), "");
    }

    #[test]
    fn unchanged_fields_and_files_render_the_same_bytes() {
        let d = Dirs::new();
        d.write("project/rules.md", "be terse");
        let render = || d.render("rules.md", "Reply in French.").unwrap();
        assert_eq!(render(), render());
    }

    #[cfg(unix)]
    #[test]
    fn tilde_is_the_home_dir_and_a_relative_path_the_projects() {
        let (project, home) = (Path::new("/proj"), Path::new("/home/u"));
        let resolved = |entry| resolve(entry, Some(project), Some(home)).unwrap();
        assert_eq!(resolved("~/notes/a.md"), Path::new("/home/u/notes/a.md"));
        assert_eq!(resolved("docs/a.md"), Path::new("/proj/docs/a.md"));
        assert_eq!(resolved("../a.md"), Path::new("/proj/../a.md"));
        assert_eq!(resolved("/etc/a.md"), Path::new("/etc/a.md"));
        assert!(resolve("a.md", None, Some(home)).is_err());
        assert!(resolve("~/a.md", Some(project), None).is_err());
    }

    #[test]
    fn a_bad_file_fails_the_render_and_is_named() {
        let d = Dirs::new();
        d.write("project/ok.md", "fine");
        d.write("project/binary.bin", [0xff, 0xfe, 0x00]);
        let err = d.render("ok.md,missing.md", "text").unwrap_err();
        assert!(
            err.starts_with("system prompt file `missing.md`: "),
            "{err}"
        );
        let err = d.render("binary.bin", "").unwrap_err();
        assert_eq!(err, "system prompt file `binary.bin`: not UTF-8 text");
    }

    #[test]
    fn only_a_regular_file_is_read() {
        let d = Dirs::new();
        let err = d.render("docs", "").unwrap_err();
        assert_eq!(err, "system prompt file `docs`: not a regular file");
    }

    #[cfg(unix)]
    #[test]
    fn a_device_or_a_fifo_is_refused_without_blocking() {
        let d = Dirs::new();
        let fifo = d.root.join("project/fifo");
        let c_path = std::ffi::CString::new(fifo.to_str().unwrap()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(c_path.as_ptr(), 0o600) }, 0);
        for special in ["fifo", "/dev/zero"] {
            let err = d.render(special, "").unwrap_err();
            assert!(err.ends_with(": not a regular file"), "{err}");
        }
    }

    #[test]
    fn an_undecryptable_prompt_fails_the_render() {
        let prompt = UserSystemPrompt {
            text: Some("be terse".to_string()),
            undecryptable: true,
            ..Default::default()
        };
        assert_eq!(
            prompt.render(None).unwrap_err(),
            "can't decrypt the chat's system prompt"
        );
    }

    #[test]
    fn windows_escaping_counts_quotes_and_backslashes() {
        assert_eq!(windows_arg_bytes("plain"), 5);
        assert_eq!(windows_arg_bytes(r#"a "b" c:\d\"#), 11 + 4);
    }

    #[test]
    fn the_cap_covers_files_and_text_together() {
        let d = Dirs::new();
        d.write("project/half.md", "x".repeat(MAX_BYTES / 2));
        d.write("project/big.md", "x".repeat(MAX_BYTES + 1));
        assert!(d.render("half.md", "short").is_ok());
        let err = d.render("half.md", &"y".repeat(MAX_BYTES / 2)).unwrap_err();
        assert!(err.contains("the limit is"), "{err}");
        let err = d.render("big.md", "").unwrap_err();
        assert!(
            err.starts_with("system prompt file `big.md`: over the "),
            "{err}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn the_cap_counts_shell_escaping() {
        let d = Dirs::new();
        assert!(d.render("", &"x".repeat(MAX_BYTES / 2)).is_ok());
        assert!(d.render("", &"'".repeat(MAX_BYTES / 2)).is_err());
    }
}
