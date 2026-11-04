//! Resident agents run in their own process group, so they outlive a spawner
//! that crashed or was SIGKILLed. Each live resident leaves a
//! `residents/<pgid>` file holding its leader's start time; the next boot kills
//! those groups. The start time guards against pid reuse. macOS only (see
//! `sys::process_start_time`): elsewhere nothing is recorded.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use tracing::{info, warn};

use crate::sys;

fn dir() -> PathBuf {
    crate::zucchini_spawner_dir().join("residents")
}

pub fn record(pgid: u32) {
    record_in(&dir(), pgid);
}

pub fn forget(pgid: u32) {
    let _ = std::fs::remove_file(dir().join(pgid.to_string()));
}

/// How long orphaned groups get to exit on SIGTERM before SIGKILL. Claude
/// reaps its Bash-tool children (they run in groups of their own) only on
/// SIGTERM.
const SIGTERM_GRACE: Duration = Duration::from_secs(5);

/// Kill the process groups a previous spawner left behind. Call only while
/// holding the single-instance lock, before any agent spawns.
pub fn sweep() {
    sweep_in(&dir(), SIGTERM_GRACE);
}

fn record_in(dir: &Path, pgid: u32) {
    let Some(start) = sys::process_start_time(pgid) else {
        return;
    };
    if let Err(e) = std::fs::create_dir_all(dir)
        .and_then(|()| std::fs::write(dir.join(pgid.to_string()), start.to_string()))
    {
        warn!(error = %e, pgid, "failed to record resident process group");
    }
}

fn sweep_in(dir: &Path, grace: Duration) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut groups = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let pgid = path
            .file_name()
            .and_then(|n| n.to_str()?.parse::<u32>().ok())
            .filter(|&pgid| pgid > 1);
        let recorded = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        if let (Some(pgid), Some(start)) = (pgid, recorded) {
            // A dead leader's pid can't be reused while its group lives, so
            // no such process means any group members left are ours.
            let ours = match sys::process_start_time(pgid) {
                Some(now) => now == start,
                None => !sys::process_exists(pgid),
            };
            if ours {
                info!(pgid, "terminating orphaned resident agent process group");
                sys::terminate_process_group(pgid);
                groups.push(pgid);
            }
        }
        let _ = std::fs::remove_file(&path);
    }
    let deadline = Instant::now() + grace;
    while !groups.is_empty() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
        groups.retain(|&pgid| sys::process_group_exists(pgid));
    }
    for pgid in groups {
        warn!(pgid, "orphaned process group survived SIGTERM, killing");
        sys::kill_process_group(pgid);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use std::io::BufRead;
    use std::os::unix::process::{CommandExt, ExitStatusExt};

    const GRACE: Duration = Duration::from_millis(200);

    fn sleeper() -> std::process::Child {
        std::process::Command::new("sleep")
            .arg("30")
            .process_group(0)
            .spawn()
            .expect("spawn sleep")
    }

    #[test]
    fn sweep_kills_a_recorded_group_but_not_a_reused_pid() {
        let dir = std::env::temp_dir().join(format!("zucchini-orphans-{}", std::process::id()));
        let mut orphan = sleeper();
        let mut reused = sleeper();
        record_in(&dir, orphan.id());
        std::fs::write(dir.join(reused.id().to_string()), "1").unwrap();

        sweep_in(&dir, GRACE);

        assert_eq!(orphan.wait().unwrap().signal(), Some(libc::SIGTERM));
        assert!(
            reused.try_wait().unwrap().is_none(),
            "start time mismatch spares it"
        );
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        reused.kill().unwrap();
        let _ = reused.wait();
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn sweep_kills_a_group_whose_leader_died() {
        use std::io::Read;
        let dir = std::env::temp_dir().join(format!(
            "zucchini-orphans-leaderless-{}",
            std::process::id()
        ));
        let mut leader = std::process::Command::new("sh")
            .args(["-c", "sleep 30 >/dev/null & echo $!"])
            .process_group(0)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut out = String::new();
        leader
            .stdout
            .take()
            .unwrap()
            .read_to_string(&mut out)
            .unwrap();
        let member: u32 = out.trim().parse().unwrap();
        leader.wait().unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(leader.id().to_string()), "1").unwrap();

        sweep_in(&dir, GRACE);

        let gone = (0..100).any(|_| {
            std::thread::sleep(Duration::from_millis(20));
            !sys::process_exists(member)
        });
        assert!(gone, "the leaderless group's member was killed");
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn sweep_kills_a_group_that_ignores_sigterm() {
        let dir =
            std::env::temp_dir().join(format!("zucchini-orphans-stubborn-{}", std::process::id()));
        let mut stubborn = std::process::Command::new("sh")
            .args(["-c", "trap '' TERM; echo ready; exec sleep 30"])
            .process_group(0)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let mut ready = String::new();
        std::io::BufReader::new(stubborn.stdout.take().unwrap())
            .read_line(&mut ready)
            .unwrap();
        record_in(&dir, stubborn.id());

        sweep_in(&dir, GRACE);

        assert_eq!(stubborn.wait().unwrap().signal(), Some(libc::SIGKILL));
        let _ = std::fs::remove_dir(&dir);
    }
}
