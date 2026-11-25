//! The chat list's copy of an agent row. The backend keeps a chat's last agent
//! row in `chats.last_agent_body`, which every chats sync ships again, though
//! only the clients' list line reads it (while the agent runs). So the writer
//! also sends this short copy, as `preview`, and the backend keeps that instead.

use serde_json::{Map, Value};

/// The list shows one line, so longer strings are cut.
const MAX_CHARS: usize = 200;

/// The `tool_use` input keys a client's one-line tool summary reads.
const TOOL_INPUT_KEYS: [&str; 6] = [
    "description",
    "command",
    "file_path",
    "pattern",
    "query",
    "to",
];

/// A copy of an agent row that every client's list line reads as current
/// clients read the row, every string `cut`: a frame keeps only the keys that
/// line reads (a Bash description as the command), plain text (shown as it
/// is) is cut. `None` when it is not shorter, or for a body with a `{` that
/// is not a JSON object: clients still parse an object after a BOM, and
/// Android also after a `#` or `//` comment.
pub(crate) fn list_preview(content: &str) -> Option<String> {
    let preview = match serde_json::from_str(content) {
        Ok(Value::Object(frame)) => slim_frame(frame)?.to_string(),
        _ if content.contains('{') => return None,
        _ => cut(content).to_string(),
    };
    (preview.len() < content.len()).then_some(preview)
}

/// `None` without a string `type`, or with a top-level `text`: clients read
/// such a body as a message envelope.
fn slim_frame(mut frame: Map<String, Value>) -> Option<Value> {
    if frame.contains_key("text") {
        return None;
    }
    let kept: &[&str] = match frame.get("type")?.as_str()? {
        "assistant" => {
            if let Some(Value::Object(message)) = frame.get_mut("message") {
                message.retain(|key, _| key == "content");
                if let Some(Value::Array(blocks)) = message.get_mut("content") {
                    blocks.retain_mut(slim_block);
                }
            }
            &["message"]
        }
        "result" => &["subtype", "duration_ms"],
        "system" => match frame.get("subtype").and_then(Value::as_str) {
            Some("context_pruned") => &["subtype", "summary"],
            Some("peer_message") => &["subtype", "name", "body"],
            Some("status") => &["subtype", "status"],
            Some("chat_suggestion") => &["subtype", "title", "message"],
            Some("task_finished") => &["subtype", "kind", "status", "description", "summary"],
            _ => &["subtype"],
        },
        _ => &[],
    };
    frame.retain(|key, _| key == "type" || kept.contains(&key.as_str()));
    let mut slim = Value::Object(frame);
    cut_strings(&mut slim);
    Some(slim)
}

/// Keeps a text or tool_use block's list keys and drops any other object
/// block (thinking, ...). A non-object block stays: iOS reads no block of a
/// `content` that has one.
fn slim_block(block: &mut Value) -> bool {
    let Value::Object(block) = block else {
        return true;
    };
    let kept: &[&str] = match block.get("type").and_then(Value::as_str) {
        Some("text") => &["type", "text"],
        Some("tool_use") => &["type", "name", "input"],
        _ => return false,
    };
    block.retain(|key, _| kept.contains(&key.as_str()));
    let bash = block.get("name").and_then(Value::as_str) == Some("Bash");
    if let Some(Value::Object(input)) = block.get_mut("input") {
        input.retain(|key, _| TOOL_INPUT_KEYS.contains(&key.as_str()));
        if bash {
            describe_command(input);
        }
    }
    true
}

/// A Bash call's single-line description replaces its command: current
/// clients show the description, older ones only the command's first line,
/// so all show the description (one with a line break would be cut).
fn describe_command(input: &mut Map<String, Value>) {
    let single_line = input
        .get("description")
        .and_then(Value::as_str)
        .is_some_and(|d| !d.is_empty() && !d.contains(['\n', '\r']));
    if single_line {
        if let Some(description) = input.remove("description") {
            input.insert("command".to_string(), description);
        }
    }
}

fn cut_strings(value: &mut Value) {
    match value {
        Value::String(s) => s.truncate(cut(s).len()),
        Value::Array(items) => items.iter_mut().for_each(cut_strings),
        Value::Object(map) => map.values_mut().for_each(cut_strings),
        _ => {}
    }
}

/// Leading whitespace is kept whole and not counted: a client's list line
/// takes the first (non-empty) line of a command, title or text.
fn cut(s: &str) -> &str {
    let text = s.trim_start();
    let blank = s.len() - text.len();
    text.char_indices()
        .nth(MAX_CHARS)
        .map_or(s, |(end, _)| s.split_at(blank + end).0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn preview(content: &str) -> Option<Value> {
        list_preview(content).map(|p| serde_json::from_str(&p).expect("preview is JSON"))
    }

    #[test]
    fn a_frame_keeps_only_the_keys_its_list_line_reads() {
        let frame = json!({
            "type": "assistant",
            "message": {
                "id": "msg_1",
                "model": "claude-opus-4-6",
                "content": [
                    { "type": "thinking", "thinking": "hmm", "signature": "sig" },
                    { "type": "text", "text": "Done.", "citations": null },
                    { "type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {
                        "command": "make", "description": "Build", "timeout": 60000, "run_in_background": true,
                    } },
                ],
                "usage": { "input_tokens": 3, "cache_read_input_tokens": 41000 },
            },
            "parent_tool_use_id": null,
            "session_id": "s1",
            "uuid": "u1",
        });
        assert_eq!(
            preview(&frame.to_string()),
            Some(json!({
                "type": "assistant",
                "message": { "content": [
                    { "type": "text", "text": "Done." },
                    { "type": "tool_use", "name": "Bash", "input": { "command": "Build" } },
                ] },
            }))
        );
    }

    #[test]
    fn a_single_line_bash_description_becomes_the_command() {
        let slim_input = |name: &str, input: &Value| {
            let frame = json!({ "type": "assistant", "message": { "content": [
                { "type": "tool_use", "id": "toolu_1", "name": name, "input": input },
            ] }, "session_id": "s1" });
            preview(&frame.to_string()).expect("preview")["message"]["content"][0]["input"].clone()
        };
        for (input, command) in [
            (
                json!({ "command": "make", "description": "Build" }),
                "Build",
            ),
            (json!({ "command": "make", "description": "   " }), "   "),
            (json!({ "description": "Build" }), "Build"),
        ] {
            assert_eq!(slim_input("Bash", &input), json!({ "command": command }));
        }
        for description in [
            json!(""),
            json!(5),
            json!("Build\nand test"),
            json!("Build\rand test"),
        ] {
            let input = json!({ "command": "make", "description": description });
            assert_eq!(slim_input("Bash", &input), input);
        }
        let agent = json!({ "description": "Find the bug" });
        assert_eq!(slim_input("Agent", &agent), agent);
    }

    #[test]
    fn a_tool_input_keeps_every_summary_key() {
        let input = json!({
            "description": "d", "command": "c", "file_path": "f", "pattern": "p", "query": "q", "to": "t",
            "content": "x".repeat(500), "old_string": "o", "plan": "p",
        });
        let frame = json!({ "type": "assistant", "message": { "content": [
            { "type": "tool_use", "name": "SendMessage", "input": input },
        ] } });
        assert_eq!(
            preview(&frame.to_string()).expect("preview")["message"]["content"][0]["input"],
            json!({ "description": "d", "command": "c", "file_path": "f", "pattern": "p", "query": "q", "to": "t" })
        );
    }

    #[test]
    fn every_string_is_cut_on_a_char_boundary() {
        let long = |tail: &str| format!("{}{tail}", "a".repeat(199));
        for (text, kept) in [
            (long("👍🏽 and more"), long("👍")),
            (long("e\u{301}tude"), long("e")),
            (
                "日本語".repeat(100),
                "日本語".repeat(67).chars().take(200).collect(),
            ),
        ] {
            let frame = json!({ "type": "assistant", "message": { "content": [{ "type": "text", "text": text }] } });
            let cut = preview(&frame.to_string()).expect("preview");
            assert_eq!(cut["message"]["content"][0]["text"], kept);
        }
        let error = format!("Error: {}", "é".repeat(300));
        assert_eq!(
            list_preview(&error),
            Some(error.chars().take(200).collect())
        );
    }

    #[test]
    fn a_cut_keeps_the_leading_whitespace_whole() {
        let command = |xs| format!("{}make test {}", "\n".repeat(250), "x".repeat(xs));
        assert_eq!(cut(&command(300)), command(190));
        let spaces = "\u{3000} ".repeat(150);
        assert_eq!(cut(&spaces), spaces);
    }

    #[test]
    fn a_result_keeps_its_subtype_and_duration() {
        let frame = r#"{"type":"result","subtype":"error_max_turns","is_error":true,"duration_ms":4100,"origin":{"kind":"task-notification"},"terminal_reason":"max_turns","result":"stopped","errors":["Reached maximum number of turns"]}"#;
        assert_eq!(
            preview(frame),
            Some(json!({ "type": "result", "subtype": "error_max_turns", "duration_ms": 4100 }))
        );
    }

    #[test]
    fn a_system_frame_keeps_what_its_subtype_shows() {
        for (frame, expected) in [
            (
                json!({ "type": "system", "subtype": "context_pruned", "summary": "pruned 3 results · ~12k freed", "count": 3, "freed_tokens": 12000 }),
                json!({ "type": "system", "subtype": "context_pruned", "summary": "pruned 3 results · ~12k freed" }),
            ),
            (
                json!({ "type": "system", "subtype": "peer_message", "name": "Explore", "body": "found it", "from_agent_id": "a1b2c3d4e5f6" }),
                json!({ "type": "system", "subtype": "peer_message", "name": "Explore", "body": "found it" }),
            ),
            (
                json!({ "type": "system", "subtype": "status", "status": "compacting", "session_id": "s1", "uuid": "u1" }),
                json!({ "type": "system", "subtype": "status", "status": "compacting" }),
            ),
            (
                json!({ "type": "system", "subtype": "chat_suggestion", "message": "fix it", "title": "Fix", "project_path": "/p", "agent_kind": "claude", "attachments": [] }),
                json!({ "type": "system", "subtype": "chat_suggestion", "message": "fix it", "title": "Fix" }),
            ),
            (
                json!({ "type": "system", "subtype": "task_finished", "task_id": "a1", "kind": "agent", "status": "finished", "description": "Find callers", "summary": "Found three.", "parent_tool_use_id": "toolu_1", "agent_id": "a0" }),
                json!({ "type": "system", "subtype": "task_finished", "kind": "agent", "status": "finished", "description": "Find callers", "summary": "Found three." }),
            ),
            (
                json!({ "type": "system", "subtype": "init", "cwd": "/p", "tools": ["Bash", "Read"], "session_id": "s1" }),
                json!({ "type": "system", "subtype": "init" }),
            ),
        ] {
            assert_eq!(preview(&frame.to_string()), Some(expected));
        }
    }

    #[test]
    fn any_other_type_keeps_only_its_type() {
        let frame = json!({
            "type": "user",
            "message": { "role": "user", "content": [{ "type": "tool_result", "tool_use_id": "toolu_1", "content": "x".repeat(3000) }] },
            "tool_use_result": { "stdout": "x".repeat(3000) },
        });
        assert_eq!(preview(&frame.to_string()), Some(json!({ "type": "user" })));
    }

    #[test]
    fn plain_text_is_cut() {
        let text = format!(
            "Error: agent exited unexpectedly.\n{}",
            "line\n".repeat(100)
        );
        assert_eq!(list_preview(&text), Some(text.chars().take(200).collect()));
        let array = format!("[{}]", "1,".repeat(200) + "1");
        assert_eq!(
            list_preview(&array),
            Some(array.chars().take(200).collect())
        );
    }

    #[test]
    fn a_body_with_a_brace_that_is_no_frame_has_none() {
        let pad = "x".repeat(300);
        for body in [
            format!("{{not json {pad}"),
            format!(" \n{{\"foo\":\"{pad}\"}}"),
            format!("{{\"type\":5,\"message\":\"{pad}\"}}"),
            format!("{{\"type\":\"assistant\",\"text\":\"{pad}\"}}"),
            format!("{{\"text\":\"\",\"attachments\":[{{\"blob_key\":\"0190a0b0-0000-7000-8000-000000000001\",\"size\":12,\"name\":\"{pad}.txt\"}}]}}"),
            format!("\u{feff}{{\"type\":\"user\",\"x\":\"{pad}\"}}"),
            format!("# note\n{{\"type\":\"user\",\"x\":\"{pad}\"}}"),
            format!("Error: {pad} {{y}}"),
        ] {
            assert_eq!(list_preview(&body), None, "{body}");
        }
    }

    #[test]
    fn a_preview_that_is_not_shorter_has_none() {
        for body in [
            crate::adapter::INTERRUPTED_RESULT,
            "ok",
            "",
            r#"{"type":"assistant"}"#,
        ] {
            assert_eq!(list_preview(body), None, "{body}");
        }
    }

    #[test]
    fn content_ios_can_not_read_stays_unreadable() {
        let frame = json!({ "type": "assistant", "message": { "content": [
            "oops",
            { "type": "redacted_thinking", "data": "x" },
            { "type": "text", "text": "hi", "citations": null },
            { "text": "no type" },
        ] }, "session_id": "s1" });
        assert_eq!(
            preview(&frame.to_string()),
            Some(
                json!({ "type": "assistant", "message": { "content": ["oops", { "type": "text", "text": "hi" }] } })
            )
        );
        for (message, kept) in [
            (json!("x".repeat(300)), json!("x".repeat(200))),
            (
                json!({ "content": "y".repeat(300), "id": "msg_1" }),
                json!({ "content": "y".repeat(200) }),
            ),
            (
                json!({ "content": [{ "type": "tool_use", "name": "Bash", "input": ["z".repeat(300)] }] }),
                json!({ "content": [{ "type": "tool_use", "name": "Bash", "input": ["z".repeat(200)] }] }),
            ),
        ] {
            let wrapped = json!({ "type": "assistant", "message": message, "session_id": "s1" });
            assert_eq!(
                preview(&wrapped.to_string()).expect("preview")["message"],
                kept
            );
        }
    }

    /// A line of `tests/data/list-preview-pairs.jsonl`: synthesized rows of
    /// every shape the spawner writes, for the clients' describer checks (the
    /// file is in the public mirror). `ZUCCHINI_UPDATE_PREVIEW_PAIRS=1`
    /// rewrites the previews.
    #[derive(serde::Serialize, serde::Deserialize)]
    struct Pair {
        name: String,
        frame: String,
        preview: Option<String>,
    }

    #[test]
    fn the_golden_pairs_match_list_preview() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/list-preview-pairs.jsonl");
        let file = std::fs::read_to_string(&path).expect("read the pairs");
        let pairs: Vec<Pair> = file
            .lines()
            .map(|line| serde_json::from_str(line).expect("pair"))
            .collect();
        assert!(!pairs.is_empty(), "no pairs in {}", path.display());
        if std::env::var("ZUCCHINI_UPDATE_PREVIEW_PAIRS").as_deref() != Ok("1") {
            for pair in pairs {
                assert_eq!(list_preview(&pair.frame), pair.preview, "{}", pair.name);
            }
            return;
        }
        let updated: String = pairs
            .into_iter()
            .map(|mut pair| {
                pair.preview = list_preview(&pair.frame);
                serde_json::to_string(&pair).expect("pair JSON") + "\n"
            })
            .collect();
        std::fs::write(&path, updated).expect("write the pairs");
    }
}
