"""
ZucchiniAdapter — bridges hermes' agent loop to the zucchini-spawner Rust
process over a single local Unix-domain socket.

================================================================================
task_id isolation — verified end-to-end
================================================================================

The critical isolation question: does an arbitrary ``chat_id`` string passed
as ``task_id`` to ``agent.run_conversation`` survive end-to-end, or does it
get collapsed somewhere?

Answer (verified by reading the source):

  1. ``agent/conversation_loop.py:266-272`` — the per-turn entry point sets
     ``effective_task_id = task_id or str(uuid.uuid4())`` and
     ``agent._current_task_id = effective_task_id``. Any string we pass is
     honoured here verbatim — no regex / suffix munging.

  2. ``tools/terminal_tool.py:967-988`` — ``_resolve_container_task_id``
     collapses task_id to ``"default"`` UNLESS the task_id has a registered
     env override:

         if task_id and task_id in _task_env_overrides:
             return task_id
         return "default"

     This is the gate that decides whether a chat_id reaches
     ``_active_environments[chat_id]`` (per-chat cwd / sandbox) or piles
     into the shared ``_active_environments["default"]`` bucket.

  3. ``tools/file_tools.py:88-127`` — ``_get_live_tracking_cwd(task_id)``
     and ``_resolve_path_for_task(filepath, task_id)`` call
     ``_resolve_container_task_id`` first, then look up
     ``_active_environments[<resolved>].cwd``. So whether file tools chdir
     into the project path depends on the same override gate.

Therefore: BEFORE every turn we call
``register_task_env_overrides(chat_id, {"cwd": project_path})``. This makes
``_resolve_container_task_id(chat_id) == chat_id`` for the duration of the
turn, which in turn isolates ``_active_environments[chat_id]`` and routes
file tools to ``project_path``. We do NOT touch ``TERMINAL_CWD`` (it's a
process-global env var that races across concurrent turns; the per-task
override is the right hook for the Telegram-shape, multi-turn-concurrent
model).

We clean up with ``clear_task_env_overrides(chat_id)`` in a ``finally``
block. Re-registering with a different ``cwd`` on the next turn for the
same chat is safe — it's a plain dict mutation.

================================================================================
Wire format (newline-delimited JSON, one frame per line, both directions)
================================================================================

Socket path: env ``ZUCCHINI_SPAWNER_SOCK`` — REQUIRED. The spawner picks
one path for the whole gateway process (e.g. ``~/.zucchini-spawner/hermes.sock``)
and bakes it into our env on launch. There is no fallback default — if the
env var is missing the plugin bails on connect() with a fatal error.

Telegram-shape topology:

    spawner (Rust)          plugin (this file)            hermes
    ------------------      ------------------------      --------
    one server socket  <->  one client connection         many concurrent
    many chats fanned       (dials out from connect())    asyncio.create_task
    in via chat_id field    multiplexes turns by          per turn, isolated
                            chat_id, isolates state       via task_id=chat_id
                            via hermes task_id            -> _active_environments

Auth: filesystem perms only (socket file is chmod 0600 on bind by the
spawner). No HMAC, no tokens. Whoever can ``open(2)`` the socket can drive
the gateway.

Inbound (spawner -> plugin), one frame per line:

    {"type": "hello"}
        Optional first frame. Plugin replies with a {"type":"hello",
        "version":<plugin_version>} ack. Spawner uses this to confirm
        the socket is alive before fanning user turns through it.

    {"type": "ping"}
        Liveness probe. Plugin replies with {"type":"pong"}.

    {"type":         "turn",
     "chat_id":      str,         # zucchini chat uuid; THIS IS THE HERMES task_id
     "user_prompt":  str,         # the user's message body (plaintext, post-decrypt)
     "project_path": str,         # absolute path to chdir into (worktree-aware)
     "yolo":         bool,        # true => HERMES_YOLO_MODE bypass for this turn
     "model":        str|null,    # override model; null = gateway default
     "channel_prompt": str|null,  # appended to ephemeral_system_prompt
     "resume":       str|null,    # hermes session_id to resume; null = new
     "attachments":  [str]|null}  # absolute paths to local files (pre-resolved)
        Drive one hermes agent turn. Plugin spawns asyncio.create_task per
        turn (no lock, no per-chat queue) — many turns may be in flight at
        once, isolated by task_id=chat_id. Outbound envelopes for this
        turn are tagged with the same chat_id.

    {"type": "stop", "chat_id": str}
        User tapped Stop in the iOS app. Plugin looks up the in-flight
        Task for chat_id and calls agent.interrupt() on it; the resulting
        partial answer + a result envelope with subtype=error still get
        streamed back before the task finishes.

Outbound (plugin -> spawner), one frame per line. Every outbound frame is
wrapped:

    {"chat_id":   str,    # echo of the turn's chat_id (or proactive target)
     "proactive": bool,   # false for turn-driven envelopes; true for cron
     "event":     {...claude-shape stream-json object...}}

The inner ``event`` mirrors what ``claude --output-format stream-json``
emits, so the spawner's existing handle_line in adapter.rs can reuse
LastTokensDedup, claude_assistant_text_envelope, etc. unchanged.

Claude-shape event variants:

    {"type": "system", "subtype": "init", "session_id": str, "tools": []}
        Sent once at turn start (after task_id override registration).

    {"type": "assistant",
     "message": {"content": [{"type": "text", "text": <delta>}],
                 "usage":   {input_tokens, cache_creation_input_tokens,
                             cache_read_input_tokens, output_tokens}}}
        Streaming text delta. One envelope per stream_delta_callback hit.

    {"type": "assistant",
     "message": {"content": [{"type": "tool_use", "id": <id>, "name": <name>,
                              "input": <args_obj>}],
                 "usage":   {zero usage}}}
        Tool start. Fired from tool_executor.py:171-175 tool_start_callback.

    {"type": "user",
     "message": {"content": [{"type": "tool_result", "tool_use_id": <id>,
                              "content": <output_str>, "is_error": <bool>}]}}
        Tool result. Fired from tool_executor.py:415-419 tool_complete_callback.

    {"type": "system", "subtype": "context_tokens", "context_tokens": int}
        Live context counter (mirrors how claude reports it).

    {"type": "result",
     "subtype": "success"|"error",
     "is_error": bool,
     "result": <final_response_str>|null,
     "duration_ms": int,
     "usage": {input_tokens, output_tokens, total_tokens}}
        Terminal envelope. The spawner uses subtype + is_error to update
        chats.agent_running and emit the [result:...] line in the UI.

    {"type": "error", "message": str}
        Fatal turn error (auth failure, import failure, bug). Emitted
        instead of a "result" envelope when the run never produced a
        usable answer.

For proactive (cron-fired) envelopes, ``proactive: true`` and the
``event`` is shaped like the streaming text delta above. The spawner
routes these as unsolicited messages in the chat (PowerSync fans out;
APNs push fires).

================================================================================
Hermes API bypass — borrowed from APIServerAdapter
================================================================================

We do NOT go through ``BasePlatformAdapter._process_message`` / the gateway
``MessageEvent`` pipeline for turn frames. That path only delivers
pre-rendered strings to ``send()`` (e.g. ``"🔧 Bash: 'ls -la'"``), losing
the structured ``(tool_name, input_args, output)`` triple iOS needs for tool
bubbles.

Instead we mirror APIServerAdapter (gateway/platforms/api_server.py:1162-1220)
which constructs an ``AIAgent`` via ``_create_agent`` (api_server.py:851-912)
and runs ``agent.run_conversation`` directly in a worker thread, wiring:

  - ``tool_start_callback(tool_call_id, function_name, function_args)``
    — fired at agent/tool_executor.py:171-175
  - ``tool_complete_callback(tool_call_id, function_name, function_args,
    function_result)``
    — fired at agent/tool_executor.py:415-419
  - ``stream_delta_callback(delta_str | None)``
    — fired per token by the streaming transport

These callbacks run on the agent worker thread; we hand them an
asyncio.Queue and drain it from a writer coroutine via
``loop.call_soon_threadsafe`` (same pattern api_server.py uses for SSE).

For PROACTIVE (cron) deliveries we DO go through ``send()`` — that's the
hook both the in-process cron path (cron/scheduler.py:677, via the
runtime_adapter.send call) and the out-of-process standalone path
(``standalone_sender_fn`` registered on PlatformEntry) end up calling.
Our ``send()`` wraps the text into a proactive envelope and writes it on
the same socket.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

# Hermes core imports. These resolve at plugin-discovery time, after
# hermes_cli/plugins.py:1259-1271 imports our __init__.py via a path-based
# spec loader (the hermes-agent root is already on sys.path).
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)
from gateway.config import Platform, PlatformConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

PLUGIN_VERSION = "0.3.0"

# Newline-delimited JSON has no implicit message boundary; cap the per-line
# read to a generous-but-bounded value so a buggy peer can't OOM us with
# a 5 GB "line". 16 MB lines fit the largest realistic tool result frames.
MAX_LINE_BYTES = 16 * 1024 * 1024

# Reconnect backoff (seconds) — exponential, capped.
RECONNECT_INITIAL_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0

# Per-turn queue depth. ~4096 envelopes ≈ 30s of a chatty agent — far past
# any human-visible drain delay. Bounded so a stuck writer can't grow
# memory unboundedly.
TURN_QUEUE_MAX = 4096

# Max hermes-side iterations per turn. Override via HERMES_MAX_ITERATIONS.
DEFAULT_MAX_ITERATIONS = 90


# ---------------------------------------------------------------------------
# Envelope builders (claude-shape stream-json)
#
# These mirror zucchini-spawner/src/adapter.rs::
#   claude_assistant_envelope / claude_assistant_text_envelope /
#   claude_tool_use_envelope / claude_zero_usage
# so iOS reuses SpawnerMessageDescriber unchanged. Keep this section thin
# and easy to compare against the Rust side — every drift = a parsing bug.
# ---------------------------------------------------------------------------


def _zero_usage() -> Dict[str, int]:
    """Zero usage block in claude's snake_case shape."""
    return {
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": 0,
    }


def _system_init_event(session_id: str) -> Dict[str, Any]:
    return {
        "type": "system",
        "subtype": "init",
        "session_id": session_id,
        "tools": [],
    }


def _assistant_text_event(text: str) -> Dict[str, Any]:
    return {
        "type": "assistant",
        "message": {
            "content": [{"type": "text", "text": text}],
            "usage": _zero_usage(),
        },
    }


# Hermes tool name → (claude tool name iOS knows, {hermes_arg: claude_arg}).
# iOS's `toolSummary()` (native-ios/Zucchini/SpawnerMessage.swift) only renders
# a one-line detail (the `Bash: ./deploy.sh` part) for claude's tool vocabulary,
# keyed on a specific input field (`command`, `file_path`, `pattern`, …). Hermes
# emits its own names (`terminal`, `read_file`, …) so without this remap the
# bubble degrades to the bare name. This mirrors codex's wire-boundary rename in
# `zucchini-spawner/src/adapters/codex.rs::normalize_item_completed`; we keep it
# here because the plugin is where hermes tool knowledge lives and where the
# structured `(name, input)` triple is already in hand. Unmapped tools fall
# through untouched — their native name is still legible, just without a detail.
_TOOL_NAME_MAP: Dict[str, Any] = {
    "terminal": ("Bash", {}),  # input already has `command`
    "read_file": ("Read", {"path": "file_path"}),
    "write_file": ("Write", {"path": "file_path"}),
    "patch": ("Edit", {"path": "file_path"}),
    "search_files": ("Grep", {}),  # input already has `pattern`
    "web_search": ("WebSearch", {}),  # input already has `query`
    "delegate_task": ("Agent", {"goal": "description"}),
}


def _normalize_tool(name: str, input_args: Any) -> Any:
    """Rename a hermes tool name to its claude equivalent and surface the
    summary field iOS reads. Returns ``(claude_name, claude_input)``. The rename
    copies the value under the claude key while leaving the original key in
    place, so nothing is lost and `toolSummary()` finds what it needs."""
    mapped = _TOOL_NAME_MAP.get(name)
    if mapped is None:
        return name, input_args
    claude_name, key_renames = mapped
    if isinstance(input_args, dict) and key_renames:
        input_args = dict(input_args)
        for hermes_key, claude_key in key_renames.items():
            if hermes_key in input_args and claude_key not in input_args:
                input_args[claude_key] = input_args[hermes_key]
    return claude_name, input_args


def _tool_use_event(tool_call_id: str, name: str, input_args: Any) -> Dict[str, Any]:
    name, input_args = _normalize_tool(name, input_args)
    return {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_call_id,
                    "name": name,
                    # Tool args from tool_executor.py are already a Python dict;
                    # they round-trip through JSON cleanly. Non-dict args (rare —
                    # mostly plugin-defined tools that take a bare string) are
                    # passed through untouched; iOS treats `input` as opaque.
                    "input": input_args if input_args is not None else {},
                }
            ],
            "usage": _zero_usage(),
        },
    }


def _tool_result_event(
    tool_call_id: str, output: Any, is_error: bool
) -> Dict[str, Any]:
    if isinstance(output, (dict, list)):
        try:
            output_str = json.dumps(output, ensure_ascii=False)
        except Exception:
            output_str = repr(output)
    else:
        output_str = "" if output is None else str(output)
    return {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": output_str,
                    "is_error": bool(is_error),
                }
            ],
        },
    }


def _system_context_tokens_event(context_tokens: int) -> Dict[str, Any]:
    return {
        "type": "system",
        "subtype": "context_tokens",
        "context_tokens": int(context_tokens),
    }


def _result_event(
    subtype: str,
    result: Optional[str],
    usage: Dict[str, int],
    duration_ms: int,
) -> Dict[str, Any]:
    return {
        "type": "result",
        "subtype": subtype,
        "is_error": subtype == "error",
        "result": result,
        "duration_ms": int(duration_ms),
        "usage": usage,
    }


def _error_event(message: str) -> Dict[str, Any]:
    return {"type": "error", "message": message}


def _wrap(chat_id: str, event: Dict[str, Any], *, proactive: bool = False) -> Dict[str, Any]:
    return {"chat_id": chat_id, "proactive": proactive, "event": event}


# ---------------------------------------------------------------------------
# Per-turn handle
# ---------------------------------------------------------------------------


class _TurnHandle:
    """Wraps the per-turn task + agent so a {"type":"stop"} frame can find it."""

    __slots__ = ("chat_id", "task", "agent", "_interrupted")

    def __init__(self, chat_id: str) -> None:
        self.chat_id = chat_id
        self.task: Optional[asyncio.Task] = None
        self.agent: Any = None
        self._interrupted = False

    def interrupt(self) -> None:
        if self._interrupted:
            return
        self._interrupted = True
        agent = self.agent
        if agent is not None:
            try:
                # AIAgent.interrupt() exists on all hermes versions we target
                # (see api_server.py:1356 references to agent.interrupt()).
                agent.interrupt()
            except Exception:
                logger.debug("zucchini: agent.interrupt() raised", exc_info=True)
        # Belt-and-braces: if interrupt() isn't honored quickly, cancel the
        # asyncio task too — the run_in_executor future will leak (Python
        # can't cancel a worker-thread call) but the turn unblocks.
        task = self.task
        if task is not None and not task.done():
            # Don't cancel yet; let interrupt() drain the result envelope.
            # Caller's _stop handler can choose to escalate.
            pass


# ---------------------------------------------------------------------------
# ZucchiniAdapter
# ---------------------------------------------------------------------------


class ZucchiniAdapter(BasePlatformAdapter):
    """
    Single client connection to the spawner's Unix socket. Telegram-shape:
    one adapter handles all chats, multiplexed by chat_id; per-chat state
    isolation flows through hermes' task_id mechanism via
    register_task_env_overrides.

    Concurrent turns: each "turn" frame spawns asyncio.create_task(_run_turn).
    No locks, no per-chat queues. Hermes' task_id isolates
    _active_environments / _file_ops_cache / per-task read trackers.

    Proactive deliveries: BasePlatformAdapter.send() override emits a
    {"proactive": true, ...} envelope on the same socket. Both the
    in-process cron path (cron/scheduler.py:677) and the out-of-process
    standalone path (PlatformEntry.standalone_sender_fn) route to send().
    """

    def __init__(self, config: PlatformConfig, **_kwargs: Any) -> None:
        # Plugin platform — Platform("zucchini") goes through Platform._missing_
        # (gateway/config.py:130-173) and returns a pseudo-member because we
        # registered "zucchini" with platform_registry.is_registered.
        platform = Platform("zucchini")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}
        raw_path = os.getenv("ZUCCHINI_SPAWNER_SOCK") or extra.get("spawner_sock")
        if not raw_path:
            # connect() will surface this as a fatal error. Empty here lets
            # the rest of __init__ proceed so the adapter registers itself
            # and emits a clean failure on connect.
            self._sock_path: str = ""
        else:
            self._sock_path = os.path.expanduser(raw_path)

        # Runtime state.
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._writer_lock = asyncio.Lock()  # serialise byte writes only
        self._reader_task: Optional[asyncio.Task] = None
        # In-flight turn tasks keyed by chat_id (one per concurrent turn).
        self._turns: Dict[str, _TurnHandle] = {}
        # Connection generation — bumped on every reconnect so per-turn
        # tasks can detect a stale writer and bail.
        self._gen = 0
        self._stopped = False

    @property
    def name(self) -> str:
        return "Zucchini"

    # ------------------------------------------------------------------
    # BasePlatformAdapter required methods
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Dial the spawner's Unix socket and start the read loop."""
        if not self._sock_path:
            logger.error(
                "zucchini: ZUCCHINI_SPAWNER_SOCK is not set; spawner is expected "
                "to pass this in the gateway's env."
            )
            self._set_fatal_error(
                "missing_socket_env",
                "ZUCCHINI_SPAWNER_SOCK env var is required",
                retryable=False,
            )
            return False

        try:
            await self._dial_once()
        except Exception as e:
            logger.error("zucchini: initial dial of %s failed: %s", self._sock_path, e)
            # Start the reconnect loop anyway — the spawner may not be ready
            # yet. ``_run_loop`` handles backoff.
            self._reader_task = asyncio.create_task(
                self._run_loop_with_reconnect(),
                name="zucchini-reader",
            )
            self._mark_connected()
            return True

        # First dial succeeded — start the read loop in the background.
        self._reader_task = asyncio.create_task(
            self._run_loop_with_reconnect(),
            name="zucchini-reader",
        )
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Cancel in-flight turns, close socket, stop the reader."""
        self._stopped = True
        # Snapshot chat_ids BEFORE clearing _turns so the post-cancel cleanup
        # loop below has something to iterate. Previously this loop ran after
        # _turns.clear() and was a no-op, leaving register_task_env_overrides
        # entries behind for any turn whose `finally` block didn't run.
        cancelled_chat_ids: List[str] = []
        # Cancel all in-flight turns.
        for chat_id, handle in list(self._turns.items()):
            cancelled_chat_ids.append(chat_id)
            try:
                handle.interrupt()
            except Exception:
                logger.debug("zucchini: interrupt during shutdown failed for %s",
                             chat_id, exc_info=True)
            task = handle.task
            if task is not None and not task.done():
                task.cancel()
        self._turns.clear()

        await self._close_writer()

        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None

        # Clear any lingering task_env_overrides we registered for this
        # adapter instance (turn finallys should have done this already).
        try:
            from tools.terminal_tool import clear_task_env_overrides
            for chat_id in cancelled_chat_ids:
                clear_task_env_overrides(chat_id)
        except Exception:
            pass

        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        PROACTIVE delivery hook. Both paths route here:

        - In-process cron (cron/scheduler.py:670-723): runner.adapters.get(
          Platform("zucchini")).send(chat_id, content, metadata=...)
        - Out-of-process standalone_sender_fn (registered on PlatformEntry):
          our _standalone_send() opens a fresh asyncio.open_unix_connection,
          but since the gateway IS this process when in-process delivery
          works, the standalone path is rare. (When the gateway dies, the
          spawner has likely already noticed via socket disconnect.)

        We wrap the text as a claude-shape assistant_text event with
        proactive=true and write it on the open socket.
        """
        if self._writer is None:
            return SendResult(
                success=False,
                error="zucchini socket not connected",
                retryable=True,
            )

        event = _assistant_text_event(content)
        envelope = _wrap(chat_id, event, proactive=True)
        try:
            await self._send_envelope(envelope)
        except Exception as e:
            return SendResult(
                success=False,
                error=f"zucchini socket write failed: {e}",
                retryable=True,
            )
        return SendResult(
            success=True,
            message_id=f"zucchini-proactive-{int(time.time() * 1000)}",
            raw_response=envelope,
        )

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """The spawner owns chat metadata; we stub."""
        return {"platform": "zucchini", "chat_id": chat_id, "type": "dm", "name": chat_id}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _dial_once(self) -> None:
        """Open a single connection to the spawner's socket."""
        reader, writer = await asyncio.open_unix_connection(
            path=self._sock_path,
            limit=MAX_LINE_BYTES,
        )
        self._reader = reader
        self._writer = writer
        self._gen += 1
        logger.info("zucchini: connected to %s (gen=%d)", self._sock_path, self._gen)

        # Send hello so the spawner can confirm liveness without sending a
        # frame first.
        try:
            await self._send_envelope({
                "type": "hello",
                "version": PLUGIN_VERSION,
            })
        except Exception:
            logger.debug("zucchini: initial hello write failed", exc_info=True)

    async def _close_writer(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                try:
                    await self._writer.wait_closed()
                except Exception:
                    pass
            except Exception:
                pass
        self._writer = None
        self._reader = None

    async def _run_loop_with_reconnect(self) -> None:
        """Reader loop with exponential-backoff reconnect on disconnect."""
        delay = RECONNECT_INITIAL_DELAY
        while not self._stopped:
            try:
                if self._reader is None:
                    await self._dial_once()
                    delay = RECONNECT_INITIAL_DELAY
                await self._read_loop()
            except asyncio.CancelledError:
                raise
            except (ConnectionRefusedError, FileNotFoundError) as e:
                logger.warning(
                    "zucchini: socket %s unavailable (%s); retrying in %.1fs",
                    self._sock_path, e, delay,
                )
            except Exception:
                logger.exception("zucchini: read loop crashed; reconnecting")

            # Drop in-flight turns on disconnect — the spawner will retry.
            await self._abort_inflight_turns("connection dropped")
            await self._close_writer()

            if self._stopped:
                return

            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)

    async def _read_loop(self) -> None:
        """Read NDJSON frames from the spawner and dispatch."""
        assert self._reader is not None
        while not self._stopped:
            line = await self._read_line(self._reader)
            if line is None:
                logger.info("zucchini: spawner closed the socket")
                return
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("zucchini: malformed frame from spawner: %s", e)
                continue

            ftype = frame.get("type")
            try:
                if ftype == "turn":
                    self._dispatch_turn(frame)
                elif ftype == "stop":
                    self._dispatch_stop(frame)
                elif ftype == "hello":
                    await self._send_envelope({
                        "type": "hello",
                        "version": PLUGIN_VERSION,
                    })
                elif ftype == "ping":
                    await self._send_envelope({"type": "pong"})
                else:
                    logger.warning("zucchini: unknown frame type %r", ftype)
            except Exception:
                logger.exception("zucchini: dispatch failed for type=%s", ftype)

    async def _abort_inflight_turns(self, reason: str) -> None:
        """Cancel + clean up all in-flight turn tasks."""
        for chat_id, handle in list(self._turns.items()):
            try:
                handle.interrupt()
            except Exception:
                pass
            task = handle.task
            if task is not None and not task.done():
                task.cancel()
            # Clear any task_env_override left behind.
            try:
                from tools.terminal_tool import clear_task_env_overrides
                clear_task_env_overrides(chat_id)
            except Exception:
                pass
        self._turns.clear()
        if reason:
            logger.info("zucchini: aborted in-flight turns: %s", reason)

    # ------------------------------------------------------------------
    # Frame dispatch
    # ------------------------------------------------------------------

    def _dispatch_turn(self, frame: Dict[str, Any]) -> None:
        chat_id = frame.get("chat_id") or ""
        user_prompt = frame.get("user_prompt") or ""
        if not chat_id or not user_prompt:
            logger.warning("zucchini: turn frame missing chat_id or user_prompt")
            return

        # If a turn for this chat is already in flight, that's a spawner-side
        # ordering bug — log and refuse rather than racing. Spawner-side
        # design guarantees one user message per chat is in flight at a
        # time (iOS serialises per-chat sends).
        if chat_id in self._turns:
            logger.warning(
                "zucchini: turn already in flight for chat_id=%s; refusing",
                chat_id,
            )
            return

        handle = _TurnHandle(chat_id=chat_id)
        # Spawn the turn task. NOT awaited — many concurrent turns are
        # the whole point of the Telegram-shape model.
        handle.task = asyncio.create_task(
            self._run_turn(frame, handle),
            name=f"zucchini-turn-{chat_id[:8]}",
        )
        self._turns[chat_id] = handle

    def _dispatch_stop(self, frame: Dict[str, Any]) -> None:
        chat_id = frame.get("chat_id") or ""
        if not chat_id:
            logger.warning("zucchini: stop frame missing chat_id")
            return
        handle = self._turns.get(chat_id)
        if handle is None:
            logger.debug("zucchini: stop for unknown chat_id=%s (turn already done?)",
                         chat_id)
            return
        logger.info("zucchini: stop requested for chat_id=%s", chat_id)
        handle.interrupt()

    # ------------------------------------------------------------------
    # Turn execution — the api_server.py:1162-1220 pattern
    # ------------------------------------------------------------------

    async def _run_turn(
        self,
        frame: Dict[str, Any],
        handle: _TurnHandle,
    ) -> None:
        """
        Drive one hermes agent run for one chat. Concurrent with other
        chats' turns; isolated via task_id=chat_id + register_task_env_overrides.
        """
        chat_id = handle.chat_id
        user_prompt = frame.get("user_prompt") or ""
        project_path = frame.get("project_path") or ""
        yolo = bool(frame.get("yolo", False))
        model_override = frame.get("model")
        channel_prompt = frame.get("channel_prompt")
        resume_session_id = frame.get("resume")
        attachments = list(frame.get("attachments") or [])
        own_gen = self._gen
        t_start = time.monotonic()

        if project_path and not os.path.isdir(project_path):
            await self._send_envelope(_wrap(
                chat_id,
                _error_event(f"project_path is not a directory: {project_path}"),
            ))
            self._turns.pop(chat_id, None)
            return

        # ─── Hermes task_id isolation ────────────────────────────────
        # Registering an override for chat_id makes
        # _resolve_container_task_id(chat_id) return chat_id (not "default")
        # for the duration of the turn, which isolates
        # _active_environments[chat_id] and routes file tool cwd through it.
        # See the module docstring "task_id isolation" section for the trace.
        from tools.terminal_tool import (
            register_task_env_overrides,
            clear_task_env_overrides,
            _active_environments,
            _env_lock,
        )

        overrides: Dict[str, Any] = {}
        if project_path:
            overrides["cwd"] = project_path
        # Register even with empty overrides — the mere presence of an entry
        # in _task_env_overrides flips _resolve_container_task_id away from
        # "default" (terminal_tool.py:986-988).
        register_task_env_overrides(chat_id, overrides or {"cwd": os.getcwd()})

        # Hermes session_id — separate from chat_id. Mint upfront so init
        # envelope is well-formed; if caller asked to resume use that id.
        # Note: chats.id IS the claude session id (migration 0019), but
        # hermes session ids look different ("20260528_214220_e6100c"-style)
        # so the spawner stores them in a separate hermes_session_id column.
        session_id = resume_session_id or f"hermes-{uuid.uuid4().hex}"

        loop = asyncio.get_running_loop()
        out_q: asyncio.Queue = asyncio.Queue(maxsize=TURN_QUEUE_MAX)

        # ─── Emit init envelope before the agent starts ──────────────
        await self._send_envelope(_wrap(chat_id, _system_init_event(session_id)))

        # ─── Callbacks (run on the agent worker thread) ──────────────
        # AIAgent fires these synchronously from its worker thread; we hop
        # back to the asyncio loop via loop.call_soon_threadsafe (same
        # pattern api_server.py uses for SSE).

        # Streaming-text accumulator. Zucchini's message-frame invariant is
        # "one messages row per JSON frame, bodies final at insert, never
        # grow" (zucchini-spawner/CLAUDE.md). Emitting one assistant envelope
        # per token delta therefore produced one message bubble PER WORD. We
        # instead buffer deltas and flush them as a SINGLE assistant text
        # envelope at each natural boundary — before a tool call, and at turn
        # end — exactly like `claude --output-format stream-json` emits one
        # whole assistant message per turn segment. `text_parts` /
        # `streamed_any` are mutated only from the agent worker thread (delta /
        # tool_start callbacks are fired synchronously from it) and read on the
        # asyncio loop thread only after the worker has joined, so no lock is
        # needed.
        text_parts: List[str] = []
        streamed_any = [False]

        def _flush_text() -> None:
            # Worker-thread context. Coalesce buffered deltas into one
            # assistant bubble. Whitespace-only segments (e.g. a lone newline
            # before a tool call) are dropped so they don't create empty rows.
            if not text_parts:
                return
            text = "".join(text_parts)
            text_parts.clear()
            if not text.strip():
                return
            envelope = _wrap(chat_id, _assistant_text_event(text))
            try:
                loop.call_soon_threadsafe(out_q.put_nowait, envelope)
            except Exception:
                logger.debug("zucchini: text flush enqueue failed", exc_info=True)

        def _on_delta(delta: Optional[str]) -> None:
            # Filter the None sentinel (api_server.py:1146-1153 uses it to
            # close CLI response boxes before tool execution; here it would
            # prematurely close our stream). Accumulate into the buffer rather
            # than emitting per-token — see `text_parts` above.
            if delta is None or not delta:
                return
            streamed_any[0] = True
            text_parts.append(delta)

        def _on_tool_start(tool_call_id: str, name: str, args: Any) -> None:
            # api_server.py:1175 filters internal tools (name starts with _).
            if not tool_call_id or (isinstance(name, str) and name.startswith("_")):
                return
            # Emit any assistant text that streamed before this tool call as
            # its own bubble, preserving claude's text-then-tool_use ordering.
            _flush_text()
            envelope = _wrap(chat_id, _tool_use_event(tool_call_id, name, args))
            try:
                loop.call_soon_threadsafe(out_q.put_nowait, envelope)
            except Exception:
                logger.debug("zucchini: tool_start enqueue failed", exc_info=True)

        def _on_tool_complete(
            tool_call_id: str,
            name: str,
            args: Any,
            result: Any,
        ) -> None:
            if not tool_call_id or (isinstance(name, str) and name.startswith("_")):
                return
            # tool_executor.py:249 detects errors via _detect_tool_failure; we
            # don't have access to that flag here (the structured callback
            # discards it). Fall back to scanning the result string for the
            # conventional "Error executing tool" prefix.
            # OPEN QUESTION (upstream): propose adding is_error to the
            # tool_complete_callback signature.
            is_error = (
                isinstance(result, str)
                and result.startswith("Error executing tool")
            )
            envelope = _wrap(
                chat_id,
                _tool_result_event(tool_call_id, result, is_error),
            )
            try:
                loop.call_soon_threadsafe(out_q.put_nowait, envelope)
            except Exception:
                logger.debug("zucchini: tool_complete enqueue failed", exc_info=True)

        # ─── Ephemeral system prompt ─────────────────────────────────
        # Gateway normally composes ephemeral_system_prompt from
        # channel_prompt (gateway/run.py:16111-16113). Bypassing
        # _process_message means we own that composition.
        ephemeral_system_prompt: Optional[str] = None
        if channel_prompt and channel_prompt.strip():
            ephemeral_system_prompt = channel_prompt.strip()

        # ─── Attachments → user prompt footer ────────────────────────
        # Hermes' vision-capable tools read files by path; we append the list.
        # iOS will have already rendered the attachment thumbnails — this
        # just tells the agent where they live. Future: wire through to
        # hermes' media_urls path properly.
        full_user_message = user_prompt
        if attachments:
            attach_block = "\n\nFiles attached (paths on this machine):\n" + "\n".join(
                f"- {p}" for p in attachments
            )
            full_user_message = full_user_message + attach_block

        # ─── Gateway session key for YOLO + approval scoping ─────────
        gateway_session_key = f"zucchini:{chat_id}"

        # ─── Build the agent (api_server.py:851-912 body, replicated) ────
        try:
            from run_agent import AIAgent
            from gateway.run import (
                _resolve_runtime_agent_kwargs,
                _resolve_gateway_model,
                _load_gateway_config,
                GatewayRunner,
            )
            from hermes_cli.tools_config import _get_platform_tools
        except Exception as e:
            await self._send_envelope(_wrap(
                chat_id,
                _error_event(f"hermes import failed: {e}"),
            ))
            self._cleanup_turn(chat_id)
            return

        try:
            runtime_kwargs = _resolve_runtime_agent_kwargs()
            reasoning_config = GatewayRunner._load_reasoning_config()
            gateway_model = _resolve_gateway_model()
            user_config = _load_gateway_config()
            enabled_toolsets = sorted(_get_platform_tools(user_config, "api_server"))
            max_iterations = int(os.getenv("HERMES_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS)))
            fallback_model = GatewayRunner._load_fallback_model()
        except Exception as e:
            await self._send_envelope(_wrap(
                chat_id,
                _error_event(f"hermes config resolution failed: {e}"),
            ))
            self._cleanup_turn(chat_id)
            return

        # NOTE: We deliberately do NOT set TERMINAL_CWD. It's a process-global
        # env var that races across concurrent turns; the per-task override
        # (register_task_env_overrides above) is the correct hook for
        # multi-turn-concurrent operation. file_tools._resolve_path_for_task
        # consults _get_live_tracking_cwd(task_id) FIRST and only falls back
        # to TERMINAL_CWD when no env exists for the task.

        # YOLO bypass — set per-session bypass; the env var alone is
        # consulted by static check sites, but session-keyed approval
        # routes also test it. Restore in finally.
        prior_yolo = os.environ.get("HERMES_YOLO_MODE", "_UNSET_")
        if yolo:
            os.environ["HERMES_YOLO_MODE"] = "1"

        try:
            agent = AIAgent(
                model=(model_override or gateway_model),
                **runtime_kwargs,
                max_iterations=max_iterations,
                quiet_mode=True,
                verbose_logging=False,
                ephemeral_system_prompt=ephemeral_system_prompt,
                enabled_toolsets=enabled_toolsets,
                session_id=session_id,
                platform="zucchini",
                stream_delta_callback=_on_delta,
                tool_start_callback=_on_tool_start,
                tool_complete_callback=_on_tool_complete,
                # NOTE: _create_agent (api_server.py:907) passes session_db
                # too. We don't have that handle from a plugin; cross-session
                # memory may be lossy on resume. Tolerable for v1.
                fallback_model=fallback_model,
                reasoning_config=reasoning_config,
                gateway_session_key=gateway_session_key,
            )
        except Exception as e:
            if prior_yolo == "_UNSET_":
                os.environ.pop("HERMES_YOLO_MODE", None)
            else:
                os.environ["HERMES_YOLO_MODE"] = prior_yolo
            await self._send_envelope(_wrap(
                chat_id,
                _error_event(f"AIAgent init failed: {e}\n{traceback.format_exc()}"),
            ))
            clear_task_env_overrides(chat_id)
            self._cleanup_turn(chat_id)
            return

        handle.agent = agent

        # ─── Drainer: pop envelopes off out_q and write them ────────
        done_marker = asyncio.Event()
        drainer = asyncio.create_task(
            self._drain_queue(out_q, done_marker, own_gen),
            name=f"zucchini-drain-{chat_id[:8]}",
        )

        # ─── Run the agent on a worker thread ────────────────────────
        def _run_sync() -> Dict[str, Any]:
            # Replicate api_server.py:3005-3045's session/approval binding so
            # YOLO + approval routing work correctly per-turn.
            from gateway.session_context import clear_session_vars, set_session_vars
            from tools.approval import (
                enable_session_yolo,
                disable_session_yolo,
                reset_current_session_key,
                set_current_session_key,
            )

            approval_token = None
            session_tokens: list = []
            yolo_enabled_here = False
            try:
                approval_token = set_current_session_key(gateway_session_key)
                session_tokens = set_session_vars(
                    platform="zucchini",
                    session_key=gateway_session_key,
                    chat_id=chat_id,
                )
                if yolo:
                    enable_session_yolo(gateway_session_key)
                    yolo_enabled_here = True

                return agent.run_conversation(
                    user_message=full_user_message,
                    conversation_history=[],
                    # task_id == chat_id — this is the WHOLE isolation story.
                    # conversation_loop.py:266-272 uses this for
                    # agent._current_task_id and downstream tool calls.
                    # _resolve_container_task_id keeps it because we
                    # register_task_env_overrides(chat_id, ...) above.
                    task_id=chat_id,
                )
            finally:
                if yolo_enabled_here:
                    try:
                        disable_session_yolo(gateway_session_key)
                    except Exception:
                        pass
                if approval_token is not None:
                    try:
                        reset_current_session_key(approval_token)
                    except Exception:
                        pass
                if session_tokens:
                    try:
                        clear_session_vars(session_tokens)
                    except Exception:
                        pass

        result: Dict[str, Any] = {}
        run_error: Optional[BaseException] = None
        try:
            result = await loop.run_in_executor(None, _run_sync)
        except asyncio.CancelledError:
            run_error = RuntimeError("turn cancelled")
            # Don't re-raise — we still want to emit the result envelope.
        except Exception as e:
            run_error = e
            logger.exception("zucchini: turn crashed for chat %s", chat_id)
        finally:
            # Restore env. Race-free because HERMES_YOLO_MODE is binary —
            # concurrent turns either all want yolo or none do, set at
            # spawner config time per machine. (Sandbox bit varies per
            # user, but that's also stable per machine_users row.)
            if prior_yolo == "_UNSET_":
                os.environ.pop("HERMES_YOLO_MODE", None)
            else:
                os.environ["HERMES_YOLO_MODE"] = prior_yolo
            try:
                clear_task_env_overrides(chat_id)
            except Exception:
                pass
            # Best-effort: drop the per-chat _active_environments entry too
            # so a long-lived gateway doesn't accumulate stale envs across
            # thousands of chats. The tool_cleanup_idle path would catch
            # this eventually, but explicit cleanup keeps RAM bounded.
            try:
                with _env_lock:
                    _active_environments.pop(chat_id, None)
            except Exception:
                pass

        # ─── Flush trailing assistant text as the final bubble ───────
        # The worker thread has joined, so reading the buffer here is safe.
        # Any text streamed after the last tool boundary (or the entire answer
        # for a tool-free turn) is still buffered; emit it as ONE assistant
        # bubble. If the provider never streamed deltas at all (some backends
        # return the whole answer at once), fall back to the run's
        # final_response so the user still sees an answer. We deliberately do
        # NOT fall back when deltas DID stream, to avoid duplicating text that
        # was already flushed at tool boundaries.
        trailing = "".join(text_parts)
        text_parts.clear()
        if not trailing.strip() and not streamed_any[0] and run_error is None:
            trailing = result.get("final_response") or ""
        if trailing.strip():
            await self._enqueue(
                out_q,
                _wrap(chat_id, _assistant_text_event(trailing)),
            )

        # ─── Resolve usage / subtype / error message ─────────────────
        duration_ms = int((time.monotonic() - t_start) * 1000)
        if run_error is not None:
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            subtype = "error"
            # str(run_error) on an OpenAI/anthropic SDK exception is the bare
            # message ("'NoneType' object is not iterable", a 400/404 body,
            # etc.) — terse but the most specific thing we have.
            error_message = str(run_error)
            result_text = error_message
        else:
            usage = {
                "input_tokens": int(getattr(agent, "session_prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(agent, "session_completion_tokens", 0) or 0),
                "total_tokens": int(getattr(agent, "session_total_tokens", 0) or 0),
            }
            ctx_used = usage["input_tokens"] + usage["output_tokens"]
            if ctx_used:
                await self._enqueue(
                    out_q,
                    _wrap(chat_id, _system_context_tokens_event(ctx_used)),
                )

            failed = bool(result.get("failed")) or bool(result.get("partial"))
            subtype = "error" if failed else "success"
            error_message = result.get("error") or ""
            result_text = result.get("final_response") or error_message

        # ─── Surface the failure reason as a visible bubble ──────────
        # iOS renders a result frame as only "[result: error (Ns)]" — the
        # message is dropped (SpawnerMessageDescriber, the shared claude-shape
        # renderer used by claude/codex/hermes alike). claude only *appears* to
        # show raw errors because the claude CLI emits them as separate text
        # frames, which the describer renders verbatim; the result pill itself
        # never shows a message. So to give the hermes user the same
        # visibility we emit the underlying error as its own assistant bubble.
        # Always (not gated on whether text streamed): a partial answer
        # followed by a crash — e.g. a greeting then hermes' OpenAI-SDK codex
        # null-output TypeError — still leaves the user staring at a bare
        # error pill unless we spell out what went wrong. The bubble follows
        # any streamed text, so a real answer is preserved, not clobbered.
        if subtype == "error" and error_message.strip():
            await self._enqueue(
                out_q,
                _wrap(chat_id, _assistant_text_event(f"⚠️ hermes error: {error_message.strip()}")),
            )

        # ─── Emit terminal envelope ──────────────────────────────────
        await self._enqueue(
            out_q,
            _wrap(chat_id, _result_event(subtype, result_text, usage, duration_ms)),
        )

        # Signal the drainer that no more envelopes will arrive, wait for
        # it to flush.
        done_marker.set()
        try:
            await drainer
        except Exception:
            logger.debug("zucchini: drainer exited with error", exc_info=True)

        self._cleanup_turn(chat_id)

    def _cleanup_turn(self, chat_id: str) -> None:
        self._turns.pop(chat_id, None)

    async def _drain_queue(
        self,
        q: asyncio.Queue,
        done_marker: asyncio.Event,
        own_gen: int,
    ) -> None:
        """Pull envelopes from the per-turn queue and write them to the socket."""
        while True:
            if done_marker.is_set() and q.empty():
                return
            try:
                envelope = await asyncio.wait_for(q.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            if self._gen != own_gen:
                # Socket reconnected mid-turn — abandon this stream. The
                # spawner will retry the turn after seeing the disconnect.
                return
            try:
                await self._send_envelope(envelope)
            except (ConnectionResetError, BrokenPipeError):
                logger.info("zucchini: spawner closed socket mid-stream")
                return
            except Exception:
                logger.debug("zucchini: write envelope failed", exc_info=True)
                return

    # ------------------------------------------------------------------
    # Framing helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _read_line(reader: asyncio.StreamReader) -> Optional[str]:
        """Read one NDJSON line. Returns None on clean EOF."""
        try:
            raw = await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError as e:
            if e.partial:
                logger.debug(
                    "zucchini: discarding %d-byte tail without newline",
                    len(e.partial),
                )
            return None
        except asyncio.LimitOverrunError as e:
            await reader.read(e.consumed)
            return None
        except (ConnectionResetError, BrokenPipeError):
            return None
        except Exception:
            return None
        if len(raw) > MAX_LINE_BYTES:
            return None
        return raw.decode("utf-8", errors="replace").rstrip("\n")

    async def _send_envelope(self, envelope: Dict[str, Any]) -> None:
        """Write one NDJSON envelope. Serialized via writer lock so concurrent
        turn drainers don't interleave bytes."""
        if self._writer is None:
            raise ConnectionError("socket not connected")
        line = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n"
        data = line.encode("utf-8")
        async with self._writer_lock:
            if self._writer is None:
                raise ConnectionError("socket not connected")
            self._writer.write(data)
            await self._writer.drain()

    @staticmethod
    async def _enqueue(q: asyncio.Queue, envelope: Dict[str, Any]) -> None:
        await q.put(envelope)


# ---------------------------------------------------------------------------
# Standalone sender — out-of-process delivery (cron in a separate process)
# ---------------------------------------------------------------------------


async def _standalone_send(
    pconfig: Any,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """
    Called by tools/send_message_tool._send_via_adapter when cron runs in a
    separate process from the gateway and the in-process adapter weakref is
    None. We open a one-shot connection to the spawner's socket and write a
    {"type":"proactive_send"} frame; the spawner routes as an unsolicited
    message in the chat.

    Single-socket model: the spawner's socket path is
    ``$ZUCCHINI_SPAWNER_SOCK`` (or platform config ``spawner_sock``). If
    the socket isn't reachable, we fail — the spawner is expected to be up
    for cron deliveries to succeed.
    """
    extra = getattr(pconfig, "extra", {}) or {}
    override = os.getenv("ZUCCHINI_SPAWNER_SOCK") or extra.get("spawner_sock")
    if not override:
        return {"error": "ZUCCHINI_SPAWNER_SOCK not set; cannot find spawner socket"}
    sock_path = os.path.expanduser(override)

    if not os.path.exists(sock_path):
        return {"error": f"zucchini spawner socket not found: {sock_path}"}

    event = _assistant_text_event(message)
    envelope = _wrap(chat_id, event, proactive=True)
    # Wrap in an outer "proactive_send" frame so the spawner's read side
    # can tell standalone deliveries apart from envelopes that flowed from
    # an in-process send(). Same proactive=true bit; same outbound shape.
    # Frame type from plugin perspective is just a normal outbound envelope.
    line = json.dumps(envelope, ensure_ascii=False, separators=(",", ":")) + "\n"

    try:
        reader, writer = await asyncio.open_unix_connection(
            path=sock_path,
            limit=MAX_LINE_BYTES,
        )
    except Exception as e:
        return {"error": f"zucchini standalone connect failed: {e}"}

    try:
        writer.write(line.encode("utf-8"))
        await writer.drain()
        return {
            "success": True,
            "message_id": f"zucchini-standalone-{int(time.time() * 1000)}",
        }
    except Exception as e:
        return {"error": f"zucchini standalone write failed: {e}"}
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


def _check_requirements() -> bool:
    # Stdlib only — always available.
    return True


def _validate_config(_cfg: Any) -> bool:
    # No required config — env var is the authoritative source.
    return True


def _is_connected(_cfg: Any) -> bool:
    # Env var presence is a reasonable static "configured" signal. Real
    # connectivity is reported by _mark_connected/_mark_disconnected at
    # runtime.
    return bool(os.getenv("ZUCCHINI_SPAWNER_SOCK"))


def _env_enablement() -> Optional[dict]:
    out: Dict[str, Any] = {}
    if v := os.getenv("ZUCCHINI_SPAWNER_SOCK"):
        out["spawner_sock"] = v
    if v := os.getenv("ZUCCHINI_CRON_DELIVER"):
        out["home_channel"] = v.split(",")[0].strip()
    return out or None


def register(ctx: Any) -> None:
    """Plugin entry point — called by hermes_cli/plugins.py:_load_plugin."""
    ctx.register_platform(
        name="zucchini",
        label="Zucchini",
        adapter_factory=lambda cfg: ZucchiniAdapter(cfg),
        check_fn=_check_requirements,
        validate_config=_validate_config,
        is_connected=_is_connected,
        required_env=["ZUCCHINI_SPAWNER_SOCK"],
        install_hint="No extra packages needed (stdlib only)",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="ZUCCHINI_CRON_DELIVER",
        standalone_sender_fn=_standalone_send,
        # No allowed_users_env / allow_all_env — auth is filesystem perms.
        max_message_length=0,  # iOS handles long messages
        pii_safe=False,
        emoji="\N{AUBERGINE}",
        allow_update_command=True,
        platform_hint=(
            "You are chatting via the Zucchini iOS app. The user is using "
            "you as a coding agent on their machine. Markdown renders "
            "(code blocks, lists, links). Long file contents are fine; iOS "
            "handles them. To send a file back to the user, use the send_message "
            "tool's attachment hook; do NOT paste large file contents inline."
        ),
    )
