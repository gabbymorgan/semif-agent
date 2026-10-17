"""Voice command-gateway adapter.

The gateway is the agent's **command** surface. This adapter is the policy layer
over the neutral `VoiceDaemon` transport in `semif_agent.voice_transport`: it
listens on the local microphone, applies the wake word + VAD endpointing +
speech-to-text, and hands the transcribed command to the scheduler; the reply is
spoken back through text-to-speech. All the routing — gate/score/queue/dispatch,
`needs_input` questions, repairs, approvals — lives in `GatewayService`, exactly
as for the SimpleX and LXMF adapters.

Unlike a messenger transport there is no remote peer and no allowlist: the
microphone is local, so physical access is the authorization. A single
configured `chat_id` identifies the session, and `home_channel` defaults to it
so background notifications (timers, repair offers) are spoken too.

Gateway isolation still holds: this adapter takes commands and speaks replies,
nothing else. It never records, stores, or exposes audio; transcription is
transient and the only artifact is the same text a typed command would carry.
"""

from __future__ import annotations

import queue
import threading

from semif_agent.reading import DEFAULT_CHUNK_CHARS, chunk_text, save_bookmark
from semif_agent.voice_transport import VoiceDaemon

from .base import GatewayAdapter, InboundMessage, OutboundMessage, is_stop_command

#: The stop vocabulary lives in `gateway.base` (shared with the LXMF adapter,
#: which also streams read-aloud text). Kept under the old name for callers.
_is_stop_command = is_stop_command


class VoiceAdapter(GatewayAdapter):
    name = "voice"
    #: Speak the skill's result line only — the spoken front end should not read
    #: out queue/urgency bookkeeping or the `<skill>: ok —` wrapper. Override
    #: with `gateway.voice.result_only: false` for the full scheduler output.
    result_only = True

    def __init__(self, cfg: dict | None = None, trace=None):
        cfg = cfg or {}
        self.chat_id = str(cfg.get("chat_id", "voice") or "voice")
        self.display_name = str(cfg.get("display_name") or self.chat_id)
        #: `run()` blocks on the mic; this daemon owns capture/wake/STT/TTS.
        self.daemon = VoiceDaemon(cfg, trace=trace, name=self.name)
        #: Set by the daemon while a barge-in command is being captured, so the
        #: transcribed utterance is classified as an interrupt.
        self._barged_in = False
        self.daemon.on_barge_in = self._on_barge_in
        self._on_inbound = None
        #: Utterances are handed to a worker thread, not run on the audio
        #: thread: the scheduler is synchronous and can block on the decision
        #: engine, and a slow or failing task must neither stop the mic being
        #: read nor kill the front end. `None` asks the worker to stop.
        self._inbound: "queue.Queue[InboundMessage | None]" = queue.Queue()
        #: The outbound reply queue (set in `run`), so a "stop" can drop replies
        #: that were already queued behind the interrupted one.
        self._outbound: "queue.Queue[OutboundMessage | None] | None" = None
        #: Long-form read-aloud: the adapter streams a saved text through the
        #: daemon's Piper voice one chunk at a time, and any inbound command
        #: interrupts it. `read.enabled` (default on) gates the whole path;
        #: `read.chunk_chars` bounds one spoken chunk (clamped to the speakable
        #: budget so a chunk is never truncated).
        read_cfg = cfg.get("read", {}) or {}
        self.supports_speech_stream = bool(read_cfg.get("enabled", True))
        self._read_chunk_chars = max(
            1, int(read_cfg.get("chunk_chars", DEFAULT_CHUNK_CHARS) or DEFAULT_CHUNK_CHARS)
        )
        self._read_lock = threading.Lock()
        #: Set to ask the active read to stop; swapped per read so a stale read
        #: never observes a new read's stop signal.
        self._read_stop = threading.Event()
        #: The bookmark key of the active read (None when nothing is being read).
        self._read_key: str | None = None

    # ---- requirements ----

    def check_requirements(self) -> tuple[bool, str | None]:
        return self.daemon.check_requirements()

    def set_input_ready(self, ready) -> None:
        """Gate the wake word on the scheduler being able to accept a command.

        `ready` is a callable returning whether a new command may start now
        (`GatewayService.input_ready`). While it returns False the wake word is
        ignored, so a second command cannot be started before the current
        request has reached a result.
        """
        self.daemon.input_ready = ready

    # ---- transport ----

    def run(
        self,
        on_inbound,
        outbound: "queue.Queue[OutboundMessage | None]",
    ) -> None:
        self._on_inbound = on_inbound
        self._outbound = outbound

        def _pump() -> None:
            # Drain the service's replies onto the daemon's speaker. Speaking
            # blocks (it plays audio), so this runs off the listening thread. A
            # read-aloud stream blocks here for as long as the user listens.
            while True:
                message = outbound.get()
                if message is None:
                    self.daemon.stop()
                    return
                if message.stream is not None:
                    try:
                        self._stream(message.stream)
                    except Exception as exc:
                        # A read failure must never kill the outbound pump (and
                        # with it the front end).
                        self.daemon._event(
                            "gateway_read_failed", message=str(exc)[:300]
                        )
                    continue
                self.daemon._event("gateway_reply_dequeued", chars=len(message.text))
                self.daemon.speak(message.text)

        threading.Thread(target=_pump, name="voice-outbound-pump", daemon=True).start()
        threading.Thread(
            target=self._inbound_worker, name="voice-inbound-worker", daemon=True
        ).start()
        try:
            self.daemon.run(self._on_utterance)
        finally:
            self._inbound.put(None)

    def close(self) -> None:
        self.daemon.close()

    # ---- inbound ----

    def _inbound_worker(self) -> None:
        """Run the scheduler off the audio thread, one utterance at a time.

        The scheduler is synchronous and can block on the decision engine; a
        slow task must not stop the mic being read, and a task error is traced
        and spoken rather than propagated — one bad command must not take the
        voice front end down with it.
        """
        while True:
            message = self._inbound.get()
            if message is None:
                return
            self._handle(message)

    def _handle(self, message: InboundMessage) -> None:
        # Any inbound command interrupts an in-progress read-aloud: the user
        # either said "stop" or wants to do something else. The bookmark already
        # points at the chunk that was interrupted, so a later request resumes
        # there.
        was_reading = self._stop_read_for_command(message)
        if message.raw.get("barge_in"):
            if _is_stop_command(message.text):
                # A "stop"/"cancel" after a barge-in aborts the spoken reply and
                # is not a request: the daemon already cut the playback, and any
                # reply still queued behind it is stale output the user stopped.
                self.daemon._event("gateway_barge_in_stop", chars=len(message.text))
                self._drop_pending_speech()
                self.daemon.set_muted(False)
                return
            # A new command after a barge-in: the playback was already aborted,
            # so just run it (and unmute once it has been handled).
            self.daemon._event("gateway_barge_in_command", chars=len(message.text))
            try:
                if self._on_inbound is not None:
                    self._on_inbound(message)
            except Exception as exc:
                self.daemon._event("gateway_handler_failed", message=str(exc)[:300])
                self.daemon.speak("Sorry, something went wrong handling that.")
            finally:
                self.daemon.set_muted(False)
            return
        if was_reading and _is_stop_command(message.text):
            # A bare "stop" heard between chunks (not a barge-in) stops the
            # reading and is not submitted as a request.
            self.daemon._event("gateway_read_stopped", chars=len(message.text))
            self._drop_pending_speech()
            return
        if self._on_inbound is None:
            return
        try:
            self._on_inbound(message)
        except Exception as exc:
            self.daemon._event("gateway_handler_failed", message=str(exc)[:300])
            self.daemon.speak("Sorry, something went wrong handling that.")

    # ---- read-aloud (long-form streaming) ----

    def _read_active(self) -> bool:
        with self._read_lock:
            return self._read_key is not None

    def _stop_read_for_command(self, message: InboundMessage) -> bool:
        """Stop an active read when a command arrives; report whether one was."""
        if not self._read_active():
            return False
        reason = "barge_in" if message.raw.get("barge_in") else "new_command"
        self._stop_reading(reason)
        return True

    def _stop_reading(self, reason: str) -> None:
        """Ask the active read to stop at the next chunk boundary.

        The bookmark was already written for the chunk being spoken, so the
        position is correct without any extra bookkeeping here.
        """
        with self._read_lock:
            if self._read_key is None:
                return
            key = self._read_key
            self._read_stop.set()
        self.daemon._event("gateway_read_stop", key=key, reason=reason)

    def _stream(self, read) -> None:
        """Stream a read-aloud request through the daemon's Piper voice.

        Chunks are spoken one at a time; the bookmark is written before each
        chunk so a stop (or a crash) leaves the position at the interrupted
        chunk. The read ends when the text is finished or the user interrupts
        (a stop flag set by `_handle`, or a barge-in muting playback).
        """
        with self._read_lock:
            self._read_stop.set()  # supersede any prior read
            stop = threading.Event()
            self._read_stop = stop
            self._read_key = read.key
        try:
            try:
                with open(read.path, "r", encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
            except OSError as exc:
                self.daemon._event(
                    "gateway_read_failed", key=read.key, message=str(exc)[:200]
                )
                return
            chunk_chars = min(
                self._read_chunk_chars,
                max(1, int(read.chunk_chars or self._read_chunk_chars)),
            )
            # Never exceed the daemon's speakable budget, or a chunk would be
            # truncated and text would be lost.
            speak_limit = int(getattr(self.daemon, "max_speak_chars", 0) or 0)
            if speak_limit > 0:
                chunk_chars = min(chunk_chars, speak_limit)
            self.daemon._event(
                "gateway_read_start", key=read.key, start=read.start, chars=len(text)
            )
            spoken = 0
            for offset, chunk in chunk_text(text, read.start, chunk_chars):
                if stop.is_set() or self.daemon.stopping():
                    break
                save_bookmark(read.bookmarks_path, read.key, offset, read.title)
                self.daemon.speak(chunk)
                spoken += 1
                # A barge-in mutes playback synchronously on the audio thread;
                # break promptly so we do not skip through suppressed chunks
                # before the transcribed "stop" reaches `_handle`.
                if stop.is_set() or self.daemon.muted():
                    break
            if stop.is_set() or self.daemon.stopping():
                self.daemon._event("gateway_read_interrupted", key=read.key, chunks=spoken)
            else:
                # Finished: park the bookmark at the end so a later request
                # starts over rather than replaying the final chunk.
                save_bookmark(read.bookmarks_path, read.key, len(text), read.title)
                self.daemon._event("gateway_read_done", key=read.key, chunks=spoken)
        finally:
            with self._read_lock:
                if self._read_stop is stop:
                    self._read_key = None

    def _drop_pending_speech(self) -> None:
        """Discard replies already queued behind an interrupted one.

        The daemon aborted the in-flight playback; anything waiting on the
        outbound queue is output the user asked to stop, so it is dropped rather
        than spoken next.
        """
        outbound = self._outbound
        if outbound is None:
            return
        dropped = 0
        while True:
            try:
                message = outbound.get_nowait()
            except queue.Empty:
                break
            if message is None:
                outbound.put(None)  # preserve the pump's shutdown sentinel
                break
            dropped += 1
        if dropped:
            self.daemon._event("gateway_speech_dropped", count=dropped)

    def _on_barge_in(self, active: bool) -> None:
        """Daemon callback: mark a barge-in session (True) and its end (False)."""
        self._barged_in = bool(active)

    def _on_utterance(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        barge_in = self._barged_in
        self._barged_in = False
        self.daemon._event(
            "gateway_message",
            contact_id=self.chat_id,
            display_name=self.display_name,
            chars=len(text),
        )
        self._inbound.put(
            InboundMessage(
                text=text,
                chat_id=self.chat_id,
                chat_type="dm",
                contact_id=self.chat_id,
                display_name=self.display_name,
                raw={"barge_in": True} if barge_in else {},
            )
        )
