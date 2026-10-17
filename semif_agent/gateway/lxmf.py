"""LXMF (Reticulum) command-gateway adapter.

The gateway is the agent's **command** surface: allowlisted contacts send LXMF
messages to the bot and their text is fed to the scheduler, whose reply is sent
back. All the messaging *UX* surface (reading a contact's messages, composing on
the user's behalf, showing a contact link) would live in a separate bridge
service, never here — see AGENTS.md "gateway isolation".

This adapter is the policy layer over the neutral `LxmfDaemon` transport in
`semif_agent.lxmf_transport`: it applies the allowlist, batches rapid messages,
and hands authorized text to the scheduler. Unlike the SimpleX adapter there is
no external daemon and no event loop — Reticulum runs in-process on its own
threads, so batching uses a per-chat `threading.Timer`.

**Voice commands.** A Sideband push-to-talk message arrives as LXMF content with
an empty body and the audio in `FIELD_AUDIO` (`[audio_mode, bytes]`); the
transport surfaces it as a normalized `audio` dict. When `voice.enabled` is set,
this adapter transcribes that payload to text and dispatches the transcript as an
ordinary command — so the same allowlist, batching-free delivery, and reply path
apply. Both PTT shapes are handled: HQ Opus/OGG goes through
`WhisperTranscriber.transcribe_bytes` (faster-whisper's bundled PyAV decodes the
container), and non-HQ Codec2 is decoded to 8 kHz PCM by `voice_transport.
decode_codec2` (the optional `pycodec2` bindings), wrapped in a WAV, and
transcribed the same way. Transcription needs only `faster-whisper`
(`requirements/transcribe.txt`), never the microphone voice stack; Codec2
additionally needs `pycodec2` (`requirements/codec2.txt`). A missing transcriber,
missing Codec2 decoder, or undecodable mode is traced and reported, never
silently dropped; the audio itself is transient and never stored, exactly like
the voice gateway.

**Voice replies.** When `voice.reply_with_voice` is on (the default), a reply to
a chat that last messaged by voice is synthesized with Piper and sent back as a
Sideband HQ push-to-talk voice message (the reply text is encoded to Opus/OGG
and rides `FIELD_AUDIO`), so a voice user hears the answer. The reply *mode*
follows the chat's most recent inbound message: a text message reverts it to
text replies. It needs `piper-tts` + PyAV (`requirements/tts.txt`), never the
microphone stack; a missing synthesizer, a synthesis failure, or an encode
failure is traced and the reply falls back to text — never dropped. The reply
text is also sent as the message content, so a text-only client still sees it.
"""

from __future__ import annotations

import os
import queue
import threading
import time

from semif_agent.lxmf_transport import LxmfDaemon
from semif_agent.reading import DEFAULT_CHUNK_CHARS, chunk_text, save_bookmark
from semif_agent.voice_transport import (
    OPUS_OGG_BITRATE,
    build_synthesizer,
    build_transcriber,
    check_codec2_requirements,
    check_synthesis_requirements,
    check_transcription_requirements,
    decode_codec2,
    pcm_to_opus_ogg,
    pcm_to_wav,
)

from .base import GatewayAdapter, InboundMessage, OutboundMessage, is_stop_command

#: Sent (once) when a voice message can't be transcribed at all. A spoken or
#: typed command still works, so the user is not left guessing.
VOICE_FAILED_MESSAGE = (
    "Sorry, I couldn't make out that voice message — please resend it or type "
    "your command."
)
#: Sent when the codec is one we can't decode (the raw Opus `AM_OPUS_*` modes
#: and the binding-less 450 Codec2 modes; HQ Opus is the default and works).
VOICE_UNSUPPORTED_MESSAGE = (
    "I can't read that voice format yet — send an HQ (Opus) voice message, or "
    "type your command."
)
#: Sent for a Codec2 voice message when the optional Codec2 decoder is absent.
VOICE_CODEC2_MESSAGE = (
    "I can't read that Codec2 voice message — the codec2 decoder isn't "
    "installed. Send an HQ (Opus) voice message, or type your command."
)


class LxmfAdapter(GatewayAdapter):
    name = "lxmf"
    #: Send the skill's result line only — drop the scheduler's queue/urgency
    #: bookkeeping and the `<skill>: ok —` wrapper, so a reply (text or voice) is
    #: the answer, not the internals. Override with `gateway.lxmf.result_only:
    #: false` for the full scheduler output.
    result_only = True

    def __init__(
        self, cfg: dict | None = None, trace=None, transcriber=None, synthesizer=None
    ):
        cfg = cfg or {}
        self.allowed_users = [str(u) for u in cfg.get("allowed_users", []) or []]
        self.allow_all_users = bool(cfg.get("allow_all_users", False))
        self.batch_delay = float(cfg.get("text_batch_delay", 0.8))
        config_dir = cfg.get("config_dir") or os.path.join(
            ".runtime", "lxmf", "reticulum"
        )
        storage_path = cfg.get("storage_path") or os.path.join(
            ".runtime", "lxmf", "router"
        )
        self.daemon = LxmfDaemon(
            config_dir=config_dir,
            storage_path=storage_path,
            display_name=cfg.get("display_name", "semif"),
            announce_interval=float(cfg.get("announce_interval", 3600)),
            stamp_cost=cfg.get("stamp_cost"),
            desired_method=cfg.get("desired_method", "direct"),
            propagation_node=cfg.get("propagation_node", ""),
            trace=trace,
            name=self.name,
        )
        self.config_dir = config_dir
        self.storage_path = storage_path
        #: Voice-message transcription (`gateway.lxmf.voice`). The transcriber is
        #: built lazily from `voice.stt` (faster-whisper only) on the first
        #: voice message; a test can inject one instead.
        self.voice_cfg = cfg.get("voice", {}) or {}
        self.voice_enabled = bool(self.voice_cfg.get("enabled", False))
        #: Reply to a voice chat with a synthesized voice message (Piper). The
        #: reply mode follows the chat's most recent inbound message.
        self.reply_with_voice = bool(self.voice_cfg.get("reply_with_voice", True))
        self.voice_max_chars = int(self.voice_cfg.get("max_speak_chars", 600) or 0)
        self.voice_bitrate = int(
            self.voice_cfg.get("reply_bitrate", OPUS_OGG_BITRATE) or OPUS_OGG_BITRATE
        )
        self._transcriber = transcriber
        self._transcriber_unavailable = False
        self._synthesizer = synthesizer
        self._synthesizer_unavailable = False
        #: Chats whose most recent inbound message was voice: their replies go
        #: back as voice (until a text message reverts them).
        self._voice_chats: set[str] = set()
        self._voice_queue: "queue.Queue[tuple[dict, dict] | None]" = queue.Queue()
        self._voice_thread: threading.Thread | None = None
        self._voice_lock = threading.Lock()
        self._outbound: "queue.Queue[OutboundMessage | None] | None" = None
        self._buffers: dict[str, str] = {}
        self._buffered: dict[str, dict] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._on_inbound = None
        #: Read-aloud: a book is streamed to a voice chat as a sequence of
        #: voice messages, one Piper-synthesized chunk each, until the user
        #: sends "stop" or a new command. Only a voice chat can receive it, so
        #: a text chat keeps the skill's saved-file summary.
        read_cfg = self.voice_cfg.get("read", {}) or {}
        self._read_enabled = bool(read_cfg.get("enabled", True))
        self._read_chunk_chars = max(
            1, int(read_cfg.get("chunk_chars", DEFAULT_CHUNK_CHARS) or DEFAULT_CHUNK_CHARS)
        )
        #: How many messages may be queued on the daemon ahead of the stream
        #: (backpressure so a stop is prompt).
        self._read_max_lead = max(1, int(read_cfg.get("max_lead", 4) or 4))
        self._read_lock = threading.Lock()
        self._read_stop = threading.Event()
        self._read_key: str | None = None
        self._read_chat: str | None = None

    @property
    def supports_speech_stream(self) -> bool:
        """Whether this adapter can deliver a read-aloud at all.

        Only when voice replies are on: a read-aloud is sent as voice messages.
        Whether a *particular* chat can receive one is `speech_stream_ready`.
        """
        return bool(self.voice_enabled and self.reply_with_voice and self._read_enabled)

    def speech_stream_ready(self, chat: str) -> bool:
        """Whether `chat` should get a read-aloud (it last messaged by voice)."""
        if not self.supports_speech_stream:
            return False
        with self._lock:
            return str(chat) in self._voice_chats

    # ---- requirements ----

    def check_requirements(self) -> tuple[bool, str | None]:
        # Only the transport is mandatory: text commands work without the
        # optional transcription stack, and a voice message then traces
        # `lxmf_voice_unavailable` rather than taking the whole gateway down.
        return self.daemon.check_requirements()

    # ---- authorization ----

    def is_authorized(self, source_hash: str, display_name: str | None) -> bool:
        if self.allow_all_users:
            return True
        candidates = {str(source_hash)}
        if display_name:
            candidates.add(str(display_name))
        if not self.allowed_users:
            return False
        return any(user in candidates for user in self.allowed_users)

    # ---- transport ----

    def run(
        self,
        on_inbound,
        outbound: "queue.Queue[OutboundMessage | None]",
    ) -> None:
        self._on_inbound = on_inbound
        self._outbound = outbound

        def _pump() -> None:
            # Translate the service's OutboundMessage queue onto the daemon's
            # (chat_id, text, audio) queue without letting scheduler work touch
            # the Reticulum callback threads. Synthesis runs here (off the router
            # thread), so a slow Piper render never stalls the transport.
            while True:
                message = outbound.get()
                if message is None:
                    self.daemon.enqueue(None)
                    return
                if message.stream is not None:
                    try:
                        self._stream(message.stream)
                    except Exception as exc:
                        # A read failure must never kill the outbound pump.
                        self.daemon._event(
                            "lxmf_read_failed", message=str(exc)[:300]
                        )
                    continue
                audio = self._voice_reply(message)
                self.daemon.enqueue(message.chat_id, message.text, audio=audio)

        threading.Thread(
            target=_pump, name="lxmf-outbound-pump", daemon=True
        ).start()
        self.daemon.run(self._on_item)

    def close(self) -> None:
        with self._lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            timer.cancel()
        self._voice_queue.put(None)
        self.daemon.close()

    # ---- inbound ----

    def _on_item(self, message: dict) -> None:
        source_hash = str(message.get("source_hash") or "")
        if not source_hash:
            return
        display_name = message.get("display_name")
        if not self.is_authorized(source_hash, display_name):
            self.daemon._event(
                "gateway_denied",
                contact_id=source_hash,
                display_name=display_name,
            )
            return
        audio = message.get("audio")
        if audio:
            # A voice message puts the chat in voice-reply mode (when the reply
            # path is on); a later text message reverts it.
            if self.voice_enabled and self.reply_with_voice:
                self._mark_voice_chat(source_hash)
            self._handle_audio(message, audio)
            return
        self._unmark_voice_chat(source_hash)
        self._schedule_flush(message)

    def _schedule_flush(self, message: dict) -> None:
        chat = str(message.get("source_hash") or "")
        text = message.get("content") or ""
        with self._lock:
            existing = self._buffers.get(chat)
            self._buffers[chat] = f"{existing}\n{text}" if existing else text
            self._buffered[chat] = message
            timer = self._timers.get(chat)
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(self.batch_delay, self._flush, args=(chat,))
            timer.daemon = True
            self._timers[chat] = timer
            timer.start()

    def _flush(self, chat: str) -> None:
        with self._lock:
            text = self._buffers.pop(chat, "")
            message = self._buffered.pop(chat, None)
            self._timers.pop(chat, None)
        if message is None or not text:
            return
        self._deliver(message, text)

    def _deliver(self, message: dict, text: str) -> None:
        """Hand one command to the service (never called on the router thread)."""
        source_hash = str(message.get("source_hash") or "")
        display_name = message.get("display_name")
        # Any command interrupts a read-aloud playing to this chat; a bare
        # "stop"/"cancel" is not submitted as a request.
        if self._stop_read_for_command(source_hash):
            self.daemon._event(
                "lxmf_read_stopped", contact_id=source_hash, chars=len(text)
            )
            if is_stop_command(text):
                return
        self.daemon._event(
            "gateway_message",
            contact_id=source_hash,
            display_name=display_name,
            chars=len(text),
        )
        if self._on_inbound is None:
            return
        self._on_inbound(
            InboundMessage(
                text=text,
                chat_id=source_hash,
                chat_type="dm",
                contact_id=source_hash,
                display_name=display_name,
                raw=message.get("raw") or {},
            )
        )

    # ---- voice messages (Sideband push-to-talk) ----

    def _handle_audio(self, message: dict, audio: dict) -> None:
        """Route a received voice message to transcription, or report why not.

        Runs on the Reticulum callback thread, so it only enqueues: the actual
        (slow) transcription happens on the worker thread, keeping the router
        responsive. An ignored/undecodable payload keeps any caption text.
        """
        source_hash = str(message.get("source_hash") or "")
        codec = audio.get("codec_name")
        content = message.get("content") or ""
        if not self.voice_enabled:
            self.daemon._event(
                "lxmf_voice_ignored", contact_id=source_hash, codec=codec
            )
            if content:
                self._schedule_flush(message)
            return
        kind = audio.get("kind")
        if kind == "codec2":
            # Codec2 needs the optional pycodec2 bindings; report a missing one
            # up front rather than queueing a transcription that can't happen.
            ok, _hint = check_codec2_requirements()
            if not ok:
                self.daemon._event(
                    "lxmf_voice_codec2_unavailable", contact_id=source_hash, codec=codec
                )
                if content:
                    self._schedule_flush(message)
                else:
                    self._notify(message, VOICE_CODEC2_MESSAGE)
                return
        elif kind != "opus_ogg":
            self.daemon._event(
                "lxmf_voice_unsupported_codec", contact_id=source_hash, codec=codec
            )
            if content:
                self._schedule_flush(message)
            else:
                self._notify(message, VOICE_UNSUPPORTED_MESSAGE)
            return
        self._ensure_voice_worker()
        self._voice_queue.put((message, audio))

    def _ensure_voice_worker(self) -> None:
        with self._voice_lock:
            if self._voice_thread is not None and self._voice_thread.is_alive():
                return
            self._voice_thread = threading.Thread(
                target=self._voice_worker, name="lxmf-voice-worker", daemon=True
            )
            self._voice_thread.start()

    def _voice_worker(self) -> None:
        """Transcribe queued voice messages off the Reticulum callback thread."""
        while True:
            item = self._voice_queue.get()
            if item is None:
                return
            message, audio = item
            text = self._transcribe(audio)
            content = (message.get("content") or "").strip()
            if not text:
                if content:
                    self._deliver(message, content)
                else:
                    self._notify(message, VOICE_FAILED_MESSAGE)
                continue
            self._deliver(message, f"{content}\n{text}" if content else text)

    def _transcribe(self, audio: dict) -> str:
        """Transcribe one audio payload to text; "" (traced) on any failure."""
        transcriber = self._ensure_transcriber()
        if transcriber is None:
            return ""
        data = audio.get("data") or b""
        if not data:
            self.daemon._event("lxmf_voice_empty", codec=audio.get("codec_name"))
            return ""
        if audio.get("kind") == "codec2":
            # Codec2 decodes to raw 8 kHz PCM; wrap it in a WAV so faster-whisper's
            # PyAV decodes and resamples it like any other container.
            pcm = decode_codec2(data, audio.get("codec2_rate") or 0)
            if not pcm:
                self.daemon._event(
                    "lxmf_voice_codec2_failed", codec=audio.get("codec_name")
                )
                return ""
            payload = pcm_to_wav(pcm, 8000)
        else:
            payload = data
        try:
            text = transcriber.transcribe_bytes(payload)
        except NotImplementedError:
            self.daemon._event(
                "lxmf_voice_transcribe_unsupported", codec=audio.get("codec_name")
            )
            return ""
        except Exception as exc:
            self.daemon._event(
                "lxmf_voice_transcribe_failed", message=str(exc)[:300]
            )
            return ""
        text = (text or "").strip()
        if not text:
            self.daemon._event(
                "lxmf_voice_empty_transcript", codec=audio.get("codec_name")
            )
            return ""
        self.daemon._event(
            "lxmf_voice_transcribed", codec=audio.get("codec_name"), chars=len(text)
        )
        return text

    def _ensure_transcriber(self):
        """Build (once) and return the transcriber, or None (traced) if absent."""
        with self._voice_lock:
            if self._transcriber is not None:
                return self._transcriber
            if self._transcriber_unavailable:
                return None
            ok, hint = check_transcription_requirements()
            if not ok:
                self._transcriber_unavailable = True
                self.daemon._event("lxmf_voice_unavailable", message=hint)
                return None
            try:
                self._transcriber = build_transcriber(self.voice_cfg)
            except Exception as exc:
                self._transcriber_unavailable = True
                self.daemon._event("lxmf_voice_unavailable", message=str(exc)[:300])
                return None
            return self._transcriber

    def _notify(self, message: dict, text: str) -> None:
        """Send a transport-level notice back to the sender (best-effort).

        A transcription failure is reported to the user rather than silently
        dropped; the audio itself is never stored or forwarded.
        """
        chat = str(message.get("source_hash") or "")
        outbound = self._outbound
        if outbound is None or not chat:
            return
        outbound.put(OutboundMessage(chat_id=chat, text=text))

    # ---- voice replies (Piper -> Opus/OGG) ----

    def _mark_voice_chat(self, chat: str) -> None:
        with self._lock:
            self._voice_chats.add(chat)

    def _unmark_voice_chat(self, chat: str) -> None:
        with self._lock:
            self._voice_chats.discard(chat)

    def _voice_reply(self, message) -> bytes | None:
        """Synthesize a voice reply for a voice chat, or None to send text.

        Runs on the outbound pump thread. Returns an OGG/Opus container (a
        Sideband HQ push-to-talk payload) or None — for a text chat, an empty
        reply, or any synthesis/encode failure (traced), in which case the
        caller sends the text reply unchanged.
        """
        if not self.reply_with_voice:
            return None
        chat = str(message.chat_id)
        with self._lock:
            is_voice = chat in self._voice_chats
        if not is_voice:
            return None
        text = (message.text or "").strip()
        if not text:
            return None
        if self.voice_max_chars and len(text) > self.voice_max_chars:
            clipped = text[: self.voice_max_chars].rsplit(" ", 1)[0]
            text = clipped or text[: self.voice_max_chars]
        synthesizer = self._ensure_synthesizer()
        if synthesizer is None:
            return None
        try:
            pcm, sample_rate = synthesizer.synthesize(text)
        except Exception as exc:
            self.daemon._event("lxmf_voice_synth_failed", message=str(exc)[:300])
            return None
        if not pcm:
            self.daemon._event("lxmf_voice_synth_empty")
            return None
        audio = pcm_to_opus_ogg(pcm, sample_rate, self.voice_bitrate)
        if not audio:
            self.daemon._event("lxmf_voice_encode_failed", chars=len(text))
            return None
        self.daemon._event(
            "lxmf_voice_reply", chars=len(text), audio_bytes=len(audio)
        )
        return audio

    def _ensure_synthesizer(self):
        """Build (once) and return the Piper synthesizer, or None (traced)."""
        with self._voice_lock:
            if self._synthesizer is not None:
                return self._synthesizer
            if self._synthesizer_unavailable:
                return None
            ok, hint = check_synthesis_requirements()
            if not ok:
                self._synthesizer_unavailable = True
                self.daemon._event("lxmf_voice_tts_unavailable", message=hint)
                return None
            try:
                self._synthesizer = build_synthesizer(self.voice_cfg)
            except Exception as exc:
                self._synthesizer_unavailable = True
                self.daemon._event("lxmf_voice_tts_unavailable", message=str(exc)[:300])
                return None
            return self._synthesizer

    # ---- read-aloud (a book streamed as voice messages) ----

    def _stop_read_for_command(self, chat: str) -> bool:
        """Stop a read-aloud playing to `chat`; report whether one was active."""
        with self._read_lock:
            if self._read_key is None or self._read_chat != str(chat):
                return False
            self._read_stop.set()
        return True

    def _synthesize_chunk(self, text: str) -> bytes | None:
        """Piper-synthesize one chunk to Opus/OGG, or None (send it as text)."""
        synthesizer = self._ensure_synthesizer()
        if synthesizer is None:
            return None
        try:
            pcm, sample_rate = synthesizer.synthesize(text)
        except Exception as exc:
            self.daemon._event("lxmf_voice_synth_failed", message=str(exc)[:300])
            return None
        if not pcm:
            return None
        audio = pcm_to_opus_ogg(pcm, sample_rate, self.voice_bitrate)
        if not audio:
            self.daemon._event("lxmf_voice_encode_failed", chars=len(text))
            return None
        return audio

    def _stream(self, read) -> None:
        """Stream a read-aloud request as a sequence of voice messages.

        Runs on the outbound pump thread: each chunk is synthesized and queued
        to the daemon, and the bookmark is written before it is sent so a stop
        leaves the position at the interrupted chunk. The read ends when the
        text is finished or the user interrupts (a stop flag set by `_deliver`).
        """
        chat = str(read.source or "").split(":", 1)[-1]
        with self._read_lock:
            self._read_stop.set()  # supersede any prior read
            stop = threading.Event()
            self._read_stop = stop
            self._read_key = read.key
            self._read_chat = chat
        try:
            if not chat or not self.speech_stream_ready(chat):
                return
            try:
                with open(read.path, "r", encoding="utf-8", errors="replace") as handle:
                    text = handle.read()
            except OSError as exc:
                self.daemon._event(
                    "lxmf_read_failed", contact_id=chat, message=str(exc)[:200]
                )
                return
            chunk_chars = min(
                self._read_chunk_chars,
                max(1, int(read.chunk_chars or self._read_chunk_chars)),
            )
            if self.voice_max_chars > 0:
                chunk_chars = min(chunk_chars, self.voice_max_chars)
            self.daemon._event(
                "lxmf_read_start",
                contact_id=chat,
                key=read.key,
                start=read.start,
                chars=len(text),
            )
            sent = 0
            for offset, chunk in chunk_text(text, read.start, chunk_chars):
                # Keep only a few messages queued ahead so a stop takes effect
                # promptly instead of after a large backlog drains.
                while not stop.is_set() and self.daemon.pending_sends() >= self._read_max_lead:
                    time.sleep(0.05)
                if stop.is_set():
                    break
                save_bookmark(read.bookmarks_path, read.key, offset, read.title)
                self.daemon.enqueue(chat, chunk, audio=self._synthesize_chunk(chunk))
                sent += 1
            if stop.is_set():
                self.daemon._event(
                    "lxmf_read_interrupted", contact_id=chat, key=read.key, chunks=sent
                )
            else:
                save_bookmark(read.bookmarks_path, read.key, len(text), read.title)
                self.daemon._event(
                    "lxmf_read_done", contact_id=chat, key=read.key, chunks=sent
                )
        finally:
            with self._read_lock:
                if self._read_stop is stop:
                    self._read_key = None
                    self._read_chat = None
