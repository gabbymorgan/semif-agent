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

from semif_agent.voice_transport import VoiceDaemon

from .base import GatewayAdapter, InboundMessage, OutboundMessage

#: Words/phrases that abort a spoken reply when heard after a barge-in. The
#: match is on the whole normalized utterance (so "stop the timer" stays a real
#: command) with a first-word fallback only for a short utterance with trailing
#: filler ("stop please", "cancel that").
_STOP_PHRASES = {
    "stop",
    "stop it",
    "stop talking",
    "stop please",
    "cancel",
    "cancel that",
    "cancel please",
    "quiet",
    "quiet down",
    "be quiet",
    "silence",
    "hush",
    "shut up",
    "never mind",
    "nevermind",
    "enough",
}
_STOP_FIRST_WORDS = {"stop", "cancel", "quiet", "silence", "hush", "shut", "enough"}
_STOP_FILLER = {"please", "now", "it", "that", "talking", "down", "up", "already"}

#: Characters stripped from a spoken utterance's edges before matching.
_PUNCT = " \t\r\n.,!?;:\"'()[]{}"


def _is_stop_command(text: str) -> bool:
    """Whether an interrupt utterance means "stop talking" (vs a new command).

    Deterministic vocabulary, no model: a bare "stop"/"cancel" (or a short
    variant like "stop please") aborts the spoken reply. Anything longer is a
    real command and is submitted as a new request.
    """
    cleaned = (text or "").strip().lower().strip(_PUNCT)
    if not cleaned:
        return False
    if cleaned in _STOP_PHRASES:
        return True
    words = cleaned.split()
    if len(words) > 3:
        return False
    first = words[0].strip(_PUNCT)
    if first not in _STOP_FIRST_WORDS:
        return False
    return all(word.strip(_PUNCT) in _STOP_FILLER for word in words[1:])


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
            # blocks (it plays audio), so this runs off the listening thread.
            while True:
                message = outbound.get()
                if message is None:
                    self.daemon.stop()
                    return
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
        if self._on_inbound is None:
            return
        try:
            self._on_inbound(message)
        except Exception as exc:
            self.daemon._event("gateway_handler_failed", message=str(exc)[:300])
            self.daemon.speak("Sorry, something went wrong handling that.")

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
