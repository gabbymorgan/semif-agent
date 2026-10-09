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
        self._on_inbound = None
        #: Utterances are handed to a worker thread, not run on the audio
        #: thread: the scheduler is synchronous and can block on the decision
        #: engine, and a slow or failing task must neither stop the mic being
        #: read nor kill the front end. `None` asks the worker to stop.
        self._inbound: "queue.Queue[InboundMessage | None]" = queue.Queue()

    # ---- requirements ----

    def check_requirements(self) -> tuple[bool, str | None]:
        return self.daemon.check_requirements()

    # ---- transport ----

    def run(
        self,
        on_inbound,
        outbound: "queue.Queue[OutboundMessage | None]",
    ) -> None:
        self._on_inbound = on_inbound

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
        if self._on_inbound is None:
            return
        try:
            self._on_inbound(message)
        except Exception as exc:
            self.daemon._event("gateway_handler_failed", message=str(exc)[:300])
            self.daemon.speak("Sorry, something went wrong handling that.")

    def _on_utterance(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
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
                raw={},
            )
        )
