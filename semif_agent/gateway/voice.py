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

    def __init__(self, cfg: dict | None = None, trace=None):
        cfg = cfg or {}
        self.chat_id = str(cfg.get("chat_id", "voice") or "voice")
        self.display_name = str(cfg.get("display_name") or self.chat_id)
        #: `run()` blocks on the mic; this daemon owns capture/wake/STT/TTS.
        self.daemon = VoiceDaemon(cfg, trace=trace, name=self.name)
        self._on_inbound = None

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
                self.daemon.speak(message.text)

        threading.Thread(target=_pump, name="voice-outbound-pump", daemon=True).start()
        self.daemon.run(self._on_utterance)

    def close(self) -> None:
        self.daemon.close()

    # ---- inbound ----

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
        message = InboundMessage(
            text=text,
            chat_id=self.chat_id,
            chat_type="dm",
            contact_id=self.chat_id,
            display_name=self.display_name,
            raw={},
        )
        if self._on_inbound is not None:
            self._on_inbound(message)
