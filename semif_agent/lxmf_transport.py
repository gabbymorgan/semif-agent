"""Low-level LXMF (Reticulum) transport (neutral).

This module owns the LXMF/Reticulum mechanics only: standing up a Reticulum
instance and an `LXMRouter`, loading (or creating) the delivery identity,
announcing reachability, normalizing inbound messages into plain dicts, and
draining an outbound send queue. It knows nothing about the SemIf scheduler,
the command gateway, or any bridge — a front end builds on top of it.

`RNS` and `LXMF` are imported lazily inside the methods that need them so
importing this module stays stdlib-only on a machine without the optional
`lxmf` dependency (the gateway refuses to start with an install hint instead).

Unlike the SimpleX transport there is no external daemon binary: Reticulum and
the LXMF router run in-process. The bot's reachable address is its LXMF
delivery destination hash (32 hex chars); the identity is persisted under the
router storage path so the address is stable across restarts.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from typing import Any, Callable

#: One received direct message, normalized into a plain dict.
InboundDict = dict[str, Any]
#: Callback invoked for each received message.
MessageHandler = Callable[[InboundDict], None]
#: Zero-argument callback invoked once the router is ready to receive.
ConnectedHandler = Callable[[], None]

#: The LXMF `FIELD_AUDIO` (0x07) field a Sideband push-to-talk voice message
#: rides in. `LXMF` is optional and lazy-imported, so its stable, committed wire
#: constants are mirrored here (the same way `simplex_ws` mirrors the v7 shapes).
FIELD_AUDIO = 0x07

#: Audio modes — the first element of a `FIELD_AUDIO` value (mirrors
#: `LXMF.AM_*`). `AM_OPUS_OGG` is a complete OGG/Opus container a decoder (PyAV,
#: which `faster-whisper` bundles) reads directly; the Codec2 modes are raw
#: frames decoded by `pycodec2`.
AM_OPUS_OGG = 0x10
AUDIO_MODE_NAMES = {
    0x01: "codec2_450pwb", 0x02: "codec2_450", 0x03: "codec2_700c",
    0x04: "codec2_1200", 0x05: "codec2_1300", 0x06: "codec2_1400",
    0x07: "codec2_1600", 0x08: "codec2_2400", 0x09: "codec2_3200",
    0x10: "opus_ogg", 0x11: "opus_lbw", 0x12: "opus_mbw", 0x13: "opus_ptt",
    0x14: "opus_rt_hdx", 0x15: "opus_rt_fdx", 0x16: "opus_standard",
    0x17: "opus_hq", 0x18: "opus_broadcast", 0x19: "opus_lossless",
}
#: Codec2 audio modes -> the codec2 bitrate (bits/s) they decode at. `pycodec2`
#: has bindings for these; 450PWB/450 (0x01/0x02) have none, so they stay
#: undecodable. (Mirrors Sideband's `audioproc.codec2_modes`.)
CODEC2_MODE_RATES = {
    0x03: 700, 0x04: 1200, 0x05: 1300, 0x06: 1400,
    0x07: 1600, 0x08: 2400, 0x09: 3200,
}

#: How often the announce loop wakes to re-check the clock and interfaces.
ANNOUNCE_TICK = 1.0
#: How often to scan Reticulum interfaces for online/offline transitions.
PEER_SCAN_INTERVAL = 10.0


def audio_mode_name(codec: int) -> str:
    """A human-readable name for an audio mode id (e.g. `AM_OPUS_OGG` ->
    `opus_ogg`); an unknown id is rendered as hex."""
    return AUDIO_MODE_NAMES.get(codec, f"0x{codec:02x}")


def audio_decode_kind(codec: int) -> str:
    """The decode path for an audio mode.

    `"opus_ogg"` (a self-contained OGG/Opus container), `"codec2"` (raw Codec2
    frames), or `"unknown"` (no decode path — e.g. the raw Opus `AM_OPUS_*`
    modes and the binding-less 450 Codec2 modes).
    """
    if codec == AM_OPUS_OGG:
        return "opus_ogg"
    if codec in CODEC2_MODE_RATES:
        return "codec2"
    return "unknown"


def parse_audio_field(value: Any) -> dict[str, Any] | None:
    """Normalize an LXMF `FIELD_AUDIO` value into a plain dict.

    The wire value is `[audio_mode, data]` (see `LXMF.FIELD_AUDIO`): an integer
    codec id and the raw audio bytes. Returns
    `{"codec", "codec_name", "data", "kind", "codec2_rate"}` or None when
    malformed. `kind` is the decode path (`opus_ogg`/`codec2`/`unknown`);
    `codec2_rate` is the bitrate for a Codec2 mode, else None.
    """
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    data = value[1]
    if not isinstance(data, (bytes, bytearray)):
        return None
    try:
        codec = int(value[0])
    except (TypeError, ValueError):
        return None
    return {
        "codec": codec,
        "codec_name": audio_mode_name(codec),
        "data": bytes(data),
        "kind": audio_decode_kind(codec),
        "codec2_rate": CODEC2_MODE_RATES.get(codec),
    }


def audio_fields(audio: bytes | None) -> dict[int, Any] | None:
    """The LXMF fields for an outbound voice reply, or None for a text message.

    A voice reply rides `FIELD_AUDIO` as `[AM_OPUS_OGG, ogg_bytes]` — the
    Sideband HQ push-to-talk shape. Pure so the wire construction is testable
    without a Reticulum router.
    """
    if not audio:
        return None
    return {FIELD_AUDIO: [AM_OPUS_OGG, bytes(audio)]}


def _as_text(value: Any) -> str:
    """Best-effort UTF-8 text for a `str`/`bytes`/`None` LXMF field."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, str):
        return value
    return str(value)


def normalize_message(message: Any) -> InboundDict:
    """Normalize one LXMF message object into a plain dict.

    Tolerates both a real `LXMessage` (bytes `content`/`title`, `source_hash`,
    `get_source()`) and a lightweight fake, so the policy layer is testable
    without the optional `lxmf` package installed.
    """
    raw_source_hash = getattr(message, "source_hash", b"") or b""
    if isinstance(raw_source_hash, (bytes, bytearray)):
        source_hash = bytes(raw_source_hash).hex()
    else:
        source_hash = str(raw_source_hash)

    content_fn = getattr(message, "content_as_string", None)
    if callable(content_fn):
        try:
            content = _as_text(content_fn())
        except Exception:
            content = ""
    else:
        content = _as_text(getattr(message, "content", None))

    title_fn = getattr(message, "title_as_string", None)
    if callable(title_fn):
        try:
            title = _as_text(title_fn())
        except Exception:
            title = ""
    else:
        title = _as_text(getattr(message, "title", None))

    display_name = ""
    source = None
    get_source = getattr(message, "get_source", None)
    if callable(get_source):
        try:
            source = get_source()
        except Exception:
            source = None
    if source is None:
        source = getattr(message, "source", None)
    if source is not None:
        display_name = _as_text(getattr(source, "display_name", None))

    fields = getattr(message, "fields", None)
    if not isinstance(fields, dict):
        fields = {}
    # A Sideband push-to-talk voice message carries its Opus/Codec2 payload in
    # `FIELD_AUDIO` and an empty content; surface it as a normalized `audio` dict
    # so a front end can transcribe it without knowing the wire shape.
    audio = None
    if FIELD_AUDIO in fields:
        audio = parse_audio_field(fields.get(FIELD_AUDIO))

    return {
        "source_hash": source_hash,
        "display_name": display_name or None,
        "title": title,
        "content": content,
        "timestamp": getattr(message, "timestamp", None),
        "fields": fields,
        "audio": audio,
        "raw": message,
    }


class LxmfDaemon:
    """One in-process Reticulum instance + LXMF router.

    `run(on_message)` blocks, announcing the delivery destination periodically
    — and whenever a Reticulum interface or AutoInterface peer comes online, so
    a peer that was not yet reachable at startup (or that joined mid-interval)
    learns the destination promptly instead of waiting out `announce_interval`.
    Each received message is delivered to `on_message`. `enqueue(chat_id,
    text)` queues an outbound send (a `None` chat_id is the stop sentinel),
    drained by an internal pump thread. `address` is the bot's LXMF address.

    Beware: the identity is persisted under `storage_path`; a single profile
    must not be shared by two processes.
    """

    def __init__(
        self,
        *,
        config_dir: str,
        storage_path: str,
        display_name: str | None = "semif",
        announce_interval: float = 3600.0,
        stamp_cost: int | None = None,
        desired_method: str = "direct",
        propagation_node: str = "",
        on_connected: ConnectedHandler | None = None,
        trace=None,
        name: str = "lxmf",
    ):
        self.config_dir = str(config_dir)
        self.storage_path = str(storage_path)
        self.display_name = str(display_name) if display_name else None
        self.announce_interval = float(announce_interval or 0.0)
        self.stamp_cost = int(stamp_cost) if stamp_cost is not None else None
        self.desired_method = str(desired_method or "direct").lower()
        self.propagation_node = str(propagation_node or "")
        self.on_connected = on_connected
        self.trace = trace
        self.name = name
        self.identity_path = os.path.join(self.storage_path, "identity")
        self._router = None
        self._reticulum = None
        self._destination = None
        self._address = ""
        #: Outbound items are `(chat_id, text, audio)`; `audio` is an OGG/Opus
        #: container for a Sideband HQ push-to-talk voice reply, else None.
        self._outbound: "queue.Queue[tuple[str, str, bytes | None] | None]" = (
            queue.Queue()
        )
        self._stop = threading.Event()
        self._on_message: MessageHandler | None = None
        #: Last announce time, and whether an interface transition asked for
        #: an immediate re-announce (set by the announce loop's interface scan).
        self._last_announce = 0.0
        self._announce_due = False
        #: `str(interface) -> online`, so the scan can spot interfaces and
        #: AutoInterface peers that just came up.
        self._interface_states: dict[str, bool] = {}

    # ---- requirements ----

    def check_requirements(self) -> tuple[bool, str | None]:
        if not self.config_dir:
            return False, "lxmf config_dir is required"
        try:
            import RNS  # noqa: F401
            import LXMF  # noqa: F401
        except ImportError:
            return False, "pip install lxmf"
        return True, None

    # ---- lifecycle ----

    @property
    def address(self) -> str:
        """The bot's LXMF delivery address (32 hex chars), or "" before ready."""
        return self._address

    def ensure_reticulum(self):
        """Initialize the Reticulum instance on the calling thread.

        Reticulum installs process signal handlers, which only works on the
        main thread. Call this once from the main thread before spawning the
        transport thread so the in-process router can coexist with other
        gateway adapters; `run()` reuses the instance when it already exists.
        """
        import RNS

        existing = RNS.Reticulum.get_instance()
        if existing is not None:
            self._reticulum = existing
            return existing
        os.makedirs(self.config_dir, exist_ok=True)
        self._reticulum = RNS.Reticulum(configdir=self.config_dir)
        return self._reticulum

    def prepare(self, on_message: MessageHandler | None = None):
        """Main-thread-sensitive setup: Reticulum instance + LXMF router.

        Both `RNS.Reticulum` and `LXMF.LXMRouter` install process signal
        handlers, which only works on the main thread. `run_gateway` calls this
        once on the main thread before the transport threads start; `run()`
        calls it again (idempotent) when used standalone.
        """
        if self._router is not None:
            if on_message is not None:
                self._on_message = on_message
            return self._router
        import LXMF
        import RNS

        if on_message is not None:
            self._on_message = on_message
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.storage_path, exist_ok=True)

        self.ensure_reticulum()
        self._router = LXMF.LXMRouter(storagepath=self.storage_path)
        identity = self._load_identity(RNS)
        self._destination = self._router.register_delivery_identity(
            identity, display_name=self.display_name, stamp_cost=self.stamp_cost
        )
        if self._destination is None:
            raise RuntimeError("could not register LXMF delivery identity")
        self._address = RNS.hexrep(self._destination.hash, delimit=False)
        self._router.register_delivery_callback(self._handle)
        if self.propagation_node:
            try:
                self._router.set_outbound_propagation_node(
                    bytes.fromhex(self.propagation_node)
                )
            except Exception as exc:
                self._event("lxmf_propagation_error", message=str(exc)[:300])
        # Snapshot the interfaces Reticulum brought up so the announce loop can
        # tell which ones (and which AutoInterface peers) come online later.
        try:
            self._interface_states = {
                str(interface): bool(getattr(interface, "online", False))
                for interface in RNS.Transport.interfaces
            }
        except Exception as exc:
            self._event("lxmf_error", phase="interface_snapshot", message=str(exc)[:300])
        self._event("lxmf_ready", address=self._address)
        return self._router

    def run(self, on_message: MessageHandler) -> None:
        self.prepare(on_message)

        threading.Thread(
            target=self._pump, name="lxmf-outbound-pump", daemon=True
        ).start()
        self._announce("startup")
        if self.on_connected is not None:
            try:
                self.on_connected()
            except Exception as exc:
                self._event(
                    "lxmf_error", phase="on_connected", message=str(exc)[:300]
                )

        self._announce_loop()

    def _announce_loop(self) -> None:
        """Announce on the interval, and whenever an interface/peer comes up.

        The startup announce fires before AutoInterface has finished its
        peering handshake, so it reaches no peer; a peer discovered later (or a
        link that recovers) would otherwise have to wait out the whole
        `announce_interval`. Re-announcing on every online transition makes a
        newly-reachable peer learn the destination promptly.
        """
        last_scan = 0.0
        while not self._stop.is_set():
            try:
                now = time.time()
                if self._announce_due:
                    self._announce_due = False
                    self._announce("peer discovered")
                elif now - self._last_announce >= self._announce_wait():
                    self._announce("interval")

                if now - last_scan >= PEER_SCAN_INTERVAL:
                    last_scan = now
                    if self._scan_interfaces():
                        self._announce_due = True
            except Exception as exc:
                self._event(
                    "lxmf_error", phase="announce_loop", message=str(exc)[:300]
                )
            self._stop.wait(ANNOUNCE_TICK)

    def _scan_interfaces(self) -> bool:
        """Note interface transitions; True when one newly came online.

        AutoInterface reports `online` only after its peering wait, and every
        discovered peer is a fresh `AutoInterfacePeer` interface, so an online
        transition is exactly "we may now be reachable by someone new".
        """
        changed = False
        try:
            import RNS

            for interface in RNS.Transport.interfaces:
                key = str(interface)
                online = bool(getattr(interface, "online", False))
                prev = self._interface_states.get(key)
                self._interface_states[key] = online
                if online and prev is not True:
                    self._event("lxmf_interface_online", interface=key)
                    changed = True
        except Exception as exc:
            self._event(
                "lxmf_error", phase="interface_scan", message=str(exc)[:300]
            )
        return changed

    def enqueue(
        self, chat_id: str | None, text: str = "", audio: bytes | None = None
    ) -> None:
        """Queue an outbound send. `chat_id=None` is the stop sentinel.

        `audio` is an OGG/Opus container: it rides the LXMF `FIELD_AUDIO` field
        (a Sideband HQ push-to-talk voice reply) with `text` as the message
        content, so a text-only client still sees the reply.
        """
        if chat_id is None:
            self._outbound.put(None)
        else:
            self._outbound.put((str(chat_id), text, audio))

    def close(self) -> None:
        self._stop.set()
        self._outbound.put(None)
        try:
            import RNS

            RNS.Reticulum.exit_handler()
        except Exception:
            pass

    # ---- identity ----

    def _load_identity(self, RNS):
        os.makedirs(self.storage_path, exist_ok=True)
        if os.path.isfile(self.identity_path):
            identity = RNS.Identity.from_file(self.identity_path)
            if identity is not None:
                return identity
        identity = RNS.Identity()
        try:
            identity.to_file(self.identity_path)
            os.chmod(self.identity_path, 0o600)
        except Exception as exc:
            self._event("lxmf_identity_save_failed", message=str(exc)[:300])
        return identity

    # ---- announce ----

    def _announce_wait(self) -> float:
        if self.announce_interval <= 0:
            return 3600.0
        return self.announce_interval

    def _announce(self, reason: str = "") -> None:
        if self._router is None or self._destination is None:
            return
        try:
            self._router.announce(self._destination.hash)
            self._last_announce = time.time()
            self._event("lxmf_announced", address=self._address, reason=reason or None)
        except Exception as exc:
            self._event("lxmf_error", phase="announce", message=str(exc)[:300])

    # ---- inbound ----

    def _handle(self, message: Any) -> None:
        try:
            normalized = normalize_message(message)
        except Exception as exc:
            self._event("lxmf_parse_error", message=str(exc)[:300])
            return
        self._event(
            "lxmf_message",
            source=normalized.get("source_hash"),
            chars=len(normalized.get("content") or ""),
        )
        if self._on_message is not None:
            self._on_message(normalized)

    # ---- outbound ----

    def _pump(self) -> None:
        while True:
            item = self._outbound.get()
            if item is None:
                return
            chat_id, text, audio = item
            try:
                self.send(chat_id, text, audio=audio)
            except Exception as exc:
                self._event(
                    "lxmf_send_failed", contact=chat_id, message=str(exc)[:300]
                )

    def send(
        self,
        destination_hex: str,
        text: str,
        audio: bytes | None = None,
        timeout: float = 15.0,
    ) -> None:
        """Send `text` to an LXMF address (destination hash hex).

        When `audio` (an OGG/Opus container) is given it is attached in the
        LXMF `FIELD_AUDIO` field as a Sideband HQ push-to-talk voice message
        (`[AM_OPUS_OGG, audio]`), with `text` as the message content.

        Recalls the peer identity from the network; when the path is not yet
        known it requests it and waits briefly. Raises (never silently drops)
        when the destination cannot be resolved — a reply must be real.
        """
        if self._router is None or self._destination is None:
            raise RuntimeError("LXMF router is not ready")
        import LXMF
        import RNS

        try:
            dest_hash = bytes.fromhex(str(destination_hex).strip())
        except ValueError as exc:
            raise ValueError(f"invalid LXMF address {destination_hex!r}") from exc

        if not RNS.Transport.has_path(dest_hash):
            RNS.Transport.request_path(dest_hash)
            deadline = time.time() + timeout
            while not RNS.Transport.has_path(dest_hash) and time.time() < deadline:
                time.sleep(0.1)
        identity = RNS.Identity.recall(dest_hash)
        if identity is None:
            raise RuntimeError(f"unknown LXMF destination {destination_hex}")

        dest = RNS.Destination(
            identity, RNS.Destination.OUT, RNS.Destination.SINGLE, "lxmf", "delivery"
        )
        method = (
            LXMF.LXMessage.OPPORTUNISTIC
            if self.desired_method == "opportunistic"
            else LXMF.LXMessage.DIRECT
        )
        fields = audio_fields(audio)
        message = LXMF.LXMessage(
            dest, self._destination, text, desired_method=method, fields=fields
        )
        self._router.handle_outbound(message)
        self._event(
            "lxmf_sent", contact=destination_hex, chars=len(text), audio=bool(audio)
        )

    # ---- tracing ----

    def _event(self, kind: str, **fields) -> None:
        if self.trace is not None:
            self.trace.append(kind, "?", **fields)
