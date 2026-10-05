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

    return {
        "source_hash": source_hash,
        "display_name": display_name or None,
        "title": title,
        "content": content,
        "timestamp": getattr(message, "timestamp", None),
        "raw": message,
    }


class LxmfDaemon:
    """One in-process Reticulum instance + LXMF router.

    `run(on_message)` blocks, announcing the delivery destination periodically
    and delivering each received message to `on_message`. `enqueue(chat_id,
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
        self._outbound: "queue.Queue[tuple[str, str] | None]" = queue.Queue()
        self._stop = threading.Event()
        self._on_message: MessageHandler | None = None

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
        self._event("lxmf_ready", address=self._address)
        return self._router

    def run(self, on_message: MessageHandler) -> None:
        self.prepare(on_message)

        threading.Thread(
            target=self._pump, name="lxmf-outbound-pump", daemon=True
        ).start()
        self._announce()
        if self.on_connected is not None:
            try:
                self.on_connected()
            except Exception as exc:
                self._event(
                    "lxmf_error", phase="on_connected", message=str(exc)[:300]
                )

        while not self._stop.wait(self._announce_wait()):
            self._announce()

    def enqueue(self, chat_id: str | None, text: str = "") -> None:
        """Queue an outbound send. `chat_id=None` is the stop sentinel."""
        if chat_id is None:
            self._outbound.put(None)
        else:
            self._outbound.put((str(chat_id), text))

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

    def _announce(self) -> None:
        if self._router is None or self._destination is None:
            return
        try:
            self._router.announce(self._destination.hash)
            self._event("lxmf_announced", address=self._address)
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
            chat_id, text = item
            try:
                self.send(chat_id, text)
            except Exception as exc:
                self._event(
                    "lxmf_send_failed", contact=chat_id, message=str(exc)[:300]
                )

    def send(self, destination_hex: str, text: str, timeout: float = 15.0) -> None:
        """Send `text` to an LXMF address (destination hash hex).

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
        message = LXMF.LXMessage(dest, self._destination, text, desired_method=method)
        self._router.handle_outbound(message)
        self._event("lxmf_sent", contact=destination_hex, chars=len(text))

    # ---- tracing ----

    def _event(self, kind: str, **fields) -> None:
        if self.trace is not None:
            self.trace.append(kind, "?", **fields)
