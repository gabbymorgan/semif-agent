"""Bridge service contract + the shared secure HTTP layer.

Every bridge stands up the same thin HTTP surface: JSON in, JSON out, bound to
localhost, optionally guarded by a shared-secret header. A subclass only
implements its routes (`handle_get` / `handle_post`) and, if it needs a service
daemon, starts it in `start()`. The HTTP/JSON/auth plumbing lives here once so
each new bridge is small and hard to get wrong.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


@dataclass(frozen=True)
class BridgeInfo:
    """Static, prompt-safe metadata about a bridge.

    Fed to the code-generation model so it knows what it can build against,
    and used to describe the service in docs. Holds no secrets and needs no
    running process.
    """

    name: str
    service: str
    description: str
    url_config_var: str
    #: Human-readable `METHOD /path` -> purpose lines.
    endpoints: tuple[str, ...] = ()
    #: Other `ctx.config` keys a body built on this bridge may read.
    config_vars: tuple[str, ...] = field(default=())
    #: One-line semantic description per config var: `(name, description)`.
    #: Every entry in `config_vars` must be documented here.
    config_var_docs: tuple[tuple[str, str], ...] = ()
    #: Optional shared-secret header the bridge enforces when its token is set
    #: (e.g. `X-Semif-Token`), and the top-level `ctx.config` var holding it.
    auth_header: str = ""
    auth_config_var: str = ""


class BridgeService(ABC):
    """Base class for a standalone third-party API bridge."""

    #: short snake_case id (`simplex`)
    name: str = "bridge"
    #: static catalog entry
    info: BridgeInfo

    def __init__(self, config: dict | None = None, trace=None):
        cfg = config or {}
        self.host = str(cfg.get("host", "127.0.0.1"))
        self.port = int(cfg.get("port", 0))
        self.token = str(cfg.get("token", "") or "")
        self.trace = trace
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----

    @abstractmethod
    def check_requirements(self) -> tuple[bool, str | None]:
        """Return `(ok, install_hint)`. Must never raise on a missing dep."""

    def start(self) -> int:
        """Start serving HTTP; returns the bound port (`port=0` is ephemeral)."""
        server = ThreadingHTTPServer((self.host, self.port), _BridgeHTTPHandler)
        server.bridge = self  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name=f"bridge-{self.name}",
            daemon=True,
        )
        self._thread.start()
        return int(server.server_address[1])

    def stop(self) -> None:
        self._stop_transport()
        server, self._server = self._server, None
        if server is not None:
            server.shutdown()
            server.server_close()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def _stop_transport(self) -> None:
        """Hook for subclasses that own a background transport (daemon)."""

    # ---- routes ----

    def handle_get(self, path: str, query: dict) -> tuple[int, dict]:
        return 404, {"error": "not found"}

    def handle_post(self, path: str, payload: dict) -> tuple[int, dict]:
        return 404, {"error": "not found"}

    # ---- helpers ----

    def _event(self, kind: str, **fields) -> None:
        if self.trace is not None:
            self.trace.append(kind, "?", **fields)


class _BridgeHTTPHandler(BaseHTTPRequestHandler):
    """JSON request handler; the owning bridge is on `self.server.bridge`."""

    def log_message(self, *args) -> None:  # silence the default stderr spam
        pass

    def _bridge(self) -> BridgeService:
        return self.server.bridge  # type: ignore[attr-defined]

    def _authorized(self) -> bool:
        token = self._bridge().token
        if not token:
            return True
        return self.headers.get("X-Semif-Token") == token

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._json(401, {"error": "unauthorized"})
        return False

    def do_GET(self) -> None:
        if not self._require_auth():
            return
        url = urlparse(self.path)
        status, payload = self._bridge().handle_get(url.path, parse_qs(url.query))
        self._json(status, payload)

    def do_POST(self) -> None:
        if not self._require_auth():
            return
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except ValueError:
            self._json(400, {"error": "invalid JSON body"})
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": "body must be a JSON object"})
            return
        status, response = self._bridge().handle_post(url.path, payload)
        self._json(status, response)
