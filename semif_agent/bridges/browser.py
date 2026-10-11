"""Browser bridge: real browser automation (Playwright) behind HTTP.

A codegen-authored skill body must never drive a browser directly, and it must
never bundle a heavy, platform-specific automation stack. This bridge owns the
browser — a single Chromium (or Firefox/WebKit) instance with a persistent
profile — and exposes one uniform, token-guarded localhost JSON API for reading
a page (text/links/screenshot) and acting on it (goto/click/fill/press/select/
wait/back/forward), so a body only reads `browser_bridge_url` and
`browser_bridge_token` from `ctx.config` and calls a documented route.

Unlike Nextcloud/Outlook this bridge needs a real optional dependency
(`playwright` plus browser binaries), so it is **disabled by default**; install
it with `scripts/bootstrap.sh --browser` (browsers land under
`.runtime/browser`). The dependency and the browser are lazy: a host without
them still runs every other bridge. `check_requirements()` deliberately always
returns ok — `run_bridges()` aborts *all* bridges when one's requirements fail,
and this dependency is optional, so a missing Playwright must degrade to a
per-call 503 (with an install hint), never take down simplex/llm/nextcloud.

The actual Playwright work lives in `browser_client.py`; the routes here
validate input and map a real failure (`BrowserError`) to a 502, a missing
dependency/browser (`BrowserUnavailable`) to a 503, and bad input to a 400 —
never a fabricated success.
"""

from __future__ import annotations

from pathlib import Path

from .base import BridgeInfo, BridgeService
from .browser_client import (
    DEFAULT_MAX_TEXT_CHARS,
    DEFAULT_TIMEOUT,
    ENGINES,
    WAIT_UNTIL,
    BrowserDriver,
    BrowserError,
    BrowserUnavailable,
)

#: The checkout root (`semif_agent/bridges/browser.py` -> repo root), used to
#: anchor the browser profile, binaries, and screenshots under the gitignored
#: `.runtime/` tree (the containment rule).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BROWSER_ROOT = str(_REPO_ROOT / ".runtime" / "browser")
DEFAULT_PROFILE_DIR = str(Path(DEFAULT_BROWSER_ROOT) / "profile")
DEFAULT_SCREENSHOT_DIR = str(Path(DEFAULT_BROWSER_ROOT) / "screenshots")


class BrowserBridge(BridgeService):
    name = "browser"
    INFO = BridgeInfo(
        name="browser",
        service="browser",
        description=(
            "Drive a real web browser (Playwright): open a URL, read a page's "
            "text or links, take a screenshot, and click/fill/press/select to "
            "navigate a site and complete a flow. Keeps a persistent profile so "
            "logins and cookies survive across calls."
        ),
        url_config_var="browser_bridge_url",
        endpoints=(
            "GET /health -> 200 {\"ok\": true, \"platform\": \"browser\", "
            "\"engine\", \"headless\"} — the bridge is up",
            "GET /state -> 200 {\"url\", \"title\"} — the current page",
            "GET /content?format=text|html&max_chars=<n> -> 200 {\"url\", "
            "\"title\", \"format\", \"content\", \"truncated\"} — the whole page's "
            "rendered text (format defaults to text, the default when you just "
            "want to read a page); content is truncated to max_chars",
            "GET /text?selector=<css> -> 200 {\"url\", \"selector\", \"count\", "
            "\"matches\": [{\"text\"}]} — text of every element matching a CSS "
            "selector",
            "GET /links?query=<text>&limit=<n> -> 200 {\"url\", \"links\": "
            "[{\"text\", \"href\"}]} — anchors on the page, optionally filtered "
            "by a substring of the label or href",
            "GET /screenshot?full_page=<bool>&selector=<css>&inline=<bool> -> 200 "
            "{\"ok\": true, \"path\", \"bytes\", \"url\", \"title\"} — saves a PNG "
            "under the checkout's .runtime/browser/screenshots (add inline=true "
            "for base64 in the reply)",
            "POST /goto {\"url\", \"wait_until\"?: commit|domcontentloaded|load|"
            "networkidle, \"timeout\"?: seconds} -> 200 {\"url\", \"title\", "
            "\"status\"} — navigates; a navigation error (bad host, TLS, timeout) "
            "is a 502",
            "POST /click {\"selector\"?| \"text\"?, \"index\"?: 0, \"timeout\"?} "
            "-> 200 {\"url\", \"title\"} — clicks by CSS selector or by visible "
            "text (index picks among several matches)",
            "POST /fill {\"selector\", \"value\", \"submit\"?: bool, "
            "\"timeout\"?} -> 200 {\"url\", \"title\"} — types into an input; "
            "submit=true presses Enter",
            "POST /press {\"key\", \"selector\"?, \"timeout\"?} -> 200 {\"url\", "
            "\"title\"} — presses a keyboard key (e.g. Enter, Escape, Tab)",
            "POST /select {\"selector\", \"value\"?|\"label\"?, \"timeout\"?} -> "
            "200 {\"url\", \"title\"} — chooses an option in a <select>",
            "POST /wait {\"selector\"?| \"text\"?, \"timeout\"?} -> 200 {\"url\", "
            "\"title\"} — waits for an element/text to appear",
            "POST /back | /forward {\"timeout\"?} -> 200 {\"url\", \"title\"}",
            "POST /reset {\"purge\"?: bool default true} -> 200 {\"ok\", "
            "\"purged\"} — closes the browser; purge=true (default) also deletes "
            "the stored profile so the next call starts logged out",
            "any request -> 401 {\"error\": \"unauthorized\"} when the token is "
            "configured and the auth header is missing or wrong",
            "503 {\"error\": \"...\"} when Playwright or its browser binaries are "
            "not installed (run scripts/bootstrap.sh --browser); 502 {\"error\": "
            "\"...\"} on a real browser failure (navigation, timeout, bad "
            "selector)",
        ),
        config_vars=("browser_bridge_url", "browser_bridge_token"),
        config_var_docs=(
            (
                "browser_bridge_url",
                "Base URL of the local browser bridge (e.g. "
                "http://127.0.0.1:5232); never hardcode it in the body.",
            ),
            (
                "browser_bridge_token",
                "Shared secret for the bridge, if one is configured; sent as the "
                "X-Semif-Token header. Leave blank when the bridge requires no auth.",
            ),
        ),
        auth_header="X-Semif-Token",
        auth_config_var="browser_bridge_token",
    )

    def __init__(self, config: dict | None = None, trace=None, client=None):
        super().__init__(config, trace)
        cfg = config or {}
        engine = str(cfg.get("engine", "chromium") or "chromium").strip().lower()
        self.engine = engine if engine in ENGINES else "chromium"
        self.headless = bool(cfg.get("headless", True))
        self.profile_dir = str(cfg.get("profile_dir") or DEFAULT_PROFILE_DIR)
        self.browsers_path = str(cfg.get("browsers_path") or DEFAULT_BROWSER_ROOT)
        self.screenshot_dir = str(cfg.get("screenshot_dir") or DEFAULT_SCREENSHOT_DIR)
        self.default_timeout = float(cfg.get("default_timeout", DEFAULT_TIMEOUT))
        self.max_text_chars = int(cfg.get("max_text_chars", DEFAULT_MAX_TEXT_CHARS))
        #: A test seam: inject a driver double. Production builds the real one.
        self.client = client or BrowserDriver(
            engine=self.engine,
            headless=self.headless,
            profile_dir=self.profile_dir,
            browsers_path=self.browsers_path,
            screenshot_dir=self.screenshot_dir,
            default_timeout=self.default_timeout,
            max_text_chars=self.max_text_chars,
        )

    # ---- lifecycle ----

    def check_requirements(self) -> tuple[bool, str | None]:
        # Deliberately never blocks startup: a missing Playwright/browser is an
        # optional-dependency problem and must not take down the other bridges
        # (run_bridges aborts them all on a failed check). It surfaces as a 503
        # on the call instead.
        return True, None

    def _stop_transport(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    # ---- routes ----

    def handle_get(self, path: str, query: dict) -> tuple[int, dict]:
        if path == "/health":
            return 200, {
                "ok": True,
                "platform": self.name,
                "engine": self.engine,
                "headless": self.headless,
            }
        try:
            return self._get(path, query)
        except BrowserUnavailable as exc:
            return 503, {"error": str(exc)[:400]}
        except BrowserError as exc:
            return 502, {"error": str(exc)[:300]}
        except ValueError as exc:
            return 400, {"error": str(exc)[:300]}

    def handle_post(self, path: str, payload: dict) -> tuple[int, dict]:
        try:
            return self._post(path, payload)
        except BrowserUnavailable as exc:
            return 503, {"error": str(exc)[:400]}
        except BrowserError as exc:
            return 502, {"error": str(exc)[:300]}
        except ValueError as exc:
            return 400, {"error": str(exc)[:300]}

    def _get(self, path: str, query: dict) -> tuple[int, dict]:
        if path == "/state":
            return 200, self.client.state()
        if path == "/content":
            fmt = (self._q(query, "format", "text") or "text").strip().lower()
            if fmt not in ("text", "html"):
                raise ValueError("format must be text or html")
            max_chars = self._int_query_opt(query, "max_chars")
            return 200, self.client.content(fmt, max_chars)
        if path == "/text":
            return 200, self.client.text(self._required_query(query, "selector"))
        if path == "/links":
            return 200, self.client.links(
                self._q(query, "query", "") or "", self._int_query(query, "limit", 50)
            )
        if path == "/screenshot":
            return 200, self.client.screenshot(
                self._bool_query(query, "full_page", False),
                self._q(query, "selector", "") or "",
                self._bool_query(query, "inline", False),
            )
        return 404, {"error": "not found"}

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        if path == "/goto":
            wait_until = (
                self._optional_str(payload, "wait_until", "load") or "load"
            ).strip().lower()
            if wait_until not in WAIT_UNTIL:
                raise ValueError(f"wait_until must be one of {', '.join(WAIT_UNTIL)}")
            return 200, self.client.goto(
                self._required(payload, "url"),
                wait_until,
                self._optional_float(payload, "timeout"),
            )
        if path == "/click":
            selector = self._optional_str(payload, "selector")
            text = self._optional_str(payload, "text")
            if not selector and not text:
                raise ValueError("selector or text is required")
            return 200, self.client.click(
                selector,
                text,
                self._optional_int(payload, "index", 0) or 0,
                self._optional_float(payload, "timeout"),
            )
        if path == "/fill":
            return 200, self.client.fill(
                self._required(payload, "selector"),
                self._required(payload, "value", allow_empty=True),
                self._optional_bool(payload, "submit", False),
                self._optional_float(payload, "timeout"),
            )
        if path == "/press":
            return 200, self.client.press(
                self._required(payload, "key"),
                self._optional_str(payload, "selector"),
                self._optional_float(payload, "timeout"),
            )
        if path == "/select":
            value = self._optional_str(payload, "value") or None
            label = self._optional_str(payload, "label") or None
            if value is None and label is None:
                raise ValueError("value or label is required")
            return 200, self.client.select(
                self._required(payload, "selector"),
                value,
                label,
                self._optional_float(payload, "timeout"),
            )
        if path == "/wait":
            selector = self._optional_str(payload, "selector")
            text = self._optional_str(payload, "text")
            if not selector and not text:
                raise ValueError("selector or text is required")
            return 200, self.client.wait_for(
                selector, text, self._optional_float(payload, "timeout")
            )
        if path == "/back":
            return 200, self.client.back(self._optional_float(payload, "timeout"))
        if path == "/forward":
            return 200, self.client.forward(self._optional_float(payload, "timeout"))
        if path == "/reset":
            return 200, self.client.reset(self._optional_bool(payload, "purge", True))
        return 404, {"error": "not found"}

    # ---- input helpers ----

    @staticmethod
    def _q(query: dict, key: str, default=None):
        values = query.get(key)
        if not values:
            return default
        return values[0]

    @staticmethod
    def _required_query(query: dict, key: str) -> str:
        value = BrowserBridge._q(query, key)
        if value is None or not str(value).strip():
            raise ValueError(f"{key} is required")
        return str(value).strip()

    @staticmethod
    def _int_query(query: dict, key: str, default: int) -> int:
        value = BrowserBridge._q(query, key)
        if value is None or str(value).strip() == "":
            return default
        try:
            return int(str(value))
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc

    @staticmethod
    def _int_query_opt(query: dict, key: str):
        value = BrowserBridge._q(query, key)
        if value is None or str(value).strip() == "":
            return None
        try:
            return int(str(value))
        except ValueError as exc:
            raise ValueError(f"{key} must be an integer") from exc

    @staticmethod
    def _bool_query(query: dict, key: str, default: bool) -> bool:
        value = BrowserBridge._q(query, key)
        if value is None or str(value).strip() == "":
            return default
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _required(payload: dict, key: str, allow_empty: bool = False) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or (not allow_empty and not value.strip()):
            raise ValueError(f"{key} is required")
        return value if allow_empty else value.strip()

    @staticmethod
    def _optional_str(payload: dict, key: str, default: str = "") -> str:
        value = payload.get(key)
        if value is None:
            return default
        if not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        return value

    @staticmethod
    def _optional_bool(payload: dict, key: str, default: bool) -> bool:
        value = payload.get(key)
        if value is None:
            return default
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
        return value

    @staticmethod
    def _optional_int(payload: dict, key: str, default: int | None = None):
        value = payload.get(key)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        return value

    @staticmethod
    def _optional_float(payload: dict, key: str):
        value = payload.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        return float(value)
