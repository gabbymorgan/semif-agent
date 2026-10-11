"""Pure-stdlib tests for the browser (Playwright) bridge.

The bridge is exercised over a real loopback HTTP server with a fake driver
standing in for the Playwright browser — the decision engine, the browser, and
Playwright itself are never involved. These prove the mechanics a skill depends
on: route wiring/validation, the token/auth surface, and the status mapping
(400 bad input, 503 unavailable, 502 real failure). Passing these proves
mechanics only; a real run against a real browser is the only proof the
integration works.
"""

import json
import urllib.error
import urllib.request

import pytest

from semif_agent.bridges.browser import BrowserBridge
from semif_agent.bridges.browser_client import (
    BrowserDriver,
    BrowserError,
    BrowserUnavailable,
)
from semif_agent.bridges.registry import (
    build_bridge,
    derived_config_vars,
    describe_bridges,
    known_infos,
)


class FakeDriver:
    """Stands in for the Playwright browser, recording every call."""

    def __init__(self, error=None, unavailable=None):
        self.error = error
        self.unavailable = unavailable
        self.calls = []
        self.closed = False

    def _maybe_fail(self):
        if self.unavailable is not None:
            raise BrowserUnavailable(self.unavailable)
        if self.error is not None:
            raise BrowserError(self.error)

    def state(self):
        self.calls.append(("state",))
        self._maybe_fail()
        return {"url": "https://example.com/", "title": "Example"}

    def goto(self, url, wait_until="load", timeout=None):
        self.calls.append(("goto", url, wait_until, timeout))
        self._maybe_fail()
        return {"url": url, "title": "Example", "status": 200}

    def content(self, fmt="text", max_chars=None):
        self.calls.append(("content", fmt, max_chars))
        self._maybe_fail()
        return {"url": "https://example.com/", "title": "Example", "format": fmt, "content": "hi", "truncated": False}

    def text(self, selector):
        self.calls.append(("text", selector))
        self._maybe_fail()
        return {"url": "https://example.com/", "selector": selector, "count": 1, "matches": [{"text": "hi"}]}

    def links(self, query="", limit=50):
        self.calls.append(("links", query, limit))
        self._maybe_fail()
        return {"url": "https://example.com/", "links": [{"text": "More", "href": "/more"}]}

    def screenshot(self, full_page=False, selector="", inline=False):
        self.calls.append(("screenshot", full_page, selector, inline))
        self._maybe_fail()
        return {"ok": True, "path": "/tmp/shot.png", "bytes": 12, "url": "https://example.com/", "title": "Example"}

    def click(self, selector="", text="", index=0, timeout=None):
        self.calls.append(("click", selector, text, index, timeout))
        self._maybe_fail()
        return {"url": "https://example.com/", "title": "Example"}

    def fill(self, selector, value, submit=False, timeout=None):
        self.calls.append(("fill", selector, value, submit, timeout))
        self._maybe_fail()
        return {"url": "https://example.com/", "title": "Example"}

    def press(self, key, selector="", timeout=None):
        self.calls.append(("press", key, selector, timeout))
        self._maybe_fail()
        return {"url": "https://example.com/", "title": "Example"}

    def select(self, selector, value=None, label=None, timeout=None):
        self.calls.append(("select", selector, value, label, timeout))
        self._maybe_fail()
        return {"url": "https://example.com/", "title": "Example"}

    def wait_for(self, selector="", text="", timeout=None):
        self.calls.append(("wait_for", selector, text, timeout))
        self._maybe_fail()
        return {"url": "https://example.com/", "title": "Example"}

    def back(self, timeout=None):
        self.calls.append(("back", timeout))
        self._maybe_fail()
        return {"url": "https://example.com/", "title": "Example"}

    def forward(self, timeout=None):
        self.calls.append(("forward", timeout))
        self._maybe_fail()
        return {"url": "https://example.com/", "title": "Example"}

    def reset(self, purge=True):
        self.calls.append(("reset", purge))
        self._maybe_fail()
        return {"ok": True, "purged": bool(purge)}

    def close(self):
        self.closed = True


def start_bridge(driver=None, **config):
    driver = driver or FakeDriver()
    bridge = BrowserBridge(config, client=driver)
    port = bridge.start()
    return bridge, driver, f"http://127.0.0.1:{port}"


def get(url, token=None):
    headers = {"X-Semif-Token": token} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def post(url, payload, token=None):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Semif-Token"] = token
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def http_error(url, payload=None, token=None):
    if payload is None:
        request = urllib.request.Request(url)
    else:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["X-Semif-Token"] = token
        request = urllib.request.Request(url, data=body, headers=headers)
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=5)
    return exc.value


# ---- routes ----

def test_health_reports_engine_without_a_browser():
    bridge, _driver, base = start_bridge(headless=False, engine="firefox")
    try:
        assert get(f"{base}/health") == {
            "ok": True,
            "platform": "browser",
            "engine": "firefox",
            "headless": False,
        }
    finally:
        bridge.stop()


def test_read_routes_wire_to_the_driver():
    bridge, driver, base = start_bridge()
    try:
        assert get(f"{base}/state") == {"url": "https://example.com/", "title": "Example"}
        assert get(f"{base}/content?format=html&max_chars=500")["content"] == "hi"
        assert get(f"{base}/text?selector=h1")["count"] == 1
        assert get(f"{base}/links?query=more&limit=5")["links"][0]["href"] == "/more"
        assert get(f"{base}/screenshot?full_page=true&inline=true")["path"] == "/tmp/shot.png"
        assert driver.calls == [
            ("state",),
            ("content", "html", 500),
            ("text", "h1"),
            ("links", "more", 5),
            ("screenshot", True, "", True),
        ]
    finally:
        bridge.stop()


def test_action_routes_wire_to_the_driver():
    bridge, driver, base = start_bridge()
    try:
        assert post(f"{base}/goto", {"url": "https://a.test/", "wait_until": "networkidle"})["status"] == 200
        post(f"{base}/click", {"text": "Sign in"})
        post(f"{base}/fill", {"selector": "#q", "value": "hi", "submit": True})
        post(f"{base}/press", {"key": "Enter"})
        post(f"{base}/select", {"selector": "select", "label": "Two"})
        post(f"{base}/wait", {"text": "Done", "timeout": 5})
        post(f"{base}/back", {})
        post(f"{base}/forward", {})
        post(f"{base}/reset", {"purge": False})
        assert driver.calls == [
            ("goto", "https://a.test/", "networkidle", None),
            ("click", "", "Sign in", 0, None),
            ("fill", "#q", "hi", True, None),
            ("press", "Enter", "", None),
            ("select", "select", None, "Two", None),
            ("wait_for", "", "Done", 5.0),
            ("back", None),
            ("forward", None),
            ("reset", False),
        ]
    finally:
        bridge.stop()


def test_missing_url_is_400():
    bridge, driver, base = start_bridge()
    try:
        exc = http_error(f"{base}/goto", {})
        assert exc.code == 400
        assert driver.calls == [], "a rejected goto must not touch the browser"
    finally:
        bridge.stop()


def test_bad_wait_until_and_content_format_are_400():
    bridge, _driver, base = start_bridge()
    try:
        assert http_error(f"{base}/goto", {"url": "https://a.test", "wait_until": "whenever"}).code == 400
        assert http_error(f"{base}/content?format=pdf").code == 400
    finally:
        bridge.stop()


def test_click_requires_selector_or_text():
    bridge, _driver, base = start_bridge()
    try:
        assert http_error(f"{base}/click", {}).code == 400
    finally:
        bridge.stop()


def test_select_requires_value_or_label():
    bridge, _driver, base = start_bridge()
    try:
        assert http_error(f"{base}/select", {"selector": "s"}).code == 400
    finally:
        bridge.stop()


def test_real_browser_failure_is_502():
    bridge, _driver, base = start_bridge(driver=FakeDriver(error="net::ERR_NAME_NOT_RESOLVED"))
    try:
        exc = http_error(f"{base}/goto", {"url": "https://nope.invalid"})
        assert exc.code == 502
        assert b"ERR_NAME_NOT_RESOLVED" in exc.read()
    finally:
        bridge.stop()


def test_unavailable_browser_is_503_with_install_hint():
    driver = FakeDriver(unavailable="install the browser bridge stack")
    bridge, _driver, base = start_bridge(driver=driver)
    try:
        exc = http_error(f"{base}/state")
        assert exc.code == 503
        assert b"install the browser bridge stack" in exc.read()
        # /health still answers, so the bridge is diagnosable without a browser.
        assert get(f"{base}/health")["ok"] is True
    finally:
        bridge.stop()


def test_unknown_route_is_404():
    bridge, _driver, base = start_bridge()
    try:
        assert http_error(f"{base}/nope").code == 404
    finally:
        bridge.stop()


def test_token_guards_the_bridge():
    bridge, _driver, base = start_bridge(token="sekret")
    try:
        assert http_error(f"{base}/state", token=None).code == 401
        assert get(f"{base}/health", token="sekret")["platform"] == "browser"
    finally:
        bridge.stop()


def test_stop_closes_the_driver():
    bridge, driver, _base = start_bridge()
    bridge.stop()
    assert driver.closed is True


def test_check_requirements_never_blocks_startup():
    # An optional dependency must not take down the other bridges, so the check
    # is honest-but-permissive and a missing browser surfaces as a 503 on use.
    assert BrowserBridge({}).check_requirements() == (True, None)


def test_bad_engine_falls_back_to_chromium():
    assert BrowserBridge({"engine": "netscape"}).engine == "chromium"
    assert BrowserBridge({"engine": "WebKit"}).engine == "webkit"


# ---- driver dependency handling ----

def test_driver_unavailable_without_playwright(monkeypatch):
    import semif_agent.bridges.browser_client as browser_client

    monkeypatch.setattr(browser_client.importlib.util, "find_spec", lambda name: None)
    driver = BrowserDriver()
    assert driver.configured is False
    with pytest.raises(BrowserUnavailable):
        driver.state()
    # No dangling loop/thread was started.
    driver.close()


# ---- catalog / codegen surface ----

def test_known_infos_include_browser():
    infos = {info.name: info for info in known_infos()}
    assert "browser" in infos
    info = infos["browser"]
    assert info.service == "browser"
    assert info.url_config_var == "browser_bridge_url"
    assert info.auth_header == "X-Semif-Token"
    assert info.auth_config_var == "browser_bridge_token"
    assert any("/goto" in endpoint for endpoint in info.endpoints)


def test_describe_bridges_covers_browser():
    text = describe_bridges()
    assert "browser_bridge_url" in text
    assert "/screenshot" in text
    assert "Playwright" in text


def test_derived_config_vars_for_browser():
    config = {"bridges": {"browser": {"host": "127.0.0.1", "port": 5232, "token": "b"}}}
    derived = derived_config_vars(config)
    assert derived["browser_bridge_url"] == "http://127.0.0.1:5232"
    assert derived["browser_bridge_token"] == "b"


def test_build_bridge_returns_browser():
    bridge = build_bridge({"bridges": {"browser": {"port": 5232}}}, "browser")
    assert isinstance(bridge, BrowserBridge)
