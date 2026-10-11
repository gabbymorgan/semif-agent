"""A real browser driver for the browser bridge, backed by Playwright.

The browser bridge calls this instead of a skill body driving a browser
directly. It owns a single Chromium (or Firefox/WebKit) instance with a
**persistent profile** under the checkout's `.runtime/browser/profile`, so a
logged-in session survives across calls and restarts, and exposes plain methods
that return dicts.

Playwright's synchronous API is bound to the thread that created it and is not
thread-safe, but the bridge's HTTP server handles each request on a new thread.
So this driver runs the **async** API on one dedicated event-loop thread and
serializes whole operations behind an `asyncio.Lock`; route handlers submit a
coroutine onto that loop with `run_coroutine_threadsafe` and wait for the
result. Browser automation is inherently serial, so one loop is enough.

Only the standard library is imported eagerly; `playwright` is imported lazily
the first time the browser is actually needed, so a host without it still runs
every other bridge. A missing package (or a missing browser binary) raises
`BrowserUnavailable`, which the bridge maps to a 503 install hint; every other
failure is a `BrowserError`, mapped to a 502 — never a fabricated success.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path

DEFAULT_ENGINE = "chromium"
ENGINES = ("chromium", "firefox", "webkit")
DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_TEXT_CHARS = 20000
#: Cap on how many elements a single `text` / `links` call returns.
DEFAULT_MAX_MATCHES = 50
WAIT_UNTIL = ("commit", "domcontentloaded", "load", "networkidle")
INSTALL_HINT = (
    "install the browser bridge stack: pip install -r requirements/browser.txt "
    "&& PLAYWRIGHT_BROWSERS_PATH=<repo>/.runtime/browser python -m playwright "
    "install chromium"
)


class BrowserError(Exception):
    """A real failure driving the browser (navigation, timeout, bad selector)."""


class BrowserUnavailable(BrowserError):
    """Playwright or its browser binaries are not installed / cannot launch."""


class BrowserDriver:
    """One persistent browser, driven from a single dedicated event-loop thread."""

    def __init__(
        self,
        engine: str = DEFAULT_ENGINE,
        headless: bool = True,
        profile_dir: str = "",
        browsers_path: str = "",
        screenshot_dir: str = "",
        default_timeout: float = DEFAULT_TIMEOUT,
        max_text_chars: int = DEFAULT_MAX_TEXT_CHARS,
        channel: str = "",
        executable_path: str = "",
    ):
        self.engine = engine if engine in ENGINES else DEFAULT_ENGINE
        self.headless = bool(headless)
        self.profile_dir = str(profile_dir or "")
        self.browsers_path = str(browsers_path or "")
        self.screenshot_dir = str(screenshot_dir or "")
        self.default_timeout = float(default_timeout)
        self.max_text_chars = max(200, int(max_text_chars))
        #: Use an already-installed browser instead of Playwright's own build:
        #: `channel` names a Playwright channel (`chrome`, `msedge`, ...) and
        #: `executable_path` points at a specific binary. Both bypass the
        #: managed-download requirement, useful where the CDN is unreachable.
        self.channel = str(channel or "").strip()
        self.executable_path = str(executable_path or "").strip()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock: asyncio.Lock | None = None
        self._start_lock = threading.Lock()
        self._playwright = None
        self._context = None
        self._page = None

    # ---- availability ----

    @property
    def configured(self) -> bool:
        """Whether the `playwright` package is importable (cheap; no browser)."""
        return importlib.util.find_spec("playwright") is not None

    # ---- lifecycle ----

    def _ensure_started(self) -> None:
        with self._start_lock:
            if self._loop is not None:
                return
            if not self.configured:
                raise BrowserUnavailable(INSTALL_HINT)
            # Locate browser binaries under the checkout before Playwright reads
            # the env var. The bridge cannot rely on the shell exporting it.
            if self.browsers_path:
                os.environ["PLAYWRIGHT_BROWSERS_PATH"] = self.browsers_path
            loop = asyncio.new_event_loop()
            self._loop = loop
            ready = threading.Event()

            def runner() -> None:
                asyncio.set_event_loop(loop)
                self._lock = asyncio.Lock()
                ready.set()
                loop.run_forever()

            self._thread = threading.Thread(target=runner, name="browser-driver", daemon=True)
            self._thread.start()
            ready.wait(5.0)

    def _run(self, coro, timeout: float | None = None):
        """Schedule `coro` on the driver loop and wait for its result."""
        if self._loop is None:
            coro.close()
            raise BrowserUnavailable(INSTALL_HINT)
        budget = (timeout if timeout else self.default_timeout) + 15.0
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(budget)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise BrowserError(f"browser operation timed out after {budget:.0f}s") from exc

    def _operation(self, fn, timeout: float | None = None):
        """Run `fn(page)` under the serialization lock, launching if needed."""
        self._ensure_started()

        async def runner():
            assert self._lock is not None
            async with self._lock:
                page = await self._ensure_context()
                return await fn(page)

        return self._run(runner(), timeout=timeout)

    async def _ensure_context(self):
        if self._page is not None and not self._page.is_closed():
            return self._page
        if self._playwright is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
        browser_type = getattr(self._playwright, self.engine, None)
        if browser_type is None:
            raise BrowserError(f"unknown browser engine {self.engine!r}")
        if self.profile_dir:
            Path(self.profile_dir).mkdir(parents=True, exist_ok=True)
            user_data_dir = self.profile_dir
        else:
            # No persistent profile configured: still a real context, just
            # ephemeral (Playwright rejects a None user_data_dir).
            user_data_dir = tempfile.mkdtemp(prefix="semif-browser-")
        try:
            # A persistent context keeps cookies/logins on disk across calls and
            # restarts; it is the browser analog of the SimpleX bridge owning its
            # own profile.
            launch_options = {
                "headless": self.headless,
                "viewport": {"width": 1280, "height": 800},
                "timeout": self.default_timeout * 1000,
            }
            if self.executable_path:
                launch_options["executable_path"] = self.executable_path
            elif self.channel:
                launch_options["channel"] = self.channel
            self._context = await browser_type.launch_persistent_context(
                user_data_dir, **launch_options
            )
        except BrowserUnavailable:
            raise
        except Exception as exc:  # missing browser binary, launch failure, ...
            raise BrowserUnavailable(
                f"could not launch {self.engine} (is the browser installed? "
                f"{INSTALL_HINT}): {exc}"
            ) from exc
        pages = self._context.pages
        self._page = pages[0] if pages else await self._context.new_page()
        return self._page

    async def _close_context(self) -> None:
        self._page = None
        context, self._context = self._context, None
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass

    def close(self) -> None:
        """Tear down the browser and the driver loop (from `_stop_transport`)."""
        loop, self._loop = self._loop, None
        if loop is None:
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
            future.result(timeout=15.0)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5.0)
        try:
            loop.close()
        except Exception:
            pass

    async def _shutdown(self) -> None:
        await self._close_context()
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    # ---- operations (each returns a plain dict) ----

    def state(self) -> dict:
        async def _do(page):
            return {"url": page.url, "title": await page.title()}

        return self._operation(_do)

    def goto(self, url: str, wait_until: str = "load", timeout: float | None = None) -> dict:
        async def _do(page):
            response = await page.goto(
                url, wait_until=wait_until, timeout=(timeout or self.default_timeout) * 1000
            )
            return {
                "url": page.url,
                "title": await page.title(),
                "status": response.status if response is not None else None,
            }

        return self._operation(_do, timeout=timeout)

    def content(self, fmt: str = "text", max_chars: int | None = None) -> dict:
        limit = max(200, int(max_chars)) if max_chars else self.max_text_chars

        async def _do(page):
            if fmt == "html":
                raw = await page.content()
            else:
                raw = await page.inner_text("body")
            text = raw or ""
            return {
                "url": page.url,
                "title": await page.title(),
                "format": fmt,
                "content": text[:limit],
                "truncated": len(text) > limit,
            }

        return self._operation(_do)

    def text(self, selector: str) -> dict:
        async def _do(page):
            locator = page.locator(selector)
            count = await locator.count()
            matches = []
            for index in range(min(count, DEFAULT_MAX_MATCHES)):
                value = await locator.nth(index).inner_text()
                matches.append({"text": (value or "").strip()[:2000]})
            return {
                "url": page.url,
                "selector": selector,
                "count": count,
                "matches": matches,
            }

        return self._operation(_do)

    def links(self, query: str = "", limit: int = 50) -> dict:
        max_links = max(1, min(int(limit), 200))

        async def _do(page):
            anchors = await page.query_selector_all("a")
            needle = query.strip().lower()
            out = []
            for element in anchors:
                href = await element.get_attribute("href")
                if not href:
                    continue
                label = ((await element.inner_text()) or "").strip()
                if needle and needle not in f"{label} {href}".lower():
                    continue
                out.append({"text": label[:300], "href": href})
                if len(out) >= max_links:
                    break
            return {"url": page.url, "links": out}

        return self._operation(_do)

    def screenshot(
        self, full_page: bool = False, selector: str = "", inline: bool = False
    ) -> dict:
        async def _do(page):
            directory = self.screenshot_dir or "/tmp"
            Path(directory).mkdir(parents=True, exist_ok=True)
            path = os.path.join(directory, f"{uuid.uuid4().hex}.png")
            if selector:
                await page.locator(selector).first.screenshot(path=path)
            else:
                await page.screenshot(path=path, full_page=bool(full_page))
            result = {
                "ok": True,
                "path": path,
                "bytes": os.path.getsize(path),
                "url": page.url,
                "title": await page.title(),
            }
            if inline:
                with open(path, "rb") as handle:
                    result["base64"] = base64.b64encode(handle.read()).decode("ascii")
            return result

        return self._operation(_do)

    def click(
        self,
        selector: str = "",
        text: str = "",
        index: int = 0,
        timeout: float | None = None,
    ) -> dict:
        async def _do(page):
            if selector:
                locator = page.locator(selector).nth(int(index))
            else:
                locator = page.get_by_text(text, exact=False).nth(int(index))
            await locator.click(timeout=(timeout or self.default_timeout) * 1000)
            return {"url": page.url, "title": await page.title()}

        return self._operation(_do, timeout=timeout)

    def fill(
        self,
        selector: str,
        value: str,
        submit: bool = False,
        timeout: float | None = None,
    ) -> dict:
        async def _do(page):
            budget = (timeout or self.default_timeout) * 1000
            await page.fill(selector, value, timeout=budget)
            if submit:
                await page.press(selector, "Enter", timeout=budget)
            return {"url": page.url, "title": await page.title()}

        return self._operation(_do, timeout=timeout)

    def press(self, key: str, selector: str = "", timeout: float | None = None) -> dict:
        async def _do(page):
            await page.press(
                selector or "body", key, timeout=(timeout or self.default_timeout) * 1000
            )
            return {"url": page.url, "title": await page.title()}

        return self._operation(_do, timeout=timeout)

    def select(
        self,
        selector: str,
        value: str | None = None,
        label: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        async def _do(page):
            budget = (timeout or self.default_timeout) * 1000
            if value is not None:
                await page.select_option(selector, value=value, timeout=budget)
            else:
                await page.select_option(selector, label=label, timeout=budget)
            return {"url": page.url, "title": await page.title()}

        return self._operation(_do, timeout=timeout)

    def wait_for(
        self, selector: str = "", text: str = "", timeout: float | None = None
    ) -> dict:
        async def _do(page):
            budget = (timeout or self.default_timeout) * 1000
            if selector:
                await page.wait_for_selector(selector, timeout=budget)
            else:
                await page.get_by_text(text, exact=False).first.wait_for(timeout=budget)
            return {"url": page.url, "title": await page.title()}

        return self._operation(_do, timeout=timeout)

    def back(self, timeout: float | None = None) -> dict:
        async def _do(page):
            await page.go_back(
                wait_until="load", timeout=(timeout or self.default_timeout) * 1000
            )
            return {"url": page.url, "title": await page.title()}

        return self._operation(_do, timeout=timeout)

    def forward(self, timeout: float | None = None) -> dict:
        async def _do(page):
            await page.go_forward(
                wait_until="load", timeout=(timeout or self.default_timeout) * 1000
            )
            return {"url": page.url, "title": await page.title()}

        return self._operation(_do, timeout=timeout)

    def reset(self, purge: bool = True) -> dict:
        """Close the browser; delete the stored profile when `purge` is true."""
        self._ensure_started()

        async def _do():
            assert self._lock is not None
            async with self._lock:
                await self._close_context()
                purged = False
                if purge and self.profile_dir:
                    shutil.rmtree(self.profile_dir, ignore_errors=True)
                    purged = True
                return {"ok": True, "purged": purged}

        return self._run(_do())
