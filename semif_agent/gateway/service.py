"""GatewayService: map external messenger traffic onto the SemIf scheduler.

The scheduler is a single execution slot with one global `pending` run. The
service does not fake concurrency: inbound text is fed through the existing
gate (`submit_request` -> gate/score/queue/dispatch), results and deferred
questions are routed back to the originating chat, and while one chat owns a
`needs_input` pause that chat answers it. Another chat messaging during that
pause is told to wait rather than silently abandoning the first chat's run.

One service can front several transports at once (SimpleX, LXMF, ...). Each
adapter is addressed by its platform name, so a request source is
`"<platform>:<chat_id>"`, the single pending-run guard is shared across
platforms, and every reply is sent back through the queue of the platform that
originated it.

Ownership: a request id maps to the `(platform, chat)` that sent it. A run
re-queued by the scheduler (updated request) mints a new id; the scheduler's
`on_request_requeued` hook copies the owner across so the completion still
reaches the right chat.
"""

from __future__ import annotations

import queue
import re
import threading
import time

from .base import InboundMessage, OutboundMessage
from .humanize import Humanizer

_LEADING_RUN_ID = re.compile(r"^\[[0-9a-f]{6,}\]\s*")

#: A result summary starts with the skill ref (`category.name:`), optionally
#: preceded by a `[resumed]` marker. Used to look up a skill's own config
#: (`data/skills/<cat>/<name>/config.json`) for a per-skill humanize override.
_SKILL_REF = re.compile(
    r"^(?:\[resumed\]\s*)?([A-Za-z0-9_][A-Za-z0-9_-]*\.[A-Za-z0-9_][A-Za-z0-9_-]*):\s"
)

#: The scheduler's run summary wraps the real result as
#: `"<skill>: <ok|failed> — <detail>"`. A result-only platform (the voice
#: gateway) speaks just `<detail>`.
_RESULT_SEP = " — "

_REPAIR_ACTIONS = {
    "retry": "retry",
    "repair": "repair_skill",
    "repair_skill": "repair_skill",
    "ask": "ask_user",
    "ask_user": "ask_user",
    "no": "no_repair",
    "skip": "no_repair",
    "no_repair": "no_repair",
}

_APPROVAL_ACTIONS = {
    "yes": True,
    "y": True,
    "approve": True,
    "ok": True,
    "no": False,
    "n": False,
    "deny": False,
    "skip": False,
}

#: An owner target is `(platform, chat_id)`.
Target = tuple[str, str]


def _strip_run_id(text: str) -> str:
    return _LEADING_RUN_ID.sub("", text or "")


def _skill_ref(text: str) -> str | None:
    """The `category.name` a result summary belongs to, or None."""
    match = _SKILL_REF.match(text or "")
    return match.group(1) if match else None


def _leading_run_id(text: str) -> str:
    match = _LEADING_RUN_ID.match(text or "")
    return match.group(0)[1:-2] if match else ""


def _result_detail(text: str) -> str:
    """Unwrap the scheduler's `"<skill>: <ok|failed> — <detail>"` summary.

    Returns the detail the skill actually produced. A line with no wrapper is
    returned unchanged.
    """
    if _RESULT_SEP in text:
        return text.split(_RESULT_SEP, 1)[1].strip()
    return text


def _reply_kind(status: str) -> str:
    """Classify a scheduler outcome for a result-only front end."""
    if status in ("running", "ran"):
        return "result"
    if status == "needs_input":
        return "question"
    if status == "queued":
        return "status"
    return "notice"


def _normalize_adapters(adapters) -> dict:
    """Accept a single adapter or a list/dict of them, keyed by platform name."""
    if isinstance(adapters, dict):
        return {str(name): adapter for name, adapter in adapters.items()}
    if hasattr(adapters, "name") and not isinstance(
        adapters, (list, tuple, set, frozenset)
    ):
        return {adapters.name: adapters}
    result: dict = {}
    for adapter in adapters:
        result[adapter.name] = adapter
    return result


class GatewayService:
    def __init__(
        self,
        scheduler,
        adapters,
        config: dict | None = None,
        poll_interval: float = 1.0,
        humanizer: Humanizer | None = None,
    ):
        self.scheduler = scheduler
        self.adapters = _normalize_adapters(adapters)
        self.platforms = list(self.adapters.keys())
        self.config = config or {}
        self.poll_interval = float(poll_interval)
        #: Per-platform outbound queue; the adapter drains its own.
        self.outbound: "dict[str, queue.Queue[OutboundMessage | None]]" = {
            platform: queue.Queue() for platform in self.platforms
        }
        self._platform_cfg: dict[str, dict] = {
            platform: self._config_for(platform) for platform in self.platforms
        }
        #: Optional LLM result cleanup (gateway-only presentation). Built from
        #: the scheduler's `llm` client unless one is injected; a no-op unless
        #: `gateway.humanize.enabled` (or a per-platform override) turns it on.
        if humanizer is not None:
            self.humanizer = humanizer
            self._humanize_default = bool(humanizer.enabled)
        else:
            humanize_cfg = self.config.get("humanize")
            humanize_cfg = humanize_cfg if isinstance(humanize_cfg, dict) else {}
            self._humanize_default = bool(humanize_cfg.get("enabled", False))
            self.humanizer = self._build_humanizer(humanize_cfg)
        self._owners: dict[str, Target] = {}
        self._parents: dict[str, str] = {}
        self._pending_owner: Target | None = None
        self._question_for_chat: dict[Target, str] = {}
        self._repair_for_chat: dict[Target, str] = {}
        self._approval_for_chat: dict[Target, str] = {}
        self._surfaced: set[str] = set()
        self._route_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- config ----

    def _config_for(self, platform: str) -> dict:
        """Per-platform settings from a nested (`{platform: {...}}`) or, for a
        single-adapter service, a flat config."""
        nested = self.config.get(platform)
        if isinstance(nested, dict):
            return nested
        if len(self.platforms) == 1:
            return self.config
        return {}

    def outbound_for(self, platform: str) -> "queue.Queue[OutboundMessage | None]":
        return self.outbound[platform]

    def _home_channel(self, platform: str) -> str:
        return str(self._platform_cfg.get(platform, {}).get("home_channel", "") or "")

    def _result_only(self, platform: str) -> bool:
        """Whether this platform speaks only the skill result line.

        The adapter declares the default (the voice adapter sets `result_only`);
        a per-platform config `result_only` overrides it either way.
        """
        cfg = self._platform_cfg.get(platform, {})
        if "result_only" in cfg:
            return bool(cfg["result_only"])
        return bool(getattr(self.adapters.get(platform), "result_only", False))

    # ---- optional LLM result cleanup ----

    def _build_humanizer(self, cfg: dict) -> Humanizer | None:
        """Build the result humanizer from config + the scheduler's `llm`.

        Global defaults live in `gateway.humanize`; a per-platform
        `gateway.<platform>.humanize` (bool or `{enabled, timeout, max_tokens,
        max_chars}`) overrides the enable flag. Returns None when no `llm`
        client is available, so a client-less scheduler degrades to no cleanup.
        The instance is always built `enabled`; the global default is kept
        separately so a per-platform override can turn it on or off.
        """
        client = getattr(self.scheduler, "llm", None)
        if client is None:
            return None
        return Humanizer(
            client,
            enabled=True,
            timeout=float(cfg.get("timeout", 20.0)),
            max_tokens=int(cfg.get("max_tokens", 200)),
            max_chars=int(cfg.get("max_chars", 600)),
        )

    def _humanize_enabled(self, platform: str, skill_ref: str | None = None) -> bool:
        """Whether this result is cleaned up.

        Precedence: the **skill's own config** (`data/skills/<category>/<name>/
        config.json` -> `{"humanize": bool}`) wins; otherwise the per-platform
        override (`gateway.<platform>.humanize`), otherwise the global default
        (`gateway.humanize.enabled`). A missing `llm` client disables it.
        """
        if self.humanizer is None:
            return False
        skill_override = self._skill_humanize_override(skill_ref)
        if skill_override is not None:
            return skill_override
        override = self._platform_cfg.get(platform, {}).get("humanize")
        if isinstance(override, bool):
            return override
        if isinstance(override, dict) and "enabled" in override:
            return bool(override["enabled"])
        return self._humanize_default

    def _skill_humanize_override(self, skill_ref: str | None) -> bool | None:
        """The skill's own `humanize` config value, or None when unset.

        A skill opts in/out of result cleanup regardless of the platform default
        via its recorded config: `data/skills/<category>/<name>/config.json` ->
        `{"humanize": false}`. Read from the scheduler's skill store; a missing
        store/config/key (or an unparseable ref) means no override.
        """
        if not skill_ref or "." not in skill_ref:
            return None
        store = getattr(self.scheduler, "body_store", None)
        if store is None:
            return None
        category, name = skill_ref.split(".", 1)
        try:
            config = store.read_config(category, name) or {}
        except Exception:
            return None
        if "humanize" in config:
            return bool(config["humanize"])
        return None

    def _fallback_target(self) -> Target | None:
        for platform in self.platforms:
            home = self._home_channel(platform)
            if home:
                return (platform, home)
        return None

    # ---- lifecycle ----

    def start(self) -> None:
        """Start the background queue-drain loop (codegen re-queues, etc.)."""
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for platform in self.platforms:
            self.outbound[platform].put(None)

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                with self._route_lock:
                    self.drain()
                    self.surface()
            except Exception:
                self.scheduler.trace.append("gateway_error", "?", phase="poll")

    # ---- routing ----

    def source_for(self, platform: str, chat_id: str) -> str:
        return f"{platform}:{chat_id}"

    def _owner_from_source(self, source: str) -> Target | None:
        for platform in self.platforms:
            prefix = f"{platform}:"
            if source.startswith(prefix):
                return (platform, source[len(prefix):])
        return None

    def handle_inbound(self, msg: InboundMessage, platform: str | None = None) -> None:
        """Handle one authorized inbound command. Called by the adapter."""
        if platform is None:
            platform = self.platforms[0] if self.platforms else ""
        with self._route_lock:
            reply_to: Target = (platform, msg.chat_id)
            pending = self.scheduler.pending
            if pending is not None and self._pending_owner is None:
                self._pending_owner = self._owner_from_source(pending.request.source)
            if pending is not None and pending.request.source == self.source_for(
                platform, msg.chat_id
            ):
                status, detail = self.scheduler.answer(msg.text)
                if self.scheduler.pending is None:
                    self._pending_owner = None
                self._reply(*reply_to, detail, kind=_reply_kind(status))
            elif pending is not None and self._pending_owner not in (None, reply_to):
                self._reply(
                    *reply_to,
                    "I'm waiting for a reply in another conversation; please resend in a moment.",
                    kind="notice",
                )
            elif reply_to in self._question_for_chat:
                question_id = self._question_for_chat.pop(reply_to)
                status, detail = self.scheduler.answer_question(question_id, msg.text)
                self._surfaced.add(question_id)
                self._reply(*reply_to, detail, kind="notice")
            elif reply_to in self._repair_for_chat and msg.text.strip().lower() in _REPAIR_ACTIONS:
                offer_id = self._repair_for_chat.pop(reply_to)
                action = _REPAIR_ACTIONS[msg.text.strip().lower()]
                status, detail = self.scheduler.resolve_repair(offer_id, action)
                self._surfaced.add(offer_id)
                self._reply(*reply_to, detail, kind="notice")
            elif reply_to in self._approval_for_chat and msg.text.strip().lower() in _APPROVAL_ACTIONS:
                approval_id = self._approval_for_chat.pop(reply_to)
                approved = _APPROVAL_ACTIONS[msg.text.strip().lower()]
                status, detail = self.scheduler.answer_approval(approval_id, approved)
                self._surfaced.add(approval_id)
                self._reply(*reply_to, detail, kind="notice")
            else:
                status, detail, run_id = self.scheduler.submit_request(
                    msg.text, source=self.source_for(platform, msg.chat_id)
                )
                if run_id:
                    self._owners[run_id] = reply_to
                if status == "needs_input":
                    self._pending_owner = reply_to
                self._reply(*reply_to, detail, kind=_reply_kind(status))
            self.drain()
            self.surface(origin=reply_to)

    def drain(self) -> None:
        """Run queued work and route each outcome back to its owner."""
        for _ in range(100):
            results = self.scheduler.run_queue()
            if not results:
                break
            for status, detail in results:
                run_id = _leading_run_id(detail)
                owner = self._owner_of(run_id)
                text = _strip_run_id(detail)
                if status == "needs_input" and owner:
                    self._pending_owner = owner
                target = owner or self._fallback_target()
                if target:
                    self._reply(*target, text, kind=_reply_kind(status))

    def surface(self, origin: Target | None = None) -> None:
        """Send any newly posted authoring questions / repair offers."""
        questions = self.scheduler.pending_questions()
        live_ids = {q["id"] for q in questions}
        self._question_for_chat = {
            chat: qid for chat, qid in self._question_for_chat.items() if qid in live_ids
        }
        for item in questions:
            if item["id"] in self._surfaced:
                continue
            self._surfaced.add(item["id"])
            target = self._resolve_target(item.get("run_id"), origin)
            if target:
                self._question_for_chat[target] = item["id"]
                self._reply(
                    *target,
                    f"(authoring {item['category']}.{item['skill']}) {item['question']}",
                    kind="question",
                )

        repairs = self.scheduler.pending_repairs()
        live_ids = {r["id"] for r in repairs}
        self._repair_for_chat = {
            chat: rid for chat, rid in self._repair_for_chat.items() if rid in live_ids
        }
        for offer in repairs:
            if offer["id"] in self._surfaced:
                continue
            self._surfaced.add(offer["id"])
            target = self._resolve_target(offer.get("run_id"), origin)
            if target:
                self._repair_for_chat[target] = offer["id"]
                self._reply(
                    *target,
                    (
                        f"{offer['category']}.{offer['skill']} failed: "
                        f"{offer['failure'][:200]}\nReply retry / repair / ask / no."
                    ),
                    kind="notice",
                )

        approvals = self.scheduler.pending_approvals()
        live_ids = {a["id"] for a in approvals}
        self._approval_for_chat = {
            chat: aid for chat, aid in self._approval_for_chat.items() if aid in live_ids
        }
        for item in approvals:
            if item["id"] in self._surfaced:
                continue
            self._surfaced.add(item["id"])
            target = self._resolve_target(item.get("run_id"), origin)
            if target:
                self._approval_for_chat[target] = item["id"]
                label = (
                    item["category"]
                    if item["kind"] == "category"
                    else f"{item['category']}.{item['skill']}"
                )
                self._reply(
                    *target,
                    (
                        f"Approve creating new {item['kind']} {label}: "
                        f"{item['description']}\nReply yes / no."
                    ),
                    kind="notice",
                )

        for fired in self.scheduler.timers.drain():
            target = self._resolve_target(fired.run_id, origin)
            if target:
                self._reply(*target, fired.message, kind="notice")

    # ---- scheduler requeue hook (called under the scheduler lock) ----

    def on_request_requeued(self, parent_id: str, child_id: str) -> None:
        """Copy run ownership across a scheduler requeue. Must not lock."""
        owner = self._owners.get(parent_id)
        if owner:
            self._owners[child_id] = owner
            self._parents[child_id] = parent_id

    def _owner_of(self, run_id: str) -> Target | None:
        seen = 0
        while run_id and seen < 16:
            owner = self._owners.get(run_id)
            if owner:
                return owner
            run_id = self._parents.get(run_id)
            seen += 1
        return None

    def _resolve_target(self, run_id: str | None, origin: Target | None) -> Target | None:
        owner = self._owner_of(run_id or "")
        if owner:
            return owner
        if origin:
            return origin
        return self._fallback_target()

    # ---- outbound ----

    def _trace(self, kind: str, **fields) -> None:
        """Best-effort trace (a test double's scheduler may have no trace log)."""
        trace = getattr(self.scheduler, "trace", None)
        if trace is not None:
            trace.append(kind, "?", **fields)

    def _reply(
        self, platform: str, chat_id: str, text: str, kind: str = "reply"
    ) -> None:
        text = _strip_run_id(text)
        result_only = self._result_only(platform)
        skill_ref = _skill_ref(text) if kind == "result" else None
        humanize = kind == "result" and self._humanize_enabled(platform, skill_ref)
        if result_only:
            # A spoken front end says the skill's result, not the scheduler's
            # bookkeeping: drop queue/urgency chatter.
            if kind == "status":
                return
        if kind == "result" and (result_only or humanize):
            # Unwrap `<skill>: <ok|failed> — <detail>` to the skill's own
            # result before optionally cleaning it up.
            text = _result_detail(text)
            if humanize:
                started = time.time()
                text = self.humanizer.humanize(text)
                self._trace(
                    "humanize",
                    elapsed_s=round(time.time() - started, 3),
                    chars=len(text),
                )
        prefix = str(self._platform_cfg.get(platform, {}).get("reply_prefix", "") or "")
        if prefix:
            text = f"{prefix}{text}"
        if not text:
            return
        target_queue = self.outbound.get(platform)
        if target_queue is None:
            return
        self._trace("gateway_reply_queued", reply_kind=kind, chars=len(text))
        target_queue.put(OutboundMessage(chat_id=str(chat_id), text=text))
