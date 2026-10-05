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

from .base import InboundMessage, OutboundMessage

_LEADING_RUN_ID = re.compile(r"^\[[0-9a-f]{6,}\]\s*")

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


def _leading_run_id(text: str) -> str:
    match = _LEADING_RUN_ID.match(text or "")
    return match.group(0)[1:-2] if match else ""


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
                self._reply(*reply_to, detail)
            elif pending is not None and self._pending_owner not in (None, reply_to):
                self._reply(
                    *reply_to,
                    "I'm waiting for a reply in another conversation; please resend in a moment.",
                )
            elif reply_to in self._question_for_chat:
                question_id = self._question_for_chat.pop(reply_to)
                status, detail = self.scheduler.answer_question(question_id, msg.text)
                self._surfaced.add(question_id)
                self._reply(*reply_to, detail)
            elif reply_to in self._repair_for_chat and msg.text.strip().lower() in _REPAIR_ACTIONS:
                offer_id = self._repair_for_chat.pop(reply_to)
                action = _REPAIR_ACTIONS[msg.text.strip().lower()]
                status, detail = self.scheduler.resolve_repair(offer_id, action)
                self._surfaced.add(offer_id)
                self._reply(*reply_to, detail)
            elif reply_to in self._approval_for_chat and msg.text.strip().lower() in _APPROVAL_ACTIONS:
                approval_id = self._approval_for_chat.pop(reply_to)
                approved = _APPROVAL_ACTIONS[msg.text.strip().lower()]
                status, detail = self.scheduler.answer_approval(approval_id, approved)
                self._surfaced.add(approval_id)
                self._reply(*reply_to, detail)
            else:
                status, detail, run_id = self.scheduler.submit_request(
                    msg.text, source=self.source_for(platform, msg.chat_id)
                )
                if run_id:
                    self._owners[run_id] = reply_to
                if status == "needs_input":
                    self._pending_owner = reply_to
                self._reply(*reply_to, detail)
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
                    self._reply(*target, text)

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
                )

        for fired in self.scheduler.timers.drain():
            target = self._resolve_target(fired.run_id, origin)
            if target:
                self._reply(*target, fired.message)

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

    def _reply(self, platform: str, chat_id: str, text: str) -> None:
        text = _strip_run_id(text)
        prefix = str(self._platform_cfg.get(platform, {}).get("reply_prefix", "") or "")
        if prefix:
            text = f"{prefix}{text}"
        if not text:
            return
        target_queue = self.outbound.get(platform)
        if target_queue is None:
            return
        target_queue.put(OutboundMessage(chat_id=str(chat_id), text=text))
