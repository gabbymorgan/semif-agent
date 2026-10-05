"""GatewayService: map external messenger traffic onto the SemIf scheduler.

The scheduler is a single execution slot with one global `pending` run. The
service does not fake concurrency: inbound text is fed through the existing
gate (`submit_request` -> gate/score/queue/dispatch), results and deferred
questions are routed back to the originating chat, and while one chat owns a
`needs_input` pause that chat answers it. Another chat messaging during that
pause is told to wait rather than silently abandoning the first chat's run.

Ownership: a request id maps to the chat that sent it. A run re-queued by the
scheduler (updated request) mints a new id; the scheduler's
`on_request_requeued` hook copies the owner across so the completion still
reaches the right chat.
"""

from __future__ import annotations

import queue
import re
import threading

from .base import GatewayAdapter, InboundMessage, OutboundMessage

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


def _strip_run_id(text: str) -> str:
    return _LEADING_RUN_ID.sub("", text or "")


def _leading_run_id(text: str) -> str:
    match = _LEADING_RUN_ID.match(text or "")
    return match.group(0)[1:-2] if match else ""


class GatewayService:
    def __init__(
        self,
        scheduler,
        adapter: GatewayAdapter,
        config: dict | None = None,
        poll_interval: float = 1.0,
    ):
        self.scheduler = scheduler
        self.adapter = adapter
        self.platform = adapter.name
        cfg = config or {}
        self.home_channel = str(cfg.get("home_channel", "") or "")
        self.reply_prefix = str(cfg.get("reply_prefix", "") or "")
        self.poll_interval = float(poll_interval)
        self.outbound: "queue.Queue[OutboundMessage | None]" = queue.Queue()
        self._owners: dict[str, str] = {}
        self._parents: dict[str, str] = {}
        self._pending_owner: str | None = None
        self._question_for_chat: dict[str, str] = {}
        self._repair_for_chat: dict[str, str] = {}
        self._approval_for_chat: dict[str, str] = {}
        self._surfaced: set[str] = set()
        self._route_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- lifecycle ----

    def start(self) -> None:
        """Start the background queue-drain loop (codegen re-queues, etc.)."""
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.outbound.put(None)

    def _poll_loop(self) -> None:
        while not self._stop.wait(self.poll_interval):
            try:
                with self._route_lock:
                    self.drain()
                    self.surface()
            except Exception:
                self.scheduler.trace.append("gateway_error", "?", phase="poll")

    # ---- routing ----

    def source_for(self, chat_id: str) -> str:
        return f"{self.platform}:{chat_id}"

    def _owner_from_source(self, source: str) -> str | None:
        prefix = f"{self.platform}:"
        if source.startswith(prefix):
            return source[len(prefix):]
        return None

    def handle_inbound(self, msg: InboundMessage) -> None:
        """Handle one authorized inbound command. Called by the adapter."""
        with self._route_lock:
            reply_to = msg.chat_id
            pending = self.scheduler.pending
            if pending is not None and self._pending_owner is None:
                self._pending_owner = self._owner_from_source(pending.request.source)
            if pending is not None and pending.request.source == self.source_for(msg.chat_id):
                status, detail = self.scheduler.answer(msg.text)
                if self.scheduler.pending is None:
                    self._pending_owner = None
                self._reply(reply_to, detail)
            elif pending is not None and self._pending_owner not in (None, msg.chat_id):
                self._reply(
                    reply_to,
                    "I'm waiting for a reply in another conversation; please resend in a moment.",
                )
            elif msg.chat_id in self._question_for_chat:
                question_id = self._question_for_chat.pop(msg.chat_id)
                status, detail = self.scheduler.answer_question(question_id, msg.text)
                self._surfaced.add(question_id)
                self._reply(reply_to, detail)
            elif msg.chat_id in self._repair_for_chat and msg.text.strip().lower() in _REPAIR_ACTIONS:
                offer_id = self._repair_for_chat.pop(msg.chat_id)
                action = _REPAIR_ACTIONS[msg.text.strip().lower()]
                status, detail = self.scheduler.resolve_repair(offer_id, action)
                self._surfaced.add(offer_id)
                self._reply(reply_to, detail)
            elif msg.chat_id in self._approval_for_chat and msg.text.strip().lower() in _APPROVAL_ACTIONS:
                approval_id = self._approval_for_chat.pop(msg.chat_id)
                approved = _APPROVAL_ACTIONS[msg.text.strip().lower()]
                status, detail = self.scheduler.answer_approval(approval_id, approved)
                self._surfaced.add(approval_id)
                self._reply(reply_to, detail)
            else:
                status, detail, run_id = self.scheduler.submit_request(
                    msg.text, source=self.source_for(msg.chat_id)
                )
                if run_id:
                    self._owners[run_id] = msg.chat_id
                if status == "needs_input":
                    self._pending_owner = msg.chat_id
                self._reply(reply_to, detail)
            self.drain()
            self.surface(origin=msg.chat_id)

    def drain(self) -> None:
        """Run queued work and route each outcome back to its owner."""
        for _ in range(100):
            results = self.scheduler.run_queue()
            if not results:
                break
            for status, detail in results:
                run_id = _leading_run_id(detail)
                chat = self._owner_of(run_id)
                text = _strip_run_id(detail)
                if status == "needs_input" and chat:
                    self._pending_owner = chat
                if chat is None:
                    chat = self.home_channel
                if chat:
                    self._reply(chat, text)

    def surface(self, origin: str | None = None) -> None:
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
            chat = self._owner_of(item["run_id"]) or origin or self.home_channel
            if chat:
                self._question_for_chat[chat] = item["id"]
                self._reply(
                    chat,
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
            chat = self._owner_of(offer["run_id"]) or origin or self.home_channel
            if chat:
                self._repair_for_chat[chat] = offer["id"]
                self._reply(
                    chat,
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
            chat = self._owner_of(item["run_id"]) or origin or self.home_channel
            if chat:
                self._approval_for_chat[chat] = item["id"]
                target = (
                    item["category"]
                    if item["kind"] == "category"
                    else f"{item['category']}.{item['skill']}"
                )
                self._reply(
                    chat,
                    (
                        f"Approve creating new {item['kind']} {target}: "
                        f"{item['description']}\nReply yes / no."
                    ),
                )

        for fired in self.scheduler.timers.drain():
            chat = self._owner_of(fired.run_id) or origin or self.home_channel
            if chat:
                self._reply(chat, fired.message)

    # ---- scheduler requeue hook (called under the scheduler lock) ----

    def on_request_requeued(self, parent_id: str, child_id: str) -> None:
        """Copy run ownership across a scheduler requeue. Must not lock."""
        owner = self._owners.get(parent_id)
        if owner:
            self._owners[child_id] = owner
            self._parents[child_id] = parent_id

    def _owner_of(self, run_id: str) -> str | None:
        seen = 0
        while run_id and seen < 16:
            owner = self._owners.get(run_id)
            if owner:
                return owner
            run_id = self._parents.get(run_id)
            seen += 1
        return None

    # ---- outbound ----

    def _reply(self, chat_id: str, text: str) -> None:
        text = _strip_run_id(text)
        if self.reply_prefix:
            text = f"{self.reply_prefix}{text}"
        if text:
            self.outbound.put(OutboundMessage(chat_id=str(chat_id), text=text))
