"""Optional LLM cleanup of a skill's result line for the messenger gateway.

The scheduler's run summary is deterministic and technical — ``<skill>:
<ok|failed> — <action_log>``, where the action log is whatever the skill body
wrote (often mechanical: ``PUT 201 …``, ``sent via /_send to contact 12``).
This layer rewrites only that result detail into one grounded, friendly
sentence for a chat front end.

It is generation, never a decision: the success/failure verdict and the routing
are already settled by SemIf and the runner, and the rewrite is never fed back
into assessment, tracing, or the decision log. The prompt forbids adding,
inferring, or dropping facts, and `humanize` falls back to the original text on
any error, timeout, empty reply, refusal, or runaway length — a cleanup failure
must never swallow or corrupt a real result.

It reuses the scheduler's configured ``llm`` client (its endpoint/model/sampler)
and passes a short per-call ``timeout`` so a slow cleanup cannot hold the
gateway's poll loop for the client's full authoring budget. It is deliberately
gateway-only: the runner and the REPL keep the raw, deterministic summary.
"""

from __future__ import annotations

#: The strict rephrasing instruction. Grounding is enforced by the prompt and
#: by the caller's fallback, not by a second model call.
SYSTEM_PROMPT = (
    "You rewrite an assistant's raw action result into one short, friendly "
    "sentence for a chat reply.\n"
    "Rules:\n"
    "- Use ONLY the facts in the result. Do not add, guess, or infer anything.\n"
    "- Preserve whether the action succeeded or failed exactly as stated.\n"
    "- Do not mention servers, HTTP status codes, command names, file paths, "
    "or internal bookkeeping unless that is the only content.\n"
    "- One sentence. No preamble, no explanation, no quotes, no markdown.\n"
    "- If the result is already a plain sentence, return it unchanged."
)

#: Leading phrases that mark a meta/refusal reply rather than a rewrite.
_REFUSAL_PREFIXES = (
    "i cannot",
    "i can't",
    "i'm sorry",
    "i am sorry",
    "i'm unable",
    "i am unable",
    "as an ai",
    "sorry,",
)


def _clean_reply(raw: str) -> str:
    """Normalize a model reply into one bare sentence, or "" if unusable."""
    text = (raw or "").strip()
    if not text:
        return ""
    # Drop a wrapping pair of quotes the model may add.
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”‘’":
        text = text[1:-1].strip()
    # Collapse newlines / runs of whitespace to a single line.
    return " ".join(text.split())


class Humanizer:
    """Rewrite a result line with an LLM, falling back to the source verbatim."""

    def __init__(
        self,
        client,
        *,
        enabled: bool = True,
        timeout: float = 20.0,
        max_tokens: int = 200,
        max_chars: int = 600,
    ):
        self.client = client
        self.enabled = bool(enabled) and client is not None
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.max_chars = int(max_chars)

    def humanize(self, text: str) -> str:
        """Return a grounded rewrite of `text`, or `text` unchanged.

        Never raises: any failure (unreachable client, timeout, empty or
        unusable reply, refusal, runaway length) returns the original text.
        """
        source = (text or "").strip()
        if not self.enabled or not source:
            return text
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": source},
        ]
        try:
            raw = self.client.chat(
                messages, max_tokens=self.max_tokens, timeout=self.timeout
            )
        except Exception:
            return text
        cleaned = _clean_reply(raw)
        if not cleaned:
            return text
        if cleaned.lower().startswith(_REFUSAL_PREFIXES):
            return text
        if len(cleaned) > self.max_chars:
            return text
        return cleaned
