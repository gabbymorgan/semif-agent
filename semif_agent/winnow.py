"""Remote Winnow decision engine: a `/v1/systemone` typed-decision HTTP API.

An alternative to the local SemIf llama.cpp engine (`engine.SemIfEngine`),
selected per machine with ``engine.provider == "winnow"`` in config.json — the
same provider pattern as ``llm`` / ``codegen``. The agent's flat decision
contract (state + question + typed options) maps onto Winnow's ``choice``
question type: the options become the question's ``criteria`` (id -> description)
and the question becomes its ``instructions``; the server returns a probability
over exactly those keys.

Winnow runs as its own process (``winnow-server``); see ``local/ENVIRONMENT.md``
for this deployment. The engine is real: any network, HTTP, or distribution
failure raises `EngineUnavailable`, which the scheduler treats as fatal — there
is no fallback model and no fabricated reply.
"""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .decisions import DecisionRequest, DecisionResult
from .engine import DecisionEngine, EngineUnavailable


@dataclass
class WinnowConfig:
    """Per-machine settings for the remote Winnow engine (`engine.winnow`)."""

    base_url: str = "http://127.0.0.1:8091"
    model: str = "Winnow-12B"
    timeout: float = 120.0
    # Transient-failure retries before declaring the engine unavailable. 0 means
    # a single attempt: an unreachable remote engine is fatal by contract.
    retries: int = 0
    api_key: str = ""
    extra_headers: dict | None = None
    user_agent: str = "semif-agent"


class WinnowEngine(DecisionEngine):
    """One Winnow `/v1/systemone` endpoint, used for every decision.

    Scoring maps `DecisionRequest` -> a single ``choice`` question and reads the
    returned distribution back over the requested option ids. The endpoint is a
    single-slot server, so calls serialize on an internal lock exactly as
    `SemIfEngine` serializes on its llama.cpp context.
    """

    label = "winnow"

    def __init__(self, config: WinnowConfig):
        self.config = config
        self._lock = threading.Lock()
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    # ---- transport ----

    def _headers(self) -> dict:
        headers = {
            "content-type": "application/json",
            "accept": "application/json",
        }
        if self.config.user_agent:
            headers["user-agent"] = self.config.user_agent
        if self.config.api_key:
            headers["authorization"] = f"Bearer {self.config.api_key}"
        for key, value in (self.config.extra_headers or {}).items():
            headers[str(key)] = str(value)
        return headers

    def _post(self, path: str, payload: dict) -> dict:
        """POST JSON, returning the parsed object or raising `EngineUnavailable`."""
        url = self.config.base_url.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")
        attempts = max(0, int(self.config.retries)) + 1
        last: Exception | None = None
        for attempt in range(attempts):
            request = urllib.request.Request(
                url, data=data, method="POST", headers=self._headers()
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.config.timeout
                ) as response:
                    body = response.read().decode("utf-8", errors="replace")
                parsed = json.loads(body)
                if not isinstance(parsed, dict):
                    raise EngineUnavailable(
                        f"Winnow response from {url} is not a JSON object"
                    )
                return parsed
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", errors="replace")[:300]
                except Exception:
                    pass
                last = EngineUnavailable(
                    f"HTTP {exc.code} from {url}: {detail}".strip()
                )
                # A 4xx (except 429) will not succeed on retry.
                if 400 <= exc.code < 500 and exc.code != 429:
                    break
            except EngineUnavailable:
                raise
            except Exception as exc:  # URLError, timeout, socket, JSONDecodeError
                last = exc
            if attempt + 1 < attempts:
                time.sleep(min(2.0**attempt, 5.0))
        if isinstance(last, EngineUnavailable):
            raise last
        raise EngineUnavailable(
            f"Winnow request to {url} failed: {type(last).__name__}: {last}"
        )

    # ---- DecisionEngine ----

    def warm(self) -> None:
        """Verify the endpoint is reachable (`GET /health`), else raise."""
        url = self.config.base_url.rstrip("/") + "/health"
        request = urllib.request.Request(url, method="GET", headers=self._headers())
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout
            ) as response:
                if getattr(response, "status", 200) != 200:
                    raise EngineUnavailable(
                        f"Winnow health check returned HTTP {response.status}"
                    )
        except EngineUnavailable:
            raise
        except Exception as exc:
            raise EngineUnavailable(f"Winnow unreachable at {url}: {exc}") from exc
        self._loaded = True

    def call(self, request: DecisionRequest) -> DecisionResult:
        option_ids = [option.id for option in request.options]
        payload = {
            "state": request.state,
            "model": self.config.model,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": request.question,
                    "criteria": {
                        option.id: option.description for option in request.options
                    },
                }
            },
        }
        with self._lock:
            started = time.perf_counter()
            body = self._post("/v1/systemone", payload)
            elapsed = time.perf_counter() - started

        answers = body.get("answers")
        answer = answers.get("decision") if isinstance(answers, dict) else None
        if not isinstance(answer, dict):
            raise EngineUnavailable("Winnow response is missing answers.decision")
        if answer.get("type") not in (None, "choice"):
            raise EngineUnavailable(
                f"Winnow answer type {answer.get('type')!r} is not 'choice'"
            )
        raw = answer.get("probabilities")
        if not isinstance(raw, dict):
            raise EngineUnavailable("Winnow answer is missing probabilities")
        if set(raw) != set(option_ids):
            raise EngineUnavailable(
                "Winnow probabilities do not match the requested options: "
                f"{sorted(raw)} != {sorted(option_ids)}"
            )
        try:
            values = [float(raw[option_id]) for option_id in option_ids]
        except (TypeError, ValueError) as exc:
            raise EngineUnavailable(
                f"Winnow probabilities are not numeric: {exc}"
            ) from exc
        if any((not math.isfinite(value)) or value < 0.0 or value > 1.0 for value in values):
            raise EngineUnavailable(
                "Winnow returned a non-finite or out-of-range probability"
            )
        total = sum(values)
        if total <= 0.0:
            raise EngineUnavailable("Winnow returned an all-zero distribution")
        probabilities = [value / total for value in values]
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        self._loaded = True
        return DecisionResult(
            request=request,
            option_ids=option_ids,
            probabilities=probabilities,
            extra={
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "forward_seconds": elapsed,
                "total_seconds": elapsed,
                "raw_probability_sum": total,
                "served_model": body.get("model"),
                "confidence": answer.get("confidence"),
            },
        )

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Text generation via the same server's `/v1/chat/completions`.

        Backs `llm.provider == "semif"` when the decision engine is Winnow.
        """
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        with self._lock:
            body = self._post("/v1/chat/completions", payload)
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise EngineUnavailable("Winnow chat response has no choices")
        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise EngineUnavailable("Winnow chat response has no message content")
        self._loaded = True
        return content.strip()
