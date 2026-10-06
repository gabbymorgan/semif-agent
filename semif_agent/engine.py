"""Real SemIf decision engine (no mocking).

Wraps the SemIf package (`semif_phase1`). The import and model load happen
lazily so the rest of the agent is pure stdlib and testable without SemIf
installed. On hardware that is neither CUDA nor Apple, use the shipped
llama.cpp backend: a local GGUF scored on CPU (or Vulkan if your llama.cpp
build enables it). This engine is only usable on a machine with SemIf and the
pinned GGUF available; elsewhere calls raise EngineUnavailable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from .decisions import DecisionRequest, DecisionResult


class EngineUnavailable(RuntimeError):
    """SemIf is not installed or the configured model is missing."""


@dataclass
class EngineConfig:
    backend: str = "llamacpp"
    source: str = "Qwen/Qwen3.5-4B"
    revision: str = ""
    gguf: str = ""
    context_tokens: int = 4096
    threads: int | None = None


class SemIfEngine:
    """One pinned SemIf model, loaded once and used for every decision."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._metadata = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> tuple:
        if self._model is not None:
            return self._model, self._tokenizer, self._metadata
        if self.config.backend != "llamacpp":
            raise EngineUnavailable(
                f"Unsupported backend {self.config.backend!r}; use 'llamacpp'."
            )
        if not self.config.gguf or not Path(self.config.gguf).is_file():
            raise EngineUnavailable(
                f"GGUF not found: {self.config.gguf!r}. Set engine.gguf in config.json."
            )
        try:
            from semif_phase1 import llamacpp_backend as backend
        except ImportError as exc:
            raise EngineUnavailable(
                "SemIf is not installed here. Install it on the target box with "
                "`pip install -e '.[test,llamacpp]'`."
            ) from exc
        try:
            model, tokenizer, metadata = backend.load_model(
                self.config.source,
                self.config.revision,
                self.config.gguf,
                threads=self.config.threads,
                context_tokens=self.config.context_tokens,
            )
        except Exception as exc:
            raise EngineUnavailable(f"Failed to load the SemIf model: {exc}") from exc
        self._model, self._tokenizer, self._metadata = model, tokenizer, metadata
        return model, tokenizer, metadata

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def call(self, request: DecisionRequest) -> DecisionResult:
        with self._lock:
            model, tokenizer, metadata = self._ensure_loaded()
            from semif_phase1 import llamacpp_backend as backend

            row = request.to_semif_row()
            result = backend.score(model, tokenizer, row, metadata)
            return DecisionResult(
                request=request,
                option_ids=list(result["option_ids"]),
                probabilities=list(result["probabilities"]),
                extra={
                    "prompt_sha256": result.get("prompt_sha256"),
                    "input_tokens": result.get("input_tokens"),
                    "forward_seconds": result.get("forward_seconds"),
                    "total_seconds": result.get("total_seconds"),
                },
            )

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Drive the pinned decision model in the normal way: text generation.

        SemIf scoring reads option logits directly; this instead autoregressively
        samples from the same llama.cpp context, e.g. for skill-tree authoring.
        Generation decodes the chat template through the backend's low-level
        context (there is no high-level chat-completion object on the CPU
        backend), stopping at the tokenizer's eos token. The KV cache is cleared
        at the start, so interleaving scoring and generation on one model is safe.

        It shares `self._lock` with `call`, so a generation serializes against
        decisions on the one loaded model — short JSON replies are seconds of
        contention, and there is only ever one copy of the GGUF.
        """
        with self._lock:
            model, tokenizer, metadata = self._ensure_loaded()
            try:
                import numpy

                engine = model.engine
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
                engine.clear()
                logits = engine._decode(prompt_ids, 0, 0, True)
                generated: list[int] = []
                rng = numpy.random.default_rng()
                for position in range(max_tokens):
                    token = _sample_token(logits, temperature, rng)
                    if token == tokenizer.eos_token_id:
                        break
                    generated.append(token)
                    logits = engine._decode([token], len(prompt_ids) + position, 0, True)
                return tokenizer.decode(generated).strip()
            except EngineUnavailable:
                raise
            except Exception as exc:
                raise EngineUnavailable(f"generation failed: {exc}") from exc


def _sample_token(logits, temperature: float, rng) -> int:
    """Pick the next token from next-position logits: greedy or temperature."""
    import numpy

    if temperature <= 0.0:
        return int(numpy.argmax(logits))
    scaled = numpy.asarray(logits, dtype=numpy.float64) / max(temperature, 1e-6)
    scaled = scaled - scaled.max()
    probabilities = numpy.exp(scaled)
    probabilities /= probabilities.sum()
    return int(rng.choice(probabilities.size, p=probabilities))

