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

