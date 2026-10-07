"""The decision-engine contract and the local SemIf implementation.

`DecisionEngine` is the pluggable contract every provider satisfies; the
provider is selected per machine by `engine.provider` in config.json, exactly
like `llm.provider` / `codegen.provider`. `SemIfEngine` is the default: it wraps
the SemIf package (`semif_phase1`). The import and model load happen
lazily so the rest of the agent is pure stdlib and testable without SemIf
installed. The shipped llama.cpp backend scores a local GGUF; it runs on CPU by
default, and `EngineConfig.gpu_layers` offloads to a GPU when llama-cpp-python
was built with a GPU backend (Vulkan/ROCm/CUDA). This engine is only usable on a
machine with SemIf and the pinned GGUF available; elsewhere calls raise
EngineUnavailable. The remote alternative lives in `semif_agent.winnow`.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .decisions import DecisionRequest, DecisionResult


class EngineUnavailable(RuntimeError):
    """The decision engine is not installed, not reachable, or failed.

    This is the one error a decision engine raises. It is fatal by contract:
    the scheduler records it (`_mark_fatal`) and the front ends exit, because
    the agent cannot make decisions without a real engine.
    """


class DecisionEngine(ABC):
    """The pluggable decision-engine contract.

    A decision engine answers a `DecisionRequest` (state + question + typed
    options) with a `DecisionResult` (probabilities over exactly those options).
    It is selected per machine by `engine.provider` in config.json, mirroring
    `llm.provider` / `codegen.provider`:

    * ``"semif"`` (default) — the local llama.cpp GGUF (`SemIfEngine`).
    * ``"winnow"`` — a remote ``/v1/systemone`` typed-decision API
      (`semif_agent.winnow.WinnowEngine`).

    The engine is always real: any failure to produce a decision raises
    `EngineUnavailable`, which the scheduler treats as fatal. `generate` is the
    optional text-generation face of the same model (used by
    `llm.provider == "semif"`); an engine that cannot generate text raises
    `EngineUnavailable` there rather than inventing a reply.
    """

    label: str = "engine"

    @property
    @abstractmethod
    def loaded(self) -> bool:
        """Whether the underlying model is loaded/warm."""

    @abstractmethod
    def warm(self) -> None:
        """Load/warm the model, raising `EngineUnavailable` on failure."""

    @abstractmethod
    def call(self, request: DecisionRequest) -> DecisionResult:
        """Score one decision, raising `EngineUnavailable` on failure."""

    def generate(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Autoregressively generate text from the same model, if supported."""
        raise EngineUnavailable(
            f"{self.label} engine does not support text generation"
        )


@dataclass
class EngineConfig:
    backend: str = "llamacpp"
    source: str = "Qwen/Qwen3.5-4B"
    revision: str = ""
    gguf: str = ""
    context_tokens: int = 4096
    threads: int | None = None
    # GPU offload (llama.cpp). 0 keeps the CPU-only default; a positive count
    # offloads that many layers, -1 offloads all of them. Requires a
    # llama-cpp-python built with a GPU backend (Vulkan/ROCm/CUDA).
    gpu_layers: int = 0
    gpu_device: int | None = None
    gpu_split: int = 0


def _require_gpu_offload() -> None:
    """Fail loudly when GPU layers are requested but the build cannot offload."""
    import llama_cpp

    if not llama_cpp.llama_supports_gpu_offload():
        raise EngineUnavailable(
            "engine.gpu_layers is set but this llama-cpp-python build has no GPU "
            "backend. Rebuild it with one, e.g. "
            "CMAKE_ARGS='-DGGML_VULKAN=on' (Vulkan), '-DGGML_HIP=on' (ROCm) or "
            "'-DGGML_CUDA=on' (CUDA)."
        )


def _install_gpu_state_reset(model) -> None:
    """Force a full cache reset between scored prompts when running on the GPU.

    SemIf's `_Engine.clear` calls `llama_memory_clear(memory, False)`: it resets
    the cache metadata but leaves the stored data. The CPU backend tolerates that
    (the slots are overwritten before use), but the Vulkan backend keeps stale
    hybrid-attention/recurrent state, so repeated `score` calls drift and return
    wrong probabilities. Clearing the data as well restores determinism and
    matches the CPU path. Applied only when offloading, so CPU behaviour is
    untouched.
    """
    engine = model.engine
    library = engine.lib

    def clear():
        library.llama_memory_clear(engine.memory, True)

    engine.clear = clear


def _offloading_params(original, config: EngineConfig):
    """Wrap SemIf's CPU-only model-params factory to offload layers to the GPU.

    `semif_phase1.llamacpp_backend` is CPU-only by construction: its
    `_cpu_model_params` hardcodes `n_gpu_layers = 0`. The backend looks that name
    up on its own module while loading, so wrapping it for the duration of the
    load lets the same path offload without patching (and losing) the re-cloned
    engine source. The caller restores the original afterwards, so a CPU engine
    loaded later in the same process is unaffected.
    """

    def offloading(library):
        params = original(library)
        params.n_gpu_layers = config.gpu_layers
        params.split_mode = config.gpu_split
        if config.gpu_device is not None:
            params.main_gpu = config.gpu_device
        return params

    return offloading


class SemIfEngine(DecisionEngine):
    """One pinned SemIf model, loaded once and used for every decision.

    The default `engine.provider`: the local llama.cpp GGUF. Text generation
    from the same model backs `llm.provider == "semif"`.
    """

    label = "semif"

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
        if self.config.gpu_layers:
            _require_gpu_offload()
        original_params = backend._cpu_model_params
        if self.config.gpu_layers:
            backend._cpu_model_params = _offloading_params(original_params, self.config)
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
        finally:
            backend._cpu_model_params = original_params
        if self.config.gpu_layers:
            _install_gpu_state_reset(model)
        # The backend hardcodes n_gpu_layers=0 in its metadata; record what we
        # actually asked llama.cpp to offload instead.
        metadata = dict(metadata)
        metadata["n_gpu_layers"] = self.config.gpu_layers
        if self.config.gpu_device is not None:
            metadata["main_gpu"] = self.config.gpu_device
        self._model, self._tokenizer, self._metadata = model, tokenizer, metadata
        return model, tokenizer, metadata

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def warm(self) -> None:
        """Load the model (idempotent), raising `EngineUnavailable` on failure."""
        self._ensure_loaded()

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

