"""Engine config wiring: the per-machine GPU-offload knobs and the loader patch.

Pure stdlib: no SemIf, no llama.cpp, no GGUF. The SemIf backend is CPU-only by
construction, so the patch that lets it offload is exercised against a fake
backend module.
"""

from __future__ import annotations

from types import SimpleNamespace

from semif_agent.cli import build_engine_config
from semif_agent.engine import (
    EngineConfig,
    _install_gpu_state_reset,
    _offloading_params,
)

PINS = {
    "engine": {
        "source": "Qwen/Qwen3.5-4B",
        "revision": "0" * 40,
        "gguf_url": "https://example.com/Qwen_Qwen3.5-4B-Q4_K_M.gguf",
    }
}


def test_engine_config_defaults_to_cpu():
    cfg = build_engine_config({"engine": {"backend": "llamacpp"}}, PINS)
    assert cfg.gpu_layers == 0
    assert cfg.gpu_device is None
    assert cfg.gpu_split == 0


def test_engine_config_reads_gpu_knobs():
    cfg = build_engine_config(
        {"engine": {"gpu_layers": -1, "gpu_device": 1, "gpu_split": 0}}, PINS
    )
    assert cfg.gpu_layers == -1
    assert cfg.gpu_device == 1
    assert cfg.gpu_split == 0


def test_gpu_offload_wrapper_rewrites_model_params():
    def original(library):
        return SimpleNamespace(n_gpu_layers=0, main_gpu=0, split_mode=1)

    offloading = _offloading_params(
        original, EngineConfig(gpu_layers=99, gpu_device=1, gpu_split=0)
    )
    params = offloading(object())
    assert params.n_gpu_layers == 99
    assert params.main_gpu == 1
    assert params.split_mode == 0


def test_gpu_offload_wrapper_leaves_main_gpu_when_device_unset():
    def original(library):
        return SimpleNamespace(n_gpu_layers=0, main_gpu=7, split_mode=1)

    offloading = _offloading_params(original, EngineConfig(gpu_layers=-1))
    params = offloading(object())
    assert params.n_gpu_layers == -1
    assert params.main_gpu == 7  # untouched
    assert params.split_mode == 0


def test_gpu_state_reset_clears_cache_data():
    calls = []

    class FakeLib:
        def llama_memory_clear(self, memory, data):
            calls.append((memory, data))

    engine = SimpleNamespace(lib=FakeLib(), memory="mem")
    engine.clear = lambda: engine.lib.llama_memory_clear(engine.memory, False)
    _install_gpu_state_reset(SimpleNamespace(engine=engine))
    engine.clear()
    assert calls == [("mem", True)]
