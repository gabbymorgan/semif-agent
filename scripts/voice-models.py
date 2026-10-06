#!/usr/bin/env python3
"""Download the voice-gateway models into the checkout's `.runtime/voice` tree.

Run after installing `requirements/voice.txt` (bootstrap does this with
`--voice`). Idempotent: a model already on disk is skipped. Reads
`gateway.voice` from config.json so the downloaded models match the configured
names, or take explicit flags.

- **Wake word** (openWakeWord): downloaded by `openwakeword.utils.download_models`
  into the venv's package resources dir (inside `.runtime/venv/`, so still
  contained in the checkout). The detector resolves the model by name there.
- **TTS** (Piper): the voice `.onnx` + `.onnx.json` are fetched from the
  `rhasspy/piper-voices` HF repo into `.runtime/voice/tts/`.
- **STT** (faster-whisper): instantiating the model warms it into the HF cache
  (`.runtime/hf`); the runner reuses that cache at runtime.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME = REPO_ROOT / ".runtime" / "voice"
PIPER_REPO = "rhasspy/piper-voices"


def piper_hf_path(voice: str) -> str:
    """Map a Piper voice id to its file path in the `rhasspy/piper-voices` repo.

    `en_US-lessac-medium` -> `en/en_US/lessac/medium/en_US-lessac-medium.onnx`
    (the voice name may itself contain dashes).
    """
    parts = voice.split("-")
    if len(parts) < 3:
        raise ValueError(f"unrecognized piper voice id {voice!r}")
    locale, quality = parts[0], parts[-1]
    name = "-".join(parts[1:-1])
    lang = locale.split("_")[0]
    return f"{lang}/{locale}/{name}/{quality}/{voice}.onnx"


def download_wake(name: str) -> None:
    import openwakeword.utils as utils

    print(f"== wake word model: {name}")
    # Default target: the venv's package resources dir (inside .runtime/venv).
    # openWakeWord resolves the model (and its shared mel/embedding models) by
    # name from there.
    utils.download_models(model_names=[name])


def download_voice(voice: str, target: Path) -> None:
    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    rel = piper_hf_path(voice)
    for suffix in ("", ".json"):
        dest = target / f"{voice}.onnx{suffix}"
        if dest.is_file():
            print(f"== tts voice: {dest.name} (present)")
            continue
        print(f"== tts voice: {voice}{suffix or '.onnx'} -> {dest}")
        cached = hf_hub_download(repo_id=PIPER_REPO, filename=rel + suffix)
        shutil.copyfile(cached, dest)


def download_stt(model: str) -> None:
    from faster_whisper import WhisperModel

    print(f"== stt model: {model} (warming HF cache)")
    WhisperModel(model, device="cpu", compute_type="int8")


def _config_voice(config_path: Path) -> dict:
    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return {}
    return (config.get("gateway", {}) or {}).get("voice", {}) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "config.json"))
    parser.add_argument("--wake", default=None, help="openWakeWord model name")
    parser.add_argument("--stt", default=None, help="faster-whisper model size/path")
    parser.add_argument("--tts", default=None, help="Piper voice id")
    parser.add_argument("--tts-dir", default=str(RUNTIME / "tts"))
    parser.add_argument("--skip-wake", action="store_true")
    parser.add_argument("--skip-stt", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    args = parser.parse_args(argv)

    voice_cfg = _config_voice(Path(args.config))
    wake = args.wake or (voice_cfg.get("wake", {}) or {}).get("model") or "hey_mycroft"
    stt = args.stt or (voice_cfg.get("stt", {}) or {}).get("model") or "base.en"
    tts = args.tts or (voice_cfg.get("tts", {}) or {}).get("voice") or "en_US-lessac-medium"

    if not args.skip_wake:
        download_wake(wake)
    if not args.skip_tts:
        download_voice(tts, Path(args.tts_dir))
    if not args.skip_stt:
        download_stt(stt)
    print("== voice models ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
