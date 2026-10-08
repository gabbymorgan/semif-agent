#!/usr/bin/env python3
"""Record a personal Piper TTS dataset (LJSpeech format) from the microphone.

This is the recording front end for the voice-training pipeline: it reads a
prompts file (default `scripts/voice-prompts-en.txt`, the public-domain Harvard
sentences) and walks the prompts one at a time, writing an LJSpeech dataset
(`metadata.csv` + `wavs/NNNN.wav`) that `scripts/voice-train.sh` consumes on the
training host. See `docs/VOICE-TRAIN.md`.

Two modes:

- **Interactive** (default): a mic check, then each prompt is shown; `Enter`
  records, then `Enter` accept / `r` redo / `p` play / `s` skip / `q` quit.
- **Hands-free** (`--speak-prompts` or `--read-prompts`): no keypresses. With
  `--speak-prompts` the recorder reads each sentence aloud and you repeat it;
  with `--read-prompts` it **prints** each line and beeps (no speech) — read it
  aloud. Either way it records until you stop speaking and moves on. It stops
  after `--max-misses` consecutive silent prompts.

Recordings are **biometric** and stay under the checkout's gitignored
`.runtime/` tree (default output) — never commit them.

Run it from the checkout with the voice stack installed:

    .runtime/venv/bin/python scripts/voice-record.py --list-devices
    .runtime/venv/bin/python scripts/voice-record.py --speak-prompts

`Ctrl-C` stops and keeps what is already saved; re-running resumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from semif_agent.voice_train import (  # noqa: E402
    DEFAULT_MAX_SECONDS,
    DEFAULT_MIN_SPEECH_MS,
    DEFAULT_SILENCE_MS,
    SAMPLE_RATE,
    DatasetStore,
    duration_s,
    level_report,
    load_prompts,
    record_until_silence,
)
from semif_agent.voice_transport import (  # noqa: E402
    PiperSynthesizer,
    SoundDeviceAudio,
    _tone_pcm,
    resolve_model_path,
)

DEFAULT_PROMPT_VOICE = "en_US-lessac-medium"


def _voice_config(config_path: Path) -> dict:
    try:
        return ((json.loads(config_path.read_text()).get("gateway", {}) or {}).get(
            "voice", {}
        ) or {})
    except Exception:
        return {}


def _voice_audio_defaults(config_path: Path) -> tuple[str, str]:
    """The voice gateway's configured input/output devices, if present."""
    audio = (_voice_config(config_path).get("audio", {}) or {})
    return (
        str(audio.get("input_device", "") or ""),
        str(audio.get("output_device", "") or ""),
    )


def _prompt_voice_default(config_path: Path) -> str:
    tts = (_voice_config(config_path).get("tts", {}) or {})
    return str(tts.get("voice", "") or DEFAULT_PROMPT_VOICE)


def _list_devices() -> int:
    try:
        import sounddevice as sd
    except Exception as exc:  # pragma: no cover - depends on the host stack
        print(f"sounddevice is not installed: {exc}")
        print("install the voice stack: scripts/bootstrap.sh --voice")
        return 2
    print(sd.query_devices())
    return 0


def _play(audio: SoundDeviceAudio, pcm16: bytes, sample_rate: int) -> None:
    if not pcm16:
        return
    try:
        audio.write(pcm16, sample_rate)
    except Exception as exc:
        print(f"  (playback failed: {exc})")


def _level_note(report: dict) -> str:
    if report["peak_dbfs"] is None:
        return ""
    if report["peak_dbfs"] > -1.0:
        return "  WARNING: clipping — lower the mic gain."
    if report["peak_dbfs"] < -30.0:
        return "  WARNING: very quiet — raise the mic gain."
    return ""


# --------------------------------------------------------------------------- #
# interactive mode
# --------------------------------------------------------------------------- #


def _mic_check(audio: SoundDeviceAudio, args: argparse.Namespace) -> bool:
    """Record a short sample, report levels, and let the user accept or retry."""
    while True:
        print(
            f"\nMic check: say a sentence (ends on ~{args.silence_ms} ms of silence, "
            f"max {args.mic_check_seconds:.0f}s)."
        )
        input("  press Enter to record… ")
        pcm = record_until_silence(
            audio,
            sample_rate=args.sample_rate,
            frame_ms=args.frame_ms,
            max_seconds=args.mic_check_seconds,
            silence_ms=args.silence_ms,
            min_speech_ms=args.min_speech_ms,
            threshold=args.vad_threshold,
            read_timeout=args.read_timeout,
        )
        if not pcm:
            print("  no speech detected. Check the input device and mic level")
            print("  (alsamixer; PortAudio has no volume control), then retry.")
        else:
            report = level_report(pcm, args.sample_rate)
            print(
                f"  captured {report['seconds']}s  rms={report['rms']}  "
                f"peak={report['peak']} ({report['peak_dbfs']} dBFS)"
                + _level_note(report)
            )
        choice = input("  [Enter] start recording, p play back, r retry, q quit: ").strip().lower()
        if choice == "q":
            return False
        if choice == "p":
            _play(audio, pcm, args.sample_rate)
            continue
        if choice == "r":
            continue
        return True


def _record_prompt(audio: SoundDeviceAudio, args: argparse.Namespace) -> bytes:
    """Record one prompt, letting the user accept / redo / play / skip."""
    while True:
        input("  press Enter to record… ")
        pcm = record_until_silence(
            audio,
            sample_rate=args.sample_rate,
            frame_ms=args.frame_ms,
            max_seconds=args.max_seconds,
            silence_ms=args.silence_ms,
            min_speech_ms=args.min_speech_ms,
            threshold=args.vad_threshold,
            read_timeout=args.read_timeout,
        )
        if not pcm:
            print("  no speech detected; try again.")
            continue
        report = level_report(pcm, args.sample_rate)
        print(f"  recorded {report['seconds']}s  peak={report['peak_dbfs']} dBFS")
        choice = input("  [Enter] accept, r redo, p play, s skip, q quit: ").strip().lower()
        if choice == "q":
            raise KeyboardInterrupt
        if choice == "r":
            continue
        if choice == "p":
            _play(audio, pcm, args.sample_rate)
            continue
        if choice == "s":
            return b""
        return pcm


# --------------------------------------------------------------------------- #
# hands-free (read-aloud) mode
# --------------------------------------------------------------------------- #


def _say(tts, audio: SoundDeviceAudio, text: str, args: argparse.Namespace) -> None:
    """Speak `text` through the output device, then beep (mic is flushed)."""
    try:
        pcm, rate = tts.synthesize(text)
        audio.flush()
        audio.write(pcm, rate)
    except Exception as exc:
        print(f"  (prompt playback failed: {exc})")
    audio.flush()
    cue = _tone_pcm(880, 120, args.sample_rate, 0.4)
    if cue:
        try:
            audio.write(cue, args.sample_rate)
        except Exception:
            pass
        audio.flush()


def _beep(audio: SoundDeviceAudio, args: argparse.Namespace) -> None:
    """Short attention beep (the 'read now' cue) — no speech."""
    cue = _tone_pcm(880, 120, args.sample_rate, 0.4)
    if not cue:
        return
    audio.flush()
    try:
        audio.write(cue, args.sample_rate)
    except Exception as exc:
        print(f"  (beep failed: {exc})")
    audio.flush()


def _pending_prompts(prompts, store: DatasetStore, args) -> list[tuple[int, str, str]]:
    """The ``(index, uid, text)`` prompts still to record.

    A prompt is pending when its id is not in the dataset, or when `--overwrite`
    is set. `--limit` caps how many are returned. This is what the session
    prints up front so a resume over a mostly-complete dataset is never
    mistaken for a fresh run.
    """
    pending: list[tuple[int, str, str]] = []
    for index, text in enumerate(prompts, start=1):
        uid = f"{index:04d}"
        if args.overwrite or uid not in store.ids():
            pending.append((index, uid, text))
    if args.limit:
        pending = pending[: args.limit]
    return pending


def _hands_free_session(audio, tts, store: DatasetStore, prompts, args) -> int:
    """Walk prompts: speak (or show + beep) each, record the repeat.

    In `--read-prompts` mode the line is printed and a beep cues the take
    (no text-to-speech). Stops after `--max-misses` consecutive silent prompts.
    """
    speak = not args.read_prompts
    recorded = 0
    misses = 0
    pending = _pending_prompts(prompts, store, args)
    if not pending:
        print("every prompt is already recorded (pass --overwrite to redo them).")
        return 0
    print(f"{len(pending)} of {len(prompts)} prompt(s) to record.", flush=True)
    if speak:
        _say(tts, audio, "Let's begin. Repeat each sentence after the beep.", args)
    for position, (_index, uid, text) in enumerate(pending, start=1):
        print(f"\n[{position}/{len(pending)}] {text}", flush=True)
        pcm = b""
        for attempt in range(2):
            if speak:
                _say(tts, audio, text, args)
            else:
                _beep(audio, args)
            pcm = record_until_silence(
                audio,
                sample_rate=args.sample_rate,
                frame_ms=args.frame_ms,
                max_seconds=args.max_seconds,
                silence_ms=args.silence_ms,
                min_speech_ms=args.min_speech_ms,
                threshold=args.vad_threshold,
                read_timeout=args.read_timeout,
            )
            if pcm:
                break
            if attempt == 0:
                print("  no speech heard; repeating the cue…", flush=True)
        if not pcm:
            misses += 1
            print(
                f"  no speech ({misses}/{args.max_misses}) — skipping; re-run to resume.",
                flush=True,
            )
            if misses >= args.max_misses:
                print("  too many silent prompts — stopping so it doesn't run away.", flush=True)
                break
            continue
        misses = 0
        store.save(uid, text, pcm)
        recorded += 1
        report = level_report(pcm, args.sample_rate)
        print(
            f"  saved {uid}.wav  ({store.count()} total, {report['seconds']}s)"
            + _level_note(report),
            flush=True,
        )
    return recorded


def main(argv: list[str] | None = None) -> int:
    default_in, default_out = _voice_audio_defaults(REPO_ROOT / "config.json")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / ".runtime" / "voice-train" / "dataset"),
                        help="dataset directory (metadata.csv + wavs/)")
    parser.add_argument("--prompts", default=str(REPO_ROOT / "scripts" / "voice-prompts-en.txt"),
                        help="prompts file (one sentence per line; # comments skipped)")
    parser.add_argument("--device", default=default_in,
                        help="input device (PortAudio name substring or index; default: config)")
    parser.add_argument("--output-device", default=default_out,
                        help="output device for playback (default: config)")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--frame-ms", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                        help="hard cap per utterance")
    parser.add_argument("--silence-ms", type=int, default=DEFAULT_SILENCE_MS,
                        help="trailing silence that ends an utterance")
    parser.add_argument("--min-speech-ms", type=int, default=DEFAULT_MIN_SPEECH_MS)
    parser.add_argument("--vad-threshold", type=float, default=200.0,
                        help="int16 RMS above which a frame counts as speech")
    parser.add_argument("--read-timeout", type=float, default=3.0,
                        help="seconds with no audio frame before a take is ended (guards a dropped device)")
    parser.add_argument("--mic-check-seconds", type=float, default=20.0,
                        help="hard cap for the mic-check sample")
    parser.add_argument("--skip-mic-check", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="record at most N prompts")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-record prompts whose id is already saved")
    parser.add_argument("--speak-prompts", action="store_true",
                        help="hands-free: speak each prompt aloud and record the repeat (no keypresses)")
    parser.add_argument("--read-prompts", action="store_true",
                        help="hands-free: print each line and beep (no speech); read it aloud, recorded")
    parser.add_argument("--max-misses", type=int, default=3,
                        help="stop after this many consecutive silent prompts in --speak-prompts mode")
    parser.add_argument("--prompt-voice", default=_prompt_voice_default(REPO_ROOT / "config.json"),
                        help="Piper voice used to read the prompts in --speak-prompts mode")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args(argv)

    if args.list_devices:
        return _list_devices()

    prompts = load_prompts(args.prompts)
    if not prompts:
        print(f"no prompts found in {args.prompts}")
        return 2
    store = DatasetStore(args.out, sample_rate=args.sample_rate)
    print(f"dataset: {store.out}  ({store.count()} already recorded)")
    print(f"prompts: {len(prompts)}  target rate: {args.sample_rate} Hz")

    audio = SoundDeviceAudio(input_device=args.device, output_device=args.output_device)
    try:
        audio.open_input(args.sample_rate, args.frame_ms, "")
    except Exception as exc:
        print(f"could not open input device {args.device!r}: {exc}")
        print("list devices with --list-devices; install the stack with scripts/bootstrap.sh --voice")
        return 2

    tts = None
    if args.speak_prompts:
        voice_path = resolve_model_path(
            args.prompt_voice, str(REPO_ROOT / ".runtime" / "voice" / "tts"), (".onnx",)
        )
        try:
            tts = PiperSynthesizer(voice_path)
        except Exception as exc:
            print(f"could not load the prompt voice {voice_path!r}: {exc}")
            print("download it with scripts/voice-models.py, or pass --prompt-voice <path>")
            audio.close()
            return 2

    recorded = 0
    try:
        if args.speak_prompts or args.read_prompts:
            recorded = _hands_free_session(audio, tts, store, prompts, args)
        else:
            if not args.skip_mic_check and not _mic_check(audio, args):
                print("aborted at mic check.")
                return 1
            pending = _pending_prompts(prompts, store, args)
            if not pending:
                print("every prompt is already recorded (pass --overwrite to redo them).")
                return 0
            print(f"{len(pending)} of {len(prompts)} prompt(s) to record.")
            for position, (_index, uid, text) in enumerate(pending, start=1):
                print(f"\n[{position}/{len(pending)}] Say: {text}")
                pcm = _record_prompt(audio, args)
                if not pcm:
                    print("  skipped.")
                    continue
                path = store.save(uid, text, pcm)
                recorded += 1
                print(f"  saved {path.name}  ({store.count()} total, {duration_s(pcm, args.sample_rate):.1f}s)")
    except KeyboardInterrupt:
        print("\nstopping (already-recorded prompts are saved).")
    finally:
        audio.close()

    print(f"\ndone: {recorded} recorded this session, {store.count()} total in {store.out}")
    print(f"metadata: {store.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
