"""Low-level voice transport (neutral): mic capture, wake word, VAD, STT, TTS.

This module owns the audio mechanics only: a continuous 16 kHz mono capture
stream, an always-on wake-word detector, silence endpointing (VAD), speech to
text, and text to speech. It knows nothing about the SemIf scheduler or the
command gateway — a front end builds on top of it (`semif_agent.gateway.voice`).

Every heavy dependency (`sounddevice`, `openwakeword`, `faster_whisper`,
`piper`, `onnxruntime`, `webrtcvad`) is imported lazily, so importing this
module stays stdlib-only on a machine without the optional voice stack. The
gateway refuses to start with an install hint instead (`check_requirements`).

Audio contract: 16 kHz, mono, signed 16-bit little-endian PCM, read in fixed
`frame_ms` blocks. That is what openWakeWord and faster-whisper both want, so no
resampling is done in the loop.

Design notes:
- **Half-duplex by default, barge-in optional.** Capture frames are normally
  discarded while a reply is being spoken, so the agent never transcribes its
  own voice. With barge-in enabled (`barge_in.enabled`), the wake word is still
  scored during playback: the user can interrupt a spoken reply, the playback
  is aborted, and the following command is captured and handed to the front end
  (which decides whether it is a stop or a new request). Speech is muted while
  the interrupt command is captured and handled, so the interrupted reply does
  not resume over the user.
- The engine interfaces are plain classes so tests can inject fakes and drive
  the loop without hardware or the optional packages.
"""

from __future__ import annotations

import importlib.util
import io
import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import Callable

#: Callback invoked for each transcribed utterance (the spoken text).
UtteranceHandler = Callable[[str], None]

#: The one audio format the whole pipeline uses.
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2

#: Required optional packages -> install hint shown by `check_requirements`.
_REQUIRED_MODULES: tuple[tuple[str, str], ...] = (
    ("sounddevice", "sounddevice (needs the system PortAudio lib: apt install libportaudio2)"),
    ("openwakeword", "openwakeword"),
    ("faster_whisper", "faster-whisper"),
    ("piper", "piper-tts"),
    ("onnxruntime", "onnxruntime"),
)

#: The one package a transcribe-only front end needs (see
#: `check_transcription_requirements`). `faster-whisper` bundles PyAV, whose
#: ffmpeg decodes an incoming Opus/OGG voice message to PCM.
_TRANSCRIBE_MODULES: tuple[tuple[str, str], ...] = (
    ("faster_whisper", "faster-whisper"),
)

#: The packages a voice-reply front end needs (see
#: `check_synthesis_requirements`): Piper for text-to-speech and PyAV for the
#: Opus/OGG container a Sideband HQ push-to-talk reply rides in. No microphone,
#: wake word, or transcription is implied — a headless host can synthesize a
#: reply without the hardware voice stack.
_SYNTHESIS_MODULES: tuple[tuple[str, str], ...] = (
    ("piper", "piper-tts"),
    ("av", "av (PyAV, Opus/OGG encoding)"),
)

#: Sideband's HQ push-to-talk container (`LXMF.AM_OPUS_OGG`) is 48 kHz mono
#: OGG/Opus; a reply's PCM is resampled to that rate before encoding.
OPUS_OGG_SAMPLE_RATE = 48000
OPUS_OGG_BITRATE = 24000


# --------------------------------------------------------------------------- #
# Engine interfaces (fakes are injected in tests)
# --------------------------------------------------------------------------- #


class WakeWordDetector:
    """Scores one PCM frame; a score >= threshold fires the wake word."""

    def score(self, pcm16: bytes) -> float:
        raise NotImplementedError

    def reset(self) -> None:
        """Clear detector state (called after a fire and after speaking)."""

    def close(self) -> None:
        pass


class VadGate:
    """Decides whether one PCM frame contains speech (for endpointing)."""

    def is_speech(self, pcm16: bytes, sample_rate: int) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass


class Transcriber:
    """Transcribes a complete utterance (16 kHz int16 PCM) to text."""

    def transcribe(self, pcm16: bytes) -> str:
        raise NotImplementedError

    def transcribe_bytes(self, data: bytes) -> str:
        """Transcribe an encoded audio container (e.g. OGG/Opus) to text.

        Default: not supported. `WhisperTranscriber` overrides it so a received
        voice message (a Sideband PTT Opus/OGG payload) is transcribed without a
        separate decode step.
        """
        raise NotImplementedError

    def close(self) -> None:
        pass


class Synthesizer:
    """Synthesizes text to `(pcm16, sample_rate)`."""

    def synthesize(self, text: str) -> tuple[bytes, int]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class AudioIO:
    """Mic capture and speaker playback.

    `read(timeout)` returns one frame's bytes; with a `timeout` it returns
    `b""` when no frame arrived in time, and `None` once the input is closed
    (the loop exits on `None`). The timeout is how the loop detects a stalled
    capture stream and recovers from it. `open_output`/`write` may be called at
    any sample rate the synthesizer produced.
    """

    def open_input(self, sample_rate: int, frame_ms: int, device: str = "") -> None:
        raise NotImplementedError

    def read(self, timeout: float | None = None) -> bytes | None:
        raise NotImplementedError

    def open_output(self, sample_rate: int, device: str = "") -> None:
        pass

    def write(self, pcm16: bytes, sample_rate: int) -> None:
        raise NotImplementedError

    def play(self, pcm16: bytes, sample_rate: int, stop_event=None) -> bool:
        """Play PCM, returning `False` if aborted early via `stop_event`.

        The default plays the whole buffer in one `write`. The real device
        overrides this to stream in small chunks so a barge-in stops the reply
        promptly instead of after the whole buffer; a fake that only implements
        `write` still plays through here.
        """
        self.write(pcm16, sample_rate)
        return True

    def flush(self) -> None:
        """Drop any buffered capture frames (called after speaking to clear echo)."""

    def close(self) -> None:
        pass


@dataclass
class VoiceEngines:
    """The bundle of engines the loop drives."""

    wake: WakeWordDetector
    vad: VadGate
    stt: Transcriber
    tts: Synthesizer
    audio: AudioIO


# --------------------------------------------------------------------------- #
# Real engines (lazy imports)
# --------------------------------------------------------------------------- #


class OpenWakeWordDetector(WakeWordDetector):
    """openWakeWord detector. `model` is a bundled name or a `.onnx`/`.tflite` path."""

    def __init__(self, model: str, threshold: float = 0.5, inference_framework: str = "onnx"):
        from openwakeword.model import Model

        self.threshold = float(threshold)
        self._model = Model(
            wakeword_models=[model] if model else None,
            inference_framework=str(inference_framework or "onnx"),
        )

    def score(self, pcm16: bytes) -> float:
        import numpy as np

        audio = np.frombuffer(pcm16, dtype=np.int16)
        predictions = self._model.predict(audio)
        if not predictions:
            return 0.0
        return float(max(predictions.values()))

    def reset(self) -> None:
        try:
            self._model.reset()
        except Exception:
            pass


class WebRtcVadGate(VadGate):
    """`webrtcvad` endpointing. Splits a frame into 20 ms sub-frames and votes."""

    def __init__(self, aggressiveness: int = 2):
        import webrtcvad

        self._vad = webrtcvad.Vad(int(aggressiveness))

    def is_speech(self, pcm16: bytes, sample_rate: int) -> bool:
        chunk = int(sample_rate * 0.02) * BYTES_PER_SAMPLE  # 20 ms
        if chunk <= 0:
            return False
        votes = 0
        total = 0
        for offset in range(0, len(pcm16) - chunk + 1, chunk):
            total += 1
            try:
                if self._vad.is_speech(pcm16[offset : offset + chunk], sample_rate):
                    votes += 1
            except Exception:
                return False
        return total > 0 and votes * 2 >= total


class EnergyVadGate(VadGate):
    """Stdlib fallback endpointing: int16 RMS above a threshold.

    Less robust than `webrtcvad`, but keeps the loop usable when the optional
    package is absent.
    """

    def __init__(self, threshold: float = 200.0):
        self.threshold = float(threshold)

    def is_speech(self, pcm16: bytes, sample_rate: int) -> bool:
        count = len(pcm16) // BYTES_PER_SAMPLE
        if count == 0:
            return False
        total = 0
        for index in range(0, count * BYTES_PER_SAMPLE, BYTES_PER_SAMPLE):
            sample = int.from_bytes(pcm16[index : index + BYTES_PER_SAMPLE], "little", signed=True)
            total += sample * sample
        rms = (total / count) ** 0.5
        return rms >= self.threshold


class WhisperTranscriber(Transcriber):
    """faster-whisper (CTranslate2) transcription. `model` is a size or a path."""

    def __init__(
        self,
        model: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
        beam_size: int = 1,
        vad_filter: bool = True,
    ):
        from faster_whisper import WhisperModel

        self.language = str(language or "en") or None
        self.beam_size = int(beam_size)
        self.vad_filter = bool(vad_filter)
        self._model = WhisperModel(str(model or "base.en"), device=str(device or "cpu"),
                                   compute_type=str(compute_type or "int8"))

    @staticmethod
    def _join(segments) -> str:
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe(self, pcm16: bytes) -> str:
        import numpy as np

        audio = np.frombuffer(pcm16, dtype=np.int16).astype("float32") / 32768.0
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        return self._join(segments)

    def transcribe_bytes(self, data: bytes) -> str:
        """Transcribe an encoded audio container (e.g. OGG/Opus).

        faster-whisper decodes the container to 16 kHz mono internally via PyAV
        (its own bundled ffmpeg), so a PTT voice message's Opus/OGG payload is
        transcribed directly — no separate decode step and no system ffmpeg.
        `data` is the raw container bytes.
        """
        import io

        segments, _info = self._model.transcribe(
            io.BytesIO(bytes(data)),
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        return self._join(segments)


class PiperSynthesizer(Synthesizer):
    """Piper TTS. `voice` is a voice name (resolved to a `.onnx`) or a path."""

    def __init__(self, voice: str, speaker_id: int = 0, length_scale: float = 1.0,
                 volume: float = 1.0):
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(voice))
        self._syn_config = _piper_syn_config(speaker_id, length_scale, volume)

    def synthesize(self, text: str) -> tuple[bytes, int]:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_fn = getattr(self._voice, "synthesize_wav", None)
            if callable(wav_fn):
                # piper-tts >= 1.3: synthesize_wav(text, wav_file, syn_config=...)
                wav_fn(text, wav_file, syn_config=self._syn_config)
            else:
                # older piper-tts: synthesize(text, wav_file)
                try:
                    self._voice.synthesize(text, wav_file, syn_config=self._syn_config)
                except TypeError:
                    self._voice.synthesize(text, wav_file)
        buffer.seek(0)
        with wave.open(buffer, "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes()), wav_file.getframerate()


def _piper_syn_config(speaker_id: int, length_scale: float, volume: float = 1.0):
    """A piper `SynthesisConfig` when the installed version provides one."""
    try:
        from piper import SynthesisConfig
    except Exception:
        return None
    kwargs: dict = {}
    if speaker_id:
        kwargs["speaker_id"] = int(speaker_id)
    if length_scale and float(length_scale) != 1.0:
        kwargs["length_scale"] = float(length_scale)
    if volume and float(volume) != 1.0:
        kwargs["volume"] = float(volume)
    if not kwargs:
        return None
    try:
        return SynthesisConfig(**kwargs)
    except Exception:
        return None


def _remove_dc(pcm16: bytes) -> bytes:
    """Subtract the per-block mean (kills a DC offset some mics carry)."""
    import numpy as np

    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return pcm16
    samples = samples - samples.mean()
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


def _tone_pcm(frequency: float, duration_ms: int, sample_rate: int, volume: float) -> bytes:
    """Generate a short sine beep as 16-bit mono PCM (stdlib only).

    A 10 ms fade in/out avoids clicks. Used for the wake-word cue; it needs no
    audio asset and no numpy, so it works on the stdlib-only path too.
    """
    import math
    import struct

    count = int(sample_rate * max(0, duration_ms) / 1000)
    if count <= 0 or frequency <= 0:
        return b""
    amplitude = int(max(0.0, min(1.0, volume)) * 32767)
    fade = max(1, int(sample_rate * 0.01))  # 10 ms
    frames = bytearray()
    for index in range(count):
        envelope = min(1.0, index / fade, (count - index) / fade)
        value = int(amplitude * envelope * math.sin(2 * math.pi * frequency * index / sample_rate))
        frames += struct.pack("<h", value)
    return bytes(frames)


def _tone_sequence(notes, sample_rate: int, volume: float, gap_ms: int = 0) -> bytes:
    """Concatenate `_tone_pcm` notes (each `[frequency_hz, duration_ms]`).

    A short inter-note silence (`gap_ms`) makes a multi-note chime read as
    distinct notes rather than one warble. Each note already fades in and out,
    so there is no click at a boundary whether or not a gap is used.
    """
    gap = b"\x00\x00" * int(sample_rate * max(0, gap_ms) / 1000)
    parts: list[bytes] = []
    for frequency, duration in notes:
        tone = _tone_pcm(frequency, duration, sample_rate, volume)
        if not tone:
            continue
        if parts and gap:
            parts.append(gap)
        parts.append(tone)
    return b"".join(parts)


def _parse_notes(value) -> list[tuple[float, int]]:
    """Parse a cue note list (`[[frequency_hz, duration_ms], ...]`).

    Bad entries are dropped rather than raising: this is cosmetic audio config,
    not a contract. An empty/absent list yields `[]` (the single-beep fallback).
    """
    notes: list[tuple[float, int]] = []
    for item in value or []:
        try:
            frequency, duration = item
            frequency, duration = float(frequency), int(duration)
        except (TypeError, ValueError):
            continue
        if frequency > 0 and duration > 0:
            notes.append((frequency, duration))
    return notes


def _device_arg(value):
    """Coerce a config device value to a PortAudio argument.

    PortAudio takes an int index or a name substring; a numeric string like
    "5" is treated as a name and fails to match, so digits become an int.
    Empty means the system default.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lstrip("-").isdigit():
        return int(text)
    return text


def _resample_pcm16(pcm16: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linear-interpolate int16 PCM from `source_rate` to `target_rate`.

    numpy is a voice-stack dependency and this is only reached by the real
    `SoundDeviceAudio` engine, so the core stays stdlib-only. Audio hardware
    rarely accepts 16 kHz directly, so capture blocks are resampled down and
    synthesized PCM is resampled up to the device's native rate.
    """
    if source_rate == target_rate or not pcm16:
        return pcm16
    import numpy as np

    source = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    count = int(round(len(source) * target_rate / source_rate))
    if count <= 0:
        return b""
    positions = np.linspace(0.0, len(source) - 1, count)
    resampled = np.interp(positions, np.arange(len(source)), source)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


class SoundDeviceAudio(AudioIO):
    """PortAudio capture/playback via `sounddevice` (raw int16 streams).

    Real audio devices rarely accept the 16 kHz processing rate directly, so
    capture runs at the device's native rate and each block is resampled down to
    the processing rate; playback resamples the synthesized PCM up to the output
    device's native rate. The native rate defaults to the device's reported
    `default_samplerate` and can be pinned with `capture_sample_rate` /
    `output_sample_rate`.
    """

    def __init__(self, input_device: str = "", output_device: str = "",
                 capture_sample_rate: int = 0, output_sample_rate: int = 0,
                 dc_removal: bool = True):
        self.input_device = _device_arg(input_device)
        self.output_device = _device_arg(output_device)
        self.capture_sample_rate = int(capture_sample_rate or 0)
        self.output_sample_rate = int(output_sample_rate or 0)
        self.dc_removal = bool(dc_removal)
        self._stream = None
        self._process_rate = SAMPLE_RATE
        self._capture_rate = 0
        self._frames: "queue.Queue[bytes | None]" = queue.Queue(maxsize=128)
        self._closed = threading.Event()

    def _device_rate(self, device, kind: str) -> int:
        """The device's native sample rate (or a sane fallback)."""
        import sounddevice as sd

        try:
            info = sd.query_devices(device, kind)
            rate = int(info.get("default_samplerate") or 0)
        except Exception:
            rate = 0
        return rate or 48000

    def open_input(self, sample_rate: int, frame_ms: int, device: str = "") -> None:
        import sounddevice as sd

        self._closed.clear()
        while not self._frames.empty():  # drop any stale sentinel/frames
            self._frames.get_nowait()
        self._process_rate = int(sample_rate)
        dev = _device_arg(device) or self.input_device
        self._capture_rate = self.capture_sample_rate or self._device_rate(dev, "input")
        block = max(1, int(self._capture_rate * frame_ms / 1000))
        self._stream = sd.RawInputStream(
            samplerate=self._capture_rate,
            channels=1,
            dtype="int16",
            blocksize=block,
            device=dev,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        if self._closed.is_set():
            return
        data = bytes(indata)
        if self.dc_removal:
            data = _remove_dc(data)
        if self._capture_rate and self._capture_rate != self._process_rate:
            data = _resample_pcm16(data, self._capture_rate, self._process_rate)
        try:
            self._frames.put_nowait(data)
        except queue.Full:
            pass  # a slow consumer (STT) drops stale frames; the mic is real-time

    def read(self, timeout: float | None = None) -> bytes | None:
        """Block for the next frame (or up to `timeout` seconds).

        Returns `None` once the input is closed (the sentinel `close()` puts on
        the queue), or `b""` when a `timeout` elapses with no frame — so a
        caller that passes a timeout can detect a stalled capture stream
        instead of blocking forever.
        """
        if timeout is None:
            return self._frames.get()
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return b""

    def open_output(self, sample_rate: int, device: str = "") -> None:
        pass  # a raw stream is opened per utterance in `write`

    def write(self, pcm16: bytes, sample_rate: int) -> None:
        import sounddevice as sd

        rate = self.output_sample_rate or self._device_rate(
            self.output_device, "output"
        )
        if rate != int(sample_rate):
            pcm16 = _resample_pcm16(pcm16, int(sample_rate), rate)
        with sd.RawOutputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            device=self.output_device,
        ) as stream:
            stream.write(pcm16)

    def play(self, pcm16: bytes, sample_rate: int, stop_event=None) -> bool:
        """Stream PCM in small chunks so a barge-in can abort playback.

        A single `write` of the whole buffer blocks until the device drains it,
        so an interrupt could only take effect after the reply had finished. The
        buffer is played in ~50 ms chunks instead, checking `stop_event` between
        them; on a stop the stream is aborted (discarding what is still buffered)
        and `False` is returned. One output stream is opened for the whole
        buffer, so the chunks play gaplessly.
        """
        import sounddevice as sd

        rate = self.output_sample_rate or self._device_rate(
            self.output_device, "output"
        )
        if rate != int(sample_rate):
            pcm16 = _resample_pcm16(pcm16, int(sample_rate), rate)
        chunk = max(BYTES_PER_SAMPLE, int(rate * 0.05) * BYTES_PER_SAMPLE)  # 50 ms
        with sd.RawOutputStream(
            samplerate=rate,
            channels=1,
            dtype="int16",
            device=self.output_device,
        ) as stream:
            for offset in range(0, len(pcm16), chunk):
                if stop_event is not None and stop_event.is_set():
                    try:
                        stream.abort()
                    except Exception:
                        pass
                    return False
                stream.write(pcm16[offset : offset + chunk])
        return True

    def flush(self) -> None:
        try:
            while not self._frames.empty():
                self._frames.get_nowait()
        except queue.Empty:
            pass

    def close(self) -> None:
        self._closed.set()
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        try:  # clear stale frames so the sentinel is what the reader gets next
            while not self._frames.empty():
                self._frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait(None)  # unblock a pending read()
        except queue.Full:
            pass


# --------------------------------------------------------------------------- #
# Engine construction
# --------------------------------------------------------------------------- #


def resolve_model_path(name: str, directory: str, suffixes: tuple[str, ...]) -> str:
    """Return `name` as-is if it is a file, else `directory/<name><suffix>`.

    Lets config carry a short model name (downloaded by bootstrap into
    `.runtime/voice/...`) while still accepting an explicit path. A versioned
    file (`<name>_v0.1.onnx`) also matches by prefix.
    """
    if not name:
        return name
    import glob
    import os

    if os.path.isfile(name):
        return name
    for suffix in suffixes:
        candidate = os.path.join(directory, f"{name}{suffix}")
        if os.path.isfile(candidate):
            return candidate
        matches = sorted(glob.glob(os.path.join(directory, f"{name}*{suffix}")))
        if matches:
            return matches[0]
    return name


def _build_vad(cfg: dict) -> VadGate:
    vad_cfg = cfg.get("vad", {}) or {}
    if importlib.util.find_spec("webrtcvad") is not None:
        return WebRtcVadGate(aggressiveness=int(vad_cfg.get("aggressiveness", 2)))
    return EnergyVadGate(threshold=float(vad_cfg.get("energy_threshold", 200.0)))


def build_transcriber(cfg: dict) -> Transcriber:
    """Build just the speech-to-text engine from an `stt` config block.

    `build_engines` also needs a microphone, a wake word, and TTS; a headless
    front end that only transcribes received audio — the LXMF gateway handling a
    Sideband PTT voice message — uses this instead, so it needs only
    `faster-whisper` and none of the hardware voice stack.
    """
    stt_cfg = (cfg or {}).get("stt", {}) or {}
    return WhisperTranscriber(
        model=str(stt_cfg.get("model", "base.en") or "base.en"),
        device=str(stt_cfg.get("device", "cpu") or "cpu"),
        compute_type=str(stt_cfg.get("compute_type", "int8") or "int8"),
        language=str(stt_cfg.get("language", "en") or "en"),
    )


def check_transcription_requirements() -> tuple[bool, str | None]:
    """Return `(ok, install_hint)` for the transcribe-only path.

    Unlike `VoiceDaemon.check_requirements` this needs no microphone, wake word,
    or TTS — only `faster-whisper` — so a headless host can transcribe incoming
    voice messages without the hardware voice stack. Never raises.
    """
    missing = []
    for module, hint in _TRANSCRIBE_MODULES:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(hint)
    if missing:
        return False, (
            "install the transcription stack (requirements/transcribe.txt): "
            + "; ".join(missing)
        )
    return True, None


def check_synthesis_requirements() -> tuple[bool, str | None]:
    """Return `(ok, install_hint)` for the voice-reply (Piper) path.

    A voice reply needs Piper (text-to-speech) and PyAV (to encode the
    synthesized PCM into the Opus/OGG container a Sideband HQ push-to-talk
    message rides in). Neither a microphone, a wake word, nor transcription is
    implied — the headless counterpart of `VoiceDaemon.check_requirements`.
    Never raises.
    """
    missing = []
    for module, hint in _SYNTHESIS_MODULES:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(hint)
    if missing:
        return False, (
            "install the voice-reply stack (requirements/tts.txt): "
            + "; ".join(missing)
        )
    return True, None


def pcm_to_opus_ogg(
    pcm16: bytes, sample_rate: int, bitrate: int = OPUS_OGG_BITRATE
) -> bytes:
    """Encode 16-bit mono PCM into an OGG/Opus container (Sideband HQ PTT).

    A received voice message is Opus/OGG (`LXMF.AM_OPUS_OGG`), so a voice reply
    is encoded to the same container: the PCM is resampled to 48 kHz (Opus's
    fullband rate) with PyAV's `AudioResampler` and muxed into OGG by its
    bundled libopus. Returns `b""` on any failure (the caller falls back to a
    text reply) — never raises, so a missing PyAV degrades gracefully.
    """
    if not pcm16 or len(pcm16) < BYTES_PER_SAMPLE:
        return b""
    try:
        import av
    except Exception:
        return b""
    try:
        rate = int(sample_rate or OPUS_OGG_SAMPLE_RATE)
        source = av.AudioFrame(
            format="s16", layout="mono", samples=len(pcm16) // BYTES_PER_SAMPLE
        )
        source.sample_rate = rate
        source.planes[0].update(pcm16)
        if rate == OPUS_OGG_SAMPLE_RATE:
            frames = [source]
        else:
            resampler = av.AudioResampler(
                format="s16", layout="mono", rate=OPUS_OGG_SAMPLE_RATE
            )
            # PyAV's `resample` returns a list of frames (older versions a single
            # frame or None); normalize both shapes.
            frames = resampler.resample(source)
            if not isinstance(frames, list):
                frames = [frames]
            tail = resampler.resample(None)
            frames += tail if isinstance(tail, list) else [tail]
        frames = [frame for frame in frames if frame is not None]
        if not frames:
            return b""
        buffer = io.BytesIO()
        container = av.open(buffer, mode="w", format="ogg")
        stream = container.add_stream("libopus", rate=OPUS_OGG_SAMPLE_RATE)
        stream.layout = "mono"
        stream.bit_rate = int(bitrate or OPUS_OGG_BITRATE)
        try:  # voice-optimised mode (Sideband's HQ PTT uses the voice profile)
            stream.codec_context.options = {"application": "voip"}
        except Exception:
            pass
        for frame in frames:
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()
        return buffer.getvalue()
    except Exception:
        return b""


def check_codec2_requirements() -> tuple[bool, str | None]:
    """Return `(ok, install_hint)` for Codec2 voice-message decoding.

    Codec2 (Sideband's non-HQ push-to-talk) needs the optional `pycodec2`
    bindings, which compile against the system `libcodec2`. Never raises.
    """
    try:
        importlib.import_module("pycodec2")
    except Exception:
        return False, (
            "pip install pycodec2 (needs the system libcodec2: "
            "apt install libcodec2-dev)"
        )
    return True, None


def decode_codec2(data: bytes, rate: int) -> bytes | None:
    """Decode a raw Codec2 bitstream to 8 kHz mono int16 PCM.

    `rate` is the codec2 bitrate in bits/s (700..3200). Returns None when the
    optional `pycodec2` bindings (or the mode) are unavailable, so the caller
    reports the message unsupported rather than guessing. Frames shorter than a
    full codec frame are dropped; decoding mirrors Sideband's per-frame loop.
    """
    if not data or not rate:
        return None
    try:
        import pycodec2
    except Exception:
        return None
    try:
        codec = pycodec2.Codec2(int(rate))
        bytes_per_frame = int(codec.bytes_per_frame())
    except Exception:
        return None
    if bytes_per_frame <= 0:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    frames = [
        np.asarray(codec.decode(data[offset : offset + bytes_per_frame]), dtype=np.int16)
        for offset in range(0, len(data) - bytes_per_frame + 1, bytes_per_frame)
    ]
    if not frames:
        return None
    return np.concatenate(frames).astype(np.int16).tobytes()


def pcm_to_wav(pcm16: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw int16 PCM in a minimal in-memory WAV container (stdlib).

    Lets a non-container payload (Codec2 frames decoded to 8 kHz PCM) ride the
    same transcription path as an Opus/OGG message: faster-whisper's PyAV decodes
    the WAV and resamples it to 16 kHz, so the boundary needs no manual resample.
    """
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(int(channels))
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm16)
    return buffer.getvalue()


def _build_piper(tts_cfg: dict) -> PiperSynthesizer:
    """Build the Piper synthesizer from a `tts` config block (shared by the
    microphone front end and the headless LXMF voice-reply path)."""
    tts_voice = resolve_model_path(
        str(tts_cfg.get("voice", "") or ""),
        str(tts_cfg.get("voice_dir", "") or ""),
        (".onnx",),
    )
    return PiperSynthesizer(
        tts_voice,
        speaker_id=int(tts_cfg.get("speaker_id", 0) or 0),
        length_scale=float(tts_cfg.get("length_scale", 1.0) or 1.0),
        volume=float(tts_cfg.get("volume", 1.0) or 1.0),
    )


def build_synthesizer(cfg: dict) -> Synthesizer:
    """Build just the text-to-speech engine from a `tts` config block.

    `build_engines` also needs a microphone, a wake word, and speech-to-text; a
    headless front end that only synthesizes a reply — the LXMF gateway
    answering a Sideband push-to-talk voice message with a Piper voice message —
    uses this instead, so it needs only `piper-tts` (and PyAV to encode the
    reply), none of the hardware voice stack.
    """
    tts_cfg = (cfg or {}).get("tts", {}) or {}
    return _build_piper(tts_cfg)


def build_engines(cfg: dict, trace=None) -> VoiceEngines:
    """Construct the real engine bundle from a `gateway.voice` config block."""
    wake_cfg = cfg.get("wake", {}) or {}
    tts_cfg = cfg.get("tts", {}) or {}
    audio_cfg = cfg.get("audio", {}) or {}

    wake_model = resolve_model_path(
        str(wake_cfg.get("model", "hey_computer") or ""),
        str(wake_cfg.get("model_dir", "") or ""),
        (".onnx", ".tflite"),
    )
    return VoiceEngines(
        wake=OpenWakeWordDetector(
            wake_model,
            threshold=float(wake_cfg.get("threshold", 0.2)),
            inference_framework=str(wake_cfg.get("inference_framework", "onnx")),
        ),
        vad=_build_vad(cfg),
        stt=build_transcriber(cfg),
        tts=_build_piper(tts_cfg),
        audio=SoundDeviceAudio(
            input_device=str(audio_cfg.get("input_device", "") or ""),
            output_device=str(audio_cfg.get("output_device", "") or ""),
            capture_sample_rate=int(audio_cfg.get("capture_sample_rate", 0) or 0),
            output_sample_rate=int(audio_cfg.get("output_sample_rate", 0) or 0),
            dc_removal=bool(audio_cfg.get("dc_removal", True)),
        ),
    )


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


class VoiceDaemon:
    """Drives the engines: listen for the wake word, endpoint, transcribe, speak.

    `run(on_utterance)` blocks on the mic until `stop()` (or the input closes),
    calling `on_utterance(text)` for each transcribed utterance. `speak(text)`
    synthesizes and plays a reply; it is safe to call from another thread (the
    adapter's outbound pump) while `run` is listening.
    """

    def __init__(self, cfg: dict | None = None, trace=None, engines: VoiceEngines | None = None,
                 name: str = "voice", input_ready=None, on_barge_in=None):
        cfg = cfg or {}
        self.config = cfg
        self.trace = trace
        self.name = name
        #: Optional predicate: whether a new command may start now. While it
        #: returns False the wake word is ignored (the scheduler's single slot
        #: is still running the current request), so a second command cannot be
        #: started before the first reaches a result. None = always ready.
        self.input_ready = input_ready
        #: Optional callback the loop invokes around a barge-in: `True` when the
        #: wake word is heard over a spoken reply (playback aborted, listening
        #: begins), `False` once that listening session ends. The front end uses
        #: it to classify the following utterance as an interrupt.
        self.on_barge_in = on_barge_in
        self.wake_threshold = float((cfg.get("wake", {}) or {}).get("threshold", 0.2))
        #: After the wake word, how long to wait for the command's speech to
        #: begin before giving up (a natural pause after "hey <word>" — e.g.
        #: waiting for the cue beep — must not truncate the command).
        self.listen_timeout_ms = int(float(cfg.get("listen_timeout_s", 3.0) or 0) * 1000)
        audio_cfg = cfg.get("audio", {}) or {}
        self.sample_rate = int(audio_cfg.get("sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        self.frame_ms = int(audio_cfg.get("frame_ms", 80) or 80)
        self.input_device = str(audio_cfg.get("input_device", "") or "")
        #: Capture-stream watchdog. `read_timeout_s` is how long a frame read
        #: waits before it counts as a stall; when the stream stalls
        #: continuously for `stall_timeout_s` the input is closed and reopened
        #: (a wedged PortAudio/ALSA callback otherwise blocks `read()` forever).
        #: 0 for either restores the old blocking read with no recovery.
        self.read_timeout_s = float(audio_cfg.get("read_timeout_s", 1.0) or 0.0)
        self.stall_timeout_s = float(audio_cfg.get("stall_timeout_s", 5.0) or 0.0)
        self._stall_since = 0.0
        vad_cfg = cfg.get("vad", {}) or {}
        self.vad_silence_ms = int(vad_cfg.get("silence_ms", 1000) or 1000)
        self.min_speech_ms = int(vad_cfg.get("min_speech_ms", 120) or 120)
        self.max_utterance_s = float(vad_cfg.get("max_utterance_s", 30) or 30)
        #: A longer silence tolerance while the utterance is still short. The
        #: wake detector can fire inside the wake phrase (the cue beep follows),
        #: so the first speech after it is the phrase's tail, not the command;
        #: a natural pause before the command must not end the capture. Once
        #: `min_command_ms` of speech is in hand, the normal `vad_silence_ms`
        #: endpoint applies (so a real command still ends promptly).
        self.min_command_ms = int(vad_cfg.get("min_command_ms", 700) or 0)
        self.initial_silence_ms = max(
            int(vad_cfg.get("initial_silence_ms", 1500) or 0), self.vad_silence_ms
        )
        self.max_speak_chars = int(cfg.get("max_speak_chars", 600) or 0)
        cue = cfg.get("cue", {}) or {}
        self.cue_enabled = bool(cue.get("enabled", True))
        self.cue_frequency = float(cue.get("frequency", 880) or 0)
        self.cue_duration_ms = int(cue.get("duration_ms", 120) or 0)
        self.cue_volume = float(cue.get("volume", 0.25) or 0)
        self.cue_submit_frequency = float(cue.get("submit_frequency", 0) or 0)
        self.cue_submit_duration_ms = int(cue.get("submit_duration_ms", 100) or 0)
        #: A short multi-note "bling" played once a transcription lands (each
        #: entry `[frequency_hz, duration_ms]`). When empty, the single
        #: `submit_frequency` beep is used instead (silent when that is 0).
        self.cue_submit_notes = _parse_notes(cue.get("submit_notes"))
        self.cue_note_gap_ms = int(cue.get("note_gap_ms", 20) or 0)

        #: Barge-in: keep scoring the wake word while a reply is playing, abort
        #: the playback on a wake, and mute speech while the interrupt command
        #: is captured/handled. `wait_s` bounds how long to wait for the
        #: interrupted playback to release the audio device. `threshold`
        #: overrides the wake threshold for barge-in only (a higher bar keeps
        #: the agent's own TTS from self-triggering; 0 = use `wake.threshold`).
        barge_cfg = cfg.get("barge_in", {}) or {}
        self.barge_in_enabled = bool(barge_cfg.get("enabled", True))
        self.barge_in_wait_s = float(barge_cfg.get("wait_s", 2.0) or 0.0)
        self.barge_in_threshold = (
            float(barge_cfg.get("threshold", 0.0) or 0.0) or self.wake_threshold
        )

        self._engines = engines
        self._engines_lock = threading.Lock()
        self._audio_lock = threading.Lock()  # serialize cue/speak playback
        self._on_utterance: UtteranceHandler | None = None
        self._stop = threading.Event()
        self._speaking = threading.Event()
        #: Set while a reply is being played; `_interrupt` aborts that playback.
        self._interrupt = threading.Event()
        #: While set, `speak` drops text (a barge-in is being captured/handled).
        self._mute = threading.Event()

    # ---- requirements ----

    def check_requirements(self) -> tuple[bool, str | None]:
        """Return `(ok, install_hint)`. Never raises on a missing optional dep.

        Actually imports each module (not just `find_spec`): `sounddevice`
        raises `OSError` at import when the system PortAudio library is absent,
        which a spec lookup would miss.
        """
        missing = []
        for module, hint in _REQUIRED_MODULES:
            try:
                importlib.import_module(module)
            except Exception:
                missing.append(hint)
        if missing:
            return False, (
                "install the voice stack (requirements/voice.txt): " + "; ".join(missing)
            )
        return True, None

    # ---- lifecycle ----

    def _ensure_engines(self) -> VoiceEngines:
        with self._engines_lock:
            if self._engines is None:
                self._engines = build_engines(self.config, trace=self.trace)
            return self._engines

    def run(self, on_utterance: UtteranceHandler) -> None:
        self._on_utterance = on_utterance
        self._stop.clear()
        engines = self._ensure_engines()
        engines.audio.open_input(self.sample_rate, self.frame_ms, self.input_device)
        self._event("voice_listening", sample_rate=self.sample_rate, frame_ms=self.frame_ms)
        try:
            self._loop(engines)
        finally:
            self._close_engines()

    def stop(self) -> None:
        self._stop.set()
        engines = self._engines
        if engines is not None:
            try:
                engines.audio.close()  # unblocks a pending read
            except Exception:
                pass

    def close(self) -> None:
        self.stop()

    def _close_engines(self) -> None:
        engines = self._engines
        if engines is None:
            return
        for engine in (engines.wake, engines.vad, engines.stt, engines.tts, engines.audio):
            try:
                engine.close()
            except Exception:
                pass

    # ---- listening ----

    def _loop(self, engines: VoiceEngines) -> None:
        while not self._stop.is_set():
            pcm = self._read_frame(engines)
            if pcm is None:
                return
            if not pcm:
                continue
            barge_in = False
            if self._speaking.is_set():
                # Half-duplex by default: the agent's own voice is discarded.
                # With barge-in enabled the wake word is still scored, so the
                # user can interrupt a spoken reply ("hey <word>, stop").
                if not self.barge_in_enabled:
                    continue
                score = engines.wake.score(pcm)
                if score < self.barge_in_threshold:
                    continue
                barge_in = True
                self._begin_barge_in(engines, score)
            else:
                score = engines.wake.score(pcm)
                if score < self.wake_threshold:
                    continue
            delivered = False
            try:
                if self.input_ready is not None and not self.input_ready():
                    # The single scheduler slot is still running the current
                    # request: hold the wake word until it reaches a result (or
                    # pauses for the user's answer), so a new command cannot
                    # start while one is in flight. Reset the detector so the
                    # same wake phrase does not immediately re-fire once the
                    # slot frees.
                    try:
                        engines.wake.reset()
                    except Exception:
                        pass
                    self._event("voice_wake_blocked", score=round(float(score), 3))
                    continue
                if not barge_in:
                    self._event("voice_wake", score=round(float(score), 3))
                try:
                    engines.wake.reset()
                except Exception:
                    pass
                self._cue(self.cue_frequency, self.cue_duration_ms)
                # Wait for the command to start rather than assuming it is glued
                # to the wake word (the wake frame + a natural pause are
                # discarded so the command is not truncated).
                wait_for_speech_ms = self.listen_timeout_ms
                utterance = self._collect(engines, pcm, wait_for_speech_ms=wait_for_speech_ms)
                if not utterance:
                    continue
                try:
                    text = engines.stt.transcribe(utterance)
                except Exception as exc:
                    self._event("voice_stt_failed", message=str(exc)[:300])
                    continue
                text = (text or "").strip()
                if not text:
                    self._event("voice_stt_empty", samples=len(utterance) // BYTES_PER_SAMPLE)
                    continue
                self._event("voice_transcribed", chars=len(text))
                self._bling()
                delivered = True
                if self._on_utterance is not None:
                    self._on_utterance(text)
            finally:
                if barge_in:
                    self._end_barge_in(delivered)

    def _begin_barge_in(self, engines: VoiceEngines, score: float) -> None:
        """Abort the current reply and start listening for an interrupt.

        The playback is stopped (the daemon's `play` watches `_interrupt`) and
        speech is muted so the interrupted reply cannot resume over the user
        while the interrupt command is captured and handled. The front end is
        told via `on_barge_in(True)` so it can classify the next utterance.
        """
        self._event("voice_barge_in", score=round(float(score), 3))
        self._mute.set()
        self._interrupt.set()
        if self.on_barge_in is not None:
            try:
                self.on_barge_in(True)
            except Exception:
                pass
        # Wait for the interrupted playback to release the audio device so the
        # wake cue and the captured command are not played over it.
        deadline = time.time() + self.barge_in_wait_s
        while self._speaking.is_set() and time.time() < deadline:
            time.sleep(0.005)
        try:
            engines.audio.flush()  # drop the reply's echo captured while speaking
        except Exception:
            pass

    def _end_barge_in(self, delivered: bool) -> None:
        """Close a barge-in session.

        `delivered` is True when a command was transcribed and handed to the
        front end; the front end then owns un-muting (it un-mutes only after the
        interrupt has been handled, so the interrupted reply never resumes). A
        barge-in that captured nothing un-mutes here.
        """
        self._interrupt.clear()
        if self.on_barge_in is not None:
            try:
                self.on_barge_in(False)
            except Exception:
                pass
        if not delivered or self.on_barge_in is None:
            self._mute.clear()

    def _collect(self, engines: VoiceEngines, first_frame: bytes, wait_for_speech_ms: int = 0) -> bytes:
        """Accumulate frames from the wake word until trailing silence.

        With `wait_for_speech_ms > 0`, `first_frame` (the wake-word frame) and
        any leading silence are discarded and capture starts only once the
        command's speech begins — so a natural pause after the wake word does
        not truncate the command. Gives up after that many milliseconds with no
        speech (traced `voice_no_command`).
        """
        frames: list[bytes] = []
        speech_ms = 0
        silence_ms = 0
        started = time.time()
        #: Wall-clock of the most recent frame the VAD called speech — the
        #: "last word spoken" boundary the latency breakdown measures from.
        last_speech_ts = started
        waiting = wait_for_speech_ms > 0
        wait_started = started
        if not waiting:
            frames.append(first_frame)
            speech_ms = self.frame_ms
        while not self._stop.is_set():
            if self._speaking.is_set():
                break  # a reply started; stop capturing so we don't transcribe it
            if self.max_utterance_s > 0 and (time.time() - started) >= self.max_utterance_s:
                break
            pcm = self._read_frame(engines)
            if pcm is None:
                break
            if not pcm:
                # No frame within the read timeout: keep waiting, but still
                # honor the "command never started" deadline below.
                if waiting and (time.time() - wait_started) * 1000.0 >= wait_for_speech_ms:
                    self._event("voice_no_command", waited_ms=wait_for_speech_ms)
                    return b""
                continue
            is_speech = engines.vad.is_speech(pcm, self.sample_rate)
            if waiting:
                if not is_speech:
                    if (time.time() - wait_started) * 1000.0 >= wait_for_speech_ms:
                        self._event("voice_no_command", waited_ms=wait_for_speech_ms)
                        return b""
                    continue
                waiting = False
            frames.append(pcm)
            if is_speech:
                speech_ms += self.frame_ms
                silence_ms = 0
                last_speech_ts = time.time()
            else:
                silence_ms += self.frame_ms
            # Tolerate a longer pause until the command's speech is underway
            # (see `initial_silence_ms`): the wake phrase's tail is captured
            # first, and a pause before the actual command must not end it.
            limit = (
                self.initial_silence_ms
                if speech_ms < self.min_command_ms
                else self.vad_silence_ms
            )
            if silence_ms >= limit:
                break
        if speech_ms < self.min_speech_ms:
            self._event(
                "voice_short_utterance",
                frames=len(frames),
                speech_ms=speech_ms,
                silence_ms=silence_ms,
            )
            return b""
        self._event("voice_speech_end", ts=last_speech_ts, speech_ms=speech_ms)
        self._event("voice_utterance", frames=len(frames), speech_ms=speech_ms)
        return b"".join(frames)

    # ---- capture health ----

    def _read_frame(self, engines: VoiceEngines) -> bytes | None:
        """Read one capture frame, recovering from a stalled stream.

        Returns the frame bytes, `b""` when no frame arrived within
        `read_timeout_s`, or `None` once the input is closed. When the stream
        stalls continuously for `stall_timeout_s` — the PortAudio callback has
        stopped delivering, so a blocking `read()` would hang the front end
        forever — the input is closed and reopened so the loop recovers.
        """
        timeout = self.read_timeout_s if self.read_timeout_s > 0 else None
        pcm = engines.audio.read(timeout=timeout)
        if pcm is None:
            return None
        if pcm:
            self._stall_since = 0.0
            return pcm
        if timeout is None or self.stall_timeout_s <= 0:
            return b""
        now = time.time()
        if not self._stall_since:
            self._stall_since = now
            return b""
        if now - self._stall_since >= self.stall_timeout_s:
            self._stall_since = 0.0
            self._reopen_input(engines)
        return b""

    def _reopen_input(self, engines: VoiceEngines) -> None:
        """Close and reopen the capture stream after a stall (best-effort).

        A stalled PortAudio/ALSA stream never delivers another frame, so the
        blocking read would hang the whole front end. Reopening gives the
        device a fresh stream; a hard failure is traced and backed off rather
        than raised, so the loop keeps retrying instead of dying.
        """
        self._event("voice_stream_stalled", stall_s=round(self.stall_timeout_s, 3))
        try:
            engines.audio.close()
        except Exception:
            pass
        try:
            engines.audio.open_input(self.sample_rate, self.frame_ms, self.input_device)
        except Exception as exc:
            self._event("voice_stream_reopen_failed", message=str(exc)[:200])
            time.sleep(min(2.0, self.stall_timeout_s or 1.0))
            return
        try:
            engines.wake.reset()
        except Exception:
            pass
        self._event("voice_stream_reopened")

    # ---- speaking ----

    def _cue(self, frequency: float, duration_ms: int) -> None:
        """Play a short beep (wake / submit cue). Silent on failure."""
        if not self.cue_enabled or frequency <= 0 or duration_ms <= 0:
            return
        engines = self._engines
        if engines is None:
            return
        pcm = _tone_pcm(frequency, duration_ms, self.sample_rate, self.cue_volume)
        if not pcm:
            return
        with self._audio_lock:
            try:
                engines.audio.open_output(self.sample_rate)
                engines.audio.write(pcm, self.sample_rate)
                self._event("voice_cue", frequency=frequency, duration_ms=duration_ms)
            except Exception as exc:
                self._event("voice_cue_failed", message=str(exc)[:200])

    def _bling(self) -> None:
        """Play the post-transcription chime.

        A configured `submit_notes` sequence (a short multi-note "bling") wins;
        otherwise it falls back to the single `submit_frequency` beep (silent at
        the shipped default of 0).
        """
        if self.cue_submit_notes:
            self._play_tones(self.cue_submit_notes)
        else:
            self._cue(self.cue_submit_frequency, self.cue_submit_duration_ms)

    def _play_tones(self, notes) -> None:
        """Play a note sequence (the multi-note submit cue); silent on failure."""
        if not self.cue_enabled:
            return
        engines = self._engines
        if engines is None:
            return
        pcm = _tone_sequence(notes, self.sample_rate, self.cue_volume, self.cue_note_gap_ms)
        if not pcm:
            return
        with self._audio_lock:
            try:
                engines.audio.open_output(self.sample_rate)
                engines.audio.write(pcm, self.sample_rate)
                self._event("voice_cue", notes=[[float(f), int(d)] for f, d in notes])
            except Exception as exc:
                self._event("voice_cue_failed", message=str(exc)[:200])

    def set_muted(self, muted: bool) -> None:
        """Mute/unmute spoken replies (used around a barge-in)."""
        if muted:
            self._mute.set()
        else:
            self._mute.clear()

    def muted(self) -> bool:
        """Whether spoken replies are currently muted (a barge-in is in flight)."""
        return self._mute.is_set()

    def stopping(self) -> bool:
        """Whether the daemon is shutting down (a long stream should stop)."""
        return self._stop.is_set()

    def speak(self, text: str) -> None:
        """Synthesize and play `text`; called from the outbound pump thread."""
        text = (text or "").strip()
        if not text:
            return
        if self._mute.is_set():
            # A barge-in is in progress: the user interrupted and is giving a
            # new command, so a pending reply is dropped rather than played over
            # them. Replies after the interrupt is handled are spoken normally.
            self._event("voice_speak_suppressed", chars=len(text))
            return
        if self.max_speak_chars and len(text) > self.max_speak_chars:
            clipped = text[: self.max_speak_chars].rsplit(" ", 1)[0]
            text = (clipped or text[: self.max_speak_chars]) + "…"
        engines = self._ensure_engines()
        # Start from a clean wake-detector state so the reply's own audio does
        # not inherit a stale partial match.
        try:
            engines.wake.reset()
        except Exception:
            pass
        self._speaking.set()
        self._event("voice_speak_start", chars=len(text))
        try:
            synth_started = time.time()
            pcm, sample_rate = engines.tts.synthesize(text)
            self._event(
                "voice_tts_synth",
                synth_s=round(time.time() - synth_started, 3),
                chars=len(text),
                audio_s=round(len(pcm) / BYTES_PER_SAMPLE / sample_rate, 3)
                if pcm and sample_rate
                else 0.0,
            )
            if pcm:
                if self._interrupt.is_set():
                    # A barge-in landed during synthesis; skip playback.
                    self._event("voice_speak_aborted", chars=len(text))
                else:
                    with self._audio_lock:
                        engines.audio.open_output(sample_rate)
                        # The instant the first reply audio is handed to the device.
                        self._event("voice_playback_start", ts=time.time(), chars=len(text))
                        completed = engines.audio.play(
                            pcm, sample_rate, stop_event=self._interrupt
                        )
                        if not completed:
                            self._event("voice_speak_interrupted", chars=len(text))
            self._event("voice_spoke", chars=len(text))
        except Exception as exc:
            self._event("voice_speak_failed", message=str(exc)[:300])
        finally:
            self._speaking.clear()
            try:
                engines.audio.flush()  # clear any TTS echo captured while speaking
            except Exception:
                pass
            try:
                engines.wake.reset()
            except Exception:
                pass

    # ---- tracing ----

    def _event(self, kind: str, ts: float | None = None, **fields) -> None:
        if self.trace is not None:
            self.trace.append(kind, "?", ts=ts, **fields)
