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
- **Half-duplex.** Capture frames are discarded while a reply is being spoken,
  so the agent never transcribes its own voice (no barge-in yet).
- **Follow-up window.** After a reply, the next utterance is accepted without
  the wake word for `follow_up_window_s`, so answering a question is
  conversational.
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

    `read()` blocks for one frame and returns its bytes, or `None` once the
    input is closed (the loop exits on `None`). `open_output`/`write` may be
    called at any sample rate the synthesizer produced.
    """

    def open_input(self, sample_rate: int, frame_ms: int, device: str = "") -> None:
        raise NotImplementedError

    def read(self) -> bytes | None:
        raise NotImplementedError

    def open_output(self, sample_rate: int, device: str = "") -> None:
        pass

    def write(self, pcm16: bytes, sample_rate: int) -> None:
        raise NotImplementedError

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

    def transcribe(self, pcm16: bytes) -> str:
        import numpy as np

        audio = np.frombuffer(pcm16, dtype=np.int16).astype("float32") / 32768.0
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


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

        Returns `None` once the input is closed, or on a `timeout` with no
        frame — a caller that passes a timeout can detect a stalled stream
        instead of blocking forever.
        """
        if timeout is None:
            return self._frames.get()
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

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


def build_engines(cfg: dict, trace=None) -> VoiceEngines:
    """Construct the real engine bundle from a `gateway.voice` config block."""
    wake_cfg = cfg.get("wake", {}) or {}
    stt_cfg = cfg.get("stt", {}) or {}
    tts_cfg = cfg.get("tts", {}) or {}
    audio_cfg = cfg.get("audio", {}) or {}

    wake_model = resolve_model_path(
        str(wake_cfg.get("model", "hey_mycroft") or ""),
        str(wake_cfg.get("model_dir", "") or ""),
        (".onnx", ".tflite"),
    )
    tts_voice = resolve_model_path(
        str(tts_cfg.get("voice", "") or ""),
        str(tts_cfg.get("voice_dir", "") or ""),
        (".onnx",),
    )
    return VoiceEngines(
        wake=OpenWakeWordDetector(
            wake_model,
            threshold=float(wake_cfg.get("threshold", 0.5)),
            inference_framework=str(wake_cfg.get("inference_framework", "onnx")),
        ),
        vad=_build_vad(cfg),
        stt=WhisperTranscriber(
            model=str(stt_cfg.get("model", "base.en") or "base.en"),
            device=str(stt_cfg.get("device", "cpu") or "cpu"),
            compute_type=str(stt_cfg.get("compute_type", "int8") or "int8"),
            language=str(stt_cfg.get("language", "en") or "en"),
        ),
        tts=PiperSynthesizer(
            tts_voice,
            speaker_id=int(tts_cfg.get("speaker_id", 0) or 0),
            length_scale=float(tts_cfg.get("length_scale", 1.0) or 1.0),
            volume=float(tts_cfg.get("volume", 1.0) or 1.0),
        ),
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
                 name: str = "voice"):
        cfg = cfg or {}
        self.config = cfg
        self.trace = trace
        self.name = name
        self.wake_threshold = float((cfg.get("wake", {}) or {}).get("threshold", 0.5))
        audio_cfg = cfg.get("audio", {}) or {}
        self.sample_rate = int(audio_cfg.get("sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
        self.frame_ms = int(audio_cfg.get("frame_ms", 80) or 80)
        self.input_device = str(audio_cfg.get("input_device", "") or "")
        vad_cfg = cfg.get("vad", {}) or {}
        self.vad_silence_ms = int(vad_cfg.get("silence_ms", 1000) or 1000)
        self.min_speech_ms = int(vad_cfg.get("min_speech_ms", 120) or 120)
        self.max_utterance_s = float(vad_cfg.get("max_utterance_s", 30) or 30)
        self.follow_up_window = float(cfg.get("follow_up_window_s", 5.0) or 0.0)
        self.max_speak_chars = int(cfg.get("max_speak_chars", 600) or 0)
        cue = cfg.get("cue", {}) or {}
        self.cue_enabled = bool(cue.get("enabled", True))
        self.cue_frequency = float(cue.get("frequency", 880) or 0)
        self.cue_duration_ms = int(cue.get("duration_ms", 120) or 0)
        self.cue_volume = float(cue.get("volume", 0.25) or 0)
        self.cue_submit_frequency = float(cue.get("submit_frequency", 0) or 0)
        self.cue_submit_duration_ms = int(cue.get("submit_duration_ms", 100) or 0)

        self._engines = engines
        self._engines_lock = threading.Lock()
        self._audio_lock = threading.Lock()  # serialize cue/speak playback
        self._on_utterance: UtteranceHandler | None = None
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._follow_until = 0.0

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
            pcm = engines.audio.read()
            if pcm is None:
                return
            if self._speaking.is_set():
                continue
            armed = time.time() < self._follow_until
            if not armed:
                score = engines.wake.score(pcm)
                if score < self.wake_threshold:
                    continue
                self._event("voice_wake", score=round(float(score), 3))
                try:
                    engines.wake.reset()
                except Exception:
                    pass
                self._cue(self.cue_frequency, self.cue_duration_ms)
            self._follow_until = 0.0
            utterance = self._collect(engines, pcm)
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
            self._cue(self.cue_submit_frequency, self.cue_submit_duration_ms)
            if self._on_utterance is not None:
                self._on_utterance(text)

    def _collect(self, engines: VoiceEngines, first_frame: bytes) -> bytes:
        """Accumulate frames from the wake word until trailing silence."""
        frames = [first_frame]
        speech_ms = self.frame_ms
        silence_ms = 0
        started = time.time()
        #: Wall-clock of the most recent frame the VAD called speech — the
        #: "last word spoken" boundary the latency breakdown measures from.
        last_speech_ts = started
        while not self._stop.is_set():
            if self._speaking.is_set():
                break  # a reply started; stop capturing so we don't transcribe it
            if self.max_utterance_s > 0 and (time.time() - started) >= self.max_utterance_s:
                break
            pcm = engines.audio.read()
            if pcm is None:
                break
            frames.append(pcm)
            if engines.vad.is_speech(pcm, self.sample_rate):
                speech_ms += self.frame_ms
                silence_ms = 0
                last_speech_ts = time.time()
            else:
                silence_ms += self.frame_ms
            if silence_ms >= self.vad_silence_ms:
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

    def speak(self, text: str) -> None:
        """Synthesize and play `text`; called from the outbound pump thread."""
        text = (text or "").strip()
        if not text:
            return
        if self.max_speak_chars and len(text) > self.max_speak_chars:
            clipped = text[: self.max_speak_chars].rsplit(" ", 1)[0]
            text = (clipped or text[: self.max_speak_chars]) + "…"
        engines = self._ensure_engines()
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
                with self._audio_lock:
                    engines.audio.open_output(sample_rate)
                    # The instant the first reply audio is handed to the device.
                    self._event("voice_playback_start", ts=time.time(), chars=len(text))
                    engines.audio.write(pcm, sample_rate)
            self._event("voice_spoke", chars=len(text))
        except Exception as exc:
            self._event("voice_speak_failed", message=str(exc)[:300])
        finally:
            self._speaking.clear()
            try:
                engines.audio.flush()  # clear any TTS echo captured while speaking
            except Exception:
                pass
            if self.follow_up_window > 0:
                self._follow_until = time.time() + self.follow_up_window
            try:
                engines.wake.reset()
            except Exception:
                pass

    # ---- tracing ----

    def _event(self, kind: str, ts: float | None = None, **fields) -> None:
        if self.trace is not None:
            self.trace.append(kind, "?", ts=ts, **fields)
