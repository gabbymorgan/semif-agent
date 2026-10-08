"""Recording helpers for a personal Piper TTS voice (training data only).

This is the neutral, stdlib-only half of the voice-training pipeline: it turns a
list of prompts plus recorded audio into an **LJSpeech-format dataset**
(``metadata.csv`` + ``wavs/NNNN.wav``) that ``piper.train fit`` consumes, and it
provides the mic-check level math and the silence-endpointing loop. It knows
nothing about the voice gateway's runtime (`semif_agent.voice_transport`) or
about the training host.

Heavy deps stay lazy: numpy is imported only inside `resample_pcm16`, so
importing this module is safe on a machine without the optional voice stack.
`scripts/voice-record.py` is the thin CLI front end; the real capture path
reuses `voice_transport.SoundDeviceAudio` (which resamples the device's native
rate down to the target 22050 Hz for us).

Audio is mono signed 16-bit little-endian PCM throughout.

Dataset format note: piper's loader reads column 1 as the audio filename (looked
up under its ``--data.audio_dir``) and the **last** column as the text, so a row
is ``0001.wav|<sentence>`` and the wavs live in ``wavs/``.
"""

from __future__ import annotations

import math
import re
import time
import wave
from pathlib import Path
from typing import Callable

#: The sample rate the recorder targets (Piper "medium" quality).
SAMPLE_RATE = 22050
BYTES_PER_SAMPLE = 2

#: Endpointing defaults (see `record_until_silence`).
DEFAULT_SILENCE_MS = 800
DEFAULT_MIN_SPEECH_MS = 150
DEFAULT_MAX_SECONDS = 15.0
#: int16 RMS above which a frame counts as speech (the energy VAD threshold).
SPEECH_RMS_THRESHOLD = 200.0

#: An `is_speech(pcm16, sample_rate) -> bool` predicate (injected in tests).
SpeechPredicate = Callable[[bytes, int], bool]


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #


def load_prompts(path: str | Path) -> list[str]:
    """Read one prompt per line; skip blanks and ``#`` comments."""
    lines: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


# --------------------------------------------------------------------------- #
# wav io (stdlib `wave`)
# --------------------------------------------------------------------------- #


def write_wav(path: str | Path, pcm16: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    """Write mono 16-bit PCM as a WAV file (creating parent dirs)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(BYTES_PER_SAMPLE)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm16)


def read_wav(path: str | Path) -> tuple[bytes, int]:
    """Read a WAV file back as ``(pcm16, sample_rate)``."""
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes()), handle.getframerate()


def resample_pcm16(pcm16: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linear-interpolate int16 PCM from `source_rate` to `target_rate`.

    numpy is a voice-stack dependency and this is only reached by the real
    capture path, so the module stays stdlib-only on import.
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


# --------------------------------------------------------------------------- #
# levels (mic check)
# --------------------------------------------------------------------------- #


def _samples(pcm16: bytes):
    """Yield int16 sample values from little-endian PCM."""
    for offset in range(0, len(pcm16) - BYTES_PER_SAMPLE + 1, BYTES_PER_SAMPLE):
        yield int.from_bytes(
            pcm16[offset : offset + BYTES_PER_SAMPLE], "little", signed=True
        )


def rms_int16(pcm16: bytes) -> float:
    """Root-mean-square level of int16 PCM (0.0 for empty input)."""
    count = len(pcm16) // BYTES_PER_SAMPLE
    if count == 0:
        return 0.0
    total = sum(sample * sample for sample in _samples(pcm16))
    return math.sqrt(total / count)


def peak_int16(pcm16: bytes) -> int:
    """Peak absolute sample value of int16 PCM."""
    return max((abs(sample) for sample in _samples(pcm16)), default=0)


def duration_s(pcm16: bytes, sample_rate: int = SAMPLE_RATE) -> float:
    """Playback duration of int16 PCM in seconds."""
    if sample_rate <= 0:
        return 0.0
    return (len(pcm16) // BYTES_PER_SAMPLE) / float(sample_rate)


def level_report(pcm16: bytes, sample_rate: int = SAMPLE_RATE) -> dict:
    """Mic-check summary: RMS, peak, dBFS, and duration."""
    peak = peak_int16(pcm16)
    rms = rms_int16(pcm16)
    peak_dbfs = 20.0 * math.log10(peak / 32768.0) if peak > 0 else float("-inf")
    return {
        "rms": round(rms, 1),
        "peak": peak,
        "peak_dbfs": None if peak <= 0 else round(peak_dbfs, 1),
        "seconds": round(duration_s(pcm16, sample_rate), 2),
    }


# --------------------------------------------------------------------------- #
# capture loops
# --------------------------------------------------------------------------- #


def _read_frame(audio, timeout: float | None):
    """Read one frame, passing a `timeout` only if the source supports it.

    The real `SoundDeviceAudio.read(timeout=...)` returns `None` on a stall so a
    dropped device cannot hang the loop; test doubles take no argument.
    """
    if timeout is None:
        return audio.read()
    try:
        return audio.read(timeout)
    except TypeError:
        return audio.read()


def _default_is_speech(threshold: float) -> SpeechPredicate:
    def is_speech(pcm16: bytes, sample_rate: int) -> bool:
        return rms_int16(pcm16) >= threshold

    return is_speech


def record_for(
    audio,
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = 40,
    seconds: float = 5.0,
    read_timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_frame: Callable[[int, float], None] | None = None,
) -> bytes:
    """Read frames for `seconds` regardless of content (the mic check).

    `audio` is any object with a blocking ``read() -> bytes | None`` (the
    `voice_transport.AudioIO` contract); a ``None`` frame ends capture early.
    `read_timeout` bounds each read so a stalled device ends capture instead of
    blocking forever.
    """
    frames: list[bytes] = []
    started = clock()
    while True:
        pcm = _read_frame(audio, read_timeout)
        if pcm is None:
            break
        frames.append(pcm)
        if on_frame is not None:
            on_frame(len(frames), clock() - started)
        if seconds > 0 and (clock() - started) >= seconds:
            break
    return b"".join(frames)


def record_until_silence(
    audio,
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = 40,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    silence_ms: int = DEFAULT_SILENCE_MS,
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS,
    threshold: float = SPEECH_RMS_THRESHOLD,
    is_speech: SpeechPredicate | None = None,
    read_timeout: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_frame: Callable[[int, int], None] | None = None,
) -> bytes:
    """Accumulate frames until trailing silence or `max_seconds`.

    Returns the captured PCM, or ``b""`` when less than `min_speech_ms` of
    speech was heard (a false start / silence). `is_speech` defaults to an int16
    RMS threshold; tests inject a scripted predicate. `read_timeout` bounds each
    read so a stalled/dropped device ends the take instead of hanging.
    """
    predicate = is_speech or _default_is_speech(threshold)
    frames: list[bytes] = []
    speech_ms = 0
    trailing_silence_ms = 0
    started = clock()
    while True:
        pcm = _read_frame(audio, read_timeout)
        if pcm is None:
            break
        frames.append(pcm)
        if predicate(pcm, sample_rate):
            speech_ms += frame_ms
            trailing_silence_ms = 0
        else:
            trailing_silence_ms += frame_ms
        if on_frame is not None:
            on_frame(speech_ms, trailing_silence_ms)
        if speech_ms >= min_speech_ms and trailing_silence_ms >= silence_ms:
            break
        if max_seconds > 0 and (clock() - started) >= max_seconds:
            break
    if speech_ms < min_speech_ms:
        return b""
    return b"".join(frames)


# --------------------------------------------------------------------------- #
# dataset on disk
# --------------------------------------------------------------------------- #


class DatasetStore:
    """An LJSpeech-format dataset directory: ``metadata.csv`` + ``wavs/``.

    Survives across runs (the recorder resumes where it left off): the existing
    ``metadata.csv`` is parsed on construction and rewritten (sorted by id)
    after every save.
    """

    def __init__(self, out_dir: str | Path, sample_rate: int = SAMPLE_RATE):
        self.out = Path(out_dir)
        self.sample_rate = int(sample_rate)
        self.wavs = self.out / "wavs"
        self.metadata = self.out / "metadata.csv"
        self._entries: dict[str, str] = {}
        self._load()

    # ---- loading ----

    def _load(self) -> None:
        if not self.metadata.is_file():
            return
        for raw in self.metadata.read_text(encoding="utf-8").splitlines():
            if "|" not in raw:
                continue
            uid, text = raw.split("|", 1)
            uid = uid.strip()
            if uid.endswith(".wav"):
                uid = uid[:-4]  # metadata stores the filename; ids stay bare
            if uid:
                self._entries[uid] = text.strip()

    # ---- queries ----

    @property
    def entries(self) -> dict[str, str]:
        return dict(self._entries)

    def ids(self) -> set[str]:
        return set(self._entries)

    def count(self) -> int:
        return len(self._entries)

    def next_id(self) -> str:
        """The lowest zero-padded 4-digit id not yet recorded."""
        index = 1
        while f"{index:04d}" in self._entries:
            index += 1
        return f"{index:04d}"

    def wav_path(self, uid: str) -> Path:
        return self.wavs / f"{uid}.wav"

    # ---- mutation ----

    def save(self, uid: str, text: str, pcm16: bytes) -> Path:
        """Write ``wavs/<uid>.wav`` and record ``<uid>.wav|<text>``."""
        uid = str(uid)
        path = self.wav_path(uid)
        write_wav(path, pcm16, self.sample_rate)
        self._entries[uid] = text.strip()
        self.rewrite()
        return path

    def remove(self, uid: str) -> bool:
        """Drop a recording: delete ``wavs/<uid>.wav`` and its metadata row.

        Returns whether the id was present. Used by the verifier's ``--prune`` to
        take a bad take out of the dataset so the recorder re-records it (the
        recorder resumes by id). The caller may move the wav aside first; the
        unlink here tolerates it already being gone.
        """
        uid = str(uid)
        if uid not in self._entries:
            return False
        try:
            self.wav_path(uid).unlink()
        except FileNotFoundError:
            pass
        del self._entries[uid]
        self.rewrite()
        return True

    def rewrite(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        lines = [f"{uid}.wav|{self._entries[uid]}" for uid in sorted(self._entries)]
        self.metadata.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )


# --------------------------------------------------------------------------- #
# fidelity check (transcript vs. prompt)
# --------------------------------------------------------------------------- #

#: Statuses the verifier flags as needing a re-record.
BAD_STATUSES = frozenset({"missing", "empty", "mismatch"})

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

#: Number words -> digits, so Whisper's "2 plus 7 is less than 10" matches a
#: prompt that spells them out. Comparison-only; never used for output.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
}


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, digitize number words.

    Whisper's output differs from the prompt in capitalization, punctuation,
    spacing, and (for numbers) digits-vs-words; this reduces both to a comparable
    word sequence. Apostrophes are kept so ``it's`` does not silently become
    ``its``.
    """
    words = _WORD_RE.findall(text.lower())
    return " ".join(_NUMBER_WORDS.get(word, word) for word in words)


def word_tokens(text: str) -> list[str]:
    """The normalized word sequence of `text`."""
    return normalize_text(text).split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance divided by the reference word count.

    0.0 is an exact match; 1.0 means every reference word was wrong (or the
    hypothesis was empty). An empty reference scores 0.0 for an empty
    hypothesis, else 1.0.
    """
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            substitution = previous[j - 1] + (0 if ref_word == hyp_word else 1)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitution))
        previous = current
    return previous[-1] / len(ref)


def compare_transcript(
    reference: str,
    hypothesis: str,
    max_wer: float = 0.3,
    review_wer: float | None = None,
) -> dict:
    """Classify a Whisper transcript against the prompt it was read from.

    Returns ``{"status", "wer", "similarity", "ref_words", "hyp_words"}`` where
    status is one of:

    - ``empty``    — Whisper heard nothing (a silent / false-start take);
    - ``mismatch`` — word error rate above `max_wer` (a wrong or truncated take);
    - ``review``   — between `review_wer` and `max_wer`: worth a listen, because
      Whisper's own spelling/number variants live in this band (e.g. "two plus
      seven" -> "2 plus 7");
    - ``ok``       — below `review_wer` (or below `max_wer` when no review band).

    `similarity` is ``1 - min(wer, 1)``, a friendlier 0..1 score.
    """
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    if not hyp:
        return {
            "status": "empty",
            "wer": 1.0,
            "similarity": 0.0,
            "ref_words": len(ref),
            "hyp_words": 0,
        }
    wer = word_error_rate(reference, hypothesis)
    if wer > max_wer:
        status = "mismatch"
    elif review_wer is not None and wer >= review_wer:
        status = "review"
    else:
        status = "ok"
    return {
        "status": status,
        "wer": round(wer, 3),
        "similarity": round(max(0.0, 1.0 - min(wer, 1.0)), 3),
        "ref_words": len(ref),
        "hyp_words": len(hyp),
    }
