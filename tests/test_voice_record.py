"""Pure-stdlib tests for the voice-recording helpers (`semif_agent.voice_train`).

The recording front end (`scripts/voice-record.py`) drives real hardware; these
tests exercise the neutral helpers it builds on — prompt loading, WAV I/O, level
math, the endpointing loop, and the LJSpeech dataset writer — with a scripted
fake audio source, so no microphone, numpy, or sounddevice is required.
"""

import itertools

from semif_agent.voice_train import (
    DatasetStore,
    compare_transcript,
    duration_s,
    level_report,
    load_prompts,
    normalize_text,
    peak_int16,
    read_wav,
    record_for,
    record_until_silence,
    resample_pcm16,
    rms_int16,
    word_error_rate,
    word_tokens,
    write_wav,
)


class FakeAudio:
    """A scripted `AudioIO`: `read()` pops frames, then returns None."""

    def __init__(self, frames):
        self.frames = list(frames)

    def read(self):
        return self.frames.pop(0) if self.frames else None


def _pcm(*values):
    return b"".join(int(v).to_bytes(2, "little", signed=True) for v in values)


# ---- prompts ---------------------------------------------------------------


def test_load_prompts_skips_comments_and_blanks(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("# a comment\n\nHello there.\n   \nSecond line.\n", encoding="utf-8")
    assert load_prompts(path) == ["Hello there.", "Second line."]


def test_load_prompts_missing_file_raises(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path / "nope.txt")


# ---- wav io ----------------------------------------------------------------


def test_wav_roundtrip(tmp_path):
    pcm = _pcm(0, 100, -100, 32767, -32768)
    path = tmp_path / "nested" / "clip.wav"
    write_wav(path, pcm, 22050)
    assert path.is_file()
    got, rate = read_wav(path)
    assert got == pcm
    assert rate == 22050


# ---- levels ----------------------------------------------------------------


def test_rms_peak_duration():
    pcm = _pcm(3, 4)  # rms = sqrt((9+16)/2) = 3.535..., peak 4
    assert abs(rms_int16(pcm) - 3.5355) < 1e-3
    assert peak_int16(pcm) == 4
    assert duration_s(_pcm(*([0] * 22050)), 22050) == 1.0


def test_levels_empty():
    assert rms_int16(b"") == 0.0
    assert peak_int16(b"") == 0
    assert duration_s(b"", 22050) == 0.0
    assert level_report(b"")["peak_dbfs"] is None


def test_level_report_dbfs():
    report = level_report(_pcm(32767), 22050)
    assert report["peak"] == 32767
    assert report["peak_dbfs"] == 0.0
    assert report["seconds"] == 0.0


# ---- resampling ------------------------------------------------------------


def test_resample_pcm16():
    import pytest

    np = pytest.importorskip("numpy")
    pcm = np.arange(441, dtype=np.int16).tobytes()
    assert len(resample_pcm16(pcm, 44100, 16000)) // 2 == 160
    assert resample_pcm16(pcm, 44100, 44100) == pcm
    assert resample_pcm16(b"", 44100, 16000) == b""


# ---- endpointing -----------------------------------------------------------


def test_record_until_silence_stops_on_trailing_silence():
    frames = [b"s"] * 5 + [b"_"] * 25
    audio = FakeAudio(frames)
    got = record_until_silence(
        audio,
        frame_ms=40,
        silence_ms=800,  # 20 silent frames
        min_speech_ms=150,  # 4 speech frames
        is_speech=lambda pcm, rate: pcm == b"s",
    )
    assert got == b"".join(frames[:25])  # 5 speech + 20 trailing silence


def test_record_until_silence_rejects_all_silence():
    audio = FakeAudio([b"_"] * 30)
    got = record_until_silence(
        audio, frame_ms=40, min_speech_ms=150, is_speech=lambda pcm, rate: False
    )
    assert got == b""


def test_record_until_silence_max_seconds_caps():
    ticks = iter([0.0, 0.0, 1.0, 1.0, 1.0])
    audio = FakeAudio([b"s"] * 5)
    got = record_until_silence(
        audio,
        frame_ms=40,
        max_seconds=1.0,
        min_speech_ms=0,
        is_speech=lambda pcm, rate: True,
        clock=lambda: next(ticks),
    )
    assert got == b"ss"  # stopped after the second frame when elapsed hit 1.0


def test_record_for_reads_until_clock_or_end():
    ticks = iter([0.0, 0.0, 1.0, 1.0])
    audio = FakeAudio([b"a", b"b", b"c", b"d"])
    assert record_for(audio, seconds=1.0, clock=lambda: next(ticks)) == b"ab"


def test_record_until_silence_returns_on_none_frame():
    audio = FakeAudio([b"s"])  # only one frame, then None
    got = record_until_silence(
        audio, frame_ms=40, min_speech_ms=0, is_speech=lambda pcm, rate: True
    )
    assert got == b"s"


def test_record_until_silence_passes_read_timeout():
    # A source that supports `read(timeout=...)` receives the timeout, so a
    # stalled device returns None and ends the take instead of hanging.
    class TimeoutAudio:
        def __init__(self, frames):
            self.frames = list(frames)
            self.saw_timeout = False

        def read(self, timeout=None):
            self.saw_timeout = timeout is not None
            return self.frames.pop(0) if self.frames else None

    audio = TimeoutAudio([b"s"])
    got = record_until_silence(
        audio,
        frame_ms=40,
        min_speech_ms=0,
        is_speech=lambda pcm, rate: True,
        read_timeout=2.0,
    )
    assert got == b"s"
    assert audio.saw_timeout is True


def test_record_for_passes_read_timeout():
    class TimeoutAudio:
        def __init__(self, frames):
            self.frames = list(frames)
            self.saw_timeout = False

        def read(self, timeout=None):
            self.saw_timeout = timeout is not None
            return self.frames.pop(0) if self.frames else None

    audio = TimeoutAudio([b"a", b"b"])
    assert record_for(audio, seconds=0.001, read_timeout=1.5) == b"ab"
    assert audio.saw_timeout is True


# ---- dataset store ---------------------------------------------------------


def test_dataset_store_save_and_reload(tmp_path):
    store = DatasetStore(tmp_path / "ds", sample_rate=22050)
    assert store.count() == 0
    assert store.next_id() == "0001"

    store.save("0001", "Hello there.", _pcm(1, 2, 3))
    assert (tmp_path / "ds" / "wavs" / "0001.wav").is_file()
    assert store.next_id() == "0002"

    # metadata uses the piper loader's `filename|text` shape.
    lines = (tmp_path / "ds" / "metadata.csv").read_text().splitlines()
    assert lines == ["0001.wav|Hello there."]

    # A fresh store resumes from disk.
    resumed = DatasetStore(tmp_path / "ds", sample_rate=22050)
    assert resumed.ids() == {"0001"}
    assert resumed.entries == {"0001": "Hello there."}


def test_dataset_store_rewrite_sorted(tmp_path):
    store = DatasetStore(tmp_path / "ds")
    store.save("0002", "second", _pcm(0))
    store.save("0001", "first", _pcm(0))
    lines = (tmp_path / "ds" / "metadata.csv").read_text().splitlines()
    assert lines == ["0001.wav|first", "0002.wav|second"]


def test_dataset_store_next_id_fills_gaps(tmp_path):
    store = DatasetStore(tmp_path / "ds")
    store.save("0001", "a", _pcm(0))
    store.save("0003", "c", _pcm(0))
    assert store.next_id() == "0002"


def test_dataset_store_remove_drops_wav_and_row(tmp_path):
    store = DatasetStore(tmp_path / "ds")
    store.save("0001", "first", _pcm(0))
    store.save("0002", "second", _pcm(0))
    assert store.remove("0001") is True
    assert not (tmp_path / "ds" / "wavs" / "0001.wav").exists()
    assert store.ids() == {"0002"}
    assert (tmp_path / "ds" / "metadata.csv").read_text().splitlines() == ["0002.wav|second"]
    assert store.remove("9999") is False


def test_dataset_store_remove_tolerates_missing_wav(tmp_path):
    store = DatasetStore(tmp_path / "ds")
    store.save("0001", "first", _pcm(0))
    (tmp_path / "ds" / "wavs" / "0001.wav").unlink()
    assert store.remove("0001") is True
    assert store.ids() == set()


# ---- transcript fidelity ---------------------------------------------------


def test_normalize_text_strips_punctuation_and_case():
    assert normalize_text("  The birch canoe, slid!  ") == "the birch canoe slid"
    assert normalize_text("It's easy.") == "it's easy"
    assert normalize_text("") == ""
    assert word_tokens("Well-done, yes?") == ["well", "done", "yes"]


def test_normalize_text_digitizes_number_words():
    assert normalize_text("Two plus seven is less than ten.") == "2 plus 7 is less than 10"
    assert normalize_text("2 plus 7 is less than 10") == "2 plus 7 is less than 10"
    assert word_error_rate("Two plus seven is less than ten.", "2 plus 7 is less than 10") == 0.0


def test_word_error_rate():
    assert word_error_rate("the hog crawled under", "the hog crawled under") == 0.0
    assert word_error_rate("the hog crawled under", "the dog crawled under") == 0.25
    assert word_error_rate("the hog crawled under", "") == 1.0
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "extra") == 1.0


def test_compare_transcript_statuses():
    ref = "The hog crawled under the high fence."
    assert compare_transcript(ref, "The hog crawled under the high fence.")["status"] == "ok"
    # A near-miss within the tolerance still passes.
    assert compare_transcript(ref, "the hog crawled under the high fence")["status"] == "ok"
    empty = compare_transcript(ref, "")
    assert empty["status"] == "empty"
    assert empty["similarity"] == 0.0
    mismatch = compare_transcript(ref, "The whole Craig under the high sense.")
    assert mismatch["status"] == "mismatch"
    assert mismatch["wer"] > 0.3
    assert 0.0 <= mismatch["similarity"] < 1.0


def test_compare_transcript_review_band():
    ref = "The junk yard had a mouldy smell."
    # A spelling/spacing variant sits in the review band, not a failure.
    got = compare_transcript(
        ref, "The junkyard had a moldy smell.", max_wer=0.5, review_wer=0.35
    )
    assert got["status"] == "review"
    # Without a review band the same take is ok.
    assert compare_transcript(
        ref, "The junkyard had a moldy smell.", max_wer=0.5
    )["status"] == "ok"
    # Above max_wer it is a mismatch regardless of the review band.
    assert compare_transcript(
        ref, "The junkyard smell.", max_wer=0.5, review_wer=0.35
    )["status"] == "mismatch"
