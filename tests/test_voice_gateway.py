"""Pure-stdlib tests for the voice gateway (adapter + neutral transport).

The voice stack (sounddevice/openwakeword/faster-whisper/piper) is optional and
lazy-imported, so these tests inject fake engines and drive the loop without
hardware, network, or the heavy packages. The live pipeline against a real mic
is a live-integration concern.

Frame constants: the fake wake word fires on `b"WAKE"`, the fake VAD calls any
non-`b"SIL"` frame speech, and the fake STT returns a fixed string.
"""

from semif_agent.cli import _build_gateway_adapter, _resolve_platforms
from semif_agent.gateway.voice import VoiceAdapter
from semif_agent.voice_transport import VoiceDaemon, VoiceEngines


# ---- fakes -----------------------------------------------------------------


class FakeAudio:
    def __init__(self, frames):
        self.frames = list(frames)
        self.writes = []
        self.closed = False
        self.input_opened = False

    def open_input(self, sample_rate, frame_ms, device=""):
        self.input_opened = True

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def open_output(self, sample_rate, device=""):
        pass

    def write(self, pcm16, sample_rate):
        self.writes.append((pcm16, sample_rate))

    def flush(self):
        pass  # the scripted `frames` are input, not a capture buffer

    def close(self):
        self.closed = True


class FakeWake:
    def __init__(self, trigger=b"WAKE"):
        self.trigger = trigger
        self.resets = 0

    def score(self, pcm16):
        return 1.0 if pcm16 == self.trigger else 0.0

    def reset(self):
        self.resets += 1


class FakeVad:
    def is_speech(self, pcm16, sample_rate):
        return pcm16 != b"SIL"


class FakeStt:
    def __init__(self, text="hello"):
        self.text = text
        self.calls = []

    def transcribe(self, pcm16):
        self.calls.append(pcm16)
        return self.text


class FakeTts:
    def __init__(self):
        self.spoken = []

    def synthesize(self, text):
        self.spoken.append(text)
        return b"PCM", 22050


def make_daemon(frames, stt_text="hello", config=None):
    audio = FakeAudio(frames)
    engines = VoiceEngines(
        wake=FakeWake(),
        vad=FakeVad(),
        stt=FakeStt(stt_text),
        tts=FakeTts(),
        audio=audio,
    )
    cfg = {
        "audio": {"frame_ms": 80, "sample_rate": 16000},
        "vad": {"silence_ms": 160, "min_speech_ms": 80, "max_utterance_s": 30},
        "follow_up_window_s": 5.0,
        "max_speak_chars": 600,
    }
    if config:
        cfg.update(config)
    return VoiceDaemon(cfg, engines=engines), engines


# ---- requirements ----------------------------------------------------------


def test_check_requirements_never_raises():
    # Either the voice stack is installed (True) or an install hint is returned;
    # the daemon must never raise on a machine without the optional deps.
    ok, hint = VoiceDaemon({}).check_requirements()
    assert isinstance(ok, bool)
    if not ok:
        assert "requirements/voice.txt" in hint


def test_check_requirements_reports_missing(monkeypatch):
    import semif_agent.voice_transport as vt

    def boom(name):
        raise ImportError(name)

    monkeypatch.setattr(vt.importlib, "import_module", boom)
    ok, hint = VoiceDaemon({}).check_requirements()
    assert ok is False
    assert "requirements/voice.txt" in hint
    assert "sounddevice" in hint


# ---- wake gating / endpointing --------------------------------------------


def test_wake_word_then_silence_produces_one_utterance():
    daemon, engines = make_daemon([b"noise", b"WAKE", b"speech", b"SIL", b"SIL"])
    heard = []
    daemon.run(heard.append)
    assert heard == ["hello"]
    assert engines.audio.input_opened is True
    assert engines.audio.closed is True
    # The wake detector is reset after firing.
    assert engines.wake.resets >= 1


def test_no_wake_word_produces_nothing():
    daemon, engines = make_daemon([b"noise", b"noise", b"SIL"])
    heard = []
    daemon.run(heard.append)
    assert heard == []
    assert engines.stt.calls == []


def test_max_utterance_caps_collection(monkeypatch):
    import semif_agent.voice_transport as vt

    ticks = iter([0.0, 5.0, 5.0, 5.0])
    monkeypatch.setattr(vt.time, "time", lambda: next(ticks))
    daemon, engines = make_daemon(
        [b"a", b"b", b"c"],
        config={"vad": {"silence_ms": 160, "min_speech_ms": 80, "max_utterance_s": 1.0}},
    )
    utterance = daemon._collect(engines, b"first")
    assert utterance == b"first"


def test_min_speech_rejects_short_utterance():
    daemon, engines = make_daemon(
        [b"WAKE", b"SIL", b"SIL"],
        config={"vad": {"silence_ms": 160, "min_speech_ms": 200, "max_utterance_s": 30}},
    )
    heard = []
    daemon.run(heard.append)
    assert heard == []
    assert engines.stt.calls == []


# ---- half-duplex / speaking ------------------------------------------------


def test_frames_ignored_while_speaking():
    daemon, engines = make_daemon([b"WAKE", b"speech", b"SIL"])
    daemon._speaking.set()
    heard = []
    daemon.run(heard.append)
    assert heard == []
    assert engines.stt.calls == []


def test_speak_synthesizes_and_plays():
    daemon, engines = make_daemon([])
    daemon.speak("  hello there  ")
    assert engines.tts.spoken == ["hello there"]
    assert engines.audio.writes == [(b"PCM", 22050)]
    # A reply opens the follow-up window (no wake word needed next).
    assert daemon._follow_until > 0


def test_follow_up_window_accepts_without_wake():
    daemon, engines = make_daemon([b"speech", b"SIL", b"SIL"])
    daemon.speak("are you there?")  # opens the follow-up window
    heard = []
    daemon.run(heard.append)
    assert heard == ["hello"]


def test_speak_truncates_long_text():
    daemon, engines = make_daemon([], config={"max_speak_chars": 10})
    daemon.speak("one two three four five")
    assert engines.tts.spoken[0] == "one two…"


def test_speak_failure_is_traced_not_raised():
    class BoomTts(FakeTts):
        def synthesize(self, text):
            raise RuntimeError("no audio device")

    daemon, engines = make_daemon([])
    engines.tts = BoomTts()
    daemon._engines = engines
    daemon.speak("hello")  # must not raise
    assert engines.audio.writes == []


# ---- adapter ---------------------------------------------------------------


def test_adapter_defaults():
    adapter = VoiceAdapter({})
    assert adapter.name == "voice"
    assert adapter.chat_id == "voice"
    assert adapter.daemon.sample_rate == 16000


def test_adapter_emits_inbound_message():
    adapter = VoiceAdapter({"chat_id": "mic", "display_name": "desk"})
    seen = []
    adapter._on_inbound = seen.append
    adapter._on_utterance("  hello there  ")
    assert len(seen) == 1
    assert seen[0].text == "hello there"
    assert seen[0].chat_id == "mic"
    assert seen[0].display_name == "desk"
    assert seen[0].chat_type == "dm"


def test_adapter_ignores_empty_utterance():
    adapter = VoiceAdapter({})
    seen = []
    adapter._on_inbound = seen.append
    adapter._on_utterance("   ")
    assert seen == []


# ---- CLI wiring ------------------------------------------------------------


def test_build_gateway_adapter_voice_anchors_model_dirs():
    adapter = _build_gateway_adapter("voice", {"enabled": True}, None)
    assert isinstance(adapter, VoiceAdapter)
    assert adapter.daemon.config["wake"]["model_dir"].endswith(".runtime/voice/wake")
    assert adapter.daemon.config["tts"]["voice_dir"].endswith(".runtime/voice/tts")


def test_resolve_platforms_includes_enabled_voice():
    platforms = _resolve_platforms(
        "all",
        {"simplex": {"enabled": False}, "voice": {"enabled": True}},
    )
    assert platforms == ["voice"]


# ---- resampling (numpy is a voice-stack dep; skip without it) --------------


def test_resample_pcm16():
    import pytest

    np = pytest.importorskip("numpy")
    from semif_agent.voice_transport import _resample_pcm16

    pcm = np.arange(441, dtype=np.int16).tobytes()
    assert len(_resample_pcm16(pcm, 44100, 16000)) // 2 == 160
    assert _resample_pcm16(pcm, 44100, 44100) == pcm
    assert _resample_pcm16(b"", 44100, 16000) == b""
