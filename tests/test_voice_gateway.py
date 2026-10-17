"""Pure-stdlib tests for the voice gateway (adapter + neutral transport).

The voice stack (sounddevice/openwakeword/faster-whisper/piper) is optional and
lazy-imported, so these tests inject fake engines and drive the loop without
hardware, network, or the heavy packages. The live pipeline against a real mic
is a live-integration concern.

Frame constants: the fake wake word fires on `b"WAKE"`, the fake VAD calls any
non-`b"SIL"` frame speech, and the fake STT returns a fixed string.
"""

from semif_agent.cli import _build_gateway_adapter, _resolve_platforms
from semif_agent.gateway.base import OutboundMessage
from semif_agent.gateway.voice import VoiceAdapter, _is_stop_command
from semif_agent.voice_transport import VoiceDaemon, VoiceEngines


# ---- fakes -----------------------------------------------------------------


class FakeAudio:
    def __init__(self, frames, on_empty=None):
        self.frames = list(frames)
        #: What `read()` returns when the script runs out: `None` = the input is
        #: closed (the loop exits), `b""` = a read timeout (a stalled stream).
        self.on_empty = on_empty
        self.writes = []
        self.closed = False
        self.input_opens = 0
        #: Optional hook run at the start of `play` (to simulate a barge-in
        #: landing mid-playback by setting the `stop_event`).
        self.on_play = None

    def open_input(self, sample_rate, frame_ms, device=""):
        self.input_opens += 1

    def read(self, timeout=None):
        if self.frames:
            return self.frames.pop(0)
        return self.on_empty

    def open_output(self, sample_rate, device=""):
        pass

    def write(self, pcm16, sample_rate):
        self.writes.append((pcm16, sample_rate))

    def play(self, pcm16, sample_rate, stop_event=None):
        if self.on_play is not None:
            self.on_play(stop_event)
        if stop_event is not None and stop_event.is_set():
            return False
        self.writes.append((pcm16, sample_rate))
        return True

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


def make_daemon(frames, stt_text="hello", config=None, input_ready=None):
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
        "max_speak_chars": 600,
    }
    if config:
        cfg.update(config)
    return VoiceDaemon(cfg, engines=engines, input_ready=input_ready), engines


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
    assert engines.audio.input_opens == 1
    assert engines.audio.closed is True
    # The wake detector is reset after firing.
    assert engines.wake.resets >= 1


def test_no_wake_word_produces_nothing():
    daemon, engines = make_daemon([b"noise", b"noise", b"SIL"])
    heard = []
    daemon.run(heard.append)
    assert heard == []
    assert engines.stt.calls == []


# ---- input gating (hold the wake word while the scheduler is busy) ---------


def test_wake_blocked_while_input_not_ready():
    # While the scheduler slot is busy the wake word is ignored: no cue, no
    # capture, no transcription (traced `voice_wake_blocked`).
    ready = {"ok": False}
    daemon, engines = make_daemon(
        [b"WAKE", b"speech", b"SIL", b"SIL"], input_ready=lambda: ready["ok"]
    )
    events = []
    daemon._event = lambda kind, **kw: events.append(kind)
    heard = []
    daemon.run(heard.append)
    assert heard == []
    assert engines.stt.calls == []
    assert engines.audio.writes == []
    assert "voice_wake_blocked" in events
    assert "voice_wake" not in events


def test_wake_allowed_when_input_ready():
    ready = {"ok": True}
    daemon, engines = make_daemon(
        [b"WAKE", b"speech", b"SIL", b"SIL"], input_ready=lambda: ready["ok"]
    )
    heard = []
    daemon.run(heard.append)
    assert heard == ["hello"]


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


def test_command_after_pause_is_captured():
    # A pause between the wake word and the command (e.g. waiting for the cue
    # beep) must not truncate the command: capture waits for speech to begin.
    daemon, engines = make_daemon(
        [b"WAKE", b"SIL", b"SIL", b"SIL", b"command", b"SIL", b"SIL"]
    )
    heard = []
    daemon.run(heard.append)
    assert heard == ["hello"]
    assert engines.stt.calls == [b"command" + b"SIL" * 2]


def test_pause_between_wake_and_command_is_tolerated():
    # The wake detector can fire inside the wake phrase, so the frames after it
    # are the phrase's tail, not the command; a natural pause before the command
    # must not end the capture (the bug that made repeated asks fail).
    daemon, engines = make_daemon(
        [b"WAKE", b"tail", b"SIL", b"SIL", b"SIL", b"SIL", b"SIL", b"SIL",
         b"command", b"SIL", b"SIL"]
    )
    heard = []
    daemon.run(heard.append)
    assert heard == ["hello"]
    assert b"command" in engines.stt.calls[0]


def test_short_command_still_endpoints():
    # Once enough speech is in hand the normal silence endpoint applies, so a
    # command is not left waiting on the longer initial tolerance.
    daemon, engines = make_daemon(
        [b"WAKE", b"a", b"b", b"c", b"d", b"e", b"f", b"g", b"h",
         b"SIL", b"SIL", b"SIL"],
        config={"vad": {"silence_ms": 160, "min_speech_ms": 80,
                        "min_command_ms": 400, "initial_silence_ms": 2000}},
    )
    heard = []
    daemon.run(heard.append)
    assert heard == ["hello"]
    # It ended on the 160 ms endpoint (two silent frames), not on the longer
    # initial tolerance.
    assert engines.stt.calls[0] == b"abcdefgh" + b"SIL" * 2


def test_wake_without_command_times_out(monkeypatch):
    # Wake fires but the user never speaks a command: no utterance is produced
    # (traced `voice_no_command`) instead of capturing the wake frame as one.
    import semif_agent.voice_transport as vt

    ticks = iter([0.0, 0.0, 3000.0])
    monkeypatch.setattr(vt.time, "time", lambda: next(ticks))
    daemon, engines = make_daemon([b"SIL", b"SIL"])
    events = []
    daemon._event = lambda kind, **kw: events.append(kind)
    utterance = daemon._collect(engines, b"WAKE", wait_for_speech_ms=3000)
    assert utterance == b""
    assert "voice_no_command" in events


# ---- capture-stream watchdog ----------------------------------------------


def test_capture_stall_reopens_input(monkeypatch):
    # A stalled PortAudio/ALSA callback never delivers another frame; the loop
    # must reopen the input instead of blocking forever.
    import semif_agent.voice_transport as vt

    daemon, engines = make_daemon([])
    engines.audio.on_empty = b""  # every read times out (a stalled stream)
    events = []
    daemon._event = lambda kind, **kw: events.append(kind)
    clock = iter([100.0, 100.5, 106.0])
    monkeypatch.setattr(vt.time, "time", lambda: next(clock))

    assert daemon._read_frame(engines) == b""  # first timeout: stall starts
    assert daemon._read_frame(engines) == b""  # still stalled, below the limit
    assert daemon._read_frame(engines) == b""  # past stall_timeout_s -> reopen
    assert "voice_stream_stalled" in events
    assert "voice_stream_reopened" in events
    assert engines.audio.input_opens == 1
    assert engines.audio.closed is True  # closed before reopening


def test_capture_stall_reopen_failure_is_traced(monkeypatch):
    import semif_agent.voice_transport as vt

    daemon, engines = make_daemon([])
    engines.audio.on_empty = b""

    def boom(sample_rate, frame_ms, device=""):
        raise RuntimeError("device busy")

    engines.audio.open_input = boom
    events = []
    daemon._event = lambda kind, **kw: events.append(kind)
    clock = iter([100.0, 106.0])
    monkeypatch.setattr(vt.time, "time", lambda: next(clock))
    monkeypatch.setattr(vt.time, "sleep", lambda *_: None)

    daemon._read_frame(engines)
    daemon._read_frame(engines)
    assert "voice_stream_reopen_failed" in events


def test_watchdog_disabled_uses_blocking_read(monkeypatch):
    # read_timeout_s=0 restores the old blocking read (no timeout kwarg passed).
    daemon, engines = make_daemon(
        [], config={"audio": {"frame_ms": 80, "sample_rate": 16000, "read_timeout_s": 0}}
    )
    seen = {}

    def read(timeout=None):
        seen["timeout"] = timeout
        return None

    engines.audio.read = read
    assert daemon._read_frame(engines) is None
    assert seen["timeout"] is None


# ---- half-duplex / speaking ------------------------------------------------


def test_frames_ignored_while_speaking_when_barge_in_disabled():
    # Strict half-duplex: with barge-in off the wake word is not even scored
    # while a reply is playing, so the agent never transcribes its own voice.
    daemon, engines = make_daemon(
        [b"WAKE", b"speech", b"SIL"], config={"barge_in": {"enabled": False}}
    )
    daemon._speaking.set()
    heard = []
    daemon.run(heard.append)
    assert heard == []
    assert engines.stt.calls == []


def test_barge_in_wake_aborts_playback_and_captures_command():
    # A reply is playing; the wake word fires over it -> the daemon stops the
    # playback, tells the front end (on_barge_in True then False), and captures
    # the command that follows.
    daemon, engines = make_daemon([b"WAKE", b"command", b"SIL", b"SIL"])
    events = []
    daemon._event = lambda kind, **kw: events.append(kind)
    barges = []

    def on_barge(active):
        barges.append(active)
        if active:
            daemon._speaking.clear()  # the interrupted playback has stopped

    daemon.on_barge_in = on_barge
    daemon._speaking.set()  # a reply is playing when the wake word lands
    heard = []
    daemon.run(heard.append)
    assert heard == ["hello"]
    assert barges == [True, False]
    assert "voice_barge_in" in events
    assert "voice_wake" not in events  # barge-in, not a fresh wake


def test_barge_in_threshold_gates_interruption():
    # A barge-in threshold above the wake score means the reply is not
    # interrupted (the agent's own TTS cannot self-trigger a barge-in).
    daemon, engines = make_daemon(
        [b"WAKE", b"command", b"SIL", b"SIL"],
        config={"barge_in": {"threshold": 2.0}},  # FakeWake scores 1.0
    )
    daemon._speaking.set()
    heard = []
    daemon.run(heard.append)
    assert heard == []
    assert engines.stt.calls == []


def test_barge_in_without_command_releases_mute():
    # The wake word interrupts, but no command follows: the listening session
    # ends and the mute is lifted (the front end is not left muted).
    daemon, engines = make_daemon([b"WAKE", b"SIL", b"SIL", b"SIL"])
    daemon._speaking.set()

    def on_barge(active):
        if active:
            daemon._speaking.clear()

    daemon.on_barge_in = on_barge
    daemon.run(lambda text: None)
    assert daemon._mute.is_set() is False


def test_speak_muted_while_barge_in():
    daemon, engines = make_daemon([])
    events = []
    daemon._event = lambda kind, **kw: events.append(kind)
    daemon.set_muted(True)
    daemon.speak("should not be spoken")
    assert engines.tts.spoken == []
    assert engines.audio.writes == []
    assert "voice_speak_suppressed" in events


def test_speak_interrupted_mid_playback():
    daemon, engines = make_daemon([])
    events = []
    daemon._event = lambda kind, **kw: events.append(kind)

    def on_play(stop_event):
        stop_event.set()  # a barge-in lands while the buffer is playing

    engines.audio.on_play = on_play
    daemon.speak("a long reply")
    assert "voice_speak_interrupted" in events


def test_speak_synthesizes_and_plays():
    daemon, engines = make_daemon([])
    daemon.speak("  hello there  ")
    assert engines.tts.spoken == ["hello there"]
    assert engines.audio.writes == [(b"PCM", 22050)]


def test_no_follow_up_after_reply():
    # The wake word is required for every command: speech after a reply is not
    # captured until the wake word fires again.
    daemon, engines = make_daemon([b"speech", b"SIL", b"SIL"])
    daemon.speak("are you there?")
    heard = []
    daemon.run(heard.append)
    assert heard == []
    assert engines.stt.calls == []


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
    # The utterance is queued for the worker, not run on the audio thread.
    assert adapter._inbound.qsize() == 1
    adapter._handle(adapter._inbound.get())
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
    assert adapter._inbound.empty()


def test_adapter_worker_handles_then_stops():
    adapter = VoiceAdapter({})
    seen = []
    adapter._on_inbound = seen.append
    adapter._on_utterance("one")
    adapter._inbound.put(None)  # sentinel stops the worker
    adapter._inbound_worker()
    assert [m.text for m in seen] == ["one"]


def test_adapter_contains_handler_error():
    # A task that raises must not kill the front end: the worker traces it and
    # speaks an apology instead of propagating.
    adapter = VoiceAdapter({})

    def boom(message):
        raise RuntimeError("kaboom")

    adapter._on_inbound = boom
    spoken = []
    adapter.daemon.speak = lambda text: spoken.append(text)
    adapter._on_utterance("do the thing")
    adapter._handle(adapter._inbound.get())  # must not raise
    assert spoken and "wrong" in spoken[0]


def test_adapter_is_result_only():
    # The spoken front end speaks only the skill's result line, not the
    # scheduler's bookkeeping (overridable via gateway.voice.result_only).
    assert VoiceAdapter({}).result_only is True


# ---- barge-in (adapter classification) -------------------------------------


def test_is_stop_command():
    assert _is_stop_command("Stop.") is True
    assert _is_stop_command("cancel") is True
    assert _is_stop_command("stop please") is True
    assert _is_stop_command("shut up already") is True
    assert _is_stop_command("  Quiet!  ") is True
    assert _is_stop_command("never mind") is True
    # A longer utterance is a real command, not a stop.
    assert _is_stop_command("stop the timer") is False
    assert _is_stop_command("shut the door") is False
    assert _is_stop_command("turn on the lights") is False
    assert _is_stop_command("") is False


def test_adapter_barge_in_stop_drops_replies_and_does_not_forward():
    import queue

    adapter = VoiceAdapter({})
    seen = []
    adapter._on_inbound = seen.append
    outbound: "queue.Queue" = queue.Queue()
    outbound.put(OutboundMessage("voice", "one"))
    outbound.put(OutboundMessage("voice", "two"))
    adapter._outbound = outbound
    events = []
    adapter.daemon._event = lambda kind, **kw: events.append(kind)
    adapter.daemon.set_muted(True)
    adapter._barged_in = True
    adapter._on_utterance("stop")
    adapter._handle(adapter._inbound.get())
    assert seen == []  # never submitted as a request
    assert outbound.empty()  # the queued replies were dropped
    assert "gateway_barge_in_stop" in events
    assert "gateway_speech_dropped" in events
    assert adapter.daemon._mute.is_set() is False  # unmuted for later replies


def test_adapter_barge_in_stop_preserves_shutdown_sentinel():
    import queue

    adapter = VoiceAdapter({})
    adapter._on_inbound = lambda message: None
    outbound: "queue.Queue" = queue.Queue()
    outbound.put(OutboundMessage("voice", "stale"))
    outbound.put(None)
    adapter._outbound = outbound
    adapter._barged_in = True
    adapter._on_utterance("cancel")
    adapter._handle(adapter._inbound.get())
    # The stale reply is dropped but the pump's sentinel is preserved.
    assert outbound.get() is None


def test_adapter_barge_in_new_command_is_forwarded():
    adapter = VoiceAdapter({})
    seen = []
    adapter._on_inbound = seen.append
    adapter.daemon.set_muted(True)
    adapter._barged_in = True
    adapter._on_utterance("turn on the lights")
    message = adapter._inbound.get()
    assert message.raw.get("barge_in") is True
    adapter._handle(message)
    assert [m.text for m in seen] == ["turn on the lights"]
    assert adapter.daemon._mute.is_set() is False


def test_adapter_normal_utterance_is_not_a_barge_in():
    adapter = VoiceAdapter({})
    seen = []
    adapter._on_inbound = seen.append
    adapter._on_utterance("hello")
    message = adapter._inbound.get()
    assert message.raw.get("barge_in") is not True
    adapter._handle(message)
    assert [m.text for m in seen] == ["hello"]


def test_adapter_on_barge_in_callback_sets_flag():
    adapter = VoiceAdapter({})
    adapter._on_barge_in(True)
    assert adapter._barged_in is True
    adapter._on_barge_in(False)
    assert adapter._barged_in is False


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


# ---- wake cue (beep) -------------------------------------------------------


def test_transcription_requirements_never_raises():
    from semif_agent.voice_transport import check_transcription_requirements

    ok, hint = check_transcription_requirements()
    assert isinstance(ok, bool)
    if not ok:
        assert "faster-whisper" in hint


def test_check_codec2_requirements_never_raises():
    from semif_agent.voice_transport import check_codec2_requirements

    ok, hint = check_codec2_requirements()
    assert isinstance(ok, bool)
    if not ok:
        assert "pycodec2" in hint


def test_decode_codec2_without_a_full_frame_is_none():
    from semif_agent.voice_transport import decode_codec2

    # Nothing to decode (shorter than a frame, empty, or no bitrate) -> None,
    # whether or not pycodec2 is installed.
    assert decode_codec2(b"x", 2400) is None
    assert decode_codec2(b"", 2400) is None
    assert decode_codec2(b"data", 0) is None


def test_decode_codec2_uses_bindings(monkeypatch):
    import sys
    import types

    import pytest

    np = pytest.importorskip("numpy")
    calls = []

    class FakeCodec2:
        def __init__(self, rate):
            self.rate = rate

        def bytes_per_frame(self):
            return 4

        def samples_per_frame(self):
            return 8

        def decode(self, data):
            calls.append(data)
            return np.full(8, 7, dtype=np.int16)

    fake = types.ModuleType("pycodec2")
    fake.Codec2 = FakeCodec2
    monkeypatch.setitem(sys.modules, "pycodec2", fake)
    from semif_agent.voice_transport import decode_codec2

    pcm = decode_codec2(b"aaaa" * 2, 2400)  # two 4-byte frames
    assert len(pcm) == 2 * 8 * 2  # frames * samples/frame * 2 bytes
    assert calls == [b"aaaa", b"aaaa"]
    assert pcm[:2] == b"\x07\x00"


def test_pcm_to_wav_is_a_readable_wav():
    import io
    import wave

    from semif_agent.voice_transport import pcm_to_wav

    pcm = b"\x01\x00\x02\x00"
    data = pcm_to_wav(pcm, 8000)
    assert data[:4] == b"RIFF"
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 8000
        assert wav.readframes(2) == pcm


# ---- voice replies (Piper + Opus/OGG) --------------------------------------


def test_synthesis_requirements_never_raises():
    from semif_agent.voice_transport import check_synthesis_requirements

    ok, hint = check_synthesis_requirements()
    assert isinstance(ok, bool)
    if not ok:
        assert "piper" in hint


def test_build_synthesizer_reads_the_tts_block(monkeypatch):
    import semif_agent.voice_transport as vt

    captured = {}

    class FakePiper:
        def __init__(self, voice, speaker_id=0, length_scale=1.0, volume=1.0):
            captured.update(
                voice=voice, speaker_id=speaker_id,
                length_scale=length_scale, volume=volume,
            )

    monkeypatch.setattr(vt, "PiperSynthesizer", FakePiper)
    vt.build_synthesizer(
        {"tts": {"voice": "en_US-lessac-medium", "speaker_id": 3, "length_scale": 0.9}}
    )
    assert captured["voice"] == "en_US-lessac-medium"
    assert captured["speaker_id"] == 3
    assert captured["length_scale"] == 0.9


def test_pcm_to_opus_ogg_empty_input_is_empty():
    from semif_agent.voice_transport import pcm_to_opus_ogg

    assert pcm_to_opus_ogg(b"", 22050) == b""
    assert pcm_to_opus_ogg(b"\x01", 22050) == b""


def test_pcm_to_opus_ogg_encodes_when_pyav_is_present():
    import pytest

    pytest.importorskip("av")
    import io

    import av

    from semif_agent.voice_transport import pcm_to_opus_ogg

    # 0.5 s of 22.05 kHz silence -> resampled + encoded to OGG/Opus.
    pcm = b"\x00\x00" * 11025
    data = pcm_to_opus_ogg(pcm, 22050)
    assert data[:4] == b"OggS"
    container = av.open(io.BytesIO(data))
    stream = container.streams.audio[0]
    assert stream.codec_context.name == "opus"
    assert stream.sample_rate == 48000


def test_tone_pcm_length():
    from semif_agent.voice_transport import _tone_pcm

    pcm = _tone_pcm(880, 100, 16000, 0.25)
    assert len(pcm) == 1600 * 2  # 100 ms @ 16 kHz, int16
    assert _tone_pcm(0, 100, 16000, 0.25) == b""


def test_wake_plays_cue():
    daemon, engines = make_daemon([b"WAKE", b"speech", b"SIL", b"SIL"])
    daemon.run(lambda text: None)
    # exactly one cue played on wake, at the processing rate
    assert len(engines.audio.writes) == 1
    pcm, rate = engines.audio.writes[0]
    assert rate == 16000
    assert len(pcm) == int(16000 * 120 / 1000) * 2


def test_cue_disabled_suppresses_beep():
    daemon, engines = make_daemon(
        [b"WAKE", b"speech", b"SIL", b"SIL"], config={"cue": {"enabled": False}}
    )
    daemon.run(lambda text: None)
    assert engines.audio.writes == []


def test_submit_plays_bling_after_transcription():
    # A configured `submit_notes` chime plays once the utterance is transcribed,
    # as a single write (the notes are concatenated) at the processing rate.
    daemon, engines = make_daemon(
        [b"WAKE", b"speech", b"SIL", b"SIL"],
        config={"cue": {"submit_notes": [[1046.5, 70], [1568.0, 90]], "note_gap_ms": 20}},
    )
    daemon.run(lambda text: None)
    # The wake beep, then the bling.
    assert len(engines.audio.writes) == 2
    pcm, rate = engines.audio.writes[1]
    assert rate == 16000
    expected = int(16000 * (70 + 90) / 1000) * 2 + int(16000 * 20 / 1000) * 2
    assert len(pcm) == expected


def test_submit_bling_falls_back_to_single_beep():
    daemon, engines = make_daemon(
        [b"WAKE", b"speech", b"SIL", b"SIL"],
        config={"cue": {"submit_frequency": 1320, "submit_duration_ms": 80}},
    )
    daemon.run(lambda text: None)
    assert len(engines.audio.writes) == 2
    pcm, rate = engines.audio.writes[1]
    assert len(pcm) == int(16000 * 80 / 1000) * 2


def test_submit_bling_disabled_suppresses():
    daemon, engines = make_daemon(
        [b"WAKE", b"speech", b"SIL", b"SIL"],
        config={"cue": {"enabled": False, "submit_notes": [[1046.5, 70]]}},
    )
    daemon.run(lambda text: None)
    assert engines.audio.writes == []


def test_parse_notes_drops_bad_entries():
    from semif_agent.voice_transport import _parse_notes

    assert _parse_notes([[880, 100], "nope", [0, 50], [440, 0], [660.0, 30.9]]) == [
        (880.0, 100),
        (660.0, 30),
    ]
    assert _parse_notes(None) == []


# ---- resampling (numpy is a voice-stack dep; skip without it) --------------


def test_resample_pcm16():
    import pytest

    np = pytest.importorskip("numpy")
    from semif_agent.voice_transport import _resample_pcm16

    pcm = np.arange(441, dtype=np.int16).tobytes()
    assert len(_resample_pcm16(pcm, 44100, 16000)) // 2 == 160
    assert _resample_pcm16(pcm, 44100, 44100) == pcm
    assert _resample_pcm16(b"", 44100, 16000) == b""


# ---- wake-word model pinning (community model, no network) ------------------


def _load_voice_models():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "voice-models.py"
    spec = importlib.util.spec_from_file_location("voice_models_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wake_word_pin_resolves_from_model_dir(tmp_path):
    import hashlib

    from semif_agent.voice_transport import resolve_model_path

    module = _load_voice_models()
    pins = module.pinned_wake_models()
    assert "hey_computer" in pins
    ref = pins["hey_computer"]
    assert ref["url"].endswith("/hey_computer.onnx")
    assert len(ref["sha256"]) == len(hashlib.sha256(b"").hexdigest())

    stub = tmp_path / "hey_computer.onnx"
    stub.write_bytes(b"stub")
    assert resolve_model_path("hey_computer", str(tmp_path), (".onnx", ".tflite")) == str(stub)


def test_download_wake_file_verifies_and_skips_existing(tmp_path):
    import hashlib

    module = _load_voice_models()
    data = b"fake-openwakeword-model"
    source = tmp_path / "src.onnx"
    source.write_bytes(data)
    ref = {"url": source.as_uri(), "sha256": hashlib.sha256(data).hexdigest()}

    wake_dir = tmp_path / "wake"
    module.download_wake_file("hey_computer", ref, wake_dir)
    dest = wake_dir / "hey_computer.onnx"
    assert dest.read_bytes() == data
    # Idempotent: a present, matching file is not re-fetched.
    module.download_wake_file("hey_computer", ref, wake_dir)
    assert dest.read_bytes() == data


def test_download_wake_file_rejects_sha_mismatch(tmp_path):
    import pytest

    module = _load_voice_models()
    source = tmp_path / "src.onnx"
    source.write_bytes(b"tampered")
    ref = {"url": source.as_uri(), "sha256": "0" * 64}

    wake_dir = tmp_path / "wake"
    with pytest.raises(SystemExit):
        module.download_wake_file("hey_computer", ref, wake_dir)
    assert not (wake_dir / "hey_computer.onnx").exists()
