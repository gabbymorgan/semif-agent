"""Pure-stdlib tests for the neutral LXMF (Reticulum) transport.

`lxmf`/`rns` are optional and lazy-imported, so these tests exercise the
stdlib-only helpers and the missing-dependency contract without the real
package (and without touching a real Reticulum network). The live transport is
a live-integration concern.
"""

from semif_agent.gateway.lxmf import LxmfAdapter
from semif_agent.lxmf_transport import (
    AM_OPUS_OGG,
    FIELD_AUDIO,
    LxmfDaemon,
    audio_fields,
    normalize_message,
    parse_audio_field,
)


# ---- requirements ----

def test_requires_config_dir():
    ok, hint = LxmfDaemon(config_dir="", storage_path="x").check_requirements()
    assert ok is False
    assert "config_dir" in hint


def test_adapter_check_requirements_never_raises():
    # Either lxmf is installed (True) or an install hint is returned (False);
    # the adapter must never raise on a machine without the optional dep.
    ok, hint = LxmfAdapter({}).check_requirements()
    assert isinstance(ok, bool)
    if not ok:
        assert "lxmf" in hint


# ---- adapter defaults ----

def test_adapter_uses_runtime_lxmf_paths_by_default():
    adapter = LxmfAdapter({})
    assert adapter.config_dir.endswith(".runtime/lxmf/reticulum")
    assert adapter.storage_path.endswith(".runtime/lxmf/router")
    assert adapter.daemon.address == ""


# ---- normalization ----

def test_normalize_message_tolerates_fakes():
    class _Src:
        display_name = "alice"

    class _Msg:
        source_hash = bytes.fromhex("ab" * 16)
        content = b"hello"
        title = b"subj"
        timestamp = 123.0

        def get_source(self):
            return _Src()

    normalized = normalize_message(_Msg())
    assert normalized["source_hash"] == "ab" * 16
    assert normalized["content"] == "hello"
    assert normalized["title"] == "subj"
    assert normalized["display_name"] == "alice"
    assert normalized["timestamp"] == 123.0


def test_normalize_message_prefers_as_string_helpers():
    class _Msg:
        source_hash = b"\x01\x02"
        content = b"raw"
        title = b"rawtitle"

        def content_as_string(self):
            return "decoded content"

        def title_as_string(self):
            return "decoded title"

    normalized = normalize_message(_Msg())
    assert normalized["source_hash"] == "0102"
    assert normalized["content"] == "decoded content"
    assert normalized["title"] == "decoded title"
    assert normalized["display_name"] is None


def test_normalize_message_handles_none_content():
    class _Msg:
        source_hash = b""
        content = None
        title = None

        def content_as_string(self):
            return None

        def title_as_string(self):
            return None

    normalized = normalize_message(_Msg())
    assert normalized["source_hash"] == ""
    assert normalized["content"] == ""
    assert normalized["title"] == ""


# ---- voice (Sideband push-to-talk) audio field ----

def test_parse_audio_field_opus_ogg_uses_the_container_path():
    audio = parse_audio_field([0x10, b"OggS-payload"])
    assert audio == {
        "codec": 0x10,
        "codec_name": "opus_ogg",
        "data": b"OggS-payload",
        "kind": "opus_ogg",
        "codec2_rate": None,
    }


def test_parse_audio_field_codec2_maps_to_a_bitrate():
    audio = parse_audio_field([0x08, b"\x00\x01"])
    assert audio["codec_name"] == "codec2_2400"
    assert audio["kind"] == "codec2"
    assert audio["codec2_rate"] == 2400


def test_parse_audio_field_450_codec2_has_no_binding():
    # 450PWB/450 have no pycodec2 bindings, so they stay undecodable.
    audio = parse_audio_field([0x01, b"\x00\x01"])
    assert audio["kind"] == "unknown"
    assert audio["codec2_rate"] is None


def test_parse_audio_field_rejects_malformed():
    assert parse_audio_field("nope") is None
    assert parse_audio_field([0x10]) is None
    assert parse_audio_field([0x10, "not-bytes"]) is None
    assert parse_audio_field(["x", b"data"]) is None


def test_normalize_message_extracts_audio_field():
    class _Msg:
        source_hash = bytes.fromhex("ab" * 16)
        content = b""
        title = b""
        fields = {0x07: [0x10, b"OggS"]}

    normalized = normalize_message(_Msg())
    assert normalized["audio"]["codec_name"] == "opus_ogg"
    assert normalized["audio"]["data"] == b"OggS"
    assert normalized["audio"]["kind"] == "opus_ogg"
    assert normalized["fields"][0x07][0] == 0x10


def test_normalize_message_without_fields_has_no_audio():
    class _Msg:
        content = b"plain text"

    normalized = normalize_message(_Msg())
    assert normalized["audio"] is None
    assert normalized["fields"] == {}


# ---- outbound voice replies ----

def test_audio_fields_wraps_ogg_as_sideband_hq_ptt():
    fields = audio_fields(b"OggS-payload")
    assert fields == {FIELD_AUDIO: [AM_OPUS_OGG, b"OggS-payload"]}
    assert AM_OPUS_OGG == 0x10


def test_audio_fields_none_for_text_or_empty():
    assert audio_fields(None) is None
    assert audio_fields(b"") is None


def test_enqueue_carries_audio_on_the_outbound_queue():
    daemon = LxmfDaemon(config_dir="x", storage_path="y")
    daemon.enqueue("ab" * 16, "hello", audio=b"OggS")
    assert daemon._outbound.get_nowait() == ("ab" * 16, "hello", b"OggS")
    # The stop sentinel is unchanged.
    daemon.enqueue(None)
    assert daemon._outbound.get_nowait() is None
