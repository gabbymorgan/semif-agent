"""Pure-stdlib tests for the neutral LXMF (Reticulum) transport.

`lxmf`/`rns` are optional and lazy-imported, so these tests exercise the
stdlib-only helpers and the missing-dependency contract without the real
package (and without touching a real Reticulum network). The live transport is
a live-integration concern.
"""

from semif_agent.gateway.lxmf import LxmfAdapter
from semif_agent.lxmf_transport import LxmfDaemon, normalize_message


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
