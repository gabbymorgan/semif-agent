"""Core bridge tests: the shared LLM bridge and the discovery registry.

Bridge implementations ship with their packs; their hermetic tests live in the
pack (see `packs/*/bridge/test_*.py`, run by `tests/test_packs.py`). This file
covers what stays in the core: the shared, pack-less LLM bridge and the registry
surface (`known_infos`/`describe_bridges`/`derived_config_vars`) that discovers
the packs' bridges and renders the catalog codegen reads. The decision engine and
a real model are never involved.
"""

import json
import urllib.error
import urllib.request

import pytest

from semif_agent.bridges.llm import LLMBridge
from semif_agent.bridges.registry import (
    derived_config_vars,
    describe_bridges,
    known_infos,
)


def get(url, token=None):
    headers = {"X-Semif-Token": token} if token else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def post(url, payload, token=None):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Semif-Token"] = token
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))

# ---- LLM bridge ----

class FakeLLMClient:
    """Stands in for the scheduler's configured `llm` client."""

    def __init__(self, reply="ok", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def chat(self, messages, max_tokens=None):
        self.calls.append((messages, max_tokens))
        if self.error is not None:
            raise self.error
        return self.reply


def start_llm_bridge(client=None, **config):
    bridge = LLMBridge(config, client=client)
    port = bridge.start()
    return bridge, f"http://127.0.0.1:{port}"


def test_llm_bridge_chat_returns_model_text():
    client = FakeLLMClient(reply='{"title": "x", "description": ""}')
    bridge, base = start_llm_bridge(client)
    try:
        messages = [{"role": "user", "content": "add lunch tomorrow"}]
        assert post(f"{base}/chat", {"messages": messages}) == {
            "text": client.reply
        }
        assert client.calls == [(messages, None)]
    finally:
        bridge.stop()


def test_llm_bridge_forwards_max_tokens():
    client = FakeLLMClient()
    bridge, base = start_llm_bridge(client)
    try:
        post(f"{base}/chat", {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64})
        assert client.calls[0][1] == 64
    finally:
        bridge.stop()


def test_llm_bridge_rejects_bad_body():
    bridge, base = start_llm_bridge(FakeLLMClient())
    try:
        for payload in ({"messages": []}, {"messages": [{"role": "user"}]}, {}):
            with pytest.raises(urllib.error.HTTPError) as exc:
                post(f"{base}/chat", payload)
            assert exc.value.code == 400
    finally:
        bridge.stop()


def test_llm_bridge_without_a_client_is_unavailable():
    bridge, base = start_llm_bridge(None)
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(f"{base}/chat", {"messages": [{"role": "user", "content": "hi"}]})
        assert exc.value.code == 502
    finally:
        bridge.stop()


def test_llm_bridge_maps_client_error_to_502():
    bridge, base = start_llm_bridge(FakeLLMClient(error=RuntimeError("down")))
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(f"{base}/chat", {"messages": [{"role": "user", "content": "hi"}]})
        assert exc.value.code == 502
    finally:
        bridge.stop()


def test_llm_bridge_health_and_token():
    bridge, base = start_llm_bridge(FakeLLMClient(), token="sekret")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            get(f"{base}/health")
        assert exc.value.code == 401
        assert get(f"{base}/health", token="sekret") == {"ok": True, "platform": "llm"}
    finally:
        bridge.stop()



# ---- catalog / codegen surface ----

def test_known_infos_include_llm():
    infos = {info.name: info for info in known_infos()}
    assert "llm" in infos
    info = infos["llm"]
    assert info.service == "llm"
    assert info.url_config_var == "llm_bridge_url"
    assert info.auth_header == "X-Semif-Token"
    assert info.auth_config_var == "llm_bridge_token"
    assert any("/chat" in endpoint for endpoint in info.endpoints)


def test_every_bridge_documents_its_config_vars_and_auth():
    for info in known_infos():
        docs = dict(info.config_var_docs)
        assert set(info.config_vars) <= set(docs), info.name
        assert all(description.strip() for description in docs.values()), info.name
        if info.auth_header or info.auth_config_var:
            assert info.auth_header and info.auth_config_var, info.name
            assert info.auth_config_var in info.config_vars, info.name


def test_derived_config_vars_come_from_each_bridge_block():
    config = {
        "bridges": {
            "simplex": {"host": "127.0.0.1", "port": 5227, "token": "s"},
            "llm": {"host": "127.0.0.1", "port": 5229, "token": ""},
            "nextcloud": {"host": "127.0.0.1", "port": 5230, "token": "n"},
        }
    }
    derived = derived_config_vars(config)
    assert derived["simplex_bridge_url"] == "http://127.0.0.1:5227"
    assert derived["simplex_bridge_token"] == "s"
    assert derived["llm_bridge_url"] == "http://127.0.0.1:5229"
    assert derived["llm_bridge_token"] == ""
    assert derived["nextcloud_bridge_url"] == "http://127.0.0.1:5230"
    assert derived["nextcloud_bridge_token"] == "n"


def test_derived_config_vars_normalize_bind_all_host():
    config = {"bridges": {"llm": {"host": "0.0.0.0", "port": 5229}}}
    assert derived_config_vars(config)["llm_bridge_url"] == "http://127.0.0.1:5229"


def test_derived_config_vars_skip_unconfigured_bridges():
    # No block at all: nothing derived, so the contract variable stays
    # unresolved and the operator is asked for it.
    assert derived_config_vars({}) == {}
    # A block without a port is not a callable address: no URL, but the token
    # var is still satisfiable (empty = no auth).
    derived = derived_config_vars({"bridges": {"nextcloud": {"token": ""}}})
    assert "nextcloud_bridge_url" not in derived
    assert derived["nextcloud_bridge_token"] == ""


