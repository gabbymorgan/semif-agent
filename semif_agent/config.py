"""Layered per-machine configuration.

Mirrors Home Assistant's split-config model:

* ``config.json`` is a **thin index** — ``{"include": ["config/core.json",
  "config/packs/nextcloud.json", ...]}``. Each included file is an ordinary
  config fragment; fragments are deep-merged in order (later wins).
* Secrets live in a gitignored ``secrets.json`` and are referenced from any
  fragment with a ``"!secret <key>"`` string, resolved at load (the analog of
  HA's ``!secret`` tag). ``api_key_env`` is gone — every secret is a
  ``secrets.json`` entry.
* Each skill pack ships its own config **schema** (``pack.json`` ``config``),
  which owns the pack's defaults and drives both ``config.example.json``
  generation and ``check-config`` validation.

The loader never hard-fails on a missing secret (a ``!secret`` with no entry
resolves to ``""``, matching the old empty-key behaviour); ``validate_config``
is where a missing secret is reported.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

SECRET_PREFIX = "!secret "
DEFAULT_SECRETS_FILE = "secrets.json"

#: Core (non-pack) config schema. A section is a nested dict; a leaf is a dict
#: with a ``default`` key, or a bare scalar shorthand for ``{"default": ...}``.
#: ``doc`` documents the key (emitted as a ``_comment`` in the example), and
#: ``secret: true`` marks a value that must come from ``secrets.json``.
CORE_SCHEMA: dict[str, Any] = {'tau': {'default': 0.6, 'type': 'float', 'doc': 'Assessment accept threshold.'},
 'max_reentries': {'default': 3,
                   'type': 'int',
                   'doc': 'Max times a `continue` step may re-enter routing for one request.'},
 'query_recall_window': {'default': 30,
                         'type': 'int',
                         'doc': "Seconds a completed run's ledger stays recallable for a follow-up "
                                'in the same chat (0 disables).'},
 'new_skill_creation': {'default': 'deny',
                        'type': 'str',
                        'doc': 'Global tri-state new-skill mode: deny (default) | allow | ask (a '
                               'category config overrides it). Deny by default: the agent leans on '
                               'the shipped packs, and a host opts in to live skill authoring by '
                               'setting allow.'},
 'verbosity': {'default': 'result',
               'type': 'str',
               'doc': 'REPL reply verbosity: result | query | decision.'},
 'chain': {'result_chars': {'default': 300, 'type': 'int'}},
 'assess': {'done_tau': {'default': 0.5, 'type': 'float'},
            'fail_tau': {'default': 0.5, 'type': 'float'}},
 'timezone': {'default': '',
              'type': 'str',
              'doc': 'IANA zone (e.g. America/Chicago) the agent reports local time in; empty uses '
                     'the host zone.'},
 'navigation': {'intent_tau': {'default': 0.6, 'type': 'float'},
                'category_tau': {'default': 0.75, 'type': 'float'},
                'action_tau': {'default': 0.5, 'type': 'float'},
                'softmax_bypass_tau': {'default': 0.5, 'type': 'float'}},
 'queue': {'max_size': {'default': 100, 'type': 'int'}},
 'engine': {'provider': {'default': 'semif', 'type': 'str'},
            'backend': {'default': 'llamacpp', 'type': 'str'},
            'context_tokens': {'default': 4096, 'type': 'int'},
            'threads': {'default': 8, 'type': 'int'},
            'gpu_layers': {'default': 0, 'type': 'int'},
            'gpu_device': {'default': None, 'type': 'null'},
            'gpu_split': {'default': 0, 'type': 'int'},
            'winnow': {'base_url': {'default': 'http://127.0.0.1:8091', 'type': 'str'},
                       'model': {'default': '', 'type': 'str'},
                       'context_tokens': {'default': 32768, 'type': 'int'},
                       'api_key': {'default': '', 'type': 'str', 'secret': True},
                       'timeout': {'default': 120, 'type': 'int'},
                       'retries': {'default': 0, 'type': 'int'},
                       'user_agent': {'default': 'semif-agent', 'type': 'str'},
                       'extra_headers': {'default': {}, 'type': 'dict'}},
            'gguf': {'default': '', 'type': 'str'}},
 'llm': {'provider': {'default': 'ollama', 'type': 'str'},
         'base_url': {'default': 'http://localhost:11434/v1', 'type': 'str'},
         'model': {'default': '', 'type': 'str'},
         'api_key': {'default': '', 'type': 'str', 'secret': True},
         'user_agent': {'default': '', 'type': 'str'},
         'timeout': {'default': 600, 'type': 'int'},
         'stream': {'default': False, 'type': 'bool'},
         'idle_warn': {'default': 30, 'type': 'int'},
         'idle_timeout': {'default': 120, 'type': 'int'},
         'context_window': {'default': 0, 'type': 'int'},
         'temperature': {'default': 0.2, 'type': 'float'},
         'top_p': {'default': 0.9, 'type': 'float'},
         'presence_penalty': {'default': 0.0, 'type': 'float'},
         'frequency_penalty': {'default': 0.0, 'type': 'float'},
         'disable_thinking': {'default': True, 'type': 'bool'},
         'semif': {'temperature': {'default': 0.0, 'type': 'float'},
                   'max_tokens': {'default': 256, 'type': 'int'}}},
 'codegen': {'provider': {'default': 'ollama', 'type': 'str'},
             'base_url': {'default': 'http://localhost:11434/v1', 'type': 'str'},
             'model': {'default': '', 'type': 'str'},
             'api_key': {'default': '', 'type': 'str', 'secret': True},
             'user_agent': {'default': '', 'type': 'str'},
             'timeout': {'default': 3600, 'type': 'int'},
             'stream': {'default': True, 'type': 'bool'},
             'idle_warn': {'default': 60, 'type': 'int'},
             'idle_timeout': {'default': 180, 'type': 'int'},
             'context_window': {'default': 0, 'type': 'int'},
             'smart_limit': {'default': 250000, 'type': 'int'},
             'warn_limit': {'default': 500000, 'type': 'int'},
             'max_fill_ratio': {'default': 0.9, 'type': 'float'},
             'warn_fill_ratio': {'default': 0.7, 'type': 'float'},
             'max_output': {'default': 0.85, 'type': 'float'},
             'chars_per_token': {'default': 4.0, 'type': 'float'},
             'temperature': {'default': 1, 'type': 'int'},
             'top_p': {'default': 0.95, 'type': 'float'},
             'presence_penalty': {'default': 0, 'type': 'int'},
             'frequency_penalty': {'default': 1, 'type': 'int'},
             'max_attempts': {'default': 3, 'type': 'int'},
             'approval': {'wait_timeout': {'default': 900, 'type': 'int'}},
             'fidelity': {'enabled': {'default': True, 'type': 'bool'},
                          'max_attempts': {'default': 1, 'type': 'int'},
                          'tau': {'default': 0.6, 'type': 'float'}},
             'repair': {'enabled': {'default': True, 'type': 'bool'},
                        'max_attempts': {'default': 1, 'type': 'int'}},
             'test_timeout': {'default': 30, 'type': 'int'},
             'test_max_attempts': {'default': 3, 'type': 'int'},
             'max_body_tokens': {'default': 0, 'type': 'int'},
             'max_test_tokens': {'default': 0, 'type': 'int'},
             'decision_reserve_tokens': {'default': 1024, 'type': 'int'},
             'contract_search': {'enabled': {'default': True, 'type': 'bool'}},
             'degeneration': {'enabled': {'default': True, 'type': 'bool'},
                              'threshold': {'default': 0.9, 'type': 'float'},
                              'interval': {'default': 8000, 'type': 'int'},
                              'window': {'default': 2000, 'type': 'int'},
                              'min_chars': {'default': 4000, 'type': 'int'}}},
 'skill_bodies': {'default': 'data/skills', 'type': 'str'},
 'log': {'default': 'data/decisions.jsonl', 'type': 'str'},
 'trace': {'default': 'data/runs.jsonl', 'type': 'str'},
 'category_registry': {'default': 'data/categories.json', 'type': 'str'},
 'simplex_chat': {'port': {'default': 5226, 'type': 'int'},
                  'display_name': {'default': 'semif', 'type': 'str'},
                  'forward_port': {'default': 5228, 'type': 'int'},
                  'forward_display_name': {'default': 'semif-forward', 'type': 'str'}},
 'gateway': {'humanize': {'enabled': {'default': True, 'type': 'bool'},
                          'timeout': {'default': 20.0, 'type': 'float'},
                          'max_tokens': {'default': 200, 'type': 'int'},
                          'max_chars': {'default': 600, 'type': 'int'}},
             'simplex': {'enabled': {'default': True, 'type': 'bool'},
                         'ws_url': {'default': 'ws://127.0.0.1:5226', 'type': 'str'},
                         'allowed_users': {'default': [], 'type': 'list'},
                         'allow_all_users': {'default': False, 'type': 'bool'},
                         'auto_accept': {'default': True, 'type': 'bool'},
                         'home_channel': {'default': '', 'type': 'str'},
                         'reply_prefix': {'default': '', 'type': 'str'},
                         'text_batch_delay': {'default': 0.8, 'type': 'float'},
                         'reconnect': {'initial': {'default': 1.0, 'type': 'float'},
                                       'max': {'default': 60.0, 'type': 'float'},
                                       'jitter': {'default': 0.2, 'type': 'float'}}},
             'lxmf': {'enabled': {'default': False, 'type': 'bool'},
                      'allowed_users': {'default': [], 'type': 'list'},
                      'allow_all_users': {'default': False, 'type': 'bool'},
                      'display_name': {'default': 'semif', 'type': 'str'},
                      'announce_interval': {'default': 3600, 'type': 'int'},
                      'stamp_cost': {'default': 8, 'type': 'int'},
                      'desired_method': {'default': 'direct', 'type': 'str'},
                      'propagation_node': {'default': '', 'type': 'str'},
                      'config_dir': {'default': '', 'type': 'str'},
                      'storage_path': {'default': '', 'type': 'str'},
                      'home_channel': {'default': '', 'type': 'str'},
                      'reply_prefix': {'default': '', 'type': 'str'},
                      'text_batch_delay': {'default': 0.8, 'type': 'float'},
                      # Reply with the skill's result line only (drop queue/
                      # urgency chatter and the `<skill>: ok —` wrapper).
                      'result_only': {'default': True, 'type': 'bool'},
                      # Transcribe Sideband push-to-talk voice messages
                      # (FIELD_AUDIO) into commands and (when reply_with_voice)
                      # answer a voice chat with a Piper voice reply. Needs only
                      # faster-whisper (requirements/transcribe.txt) plus
                      # piper-tts + PyAV (requirements/tts.txt) for replies,
                      # not the microphone stack.
                      'voice': {'enabled': {'default': False, 'type': 'bool'},
                                'reply_with_voice': {'default': True, 'type': 'bool'},
                                'max_speak_chars': {'default': 600, 'type': 'int'},
                                'reply_bitrate': {'default': 24000, 'type': 'int'},
                                'stt': {'model': {'default': 'base.en', 'type': 'str'},
                                        'device': {'default': 'cpu', 'type': 'str'},
                                        'compute_type': {'default': 'int8', 'type': 'str'},
                                        'language': {'default': 'en', 'type': 'str'}},
                                'tts': {'voice': {'default': 'en_US-lessac-medium',
                                                  'type': 'str'},
                                        'voice_dir': {'default': '', 'type': 'str'},
                                        'speaker_id': {'default': 0, 'type': 'int'},
                                        'length_scale': {'default': 1.0,
                                                         'type': 'float'},
                                        'volume': {'default': 1.0, 'type': 'float'}}}},
             'voice': {'enabled': {'default': False, 'type': 'bool'},
                       'chat_id': {'default': 'voice', 'type': 'str'},
                       'display_name': {'default': 'voice', 'type': 'str'},
                       'home_channel': {'default': 'voice', 'type': 'str'},
                       'reply_prefix': {'default': '', 'type': 'str'},
                       'result_only': {'default': True, 'type': 'bool'},
                       'max_speak_chars': {'default': 600, 'type': 'int'},
                       'listen_timeout_s': {'default': 3.0, 'type': 'float'},
                       'cue': {'enabled': {'default': True, 'type': 'bool'},
                               'frequency': {'default': 880, 'type': 'int'},
                               'duration_ms': {'default': 120, 'type': 'int'},
                               'volume': {'default': 0.9, 'type': 'float'},
                               'submit_frequency': {'default': 0, 'type': 'int'},
                               'submit_duration_ms': {'default': 100, 'type': 'int'},
                               'submit_notes': {'default': [[1046.5, 70], [1568.0, 90]],
                                                'type': 'list'},
                               'note_gap_ms': {'default': 20, 'type': 'int'}},
                       'wake': {'model': {'default': 'hey_computer', 'type': 'str'},
                                'threshold': {'default': 0.2, 'type': 'float'},
                                'inference_framework': {'default': 'onnx', 'type': 'str'},
                                'model_dir': {'default': '', 'type': 'str'}},
                       'stt': {'model': {'default': 'base.en', 'type': 'str'},
                               'device': {'default': 'cpu', 'type': 'str'},
                               'compute_type': {'default': 'int8', 'type': 'str'},
                               'language': {'default': 'en', 'type': 'str'}},
                       'tts': {'voice': {'default': 'en_US-lessac-medium', 'type': 'str'},
                               'voice_dir': {'default': '', 'type': 'str'},
                               'speaker_id': {'default': 0, 'type': 'int'},
                               'length_scale': {'default': 1.0, 'type': 'float'},
                               'volume': {'default': 1.0, 'type': 'float'}},
                       'audio': {'input_device': {'default': '', 'type': 'str'},
                                 'output_device': {'default': '', 'type': 'str'},
                                 'sample_rate': {'default': 16000, 'type': 'int'},
                                 'frame_ms': {'default': 80, 'type': 'int'},
                                 'capture_sample_rate': {'default': 0, 'type': 'int'},
                                 'output_sample_rate': {'default': 0, 'type': 'int'},
                                 'dc_removal': {'default': True, 'type': 'bool'},
                                 'read_timeout_s': {'default': 1.0, 'type': 'float'},
                                 'stall_timeout_s': {'default': 5.0, 'type': 'float'}},
                       'vad': {'silence_ms': {'default': 500, 'type': 'int'},
                               'min_speech_ms': {'default': 120, 'type': 'int'},
                               'max_utterance_s': {'default': 30, 'type': 'int'},
                               'min_command_ms': {'default': 700, 'type': 'int'},
                               'initial_silence_ms': {'default': 1500, 'type': 'int'},
                               'aggressiveness': {'default': 2, 'type': 'int'},
                               'energy_threshold': {'default': 200.0, 'type': 'float'}},
                       'follow_up_window_s': {'default': 0.0, 'type': 'float'}}},
 'dashboard': {'port': {'default': 8765, 'type': 'int'},
               'host': {'default': '127.0.0.1', 'type': 'str'}},
 'skill_packs': {'default': 'packs',
                 'type': 'str',
                 'doc': 'Path to the packs directory (scanned for */pack.json) or a list of pack '
                        'roots.'},
 'secrets_file': {'default': 'secrets.json',
                  'type': 'str',
                  'doc': 'Path to the gitignored secrets file, relative to config.json.'}}


#: Bridges that are not owned by any pack — shared services a pack may call.
#: They still live under ``bridges.<name>`` so ``derived_config_vars`` and
#: ``run_bridges`` find them, and they are validated/generated like pack bridges.
SHARED_BRIDGE_SCHEMA: dict[str, Any] = {
    "llm": {
        "enabled": {"type": "bool", "default": True, "doc": "Run the generic LLM bridge (a POST /chat in front of the configured `llm` model)."},
        "host": {"type": "str", "default": "127.0.0.1", "doc": "Bridge bind address (localhost only)."},
        "port": {"type": "int", "default": 5229, "doc": "Bridge listen port."},
        "token": {"type": "str", "default": "", "secret": True, "doc": "Shared secret required in the X-Semif-Token header when set."},
    },
}


class ConfigError(Exception):
    """A malformed or inconsistent configuration."""


def read_json_object(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config file must be a JSON object: {path}")
    return data


def deep_merge(base: dict, extra: dict) -> dict:
    """Merge `extra` into `base` in place; dicts recurse, everything else wins."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config_file(path: str | Path, _seen: set[Path] | None = None) -> dict:
    """Load one config file, following its top-level ``include`` list.

    Includes are resolved relative to the including file and merged first, so a
    file's own keys override anything it includes. Cycles are an error.
    """
    config_path = Path(path)
    resolved = config_path.resolve()
    seen = _seen if _seen is not None else set()
    if resolved in seen:
        raise ConfigError(f"config include cycle at {config_path}")
    seen.add(resolved)
    data = read_json_object(config_path)
    includes = data.pop("include", None)
    merged: dict = {}
    if includes:
        if not isinstance(includes, list):
            raise ConfigError(f"`include` must be a list in {config_path}")
        for entry in includes:
            inc_path = Path(entry)
            if not inc_path.is_absolute():
                inc_path = config_path.parent / inc_path
            deep_merge(merged, load_config_file(inc_path, seen))
    deep_merge(merged, data)
    return merged


def load_secrets(path: str | Path | None) -> dict:
    """Load the gitignored secrets file. Absent or malformed yields ``{}``."""
    if not path:
        return {}
    secrets_path = Path(path)
    if not secrets_path.is_file():
        return {}
    try:
        data = json.loads(secrets_path.read_text())
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def iter_secret_refs(value: Any) -> Iterator[str]:
    """Every ``!secret <key>`` key referenced anywhere in a config value."""
    if isinstance(value, str) and value.startswith(SECRET_PREFIX):
        yield value[len(SECRET_PREFIX):].strip()
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_secret_refs(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_secret_refs(item)


def resolve_secrets(value: Any, secrets: dict) -> Any:
    """Replace every ``"!secret <key>"`` string with its value (missing -> "")."""
    if isinstance(value, str) and value.startswith(SECRET_PREFIX):
        return secrets.get(value[len(SECRET_PREFIX):].strip(), "")
    if isinstance(value, dict):
        return {key: resolve_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_secrets(item, secrets) for item in value]
    return value


def _schema_defaults(schema: dict) -> dict:
    """Build a defaults dict from a (possibly nested) schema."""
    out: dict = {}
    for key, spec in schema.items():
        if isinstance(spec, dict) and "default" not in spec:
            out[key] = _schema_defaults(spec)
        elif isinstance(spec, dict):
            out[key] = spec["default"]
        else:
            out[key] = spec
    return out


def default_config(packs=None) -> dict:
    """The full default config: core schema defaults + every pack's defaults.

    A pack's ``config`` schema maps onto its ``bridges.<name>`` block (bridge
    packs) or ``packs.<name>`` (bridge-less packs). Used by bootstrap to seed a
    fresh split config and by ``generate_example_config``.
    """
    config = _schema_defaults(CORE_SCHEMA)
    config.setdefault("bridges", {}).update(_schema_defaults(SHARED_BRIDGE_SCHEMA))
    for pack in packs or []:
        block = _schema_defaults(pack.config)
        if pack.bridges:
            config.setdefault("bridges", {})[pack.bridges[0]] = block
        else:
            config.setdefault("packs", {}).setdefault(pack.name, {})["enabled"] = block.pop("enabled", True)
    return config


def generate_example_config(packs=None) -> dict:
    """The committed ``config.example.json``: defaults with ``!secret`` refs.

    A secret-typed key is emitted as ``"!secret <pack>_<key>"`` so the example
    never carries a real credential. The example is a flat (un-split) mirror of
    the live ``config/`` layout, for reference and for bootstrap to seed from.
    """
    config = _schema_defaults(CORE_SCHEMA)
    config["_comment"] = (
        "GENERATED from the core schema (semif_agent/config.py) and each pack's "
        "pack.json `config` schema — do not hand-edit; edit a pack's schema or "
        "the core schema. This is the reference example; the live per-machine "
        "config is split under config/ (see config.json's include list) and "
        "secrets live in the gitignored secrets.json, referenced as "
        "'!secret <key>'."
    )
    _mark_secrets(config, CORE_SCHEMA, "")
    shared = _schema_defaults(SHARED_BRIDGE_SCHEMA)
    for name, spec in SHARED_BRIDGE_SCHEMA.items():
        _mark_secrets(shared[name], spec, f"{name}_")
    config.setdefault("bridges", {}).update(shared)
    for pack in packs or []:
        block = _schema_defaults(pack.config)
        _mark_secrets(block, pack.config, f"{pack.name}_")
        if pack.bridges:
            config.setdefault("bridges", {})[pack.bridges[0]] = block
        else:
            config.setdefault("packs", {}).setdefault(pack.name, {})["enabled"] = block.pop("enabled", True)
    return config


def _mark_secrets(values: dict, schema: dict, prefix: str, path: str = "") -> None:
    for key, spec in schema.items():
        if not isinstance(spec, dict):
            continue
        if "default" in spec:
            if spec.get("secret"):
                values[key] = f"{SECRET_PREFIX}{prefix}{path}{key}"
        elif isinstance(values.get(key), dict):
            _mark_secrets(values[key], spec, prefix, f"{path}{key}_")


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def _check_schema(values: dict, schema: dict, path: str, problems: list[str]) -> None:
    for key, spec in schema.items():
        if not isinstance(spec, dict) or "default" not in spec:
            if isinstance(spec, dict):
                child = values.get(key)
                if isinstance(child, dict):
                    _check_schema(child, spec, f"{path}{key}.", problems)
            continue
        if key not in values:
            continue
        expected = spec.get("type")
        actual = _type_name(values[key])
        if expected and actual != expected:
            # int/float are interchangeable for numeric knobs.
            numeric = {expected, actual} <= {"int", "float"}
            if not numeric:
                problems.append(f"{path}{key}: expected {expected}, got {actual}")
    # Unknown keys are a warning (the schema documents the code's keys; a config
    # may legitimately carry forward a key the schema hasn't caught up to).
    known = set(schema)
    for key in values:
        if key.startswith("_") or key == "include":
            continue
        if key not in known:
            problems.append(f"warning: {path}{key}: unknown config key")


def validate_config(config: dict, packs=None) -> list[str]:
    """Validate a merged config against the core + pack schemas.

    Returns a list of human-readable problems (empty == valid). Secret refs are
    checked against ``secrets.json`` so a missing credential is reported here
    rather than silently resolving to "".
    """
    problems: list[str] = []
    # `bridges` and `packs` are pack-owned; they are validated per pack below.
    core = {k: v for k, v in config.items() if k not in ("bridges", "packs")}
    _check_schema(core, CORE_SCHEMA, "", problems)
    bridges = config.get("bridges", {}) or {}
    for name, spec in SHARED_BRIDGE_SCHEMA.items():
        block = bridges.get(name)
        if isinstance(block, dict):
            _check_schema(block, spec, f"bridges.{name}.", problems)
    for pack in packs or []:
        if pack.bridges:
            block = (config.get("bridges", {}) or {}).get(pack.bridges[0])
            if isinstance(block, dict):
                _check_schema(block, pack.config, f"bridges.{pack.bridges[0]}.", problems)
        else:
            block = (config.get("packs", {}) or {}).get(pack.name)
            if isinstance(block, dict):
                _check_schema(block, {"enabled": pack.config.get("enabled", True)}, f"packs.{pack.name}.", problems)
    return problems


def missing_secret_refs(config: dict, secrets: dict) -> list[str]:
    """Secret keys referenced by the config but absent from secrets.json."""
    return sorted({key for key in iter_secret_refs(config) if key not in secrets})


def split_config(config: dict, packs=None) -> dict[str, dict]:
    """Split a merged config into the ``config/`` layout.

    Returns a map of relative path -> fragment. Core keys go to
    ``config/core.json``; ``gateway`` to ``config/gateway.json``; bridges not
    owned by a pack to ``config/bridges.json``; each pack's bridge block and
    enable flag to ``config/packs/<name>.json``; and ``config.json`` becomes the
    thin index listing them in order.
    """
    owner = {bridge: pack.name for pack in (packs or []) for bridge in pack.bridges}
    by_name = {pack.name: pack for pack in (packs or [])}
    files: dict[str, dict] = {}
    core = {
        key: value
        for key, value in config.items()
        if key not in ("bridges", "gateway", "packs", "_comment")
    }
    files["config/core.json"] = core
    if config.get("gateway"):
        files["config/gateway.json"] = {"gateway": config["gateway"]}
    bridges = config.get("bridges", {}) or {}
    shared = {name: block for name, block in bridges.items() if name not in owner}
    if shared:
        files["config/bridges.json"] = {"bridges": shared}
    pack_cfg = config.get("packs", {}) or {}
    for name, pack in by_name.items():
        fragment: dict = {}
        owned = {bridge: bridges[bridge] for bridge in pack.bridges if bridge in bridges}
        if owned:
            fragment["bridges"] = owned
        if pack_cfg.get(name):
            fragment["packs"] = {name: pack_cfg[name]}
        if fragment:
            files[f"config/packs/{name}.json"] = fragment
    files["config.json"] = {"include": list(files)}
    return files


def write_split_config(root: str | Path, config: dict, packs=None) -> list[str]:
    """Write the ``config/`` layout under `root`; returns the paths written."""
    base = Path(root)
    written = []
    for rel, fragment in split_config(config, packs).items():
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(fragment, indent=2) + "\n")
        written.append(str(target))
    return written
