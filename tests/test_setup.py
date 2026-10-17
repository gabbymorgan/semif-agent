"""The first-run setup interview: answers -> split config, secrets, install plan.

These exercise the two ends of the spectrum the bootstrap must support:

* **minimum** — local SemIf 4B + the ``essentials`` pack, no gateway, no
  authoring, no ``llm``/``codegen`` endpoint, nothing heavy to install;
* **maximum** — every pack, all gateways, a remote Winnow engine, remote
  ``llm``/``codegen`` endpoints, and every conditional install enabled.

The interview is driven with a scripted ``input_fn`` (no TTY); config
generation is validated with the real pack schemas.
"""

from __future__ import annotations

import json
from pathlib import Path

from semif_agent.config import load_config_file, validate_config
from semif_agent.packs import load_packs
from semif_agent.setup import (
    Answers,
    build_config,
    build_plan,
    build_secrets,
    load_pack_index,
    plan_from_config,
    run_interview,
    run_setup,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _packs():
    return load_packs({"skill_packs": str(REPO_ROOT / "packs")})


def _scripted(answers):
    """An input_fn that yields `answers` in order and raises past the end."""
    it = iter(answers)

    def _input(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise AssertionError(f"unexpected prompt: {prompt!r}") from None

    return _input


def _silent(_: str) -> None:
    pass


# --------------------------------------------------------------------------- #
# Minimum path
# --------------------------------------------------------------------------- #


def test_minimal_config_is_essentials_only_with_no_endpoints():
    config = build_config(Answers.minimal(), _packs())
    assert config["engine"]["provider"] == "semif"
    assert config["new_skill_creation"] == "deny"
    # llm reuses the loaded engine; codegen is deliberately unconfigured.
    assert config["llm"]["provider"] == "semif"
    assert config["codegen"]["model"] == ""
    # essentials on, every other pack off.
    assert config["packs"]["essentials"]["enabled"] is True
    for bridge in ("simplex", "nextcloud", "outlook", "browser"):
        assert config["bridges"][bridge]["enabled"] is False
    # no gateway, and the shared llm bridge (only useful to a bridge pack) off.
    assert config["gateway"]["simplex"]["enabled"] is False
    assert config["gateway"]["lxmf"]["enabled"] is False
    assert config["gateway"]["voice"]["enabled"] is False
    assert config["bridges"]["llm"]["enabled"] is False
    # The generated config passes schema validation.
    assert validate_config(config, _packs()) == []


def test_minimum_plan_installs_nothing_heavy():
    plan = build_plan(Answers.minimal(), _packs())
    assert plan["install_engine"] is True
    assert plan["install_bridge_service"] is False
    assert plan["install_gateway_service"] is False
    assert plan["install_simplex_binary"] is False
    assert plan["install_simplex_forward"] is False
    assert plan["install_browser"] is False
    assert plan["install_voice"] is False
    assert plan["install_transcribe"] is False
    assert plan["packs"] == ["essentials"]
    assert plan["pack_requirements"] == []


# --------------------------------------------------------------------------- #
# Maximum path
# --------------------------------------------------------------------------- #


def test_maximum_config_enables_everything():
    answers = Answers(
        engine_provider="winnow",
        winnow_base_url="http://10.0.0.5:8091",
        winnow_model="winnow-1",
        winnow_api_key="wk",
        new_skill_creation="allow",
        llm_provider="ollama",
        llm_model="small",
        codegen_provider="opencode",
        codegen_model="coder",
        codegen_api_key="sk",
        packs=["simplex", "browser", "nextcloud", "outlook"],
        gateways=["simplex", "lxmf", "voice"],
        nextcloud_app_password="nc",
        nextcloud_url="https://cloud.example.com",
        outlook_client_id="client",
    )
    config = build_config(answers, _packs())
    assert config["engine"]["provider"] == "winnow"
    assert config["engine"]["winnow"]["model"] == "winnow-1"
    assert config["engine"]["winnow"]["api_key"] == "!secret engine_winnow_api_key"
    assert config["new_skill_creation"] == "allow"
    assert config["llm"]["model"] == "small"
    assert config["codegen"]["model"] == "coder"
    assert config["codegen"]["api_key"] == "!secret codegen_api_key"
    assert config["bridges"]["simplex"]["enabled"] is True
    assert config["bridges"]["browser"]["enabled"] is True
    assert config["bridges"]["llm"]["enabled"] is True
    assert config["gateway"]["voice"]["enabled"] is True
    assert config["gateway"]["humanize"]["enabled"] is True
    assert config["bridges"]["nextcloud"]["app_password"] == (
        "!secret nextcloud_app_password"
    )
    assert validate_config(config, _packs()) == []

    plan = build_plan(answers, _packs())
    assert plan["install_engine"] is False
    assert plan["install_simplex_binary"] is True
    assert plan["install_simplex_forward"] is True
    assert plan["install_browser"] is True
    assert plan["install_voice"] is True
    assert plan["install_bridge_service"] is True
    assert plan["install_gateway_service"] is True
    # The browser pack ships its own python deps; the plan names it for bootstrap.
    assert plan["pack_requirements"] == ["browser"]

    secrets = build_secrets(answers)
    assert secrets["engine_winnow_api_key"] == "wk"
    assert secrets["codegen_api_key"] == "sk"
    assert secrets["nextcloud_app_password"] == "nc"
    # llm has no key configured, so no llm_api_key entry.
    assert "llm_api_key" not in secrets


def test_simplex_gateway_without_pack_still_needs_the_binary():
    answers = Answers(gateways=["simplex"])
    plan = build_plan(answers, _packs())
    assert plan["install_simplex_binary"] is True
    assert plan["install_simplex_forward"] is False
    assert plan["install_gateway_service"] is True
    assert plan["install_bridge_service"] is False


def test_voice_only_gateway_is_foreground_not_a_service():
    answers = Answers(gateways=["voice"])
    plan = build_plan(answers, _packs())
    # Voice is foreground-only: no systemd gateway unit, no daemon platforms.
    assert plan["install_voice"] is True
    assert plan["install_gateway_service"] is False
    assert plan["gateway_platforms"] == []
    assert plan["install_simplex_binary"] is False


def test_gateway_platforms_exclude_voice():
    answers = Answers(gateways=["simplex", "voice"])
    plan = build_plan(answers, _packs())
    assert plan["gateway_platforms"] == ["simplex"]
    assert plan["install_gateway_service"] is True


def test_lxmf_voice_opts_into_transcription_dep():
    answers = Answers(gateways=["lxmf"], lxmf_voice=True)
    config = build_config(answers, _packs())
    assert config["gateway"]["lxmf"]["voice"]["enabled"] is True
    assert config["gateway"]["lxmf"]["voice"]["reply_with_voice"] is True
    assert validate_config(config, _packs()) == []

    plan = build_plan(answers, _packs())
    assert plan["install_transcribe"] is True
    # A voice reply also pulls the Piper TTS deps.
    assert plan["install_lxmf_tts"] is True
    # The rerun path re-derives the same plan from the written config.
    assert plan_from_config(config, _packs())["install_transcribe"] is True
    assert plan_from_config(config, _packs())["install_lxmf_tts"] is True


def test_lxmf_voice_transcribe_only_skips_tts_dep():
    answers = Answers(gateways=["lxmf"], lxmf_voice=True, lxmf_voice_reply=False)
    config = build_config(answers, _packs())
    assert config["gateway"]["lxmf"]["voice"]["enabled"] is True
    assert config["gateway"]["lxmf"]["voice"]["reply_with_voice"] is False
    plan = build_plan(answers, _packs())
    assert plan["install_transcribe"] is True
    assert plan["install_lxmf_tts"] is False
    assert plan_from_config(config, _packs())["install_lxmf_tts"] is False


def test_lxmf_without_voice_does_not_install_transcription():
    answers = Answers(gateways=["lxmf"])
    config = build_config(answers, _packs())
    assert config["gateway"]["lxmf"]["voice"]["enabled"] is False
    assert build_plan(answers, _packs())["install_transcribe"] is False
    assert build_plan(answers, _packs())["install_lxmf_tts"] is False


# --------------------------------------------------------------------------- #
# Interview branching
# --------------------------------------------------------------------------- #


def test_interview_collects_authoring_endpoints_and_packs():
    # engine=1(local); authoring=2(ask); llm provider=1(semif);
    # codegen provider=1(ollama); codegen base URL='' (default);
    # codegen model='my-coder'; codegen key=''; packs='a,c' (simplex+nextcloud);
    # nextcloud url/user/pass; gateways='b' (lxmf); timezone.
    script = _scripted(
        [
            "1",  # engine: local semif
            "2",  # authoring: ask
            "1",  # llm provider: semif (in-process)
            "1",  # codegen provider: ollama
            "",  # codegen base URL: default
            "my-coder",  # codegen model
            "",  # codegen api key
            "a,c",  # packs: simplex + nextcloud
            "https://nc.example",  # nextcloud url
            "bob",  # nextcloud username
            "pw",  # nextcloud app password
            "b",  # gateways: lxmf
            "3",  # lxmf voice: transcribe + reply with voice
            "Europe/Paris",  # timezone
        ]
    )
    # The offered packs come from the pack index (letters a,b,c,d in order).
    answers = run_interview(
        input_fn=script,
        output=_silent,
        pack_choices=[
            ("simplex", "SimpleX"),
            ("browser", "Browser"),
            ("nextcloud", "Nextcloud"),
            ("outlook", "Outlook"),
        ],
    )
    assert answers.engine_provider == "semif"
    assert answers.new_skill_creation == "ask"
    assert answers.llm_provider == "semif"
    assert answers.codegen_provider == "ollama"
    assert answers.codegen_model == "my-coder"
    assert answers.packs == ["essentials", "simplex", "nextcloud"]
    assert answers.nextcloud_url == "https://nc.example"
    assert answers.gateways == ["lxmf"]
    assert answers.lxmf_voice is True
    assert answers.lxmf_voice_reply is True
    assert answers.timezone == "Europe/Paris"


def test_interview_non_interactive_returns_minimum():
    answers = run_interview(interactive=False, output=_silent)
    assert answers.new_skill_creation == "deny"
    assert answers.packs == ["essentials"]
    assert answers.gateways == []


def test_interview_without_index_offers_essentials_only():
    # No pack index -> the pack question is skipped (essentials only).
    script = _scripted(
        [
            "1",  # engine: local semif
            "1",  # authoring: deny
            "",   # gateways: none
            "",   # timezone: host
        ]
    )
    answers = run_interview(input_fn=script, output=_silent, pack_choices=[])
    assert answers.packs == ["essentials"]


# --------------------------------------------------------------------------- #
# Pack index (the source of truth for which packs exist)
# --------------------------------------------------------------------------- #


def test_load_pack_index_parses_entries(tmp_path):
    index = tmp_path / "packs.json"
    index.write_text(
        json.dumps(
            {
                "packs": [
                    {"name": "essentials", "url": "ssh://h/pack.essentials.git"},
                    {
                        "name": "simplex",
                        "url": "ssh://h/pack.simplex.git",
                        "title": "SimpleX",
                        "description": "messaging",
                    },
                    {"name": "", "url": "ssh://h/ignored.git"},
                ]
            }
        )
    )
    choices = load_pack_index(str(index))
    assert choices == [
        ("essentials", "essentials"),
        ("simplex", "SimpleX — messaging"),
    ]
    # Missing/unreadable index -> no choices (essentials-only interview).
    assert load_pack_index(None) == []
    assert load_pack_index(str(tmp_path / "nope.json")) == []


# --------------------------------------------------------------------------- #
# run_setup: files, secrets, plan, idempotency
# --------------------------------------------------------------------------- #


def test_run_setup_writes_split_config_and_secrets(tmp_path):
    plan_out = tmp_path / "plan.json"
    code = run_setup(
        root=tmp_path,
        path=tmp_path / "config.json",
        packs=_packs(),
        yes=True,
        plan_out=str(plan_out),
        output=_silent,
    )
    assert code == 0
    assert (tmp_path / "config" / "core.json").is_file()
    assert (tmp_path / "config" / "packs" / "essentials.json").is_file()
    assert (tmp_path / "secrets.json").is_file()
    plan = json.loads(plan_out.read_text())
    assert plan["packs"] == ["essentials"]
    # The written index reloads to the answered values.
    reloaded = load_config_file(tmp_path / "config.json")
    assert reloaded["engine"]["provider"] == "semif"
    assert reloaded["codegen"]["model"] == ""
    assert reloaded["packs"]["essentials"]["enabled"] is True


def test_run_setup_answers_out_writes_answers_and_no_config(tmp_path):
    # The bootstrap pre-clone pass: answers only, no packs, no config.
    answers_out = tmp_path / "bootstrap-answers.json"
    code = run_setup(
        root=tmp_path,
        path=tmp_path / "config.json",
        packs=[],
        yes=True,
        answers_out=str(answers_out),
        output=_silent,
    )
    assert code == 0
    answers = json.loads(answers_out.read_text())
    assert answers["packs"] == ["essentials"]
    assert answers["engine_provider"] == "semif"
    # Nothing else was written.
    assert not (tmp_path / "config.json").exists()
    assert not (tmp_path / "config").exists()
    assert not (tmp_path / "secrets.json").exists()


def test_run_setup_answers_file_and_secrets_merge(tmp_path):
    answers = {
        "new_skill_creation": "allow",
        "llm_provider": "ollama",
        "llm_model": "small",
        "codegen_provider": "ollama",
        "codegen_model": "coder",
        "codegen_api_key": "sk-codegen",
    }
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(json.dumps(answers))
    # A pre-existing secret must survive a merge.
    (tmp_path / "secrets.json").write_text(json.dumps({"codegen_api_key": "old"}))

    code = run_setup(
        root=tmp_path,
        path=tmp_path / "config.json",
        packs=_packs(),
        answers_file=str(answers_path),
        output=_silent,
    )
    assert code == 0
    secrets = json.loads((tmp_path / "secrets.json").read_text())
    assert secrets["codegen_api_key"] == "sk-codegen"
    reloaded = load_config_file(tmp_path / "config.json")
    assert reloaded["new_skill_creation"] == "allow"
    assert reloaded["codegen"]["model"] == "coder"


def test_run_setup_refuses_to_overwrite_without_force(tmp_path):
    run_setup(
        root=tmp_path, path=tmp_path / "config.json", packs=_packs(),
        yes=True, output=_silent,
    )
    code = run_setup(
        root=tmp_path, path=tmp_path / "config.json", packs=_packs(),
        yes=True, output=_silent,
    )
    assert code == 1
    # --force re-seeds.
    code = run_setup(
        root=tmp_path, path=tmp_path / "config.json", packs=_packs(),
        yes=True, force=True, output=_silent,
    )
    assert code == 0


def test_plan_only_matches_build_plan(tmp_path):
    answers = Answers(packs=["simplex"], gateways=["simplex"])
    run_setup(
        root=tmp_path, path=tmp_path / "config.json", packs=_packs(),
        answers_file=_answers_file(tmp_path, answers), output=_silent,
    )
    reloaded = load_config_file(tmp_path / "config.json")
    derived = plan_from_config(reloaded, _packs())
    assert derived == build_plan(answers, _packs())


def _answers_file(tmp_path: Path, answers: Answers) -> str:
    path = tmp_path / "answers-in.json"
    path.write_text(json.dumps(answers.to_dict()))
    return str(path)
