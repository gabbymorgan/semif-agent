import math

import pytest

from semif_agent.decisions import DecisionRequest, DecisionResult, Option
from semif_agent.dream import HUMAN_WEIGHT, SELF_WEIGHT, dream
from semif_agent.log import DecisionLog


def make_result(probs):
    request = DecisionRequest(
        state="state",
        question="question",
        options=[Option(key, key) for key in probs],
    )
    result = DecisionResult(
        request=request,
        option_ids=list(probs),
        probabilities=list(probs.values()),
    )
    return request, result


def test_cross_entropy_matches_hand_calculation(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    probs = {"yes": 0.8, "no": 0.2}
    request, result = make_result(probs)
    log.append(request, result, label="no")
    report = dream(log)
    assert report.cross_entropy == pytest.approx(-math.log(0.2))
    assert report.rows[0].nll == pytest.approx(-math.log(0.2))
    assert report.rows[0].weight == HUMAN_WEIGHT


def test_default_label_is_selected(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    request, result = make_result({"yes": 0.9, "no": 0.1})
    log.append(request, result)
    report = dream(log)
    row = report.rows[0]
    assert row.observed == "yes"
    assert row.correct is True
    assert row.weight == SELF_WEIGHT


def test_accuracy_and_ece(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    for probs, label in [
        ({"yes": 0.9, "no": 0.1}, "yes"),
        ({"yes": 0.6, "no": 0.4}, "no"),
        ({"yes": 0.9, "no": 0.1}, "yes"),
    ]:
        request, result = make_result(probs)
        log.append(request, result, label=label)
    report = dream(log)
    assert report.accuracy == pytest.approx(2 / 3)
    assert report.ece is not None and 0.0 <= report.ece <= 1.0


def test_relabel_human_override(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    request, result = make_result({"yes": 0.9, "no": 0.1})
    log.append(request, result)
    assert log.relabel(request.id, "no") is True
    rows = log.read()
    assert rows[0]["observed_outcome"] == "no"
    assert rows[0]["label_source"] == "human"
    assert log.relabel("missing", "yes") is False


def test_skips_unlabeled(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text('{"id": "x", "predicted_probs": {"a": 1.0}}\n')
    report = dream(log)
    assert report.skipped == 1
    assert report.rows == []
    assert report.cross_entropy is None


def test_clamped_probability_never_zero(tmp_path):
    log = DecisionLog(str(tmp_path / "decisions.jsonl"))
    log.path.parent.mkdir(parents=True, exist_ok=True)
    log.path.write_text(
        '{"id": "x", "predicted_probs": {"a": 0.0, "b": 1.0}, '
        '"selected": "b", "observed_outcome": "a", "label_source": "human"}\n'
    )
    report = dream(log)
    assert report.rows[0].nll == pytest.approx(-math.log(1e-9))