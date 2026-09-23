from semif_agent.decisions import DecisionRequest, DecisionResult, Option, Request


def test_to_semif_row_shape():
    request = DecisionRequest(
        state="some state",
        question="some question?",
        options=[Option("a", "A."), Option("b", "B.")],
        id="abc",
    )
    row = request.to_semif_row()
    assert row["id"] == "abc"
    assert row["state"] == "some state"
    assert row["question"] == "some question?"
    assert row["options"] == [
        {"id": "a", "description": "A."},
        {"id": "b", "description": "B."},
    ]


def test_result_selected_and_probs():
    request = DecisionRequest(
        state="s",
        question="q",
        options=[Option("a", "A."), Option("b", "B.")],
    )
    result = DecisionResult(
        request=request, option_ids=["a", "b"], probabilities=[0.3, 0.7]
    )
    assert result.selected == "b"
    assert result.probs == {"a": 0.3, "b": 0.7}
    assert result.prob("a") == 0.3


def test_request_requeue_preserves_state():
    original = Request(text="t", priority=0.7)
    original.resume["from_skill"] = "email.compose"
    updated = original.copy_for_requeue()
    assert updated.id == original.id
    assert updated.priority == original.priority
    assert updated.resume["from_skill"] == "email.compose"
    assert updated.reentries == original.reentries + 1