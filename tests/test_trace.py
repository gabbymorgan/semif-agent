from semif_agent.trace import TraceLog


def test_append_read_roundtrip(tmp_path):
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    trace.append("submit", "run-a", text="hello")
    trace.append("queued", "run-a", weight=0.5, label="medium")
    trace.append("submit", "run-b", text="world")

    rows = trace.read()
    assert len(rows) == 3
    assert rows[0]["kind"] == "submit"
    assert rows[0]["run_id"] == "run-a"
    assert rows[0]["text"] == "hello"
    assert rows[1]["label"] == "medium"


def test_runs_groups_by_run_id_preserving_order(tmp_path):
    trace = TraceLog(str(tmp_path / "runs.jsonl"))
    trace.append("submit", "run-a", text="x")
    trace.append("submit", "run-b", text="y")
    trace.append("assessed", "run-a", success=True)

    runs = trace.runs()
    assert list(runs) == ["run-a", "run-b"]
    assert len(runs["run-a"]) == 2
    assert runs["run-b"][0]["text"] == "y"


def test_read_missing_file_is_empty(tmp_path):
    trace = TraceLog(str(tmp_path / "none.jsonl"))
    assert trace.read() == []
    assert trace.runs() == {}