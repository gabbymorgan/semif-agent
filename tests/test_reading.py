"""Pure-stdlib tests for the reading helpers and the scheduler speech channel.

`semif_agent.reading` owns bookmark persistence and deterministic speakable
chunking; `semif_agent.speech` owns the queue a skill posts a read-aloud request
to. Neither touches audio, the scheduler, or a gateway.
"""

import json

from semif_agent.reading import (
    BOOKMARKS_FILE,
    bookmarks_file,
    chunk_text,
    load_bookmarks,
    read_offset,
    save_bookmark,
)
from semif_agent.speech import ReadAloud, SpeechService


# ---- chunking --------------------------------------------------------------


def test_chunk_text_splits_and_offsets_are_contiguous():
    text = "One. Two. Three. Four. Five. Six."
    chunks = list(chunk_text(text, 0, 12))
    assert len(chunks) > 1
    for offset, chunk in chunks:
        assert text[offset : offset + len(chunk)] == chunk
    # The concatenation (whitespace-insensitive) is the original text.
    assert "".join(chunk for _o, chunk in chunks).replace(" ", "") == text.replace(" ", "")


def test_chunk_text_resumes_deterministically_from_a_saved_offset():
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."
    chunks = list(chunk_text(text, 0, 20))
    assert len(chunks) >= 3
    resume_offset = chunks[2][0]
    # Resuming from a chunk boundary reproduces the tail exactly.
    tail = list(chunk_text(text, resume_offset, 20))
    assert tail == chunks[2:]


def test_chunk_text_respects_max_chars_without_a_good_break():
    text = "x" * 100
    chunks = list(chunk_text(text, 0, 30))
    assert [len(chunk) for _o, chunk in chunks] == [30, 30, 30, 10]


def test_chunk_text_prefers_a_paragraph_break():
    text = "A" * 40 + "\n\n" + "B" * 40
    chunks = list(chunk_text(text, 0, 60))
    # The first chunk ends at the paragraph break, not mid-word.
    assert chunks[0][1] == "A" * 40


def test_chunk_text_empty_and_start_past_end():
    assert list(chunk_text("", 0, 10)) == []
    assert list(chunk_text("short", 999, 10)) == []


def test_chunk_text_skips_leading_whitespace_at_resume():
    text = "  hello world"
    chunks = list(chunk_text(text, 0, 50))
    assert chunks == [(2, "hello world")]


# ---- bookmarks -------------------------------------------------------------


def test_bookmarks_file_lives_in_the_books_directory(tmp_path):
    assert bookmarks_file(tmp_path) == tmp_path / BOOKMARKS_FILE


def test_save_and_read_bookmark_roundtrip(tmp_path):
    path = bookmarks_file(tmp_path)
    assert read_offset(path, "gutenberg-1342") == 0
    save_bookmark(path, "gutenberg-1342", 1234, "Pride and Prejudice")
    assert read_offset(path, "gutenberg-1342") == 1234
    entry = load_bookmarks(path)["gutenberg-1342"]
    assert entry["offset"] == 1234
    assert entry["title"] == "Pride and Prejudice"
    assert "updated_at" in entry
    # A second save keeps the other keys.
    save_bookmark(path, "gutenberg-84", 7, "Frankenstein")
    assert read_offset(path, "gutenberg-1342") == 1234
    assert read_offset(path, "gutenberg-84") == 7


def test_save_bookmark_leaves_no_temp_file(tmp_path):
    path = bookmarks_file(tmp_path)
    save_bookmark(path, "k", 5)
    assert not (tmp_path / (BOOKMARKS_FILE + ".tmp")).exists()


def test_bookmark_tolerates_missing_and_malformed(tmp_path):
    path = bookmarks_file(tmp_path)
    assert load_bookmarks(path) == {}
    path.write_text("{not json")
    assert load_bookmarks(path) == {}
    assert read_offset(path, "k") == 0
    # A legacy bare-integer entry is still read.
    path.write_text(json.dumps({"k": 42}))
    assert read_offset(path, "k") == 42
    # A non-dict top level is ignored.
    path.write_text(json.dumps([1, 2, 3]))
    assert load_bookmarks(path) == {}


# ---- speech service --------------------------------------------------------


def test_speech_service_queues_and_drains():
    service = SpeechService()
    service.read_aloud(
        path="/books/x.txt",
        key="gutenberg-1",
        bookmarks_path="/books/bookmarks.json",
        run_id="r1",
        source="voice:mic",
        title="X",
        start=10,
    )
    assert len(service.pending()) == 1
    reads = service.drain()
    assert len(reads) == 1
    read = reads[0]
    assert isinstance(read, ReadAloud)
    assert read.key == "gutenberg-1"
    assert read.start == 10
    assert read.run_id == "r1"
    # Drained once; the queue is empty afterwards.
    assert service.drain() == []
    assert service.pending() == []


def test_speech_service_is_bounded():
    service = SpeechService(max_pending=2)
    for index in range(5):
        service.read_aloud(
            path=f"/b/{index}.txt", key=str(index), bookmarks_path="/b/bookmarks.json"
        )
    assert len(service.pending()) == 2  # oldest dropped
    assert [r.key for r in service.drain()] == ["3", "4"]
