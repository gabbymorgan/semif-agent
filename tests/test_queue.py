from semif_agent.queue import RequestQueue
from semif_agent.decisions import Request


def make(text):
    return Request(text=text)


def test_fifo_order():
    q = RequestQueue()
    q.push(make("first"))
    q.push(make("second"))
    q.push(make("third"))
    assert q.pop().text == "first"
    assert q.pop().text == "second"
    assert q.pop().text == "third"
    assert q.pop() is None


def test_peek_does_not_remove():
    q = RequestQueue()
    q.push(make("peek"))
    assert q.peek().text == "peek"
    assert len(q) == 1


def test_bounds_reject():
    q = RequestQueue(max_size=2)
    assert q.push(make("a"))
    assert q.push(make("b"))
    assert not q.push(make("c"))


def test_items_in_arrival_order():
    q = RequestQueue()
    q.push(make("b"))
    q.push(make("a"))
    texts = [request.text for request in q.items()]
    assert texts == ["b", "a"]
