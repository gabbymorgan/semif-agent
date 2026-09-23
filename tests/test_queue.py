import pytest

from semif_agent.queue import UrgencyQueue
from semif_agent.decisions import Request


def make(text, weight):
    return Request(text=text), weight


def test_order_by_weight_desc():
    q = UrgencyQueue()
    req_a, _ = make("low", 0.25)
    req_c, _ = make("critical", 1.0)
    req_h, _ = make("high", 0.75)
    q.push(req_a, 0.25)
    q.push(req_c, 1.0)
    q.push(req_h, 0.75)
    assert q.pop().text == "critical"
    assert q.pop().text == "high"
    assert q.pop().text == "low"
    assert q.pop() is None


def test_fifo_tiebreak():
    q = UrgencyQueue()
    req_a, _ = make("first", 0.5)
    req_b, _ = make("second", 0.5)
    q.push(req_a, 0.5)
    q.push(req_b, 0.5)
    assert q.pop().text == "first"
    assert q.pop().text == "second"


def test_fifo_beats_recency():
    q = UrgencyQueue()
    old, _ = make("older", 0.5)
    new, _ = make("newer", 0.5)
    q.push(old, 0.5, recency=1.0)
    q.push(new, 0.5, recency=2.0)
    assert q.pop().text == "older"


def test_peek_does_not_remove():
    q = UrgencyQueue()
    req, _ = make("peek", 0.9)
    q.push(req, 0.9)
    assert q.peek().text == "peek"
    assert len(q) == 1


def test_bounds_reject():
    q = UrgencyQueue(max_size=2)
    assert q.push(make("a", 1.0)[0], 1.0)
    assert q.push(make("b", 1.0)[0], 1.0)
    assert not q.push(make("c", 1.0)[0], 1.0)


def test_age_raises_priority():
    q = UrgencyQueue(age_rate=1.0)
    old, _ = make("aging", 0.2)
    new, _ = make("busy", 1.0)
    q.push(old, 0.2)
    q.push(new, 1.0)
    q.age(dt=10.0)
    assert q.peek().text == "aging"


def test_items_sorted():
    q = UrgencyQueue()
    q.push(make("b", 0.5)[0], 0.5)
    q.push(make("a", 1.0)[0], 1.0)
    weights = [weight for weight, _ in q.items()]
    assert weights == [1.0, 0.5]