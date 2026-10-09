"""Evaluate a simple arithmetic request and say the answer out loud.

Real local compute: the `compute` transport is the codebase's home for a body
that acts on the machine itself, and arithmetic is exactly that — no external
service, no configuration. The body is deliberately bounded to **two operands
and one operator**.

The decision engine (SemIf) resolves the *operator* from the request text with a
single choice over the supported operations: addition, subtraction,
multiplication, division, exponents, squares, cubes, square roots, cube roots,
fractions, and percentages. The operands are the numbers written in the request;
when the request offers more numbers than the operation consumes, SemIf chooses
which numbers are the operands (one choice per operand slot). When too few
numbers are present the body pauses and asks for them, stashing the answer in
`request.meta` so the single-phase `act` re-run does not ask twice.

The result is phrased for the ear: every number is spelled out in words and an
inexact result is announced as "approximately ...". Square, cube, square-root
and cube-root requests are unary in plain language but are still "one operator,
two operands" underneath (the base and an implied 2, 3, 1/2 or 1/3).
"""

from __future__ import annotations

import math
import re
from collections import namedtuple

from semif_agent.decisions import DecisionRequest, Option
from semif_agent.skills import ActionResult

INTEGRATION = {"service": "local_calculator", "transport": "compute", "config_vars": []}

CONTRACT = {}

# operator id -> (option description for SemIf, number of operands it consumes)
OPERATIONS: dict[str, tuple[str, int]] = {
    "add": (
        "Add two numbers together, stated as 'plus', 'added to', 'the sum of', or '+'.",
        2,
    ),
    "subtract": (
        "Subtract one number from another, stated as 'minus', 'less', 'take away', or '-'.",
        2,
    ),
    "multiply": (
        "Multiply two numbers, stated as 'times', 'multiplied by', or 'the product of'.",
        2,
    ),
    "divide": (
        "Divide one number by another, stated as 'divided by' or with a division sign.",
        2,
    ),
    "exponent": (
        "Raise a base to a power, stated as 'to the power of' or with a caret '^'.",
        2,
    ),
    "square": ("Square a single number, stated as 'squared'.", 1),
    "cube": ("Cube a single number, stated as 'cubed'.", 1),
    "square_root": ("The square root of a single number, stated as 'square root of'.", 1),
    "cube_root": ("The cube root of a single number, stated as 'cube root of'.", 1),
    "fraction": (
        "The value of a plain fraction written with a slash, such as 3/4, when the "
        "request does not say 'divided by'.",
        2,
    ),
    "percent": (
        "Take a percentage of a number, stated as 'X% of Y' or 'X percent of Y'.",
        2,
    ),
}

_ASKS = {
    "add": "What two numbers should I add? (e.g. '3 and 4')",
    "subtract": "What two numbers should I subtract? (e.g. '9 minus 4')",
    "multiply": "What two numbers should I multiply? (e.g. '6 and 7')",
    "divide": "What should I divide, and by what? (e.g. '10 divided by 4')",
    "exponent": "What base and what power? (e.g. '2 to the power of 10')",
    "square": "What number should I square?",
    "cube": "What number should I cube?",
    "square_root": "What number should I take the square root of?",
    "cube_root": "What number should I take the cube root of?",
    "fraction": "What fraction should I evaluate? (e.g. '3/4')",
    "percent": "What percentage of what number? (e.g. '20% of 50')",
}

# ---- spoken number words ----

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
    "trillion": 1000000000000,
}
_UNITS_WORD = {value: word for word, value in _UNITS.items()}
_TENS_WORD = {value: word for word, value in _TENS.items()}
_SCALE_WORDS = [
    (1000000000000, "trillion"),
    (1000000000, "billion"),
    (1000000, "million"),
    (1000, "thousand"),
]
_NUMBER_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES)
_WORD_TOKEN = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(word)
        for word in sorted(_NUMBER_WORDS | {"and", "negative"}, key=len, reverse=True)
    )
    + r")\b",
    re.I,
)
_DIGIT = re.compile(r"(?P<sign>negative\s+)?(?P<num>\d+(?:\.\d+)?|\.\d+)", re.I)


def _int_to_words(number: int) -> str:
    if number < 0:
        return "negative " + _int_to_words(-number)
    if number < 20:
        return _UNITS_WORD[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        word = _TENS_WORD[tens * 10]
        return f"{word}-{_UNITS_WORD[ones]}" if ones else word
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        word = f"{_UNITS_WORD[hundreds]} hundred"
        return f"{word} {_int_to_words(rest)}" if rest else word
    for scale, name in _SCALE_WORDS:
        if number >= scale:
            leading, rest = divmod(number, scale)
            word = f"{_int_to_words(leading)} {name}"
            return f"{word} {_int_to_words(rest)}" if rest else word
    return str(number)


def _speak_number(value: float) -> str:
    """A number as English words, so a spoken reply is unambiguous."""
    value = float(value)
    if not math.isfinite(value):
        return "an undefined value"
    if value < 0:
        return "negative " + _speak_number(-value)
    if abs(value - round(value)) < 1e-9:
        return _int_to_words(int(round(value)))
    if value >= 1e15:
        return f"{value:.6g}"
    whole = int(value)
    fraction_digits = f"{value:.10f}".split(".", 1)[1].rstrip("0")
    words = _int_to_words(whole) if whole else "zero"
    if fraction_digits:
        return f"{words} point " + " ".join(_UNITS_WORD[int(d)] for d in fraction_digits)
    return words


# ---- request parsing ----

_Candidate = namedtuple("_Candidate", "value phrase start end")


def _normalize_leading_sign(text: str) -> str:
    """Turn a leading '-5' into 'negative 5' so a sign at the start of the
    request is not confused with a spaced-out subtraction."""
    return re.sub(r"^\s*[-\u2212]\s*(?=\d)", "negative ", text, count=1)


def _parse_word_number(words: list[str]) -> float | None:
    total = 0
    current = 0
    negative = False
    seen = False
    for word in words:
        if word == "negative":
            negative = True
        elif word == "and":
            continue
        elif word in _UNITS:
            current += _UNITS[word]
            seen = True
        elif word in _TENS:
            current += _TENS[word]
            seen = True
        elif word in _SCALES:
            scale = _SCALES[word]
            if scale == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * scale
                current = 0
            seen = True
        else:
            return None
    if not seen:
        return None
    value = total + current
    return float(-value if negative else value)


def _digit_candidates(text: str) -> list[_Candidate]:
    candidates = []
    for match in _DIGIT.finditer(text):
        value = float(match.group("num"))
        if match.group("sign"):
            value = -value
        candidates.append(
            _Candidate(value, match.group(0).strip(), match.start(), match.end())
        )
    return candidates


def _word_candidates(text: str) -> list[_Candidate]:
    matches = list(_WORD_TOKEN.finditer(text))
    groups: list[list] = []
    current: list = []
    for match in matches:
        if current:
            gap = text[current[-1].end() : match.start()]
            if not re.fullmatch(r"[\s\-]*", gap):
                groups.append(current)
                current = []
        current.append(match)
    if current:
        groups.append(current)

    candidates = []
    for group in groups:
        value = _parse_word_number([match.group(0).lower() for match in group])
        if value is None:
            continue
        start, end = group[0].start(), group[-1].end()
        candidates.append(_Candidate(value, text[start:end], start, end))
    return candidates


def _find_candidates(text: str) -> list[_Candidate]:
    """Every number written in the request, in the order it appears."""
    text = _normalize_leading_sign(text or "")
    candidates = _digit_candidates(text) + _word_candidates(text)
    candidates.sort(key=lambda candidate: (candidate.start, candidate.end))
    ordered: list[_Candidate] = []
    last_end = -1
    for candidate in candidates:
        if candidate.start >= last_end:
            ordered.append(candidate)
            last_end = candidate.end
    return ordered


# ---- SemIf decisions ----

def _decide_operator(ctx, text: str, decisions: list) -> str:
    decision = DecisionRequest(
        state=text,
        question="Which arithmetic operation does the user's request ask for?",
        options=[
            Option(name, description) for name, (description, _count) in OPERATIONS.items()
        ],
    )
    result = ctx.engine.call(decision)
    decisions.append((decision, result))
    return result.selected


def _select_operands(ctx, text, candidates, needed, decisions):
    """The operand values the operation consumes.

    When the request writes exactly the numbers the operation needs they are
    read in order; when it writes more, SemIf chooses which numbers are the
    operands, one choice per operand slot.
    """
    if len(candidates) == needed:
        return [candidate.value for candidate in candidates]

    pool = list(candidates)
    chosen = []
    for slot in range(needed):
        if slot == 0:
            question = (
                "Which of these numbers is the first number to use in the calculation?"
            )
        else:
            question = (
                "Which of these numbers is the second number to use in the calculation?"
            )
        decision = DecisionRequest(
            state=text,
            question=question,
            options=[
                Option(
                    str(index),
                    f"{candidate.phrase} (equal to {_speak_number(candidate.value)})",
                )
                for index, candidate in enumerate(pool)
            ],
        )
        result = ctx.engine.call(decision)
        decisions.append((decision, result))
        try:
            selected = int(result.selected)
        except (TypeError, ValueError):
            selected = 0
        if not 0 <= selected < len(pool):
            selected = 0
        chosen.append(pool.pop(selected).value)
    return chosen


# ---- arithmetic ----

class _CalcError(Exception):
    """The request cannot be computed (bad operand, undefined result)."""


def _compute(operator: str, operands: list[float]) -> float:
    first = operands[0]
    second = operands[1] if len(operands) > 1 else None
    if operator == "square":
        return first * first
    if operator == "cube":
        return first * first * first
    if operator == "square_root":
        if first < 0:
            raise _CalcError("I can't take the square root of a negative number.")
        return math.sqrt(first)
    if operator == "cube_root":
        return math.copysign(abs(first) ** (1.0 / 3.0), first)
    if operator in ("divide", "fraction"):
        if second == 0:
            raise _CalcError("I can't divide by zero.")
        return first / second
    if operator == "percent":
        return (first / 100.0) * second
    if operator == "exponent":
        try:
            result = first ** second
        except (OverflowError, ValueError):
            raise _CalcError("That power is too large or is not a real number.")
        if isinstance(result, complex):
            raise _CalcError("That power is not a real number.")
        return result
    if operator == "add":
        return first + second
    if operator == "subtract":
        return first - second
    if operator == "multiply":
        return first * second
    raise _CalcError("I don't know how to compute that.")


def _round_for_speech(value: float) -> tuple[float, bool]:
    """The value to speak and whether it had to be rounded."""
    value = float(value)
    if not math.isfinite(value):
        return value, True
    if abs(value - round(value)) < 1e-9:
        return float(round(value)), False
    rounded = round(value, 4)
    return rounded, abs(rounded - value) > 1e-9


def _report(operator: str, operands: list[float], value: float, approximate: bool) -> str:
    first = _speak_number(operands[0])
    second = _speak_number(operands[1]) if len(operands) > 1 else None
    result = _speak_number(value)
    prefix = "approximately " if approximate else ""
    sentences = {
        "add": f"{first} plus {second} is {prefix}{result}.",
        "subtract": f"{first} minus {second} is {prefix}{result}.",
        "multiply": f"{first} times {second} is {prefix}{result}.",
        "divide": f"{first} divided by {second} is {prefix}{result}.",
        "exponent": f"{first} to the power of {second} is {prefix}{result}.",
        "square": f"{first} squared is {prefix}{result}.",
        "cube": f"{first} cubed is {prefix}{result}.",
        "square_root": f"The square root of {first} is {prefix}{result}.",
        "cube_root": f"The cube root of {first} is {prefix}{result}.",
        "fraction": f"The fraction {first} over {second} is {prefix}{result}.",
        "percent": f"{first} percent of {second} is {prefix}{result}.",
    }
    message = sentences.get(operator, f"The answer is {prefix}{result}.")
    return message[0].upper() + message[1:]


# ---- the skill phase ----

def act(ctx, request):
    answers = request.meta.setdefault("calculator_answers", {})
    awaiting = request.meta.get("calculator_awaiting")
    if request.user_input is not None and awaiting:
        answers[awaiting] = request.user_input.strip()
        request.meta.pop("calculator_awaiting", None)

    text = request.text
    decisions: list = []

    operator = request.meta.get("calculator_operator")
    if operator not in OPERATIONS:
        operator = _decide_operator(ctx, text, decisions)
        request.meta["calculator_operator"] = operator
    needed = OPERATIONS[operator][1]

    candidates = _find_candidates(text)
    if answers.get("operands"):
        candidates = candidates + _find_candidates(answers["operands"])

    if len(candidates) < needed:
        request.meta["calculator_awaiting"] = "operands"
        return ActionResult(
            action_log="waiting for the numbers to calculate",
            new_state=request.text,
            needs_input=_ASKS[operator],
            decisions=decisions,
        )

    operands = _select_operands(ctx, text, candidates, needed, decisions)
    try:
        value = _compute(operator, operands)
    except _CalcError as exc:
        message = f"{exc}"
        return ActionResult(action_log=message, new_state=message, decisions=decisions)

    if not math.isfinite(value):
        message = "That calculation does not have a finite answer."
        return ActionResult(action_log=message, new_state=message, decisions=decisions)

    rounded, approximate = _round_for_speech(value)
    message = _report(operator, operands, rounded, approximate)
    return ActionResult(action_log=message, new_state=message, decisions=decisions)
