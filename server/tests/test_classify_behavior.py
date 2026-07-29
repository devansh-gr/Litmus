"""Behavior tests: does the detector actually detect the right things?

These are the semantic guarantees the product depends on. Unlike the accuracy harness
(which measures aggregate quality), these pin specific must-hold behaviors.
"""
import pytest

NEUTRAL = [
    "The meeting is scheduled for 3pm on Tuesday in room 214.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The train departs from platform 4 every fifteen minutes.",
]

CLEAR = [
    ("Act now, this offer expires in ten minutes!", "false-urgency"),
    ("Top scientists agree this is the only correct approach.", "authority-appeal"),
    # NOTE: a clean fear example with NO urgency cue ("act now"). Sentences that mix
    # fear + a deadline land on false-urgency — a real, documented overlap (RESULTS.md).
    ("A terrifying new disease is spreading uncontrollably toward your town.", "fear-mongering"),
]


@pytest.mark.parametrize("text", NEUTRAL)
def test_neutral_text_is_none(classify, text):
    assert classify(text)["vector"] == "none"


@pytest.mark.parametrize("text,expected", CLEAR)
def test_clear_cases_hit_and_are_confident(classify, text, expected):
    r = classify(text)
    assert r["vector"] == expected, f"{text!r} -> {r['vector']} (want {expected})"
    assert r["confidence"] >= 60, f"unexpectedly unsure ({r['confidence']}%) on a clear case"


def test_determinism(classify):
    # Same input must give the same verdict (scoring is deterministic; also cached).
    text = "You must decide right this second or lose it forever."
    a, b = classify(text), classify(text)
    assert a["vector"] == b["vector"]
    assert a["confidence"] == b["confidence"]


def test_short_input_does_not_crash(classify):
    r = classify("hi")
    assert "vector" in r


def test_confidence_higher_on_blatant_than_subtle(classify):
    blatant = classify("ACT NOW! Offer expires in 5 minutes, last chance ever!")
    subtle = classify("It might be worth deciding sometime soon, if convenient.")
    # A blatant manipulation should not be LESS confident than a mild/neutral line.
    assert blatant["confidence"] >= subtle["confidence"]
