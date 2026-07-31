"""Regression guard: benign / conversational text must NOT be flagged as manipulation.

This is the bug where highlighting a Google AI Overview's "Hello! How can I help you
today?" reported "Hype 84%". The two-stage gate (manipulation? then which) fixes it —
these tests keep it fixed.
"""
import json
from pathlib import Path

import pytest

BENIGN = [json.loads(l)["text"]
          for l in (Path(__file__).parent / "data" / "benign_set.jsonl").read_text().splitlines()
          if l.strip()]

# The exact reported case + core greetings — these must ALWAYS be none.
HARD = [
    "Hello!",
    "Hello! How can I help you today?",
    "Hi there, how are you doing?",
    "What time does the library close on Sundays?",
]


@pytest.mark.parametrize("text", HARD)
def test_greetings_and_questions_are_none(classify, text):
    r = classify(text)
    assert r["vector"] == "none", f"{text!r} flagged as {r['vector']} {r['confidence']}%"


def test_benign_false_positive_rate_is_low(classify):
    flagged = [(t, classify(t)) for t in BENIGN]
    misses = [(t, r["vector"], r["confidence"]) for t, r in flagged if r["vector"] != "none"]
    # A couple of strongly positive-affect phrases may still slip; the rest must be none.
    assert len(misses) <= 2, f"too many benign false positives ({len(misses)}/{len(BENIGN)}): {misses}"
