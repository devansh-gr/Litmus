"""Regression guards for the fomo ⇄ false-urgency separation.

The benchmark's biggest technique blur was scarcity/fomo being read as false-urgency. The
sharpened definitions draw the line by pressure type: false-urgency = the CLOCK
(deadline/countdown), fomo = SCARCITY/exclusion (limited quantity, left behind). These
lock that in — allowing either the target OR its immediate cluster neighbor, since the
two genuinely overlap, but forbidding a wild miss.
"""
import pytest

# (text, expected primary, acceptable cluster) — the cluster reflects real overlap.
CASES = [
    ("Hurry, this offer expires at midnight tonight!", "false-urgency", {"false-urgency"}),
    ("Last chance — the deadline is in one hour.", "false-urgency", {"false-urgency"}),
    ("Only 3 items remain in stock, selling fast!", "fomo", {"fomo", "false-urgency"}),
    ("The train is leaving, don't miss your chance to join.", "fomo", {"fomo"}),
    ("Everyone is switching, don't get left behind.", "fomo", {"fomo", "social-proof-conformity"}),
]


@pytest.mark.parametrize("text,primary,cluster", CASES)
def test_urgency_vs_fomo(classify, text, primary, cluster):
    v = classify(text)["vector"]
    assert v in cluster, f"{text!r} -> {v}, expected {primary} (cluster {cluster})"


def test_pure_time_pressure_is_not_fomo(classify):
    assert classify("Buy before the countdown hits zero.")["vector"] != "fomo"


def test_pure_scarcity_is_not_pure_clock(classify):
    # "only N left / selling fast" should read as scarcity (fomo), not the clock.
    assert classify("Only a few left in stock — while supplies last.")["vector"] in {
        "fomo", "dopamine-bait", "social-proof-conformity"}
