"""Regression guards for technique disambiguation (the 44%→61% accuracy win).

Stage-2 used to collapse distinct techniques into false-urgency (fomo→false-urgency 55%,
dopamine recall 0.00). The sharpened definitions + tie-break rule route by the PRIMARY LEVER:
limited supply→fomo, deadline/clock→false-urgency, crowd/activity→social-proof, reward/
curiosity→dopamine, celebrity/credential→authority. These lock that in with EXACT labels for
the clear cases, and a cluster allowance only where two techniques genuinely overlap.

Hits the live server; skips if it's down (see conftest).
"""
import pytest

# (text, exact expected label) — verified reliable after the disambiguation fix.
EXACT = [
    # scarcity / limited supply -> fomo (was collapsing to false-urgency)
    ("Only 2 left in stock, order soon.", "fomo"),
    ("Almost gone — while stocks last.", "fomo"),
    ("Items in your cart are in high demand. But we have reserved yours.", "fomo"),
    # explicit deadline / clock -> false-urgency
    ("Act now, this offer expires tonight at midnight.", "false-urgency"),
    ("Sale ends in 02:14:59", "false-urgency"),
    # other people's activity / crowd -> social-proof
    ("24 people have purchased this wine today", "social-proof-conformity"),
    ("Someone in Ampang, Malaysia just bought this", "social-proof-conformity"),
    # curiosity-gap clickbait -> dopamine
    ("You won't believe what happened next", "dopamine-bait"),
    ("This one weird trick will change everything", "dopamine-bait"),
    # celebrity / anecdotal authority -> authority-appeal
    ("Michael Jordan wears Hanes underwear, so you should too!", "authority-appeal"),
    ("Because doctors smoke it must be a healthy choice.", "authority-appeal"),
]

# Genuinely dual-signal cases: allow the target or its immediate neighbor. "The train is
# leaving" carries a departure/time metaphor, so false-urgency is also a fair read there.
CLUSTER = [
    ("The train is leaving, don't miss your chance to join.",
     {"fomo", "social-proof-conformity", "false-urgency"}),
    ("Everyone is switching, don't get left behind.", {"fomo", "social-proof-conformity"}),
]


@pytest.mark.parametrize("text,label", EXACT)
def test_primary_lever_exact(classify, text, label):
    v = classify(text)["vector"]
    assert v == label, f"{text!r} -> {v}, expected {label}"


@pytest.mark.parametrize("text,cluster", CLUSTER)
def test_dual_signal_stays_in_cluster(classify, text, cluster):
    v = classify(text)["vector"]
    assert v in cluster, f"{text!r} -> {v}, expected one of {cluster}"


def test_pure_scarcity_is_not_time_urgency(classify):
    # Scarcity ("only N left / while supplies last") must not read as the CLOCK.
    assert classify("Only a few left in stock — while supplies last.")["vector"] != "false-urgency"
