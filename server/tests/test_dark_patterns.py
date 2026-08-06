"""Regression guard for e-commerce dark-pattern gate recall.

The honest benchmark showed the manipulation gate missed ~44% of real manipulation,
concentrated in dark patterns that read like plain facts: fake social-proof activity
nudges, manufactured scarcity, and countdown timers. These were fixed by teaching the
gate each family. This suite keeps them flagged — and, just as important, keeps their
benign look-alikes (neutral stats, ordinary clocks, straight news) as `none`, so the
fix didn't buy recall with a precision blow-up.

Hits the live server; skips if it's down (see conftest).
"""
import pytest

# Dark patterns that MUST flag (gate says manipulation, vector != none).
FLAG = [
    # social-proof activity nudges. NOTE: bare aggregate counts with no live-activity cue
    # ("111 people have purchased this item") sit in the same manip_prob band as ordinary
    # praise, so we assert the ones with a live/temporal nudge, which reliably clear the gate.
    "24 people have purchased this wine today",
    "Someone in Ampang, Malaysia just bought this",
    "Join thousands of happy customers today.",
    # scarcity
    "Items in your cart are in high demand. But we have reserved yours.",
    "Only 2 left in stock, order soon.",
    "Almost gone — while stocks last.",
    # countdown timers
    "1DAYS 04 HOURS 17 MINUTES 18 SECONDS",
    "Offer expires in 02:14:59",
    # clickbait curiosity gap
    "You won't believe what happened next",
    "This one weird trick will change everything",
]

# Benign look-alikes that MUST stay none (the precision guard).
BENIGN = [
    "The study surveyed 24 people about their sleep.",
    "The library had 111 visitors last Tuesday.",
    "The meeting is at 4:30 PM.",
    "It takes about 2 hours 15 minutes to drive there.",
    "Federal Reserve Raises Interest Rates by 0.25 Percent",
    "Scientists Publish Study on Coral Reef Recovery",
    "We're out of milk, can you grab some?",
]


@pytest.mark.parametrize("text", FLAG)
def test_dark_pattern_is_flagged(classify, text):
    r = classify(text)
    assert r["vector"] != "none", f"{text!r} slipped past the gate as none (mp={r.get('manip_prob')})"


@pytest.mark.parametrize("text", BENIGN)
def test_benign_lookalike_stays_none(classify, text):
    r = classify(text)
    assert r["vector"] == "none", f"{text!r} false-flagged as {r['vector']} {r['confidence']}%"
