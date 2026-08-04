"""Regression guard for the INTERPERSONAL manipulation family.

The taxonomy was originally marketing/propaganda-only, so one-on-one tactics
('if you really cared you would', 'you made me do this') misfiled onto the nearest
marketing label. These tests lock in the interpersonal family: guilt-tripping,
love-bombing, blame-shifting, and gaslighting/minimization (which folds into
critical-thinking-suppression).

Two strengths of assertion:
  - EXACT label for the clear, canonical phrasings.
  - "at least flagged" (not none) for all manipulation, since cluster-boundary
    cases (guilt vs blame vs love-bomb) can swap among interpersonal labels.
Hits the live server; skips if it's down (see conftest).
"""
import pytest

# (text, expected exact label)
EXACT = [
    ("No one will ever love you the way I do, you're my everything.", "love-bombing"),
    ("You're my soulmate, I can't live without you, we're meant to be.", "love-bombing"),
    ("If you really cared about me, you would do this.", "guilt-tripping"),
    ("After all I've done for you, this is how you repay me?", "guilt-tripping"),
    ("You made me do this. If you hadn't pushed me I never would have.", "blame-shifting"),
    ("It's your fault, you're the one who started it.", "blame-shifting"),
    ("That never happened, you're imagining it, you're too sensitive.", "critical-thinking-suppression"),
    ("Can't you take a joke? It really wasn't that bad.", "critical-thinking-suppression"),
]

# Any interpersonal manipulation must at least clear the gate (never 'none').
FLAGGED = [t for t, _ in EXACT] + [
    "Fine, I'll just do it myself then. Don't worry about me.",
    "I'm not mad, I'm just disappointed in you.",
]

# Benign interpersonal talk + ordinary praise must stay none (the love-bombing FP guard).
BENIGN = [
    "Thanks so much for dinner, I had a really nice time.",
    "You did a great job on the presentation.",
    "Nice work, well done.",
    "Let me know when you're free to catch up this week.",
]


@pytest.mark.parametrize("text,label", EXACT)
def test_interpersonal_exact_label(classify, text, label):
    r = classify(text)
    assert r["vector"] == label, f"{text!r} -> {r['vector']} {r['confidence']}% (wanted {label})"


@pytest.mark.parametrize("text", FLAGGED)
def test_interpersonal_manipulation_is_flagged(classify, text):
    r = classify(text)
    assert r["vector"] != "none", f"{text!r} slipped past the gate as none"


@pytest.mark.parametrize("text", BENIGN)
def test_benign_interpersonal_stays_none(classify, text):
    r = classify(text)
    assert r["vector"] == "none", f"{text!r} false-flagged as {r['vector']} {r['confidence']}%"
