"""Robustness: the detector must never crash on weird input.

⌘B and OCR feed arbitrary selections — long articles, emoji, other scripts, whitespace.
None of it should 500; every call must return a well-formed verdict.
"""
import json
import urllib.request

from conftest import ENDPOINT, VECTORS


def _health():
    with urllib.request.urlopen(ENDPOINT.rstrip("/") + "/health", timeout=10) as r:
        return json.loads(r.read())


def test_health_ok(classify):  # classify fixture ensures server is up / skips otherwise
    h = _health()
    assert h.get("ok") is True
    assert "device" in h


def test_long_input(classify):
    # A whole pasted article — must classify, not choke.
    text = ("Act now, this incredible once-in-a-lifetime opportunity expires soon. " * 80)
    r = classify(text)
    assert r["vector"] in VECTORS
    assert 0 <= r["confidence"] <= 100


def test_emoji_and_unicode(classify):
    for text in ("🚀🚀🚀 to the moon, don't miss out! 🌕",
                 "¡Actúa ahora o piérdelo todo para siempre!",
                 "急いで、今すぐ行動しないと全てを失う"):
        r = classify(text)
        assert r["vector"] in VECTORS


def test_whitespace_and_newlines(classify):
    r = classify("   \n\n  Everyone is switching.\n\n  Don't be left behind.  \n ")
    assert r["vector"] in VECTORS


def test_repeated_calls_stable(classify):
    # Hammer the same text; must stay well-formed and identical (cache + determinism).
    text = "Top experts agree this is beyond any dispute whatsoever."
    first = classify(text)
    for _ in range(5):
        r = classify(text)
        assert r["vector"] == first["vector"]
        assert r["confidence"] == first["confidence"]
