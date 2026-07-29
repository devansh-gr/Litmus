"""Shared fixtures for the server test suite.

Tests hit the live /classify endpoint (so they validate the real deployed path, not a
mock). If the server isn't running they SKIP rather than fail, so the suite is safe to
run anywhere.
"""
import json
import os
import urllib.request

import pytest

ENDPOINT = os.environ.get("CPD_ENDPOINT_URL", "http://127.0.0.1:8765")

VECTORS = {
    "fear-mongering", "critical-thinking-suppression", "tribal-in-group-bias",
    "dopamine-bait", "outrage", "authority-appeal", "false-urgency",
    "social-proof-conformity", "hype-hope-mongering", "fomo", "manufactured-awe",
    "none",
}


def _post(path, payload, timeout=60):
    req = urllib.request.Request(
        ENDPOINT.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


@pytest.fixture(scope="session")
def classify():
    try:
        _post("/classify", {"text": "health check ping"})
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"server not reachable at {ENDPOINT}: {e}")

    def _c(text):
        return _post("/classify", {"text": text})

    return _c
