"""Realistic-usage guard: on a representative mix (mostly benign), the detector must be
both precise (few false positives) and sensitive (catches most manipulation).

Unlike eval_set (92% manipulation), realistic_set.jsonl is 60% benign — closer to what
people actually highlight. This locks in the two-stage gate's operating point.
"""
import json
from pathlib import Path

DATA = [json.loads(l) for l in
        (Path(__file__).parent / "data" / "realistic_set.jsonl").read_text().splitlines()
        if l.strip()]


def test_benign_false_positive_rate_low(classify):
    benign = [d["text"] for d in DATA if d["label"] == "none"]
    fps = [(t, classify(t)["vector"]) for t in benign if classify(t)["vector"] != "none"]
    # The gate threshold (0.70) is a deliberate recall-favoring point, so ~20% of benign
    # may be misflagged on this self-authored set. Hard greetings ("Hello!") are guarded
    # separately in test_benign.py and must always pass.
    assert len(fps) <= max(4, len(benign) // 5), f"benign false positives: {fps}"


def test_manipulation_recall_high(classify):
    manip = [d["text"] for d in DATA if d["label"] != "none"]
    missed = [t for t in manip if classify(t)["vector"] == "none"]
    # The gate may drop a subtle case, but must catch the clear majority.
    assert len(missed) <= max(2, len(manip) // 10), f"manipulation missed by gate: {missed}"
