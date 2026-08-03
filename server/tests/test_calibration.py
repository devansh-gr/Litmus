"""Regression guards for confidence calibration + abstain.

Before Platt calibration the model said "99%" everywhere (ECE 0.37). These lock in the
softened, honest confidence and the abstain-on-unclear behavior.
"""


def test_confidence_is_not_maxed_out(classify):
    # Calibration must soften over-confidence — not every verdict is 95-100%.
    texts = [
        "This will change your life and the whole world.",
        "Only a few left, better decide.",
        "Nine out of ten experts endorse this, so the debate is over.",
        "You might want to consider this at some point.",
    ]
    confs = [classify(t)["confidence"] for t in texts]
    assert any(c < 90 for c in confs), f"calibration not softening: {confs}"


def test_clear_case_still_confident(classify):
    # Calibration shouldn't crush confidence on blatant cases.
    assert classify("Act now, this offer expires at midnight tonight!")["confidence"] >= 65


def test_abstain_invariant(classify):
    # Any low-confidence MANIPULATION verdict must carry the uncertain flag.
    for t in ("Nine out of ten experts endorse this, so the debate is over.",
              "It could perhaps be a little bit questionable, in a way."):
        r = classify(t)
        if r["vector"] != "none" and r["confidence"] < 45:
            assert r.get("uncertain") is True, f"{t!r} low-conf but not flagged uncertain"


def test_manip_prob_exposed(classify):
    # The gate probability is exposed for eval/PR-AUC.
    r = classify("Act now or lose everything!")
    assert "manip_prob" in r and 0.0 <= r["manip_prob"] <= 1.0
