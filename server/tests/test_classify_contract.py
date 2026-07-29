"""Contract tests: the /classify response shape the Swift `RemoteClassifier` decodes.

If any of these break, the app's `ClassifyResponse` decoder breaks silently — the card
shows nothing or wrong data. These pin the JSON contract.
"""
from conftest import VECTORS


def test_response_has_required_fields(classify):
    r = classify("Act now or lose everything forever.")
    for field in ("vector", "confidence", "rationale", "alternatives"):
        assert field in r, f"missing field: {field}"


def test_confidence_is_int_percent(classify):
    r = classify("Act now or lose everything forever.")
    assert isinstance(r["confidence"], int)
    assert 0 <= r["confidence"] <= 100


def test_vector_is_in_taxonomy(classify):
    # Every returned vector must be a key the Swift PersuasionVector enum knows.
    for text in ("Everyone is switching, don't be left behind.",
                 "Top scientists agree this is settled.",
                 "The train departs at noon."):
        assert classify(text)["vector"] in VECTORS


def test_alternatives_are_well_formed(classify):
    r = classify("This will change your life and the whole world forever.")
    assert isinstance(r["alternatives"], list)
    for alt in r["alternatives"]:
        assert set(("vector", "p")).issubset(alt)
        assert alt["vector"] in VECTORS
        assert 0.0 <= alt["p"] <= 1.0


def test_alternatives_sorted_descending(classify):
    r = classify("Hurry, the deal ends tonight and everyone is buying in.")
    ps = [a["p"] for a in r["alternatives"]]
    assert ps == sorted(ps, reverse=True), "alternatives must be ranked high->low"


def test_rationale_present_for_manipulation(classify):
    r = classify("Act now, this offer expires in ten minutes!")
    assert r["rationale"] and isinstance(r["rationale"], str)
