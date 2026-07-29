"""Guard against server<->app taxonomy drift.

The server's VECTORS/DEFINITIONS and the Swift PersuasionVector enum must agree on the
exact label keys. If they drift, the server can return a vector the app can't decode
(the card silently shows nothing). This parses both source files textually (no heavy
imports) and asserts the label sets match.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER_PY = ROOT / "server" / "server.py"
TAXONOMY_SWIFT = ROOT / "Sources" / "CorticalPersuasionDecoder" / "Taxonomy" / "Taxonomy.swift"


def _server_vectors() -> set[str]:
    src = SERVER_PY.read_text()
    block = re.search(r"^VECTORS\s*=\s*\[(.*?)\]", src, re.S | re.M).group(1)
    return set(re.findall(r'"([a-z][a-z-]+)"', block))


def _server_definition_keys() -> set[str]:
    src = SERVER_PY.read_text()
    block = re.search(r"^DEFINITIONS\s*=\s*\{(.*?)\n\}", src, re.S | re.M).group(1)
    return set(re.findall(r'"([a-z][a-z-]+)"\s*:', block))


def _swift_vectors() -> set[str]:
    src = TAXONOMY_SWIFT.read_text()
    # enum raw values, e.g.  case fearMongering = "fear-mongering"
    enum = re.search(r"enum PersuasionVector.*?\{(.*?)\n\}", src, re.S).group(1)
    swift = set(re.findall(r'=\s*"([a-z][a-z-]+)"', enum))
    swift.add("none")  # server has an explicit "none"; Swift models it as the neutral path
    return swift


def test_server_vectors_match_definitions():
    assert _server_vectors() == _server_definition_keys(), \
        "server VECTORS and DEFINITIONS keys disagree"


def test_server_and_swift_agree():
    server = _server_vectors()
    swift = _swift_vectors()
    assert server == swift, (
        f"taxonomy drift!\n  only in server: {server - swift}\n  only in swift:  {swift - server}"
    )


def test_all_eval_labels_are_known():
    import json
    data = (Path(__file__).parent / "data" / "eval_set.jsonl").read_text().splitlines()
    labels = {json.loads(l)["label"] for l in data if l.strip()}
    assert labels <= _server_vectors(), f"eval set uses unknown labels: {labels - _server_vectors()}"
