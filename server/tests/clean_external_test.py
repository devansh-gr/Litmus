"""Produce a CLEANED variant of external_test.jsonl by removing entries that are not
actually instances of their label — DEFINITIONS of a technique, QUIZ-QUESTION stems, and
clear CROSS-FALLACY MISLABELS from the noisy LOGIC / propaganda source datasets.

Conservative by design: it removes ONLY entries that are not manipulative text at all, or
are unambiguously a different fallacy than their label. Every genuine and every borderline
example is kept. The original `external_test.jsonl` is left untouched for provenance; this
writes `external_test_clean.jsonl` + `CLEANING_LOG.md` so every removal is auditable.

Run:  python tests/clean_external_test.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "data" / "external_test.jsonl"
OUT = HERE / "data" / "external_test_clean.jsonl"
LOG = HERE / "data" / "CLEANING_LOG.md"

# (unique substring of the text, its labeled class, reason for removal). Matched by
# substring so the exact OCR/formatting doesn't have to be reproduced here.
REMOVALS = [
    # --- definitions / meta-descriptions of a technique (not an instance of it) ---
    ("Presenting an unqualified person or institution", "authority-appeal",
     "definition of the fallacy, not an example of it"),
    ("Invoke shared values and principles", "authority-appeal",
     "definition of a different technique (appeal to shared values), not appeal-to-authority"),
    ("This type of ad taps into a person", "social-proof-conformity",
     "definition of the technique, not an instance"),
    ("This type of propaganda implies that since EVERYONE", "social-proof-conformity",
     "definition of the technique, not an instance"),
    ("This makes you think you need to believe or buy something because everyone else", "social-proof-conformity",
     "definition + quiz stem, not an instance"),
    # --- quiz-question stems (asking ABOUT a fallacy, not performing it) ---
    ("Which fallacy is used to promote something based on popularity", "social-proof-conformity",
     "quiz question stem, not manipulative text"),
    ("when evidence boils down to", "social-proof-conformity",
     "garbled meta-fragment describing the fallacy, not a clean instance"),
    # --- clear cross-fallacy mislabels (a different fallacy than the assigned label) ---
    ("This coin has landed heads-up nine times in a row", "social-proof-conformity",
     "gambler's fallacy, not appeal-to-popularity"),
    ("President Clinton is an advocate of socialized medicine", "social-proof-conformity",
     "guilt-by-association smear, not appeal-to-popularity"),
    ("Homosexuality is / ought to be morally wrong (moral property) because it is not normal", "authority-appeal",
     "naturalistic fallacy (appeal to nature), not appeal-to-authority"),
]


def main():
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    removed, kept = [], []
    used = [False] * len(REMOVALS)
    for r in rows:
        text = r.get("text", "")
        hit = None
        for i, (sub, lab, reason) in enumerate(REMOVALS):
            if sub in text and r.get("label") == lab:
                hit = (sub, lab, reason)
                used[i] = True
                break
        (removed if hit else kept).append((r, hit))

    OUT.write_text("\n".join(json.dumps(r) for r, _ in kept) + "\n")

    # audit log
    lines = ["# external_test cleaning log", "",
             f"Source: `external_test.jsonl` ({len(rows)}) -> cleaned `external_test_clean.jsonl` "
             f"({len(kept)}). Removed {len(removed)} entries that are not instances of their label.",
             "", "| removed text | labeled | why removed |", "|---|---|---|"]
    for r, (sub, lab, reason) in removed:
        t = r["text"].replace("|", "\\|")[:80]
        lines.append(f"| {t} | {lab} | {reason} |")
    lines += ["", "Removal rule: only DEFINITIONS of a technique, QUIZ-QUESTION stems, and clear "
              "CROSS-FALLACY mislabels are dropped — all genuine and borderline examples are kept. "
              "The original file is preserved for provenance; regenerate with "
              "`python tests/clean_external_test.py`."]
    LOG.write_text("\n".join(lines) + "\n")

    print(f"kept {len(kept)}  removed {len(removed)}  (from {len(rows)})")
    for i, (sub, lab, _) in enumerate(REMOVALS):
        if not used[i]:
            print(f"  WARNING: removal pattern never matched: {sub!r} ({lab})")
    print(f"wrote {OUT.name} + {LOG.name}")


if __name__ == "__main__":
    main()
