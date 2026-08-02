#!/usr/bin/env python3
"""Rigorous benchmark harness (Tier-0 metrics).

Reports what the research says a subjective, imbalanced, two-stage, confidence-scored
classifier actually needs — not just accuracy:

  - overall accuracy WITH a 95% Wilson confidence interval (N is small!)
  - macro-F1 (+ bootstrap 95% CI), weighted-F1, balanced accuracy, MCC
  - per-class precision / recall / F1 / support
  - normalized confusion matrix
  - GATE evaluated as a binary detector: precision/recall/F1 + PR-AUC + Brier
    (uses the server's `manip_prob` field)
  - calibration of the shown confidence %: ECE + reliability bins + Brier
  - two-stage attribution: gate false-negatives / false-positives

    python tests/bench_full.py --data tests/data/external_test.jsonl --name external

Requires sklearn/scipy (already in the venv for nilearn).
"""
import argparse
import json
import math
import random
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef)

random.seed(0)
np.random.seed(0)
ENDPOINT = "http://127.0.0.1:8765"


def classify(text, timeout=60):
    req = urllib.request.Request(ENDPOINT + "/classify",
                                 data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0, center - half), min(1, center + half))


def bootstrap_macro_f1(y_true, y_pred, labels, B=2000):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(B):
        idx = np.random.randint(0, n, n)
        vals.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def ece_and_bins(confs, correct, n_bins=10):
    confs, correct = np.array(confs), np.array(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if m.sum() == 0:
            continue
        acc, conf = correct[m].mean(), confs[m].mean()
        ece += (m.sum() / len(confs)) * abs(acc - conf)
        rows.append((lo, hi, m.sum(), conf, acc))
    return ece, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="set")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines() if l.strip()]
    y_true, y_pred, confs, correct, manip_p, is_manip = [], [], [], [], [], []
    for r in rows:
        resp = classify(r["text"])
        gold, pred = r["label"], resp.get("vector", "?")
        y_true.append(gold)
        y_pred.append(pred)
        confs.append(resp.get("confidence", 0) / 100)
        correct.append(int(pred == gold))
        manip_p.append(resp.get("manip_prob", 1.0 if pred != "none" else 0.0))
        is_manip.append(int(gold != "none"))

    N = len(y_true)
    labels = sorted(set(y_true) | set(y_pred))
    acc = sum(correct) / N
    lo, hi = wilson(sum(correct), N)

    print(f"\n{'='*70}\nBENCHMARK: {args.name}   (N={N})\n{'='*70}")
    print(f"accuracy            {acc:.1%}   95% CI [{lo:.1%}, {hi:.1%}]  (Wilson)")
    mf1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    blo, bhi = bootstrap_macro_f1(y_true, y_pred, labels)
    print(f"macro-F1            {mf1:.3f}   95% CI [{blo:.3f}, {bhi:.3f}]  (bootstrap)")
    print(f"weighted-F1         {f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0):.3f}")
    print(f"balanced accuracy   {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"MCC                 {matthews_corrcoef(y_true, y_pred):.3f}   (0=random, 1=perfect)")

    print(f"\nper-class:\n{classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=2)}")

    # confusion (normalized by true row)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("confusion (rows=gold, cols=pred, row-normalized %):")
    short = [l[:10] for l in labels]
    print("            " + " ".join(f"{s:>10}" for s in short))
    for i, l in enumerate(labels):
        tot = cm[i].sum() or 1
        print(f"{l[:11]:>11} " + " ".join(f"{100*cm[i][j]/tot:>10.0f}" for j in range(len(labels))))

    # ---- Gate as a binary detector ----
    print(f"\n{'-'*70}\nGATE (binary: manipulation vs none)")
    y_gate_pred = [int(p != "none") for p in y_pred]
    tp = sum(1 for a, b in zip(is_manip, y_gate_pred) if a and b)
    fp = sum(1 for a, b in zip(is_manip, y_gate_pred) if not a and b)
    fn = sum(1 for a, b in zip(is_manip, y_gate_pred) if a and not b)
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    print(f"  precision {prec:.2f}  recall {rec:.2f}  F1 {2*prec*rec/(prec+rec) if prec+rec else 0:.2f}")
    if len(set(is_manip)) == 2:
        pr_auc = average_precision_score(is_manip, manip_p)
        base = sum(is_manip) / N
        print(f"  PR-AUC (avg precision) {pr_auc:.3f}   (prevalence baseline {base:.3f})")
        print(f"  Brier (gate prob)      {brier_score_loss(is_manip, manip_p):.3f}")
    print(f"  gate false-negatives (manipulation → none): {fn}/{sum(is_manip)}")
    print(f"  gate false-positives (benign → flagged):    {fp}/{N-sum(is_manip)}")

    # ---- Calibration of the shown confidence ----
    print(f"\n{'-'*70}\nCALIBRATION of the confidence %")
    ece, brows = ece_and_bins(confs, correct)
    print(f"  ECE {ece:.3f}   Brier {brier_score_loss(correct, confs):.3f}   (lower=better)")
    print("  reliability (conf bin → mean confidence vs actual accuracy):")
    for lo_, hi_, n_, conf_, acc_ in brows:
        flag = "  <-- overconfident" if conf_ - acc_ > 0.1 else ""
        print(f"    ({lo_:.1f},{hi_:.1f}]  n={n_:<3}  conf={conf_:.0%}  acc={acc_:.0%}{flag}")


if __name__ == "__main__":
    main()
