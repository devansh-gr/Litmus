#!/usr/bin/env python3
"""Fit Platt calibration (a, b) for the displayed confidence.

Reads a dump from `bench_full.py --dump` (each row: conf, correct) on a DEV set, fits
P(correct) = sigmoid(a·logit(conf) + b) via logistic regression, and prints the env vars
to set. Also reports ECE before/after so the improvement is visible.

    python tests/fit_calibration.py <dev_dump.json>
"""
import json
import math
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression


def ece(confs, correct, n_bins=10):
    confs, correct = np.asarray(confs), np.asarray(correct, float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if m.sum():
            e += (m.sum() / len(confs)) * abs(correct[m].mean() - confs[m].mean())
    return e


def main():
    data = json.load(open(sys.argv[1]))
    conf = np.array([min(0.999, max(0.001, d["conf"])) for d in data])
    correct = np.array([d["correct"] for d in data])
    logit = np.log(conf / (1 - conf)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, correct)
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])

    def cal(p):
        p = min(0.999, max(0.001, p))
        z = a * math.log(p / (1 - p)) + b
        return 1 / (1 + math.exp(-z))

    cal_conf = [cal(c) for c in conf]
    print(f"fit on {len(data)} dev examples (raw accuracy {correct.mean():.1%})")
    print(f"  ECE before: {ece(conf, correct):.3f}")
    print(f"  ECE after : {ece(cal_conf, correct):.3f}")
    print(f"\nCPD_CALIB_A={a:.4f}")
    print(f"CPD_CALIB_B={b:.4f}")


if __name__ == "__main__":
    main()
