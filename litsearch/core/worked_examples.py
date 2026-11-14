"""Worked examples whose numbers are computed by code, not by the model (used for topics flagged `worked_example: confusion`).

A small 2x2 scenario (prevalence, sensitivity, specificity, sample size) is chosen deterministically from the date, turned into counts
and every metric the lesson mentions is computed here, so the audio never contains an arithmetic slip.
"""
from __future__ import annotations

import hashlib
import math

SCENARIOS = [
    {"label": "screening a rare outcome", "n": 1000, "prevalence": 0.05, "sensitivity": 0.80, "specificity": 0.90},
    {"label": "flagging a rare complication", "n": 2000, "prevalence": 0.02, "sensitivity": 0.70, "specificity": 0.95},
    {"label": "triaging a moderately common condition", "n": 1000, "prevalence": 0.20, "sensitivity": 0.85, "specificity": 0.80},
    {"label": "finding a very rare event in a large cohort", "n": 5000, "prevalence": 0.01, "sensitivity": 0.90, "specificity": 0.92},
]


def confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Every metric for a 2x2 table (None where undefined)."""
    n = tp + fp + fn + tn
    div = lambda a, b: a / b if b else None
    recall, spec = div(tp, tp + fn), div(tn, tn + fp)
    prec, npv = div(tp, tp + fp), div(tn, tn + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "prevalence": div(tp + fn, n), "accuracy": div(tp + tn, n),
            "recall": recall, "specificity": spec, "precision": prec, "npv": npv, "f1": f1,
            "balanced_accuracy": (recall + spec) / 2 if recall is not None and spec is not None else None,
            "mcc": ((tp * tn - fp * fn) / mcc_den) if mcc_den else None,
            "majority_accuracy": max(tp + fn, tn + fp) / n if n else None,
            "f1_if_all_positive": div(2 * (tp + fn), 2 * (tp + fn) + (tn + fp))}


def scenario_for(seed: str) -> dict:
    """A deterministic scenario with integer counts for this seed (the date, say)."""
    s = SCENARIOS[int(hashlib.sha256(seed.encode()).hexdigest(), 16) % len(SCENARIOS)]
    pos = round(s["n"] * s["prevalence"])
    neg = s["n"] - pos
    tp = round(pos * s["sensitivity"])
    tn = round(neg * s["specificity"])
    return {**s, **confusion_metrics(tp, neg - tn, pos - tp, tn)}


def describe(sc: dict) -> str:
    pct = lambda x: f"{x * 100:.1f}%"
    f = lambda x: "undefined" if x is None else f"{x:.2f}"
    return "\n".join([
        f"Scenario: {sc['label']}. {sc['n']} patients; {sc['tp'] + sc['fn']} truly positive ({pct(sc['prevalence'])}), {sc['tn'] + sc['fp']} truly negative.",
        f"The model flags {sc['tp'] + sc['fp']}: {sc['tp']} true positives, {sc['fp']} false positives; it misses {sc['fn']} (false negatives) and clears {sc['tn']} true negatives.",
        f"Accuracy {pct(sc['accuracy'])}; always predicting the majority class would score {pct(sc['majority_accuracy'])}.",
        f"Recall (sensitivity) {f(sc['recall'])}; specificity {f(sc['specificity'])}; precision (PPV) {f(sc['precision'])}; NPV {f(sc['npv'])}.",
        f"F1 {f(sc['f1'])}; balanced accuracy {f(sc['balanced_accuracy'])}; Matthews correlation coefficient {f(sc['mcc'])}.",
        f"For contrast, a model that flags everyone as positive would have recall 1.00 and F1 {f(sc['f1_if_all_positive'])}."])
