"""Paired inference across questions, not independent repeated API calls."""
from itertools import combinations

import numpy as np
import pandas as pd


def paired_report(frame, seed=2026, resamples=2000):
    """Question-cluster bootstrap of paired accuracy differences (95% percentile CI).

    Require matching run/question observations and report every exclusion.
    Intervals are exploratory, unadjusted for multiple comparisons.
    """
    required = {"run_id", "question_id", "method", "correct"}
    if not required.issubset(frame):
        raise ValueError("Paired inference requires run_id, question_id, method and correct")
    keys = ["run_id", "question_id", "method"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate run/question/method observations")
    data = frame.copy()
    if "error" in data:
        data = data[data["error"].fillna("") == ""]
    data["correct"] = data["correct"].map(
        lambda x: {"True": 1.0, "False": 0.0}.get(str(x), x)
    )
    data["correct"] = pd.to_numeric(data["correct"], errors="raise")
    pivot = data.pivot(index=["run_id", "question_id"], columns="method", values="correct")
    rng = np.random.default_rng(seed)
    comparisons = []
    for a, b in combinations(sorted(frame["method"].unique()), 2):
        pair = pivot.reindex(columns=[a, b]).dropna()
        differences = (pair[a] - pair[b]).groupby("question_id").mean().to_numpy()
        count = len(differences)
        row = {
            "method_a": a, "method_b": b, "paired_observations": len(pair),
            "unique_questions": count,
            "excluded_a": int((frame.method == a).sum() - len(pair)),
            "excluded_b": int((frame.method == b).sum() - len(pair)),
            "accuracy_difference_pp": float(differences.mean() * 100) if count else None,
            "ci95_low_pp": None, "ci95_high_pp": None,
        }
        if count >= 2:
            means = rng.choice(differences, (resamples, count), replace=True).mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975]) * 100
            row.update(ci95_low_pp=float(low), ci95_high_pp=float(high))
        comparisons.append(row)
    return {
        "seed": seed, "resamples": resamples, "unit": "question",
        "note": "Exploratory paired question-cluster bootstrap; no multiplicity correction. "
                "Repeated runs are not independent questions. No claims beyond this dataset.",
        "comparisons": comparisons,
    }

