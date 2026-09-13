"""Completeness and end-to-end accuracy, including failed observations."""
from collections import Counter


def summarize_run(frame, question_ids, methods, runs):
    records = {(int(r.run_id), int(r.question_id), r.method): r
               for r in frame.itertuples()}
    summaries = {}
    for method in methods:
        keys = [(run, question, method) for run in range(1, runs + 1)
                for question in question_ids]
        valid = correct = missing = 0
        errors = Counter()
        for key in keys:
            row = records.get(key)
            if row is None:
                missing += 1
            elif isinstance(row.error, str) and row.error:
                errors[row.error] += 1
            elif row.predicted not in {"A", "B", "C", "D"}:
                errors["InvalidAnswer"] += 1
            elif row.correct not in (0, 1, False, True):
                errors["InvalidScore"] += 1
            else:
                valid += 1
                correct += int(row.correct)
        summaries[method] = {
            "expected": len(keys), "valid": valid, "correct": correct,
            "failed": sum(errors.values()), "missing": missing, "errors": dict(errors),
            "end_to_end_accuracy": correct / len(keys) if keys else None,
            "valid_response_accuracy": correct / valid if valid else None,
        }
    return {"complete": all(s["valid"] == s["expected"] for s in summaries.values()),
            "methods": summaries}
