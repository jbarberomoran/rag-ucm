"""Deterministic option permutation without changing question semantics."""
import copy
import hashlib
import json
import random
import re
from collections import Counter


def rebalance(questions, seed=2026):
    rng = random.Random(seed)
    targets = ["ABCD"[i % 4] for i in range(len(questions))]
    rng.shuffle(targets)
    balanced, mappings = [], []
    for index, (question, target) in enumerate(zip(questions, targets, strict=True), 1):
        if any(re.search(r"\b(above|below|both [A-D]|option [A-D]|answer [A-D])\b", text,
                         re.IGNORECASE) for text in question["answers"].values()):
            raise ValueError(f"Question {index} requires manual positional-reference review")
        old_correct = question["correct_answer"]
        distractors = [letter for letter in "ABCD" if letter != old_correct]
        rng.shuffle(distractors)
        new_to_old = {letter: old_correct if letter == target else distractors.pop()
                      for letter in "ABCD"}
        updated = copy.deepcopy(question)
        updated["answers"] = {new: question["answers"][old] for new, old in new_to_old.items()}
        updated["correct_answer"] = target
        balanced.append(updated)
        mappings.append({"question_id": index, "old_correct": old_correct,
                         "new_correct": target, "new_to_old": new_to_old})
    audit = {"schema": 1, "seed": seed, "algorithm": "src.dataset_balance.rebalance",
             "original_content_sha256": hashlib.sha256(json.dumps(
                 questions, sort_keys=True, ensure_ascii=False
             ).encode()).hexdigest(),
             "before": dict(Counter(q["correct_answer"] for q in questions)),
             "after": dict(Counter(q["correct_answer"] for q in balanced)),
             "permutations": mappings}
    return balanced, audit
