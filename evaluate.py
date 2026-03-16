#!/usr/bin/env python3
"""Evaluate predictions using Exact Match and F1 score (SQuAD-style)."""

import json
import re
import string
import sys
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores.append(score)
    return max(scores)


def evaluate(predictions_path, reference_path):
    """Evaluate predictions against references."""
    # Load reference
    references = []
    with open(reference_path, "r") as f:
        for line in f:
            data = json.loads(line)
            answers = data["answer"].split("|")
            references.append({
                "question": data["question"],
                "answers": [a.strip() for a in answers],
            })

    # Load predictions
    with open(predictions_path, "r") as f:
        predictions = [line.strip() for line in f]

    assert len(predictions) == len(references), \
        f"Number of predictions ({len(predictions)}) != references ({len(references)})"

    f1_total = 0
    em_total = 0
    for pred, ref in zip(predictions, references):
        f1 = metric_max_over_ground_truths(f1_score, pred, ref["answers"])
        em = metric_max_over_ground_truths(exact_match_score, pred, ref["answers"])
        f1_total += f1
        em_total += em
        print(f"Q: {ref['question'][:80]}")
        print(f"  Pred: {pred}")
        print(f"  Gold: {ref['answers']}")
        print(f"  F1: {f1:.3f}, EM: {em}")
        print()

    n = len(references)
    print(f"Overall F1: {f1_total/n:.4f}")
    print(f"Overall EM: {em_total/n:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 evaluate.py <predictions_txt> <reference_jsonl>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
