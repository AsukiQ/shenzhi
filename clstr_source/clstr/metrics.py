from __future__ import annotations

import numpy as np


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return float(any(item in relevant_ids for item in ranked_ids[:k]))


def stop_f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def belief_mse(values_a, values_b) -> float:
    return float(np.mean((np.asarray(values_a) - np.asarray(values_b)) ** 2))
