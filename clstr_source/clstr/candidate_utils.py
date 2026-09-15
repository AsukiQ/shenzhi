from __future__ import annotations


def inject_positive_candidate(raw_candidates: list[int], positive: int, k: int) -> list[int]:
    if positive in raw_candidates:
        return list(raw_candidates[:k])
    trimmed = list(raw_candidates[:k])
    if len(trimmed) < k:
        trimmed.append(positive)
        return trimmed
    if not trimmed:
        return [positive]
    return trimmed[:-1] + [positive]
