from __future__ import annotations

import argparse
import json
import time

import torch

from clstr.memory_candidate_recall import (
    CANDIDATE_SELECTION_VERSION,
    TIE_BREAK_POLICY,
    stable_masked_topk_rows,
)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _mean_latency(callback, *, warmup: int, repeats: int, device: torch.device) -> float:
    for _ in range(max(0, int(warmup))):
        callback()
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(max(1, int(repeats))):
        callback()
    _synchronize(device)
    return (time.perf_counter() - started) / max(1, int(repeats))


def benchmark_candidate_selection(
    *,
    batch_size: int,
    skill_count: int,
    hidden_size: int,
    static_k: int,
    dynamic_extra_k: int,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, float | int | str]:
    generator = torch.Generator(device=device)
    generator.manual_seed(17)
    h = torch.randn(batch_size, hidden_size, generator=generator, device=device)
    skill_table = torch.randn(skill_count, hidden_size, generator=generator, device=device)
    logits = h @ skill_table.t()
    valid = torch.ones_like(logits, dtype=torch.bool)
    selector_k = max(0, int(static_k)) + max(0, int(dynamic_extra_k))

    selector_latency = _mean_latency(
        lambda: stable_masked_topk_rows(logits, valid, k=selector_k),
        warmup=warmup,
        repeats=repeats,
        device=device,
    )
    topk_latency = _mean_latency(
        lambda: torch.topk(logits, k=min(selector_k, skill_count), dim=-1, sorted=False),
        warmup=warmup,
        repeats=repeats,
        device=device,
    )

    selector_e2e = _mean_latency(
        lambda: stable_masked_topk_rows(h @ skill_table.t(), valid, k=selector_k),
        warmup=warmup,
        repeats=repeats,
        device=device,
    )
    topk_e2e = _mean_latency(
        lambda: torch.topk(
            h @ skill_table.t(),
            k=min(selector_k, skill_count),
            dim=-1,
            sorted=False,
        ),
        warmup=warmup,
        repeats=repeats,
        device=device,
    )

    extra_equivalents = 0.0
    peak_allocated_bytes = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        stable_masked_topk_rows(logits, valid, k=selector_k)
        _synchronize(device)
        peak_allocated_bytes = max(0, int(torch.cuda.max_memory_allocated(device) - baseline))
        score_matrix_bytes = max(1, int(logits.numel() * logits.element_size()))
        extra_equivalents = peak_allocated_bytes / score_matrix_bytes

    return {
        "candidate_selection_version": CANDIDATE_SELECTION_VERSION,
        "tie_break_policy": TIE_BREAK_POLICY,
        "device": str(device),
        "batch_size": int(batch_size),
        "skill_count": int(skill_count),
        "hidden_size": int(hidden_size),
        "static_k": int(static_k),
        "dynamic_extra_k": int(dynamic_extra_k),
        "selector_latency_seconds": float(selector_latency),
        "topk_latency_seconds": float(topk_latency),
        "selector_topk_latency_ratio": float(selector_latency / max(topk_latency, 1.0e-12)),
        "selector_end_to_end_seconds": float(selector_e2e),
        "topk_end_to_end_seconds": float(topk_e2e),
        "end_to_end_overhead_fraction": float(max(0.0, selector_e2e - topk_e2e) / max(topk_e2e, 1.0e-12)),
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "extra_score_matrix_equivalents": float(extra_equivalents),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--skill_count", type=int, default=67557)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--static_k", type=int, default=500)
    parser.add_argument("--dynamic_extra_k", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_topk_latency_ratio", type=float, default=2.0)
    parser.add_argument("--max_end_to_end_overhead_fraction", type=float, default=0.20)
    parser.add_argument("--max_extra_score_matrix_equivalents", type=float, default=1.0)
    args = parser.parse_args()

    report = benchmark_candidate_selection(
        batch_size=args.batch_size,
        skill_count=args.skill_count,
        hidden_size=args.hidden_size,
        static_k=args.static_k,
        dynamic_extra_k=args.dynamic_extra_k,
        warmup=args.warmup,
        repeats=args.repeats,
        device=torch.device(args.device),
    )
    report["thresholds"] = {
        "max_topk_latency_ratio": float(args.max_topk_latency_ratio),
        "max_end_to_end_overhead_fraction": float(args.max_end_to_end_overhead_fraction),
        "max_extra_score_matrix_equivalents": float(args.max_extra_score_matrix_equivalents),
    }
    report["passed"] = bool(
        report["selector_topk_latency_ratio"] <= args.max_topk_latency_ratio
        and report["end_to_end_overhead_fraction"] <= args.max_end_to_end_overhead_fraction
        and report["extra_score_matrix_equivalents"] <= args.max_extra_score_matrix_equivalents
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
