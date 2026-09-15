"""GPU smoke for the executable Stage2 multi-step retrieval policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from multistep_search import MultiStepPaperSearch, Stage2ActionPolicy
from paper_search import PaperSearchIndex
from retrieval_pipeline import HybridPaperSearch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--skills", required=True)
    parser.add_argument("--clstr-source", required=True)
    parser.add_argument("--query", default="graph neural network knowledge graph retrieval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    policy = Stage2ActionPolicy(
        checkpoint_path=args.checkpoint,
        skills_path=args.skills,
        clstr_source=args.clstr_source,
        device=args.device,
    )
    if not policy.enabled:
        raise RuntimeError(f"Stage2 policy failed to load: {policy.load_error}")
    selector = policy._selector
    if selector is None or len(selector.skill_ids) != 515:
        raise RuntimeError("Stage2 action inventory must contain exactly 515 skills")

    index = PaperSearchIndex(args.db)
    executor = MultiStepPaperSearch(HybridPaperSearch(index), candidate_limit=100)
    results, state = executor.run(
        args.query,
        policy=policy.new_episode(),
        top_k=args.top_k,
        max_steps=args.max_steps,
    )
    legal_errors = []
    for event in state.events:
        if event.skill_id not in event.legal_actions:
            legal_errors.append({"step": event.step_index, "skill_id": event.skill_id})
    if legal_errors:
        raise RuntimeError(f"Stage2 emitted illegal actions: {legal_errors}")
    if not state.stopped or not state.events or state.events[0].action != "SEARCH_PAPERS":
        raise RuntimeError("Stage2 executor did not complete SEARCH-first STOP-terminated loop")
    report = {
        "status": "ok",
        "protocol": "shenzhi_stage2_native_online_smoke_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "skill_count": len(selector.skill_ids),
        "device": str(selector.device),
        "query": args.query,
        "event_count": len(state.events),
        "executed_skill_ids": state.executed_skill_ids,
        "executed_actions": state.executed_actions,
        "stop_reason": state.stop_reason,
        "failure_count": len(state.failures),
        "illegal_action_count": len(legal_errors),
        "result_count": len(results),
        "events": [
            {
                "step_index": event.step_index,
                "skill_id": event.skill_id,
                "action": event.action,
                "status": event.status,
                "bound_values": event.bound_values,
                "policy_method": event.policy.get("method"),
                "stage2_enabled": event.policy.get("stage2_enabled"),
            }
            for event in state.events
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
