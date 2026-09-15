#!/usr/bin/env python3
"""
Verify current_route_rollouts.jsonl structure and completeness.

Usage:
    python scripts/verify_current_route_rollouts.py \
        --rollouts_path outputs/.../current_route_rollouts.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict


def load_jsonl(path):
    """Load JSONL file."""
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def verify_rollouts(rollouts_path):
    """Verify current_route_rollouts.jsonl structure."""
    print(f"=== Verifying: {rollouts_path} ===\n")

    if not Path(rollouts_path).exists():
        print(f"❌ File not found: {rollouts_path}")
        return False

    rollouts = load_jsonl(rollouts_path)
    print(f"✅ Loaded {len(rollouts)} rollouts\n")

    if len(rollouts) == 0:
        print("❌ No rollouts found")
        return False

    # Check required fields
    required_fields = [
        'task_id', 'outcome', 'training_ready',
        'decisions', 'task_completed', 'evaluation_success'
    ]

    missing_fields = defaultdict(int)
    for rollout in rollouts:
        for field in required_fields:
            if field not in rollout:
                missing_fields[field] += 1

        # Check nested outcome fields
        if 'outcome' in rollout:
            outcome = rollout['outcome']
            if 'policy_signal' not in outcome:
                missing_fields['outcome.policy_signal'] += 1
            if 'label' not in outcome:
                missing_fields['outcome.label'] += 1

    if missing_fields:
        print("❌ Missing required fields:")
        for field, count in missing_fields.items():
            print(f"  - {field}: missing in {count}/{len(rollouts)} rollouts")
        return False
    else:
        print("✅ All required top-level fields present\n")

    # Check decision structure (renamed from 'steps' to 'decisions')
    print("=== Decision Structure ===")
    decision_fields = [
        'candidate_skill_ids', 'ranking_mode', 'candidate_source',
        'code', 'execute_output', 'evaluation_success'
    ]

    total_decisions = sum(len(r.get('decisions', [])) for r in rollouts)
    print(f"Total decisions: {total_decisions}")

    decision_missing = defaultdict(int)
    for rollout in rollouts:
        for decision in rollout.get('decisions', []):
            for field in decision_fields:
                if field not in decision:
                    decision_missing[field] += 1

    if decision_missing:
        print("⚠️  Some decision fields missing:")
        for field, count in decision_missing.items():
            print(f"  - {field}: missing in {count}/{total_decisions} decisions")
    else:
        print("✅ All decision fields present\n")

    # Outcome distribution
    print("=== Outcome Distribution ===")
    outcomes = Counter(r.get('outcome', {}).get('label', 'unknown') for r in rollouts)
    for outcome, count in outcomes.most_common():
        print(f"  {outcome}: {count}")
    print()

    # Policy signal distribution
    print("=== Policy Signal Distribution ===")
    signals = Counter(r.get('outcome', {}).get('policy_signal', 'unknown') for r in rollouts)
    for signal, count in signals.most_common():
        print(f"  {signal}: {count}")
    print()

    # Training readiness
    print("=== Training Readiness ===")
    training_ready = sum(1 for r in rollouts if r.get('training_ready', False))
    print(f"  training_ready=true: {training_ready}/{len(rollouts)}")
    print(f"  training_ready=false: {len(rollouts) - training_ready}/{len(rollouts)}")

    not_ready_reasons = defaultdict(int)
    for rollout in rollouts:
        if not rollout.get('training_ready', False):
            reason = rollout.get('training_not_ready_reason', 'unknown')
            not_ready_reasons[reason] += 1

    if not_ready_reasons:
        print("\n  Reasons for training_ready=false:")
        for reason, count in not_ready_reasons.items():
            print(f"    - {reason}: {count}")
    print()

    # Detailed sample
    print("=== Sample Rollout (first one) ===")
    if rollouts:
        rollout = rollouts[0]
        outcome = rollout.get('outcome', {})
        print(f"task_id: {rollout.get('task_id')}")
        print(f"outcome.label: {outcome.get('label')}")
        print(f"outcome.policy_signal: {outcome.get('policy_signal')}")
        print(f"training_ready: {rollout.get('training_ready')}")
        print(f"decisions: {len(rollout.get('decisions', []))}")
        print(f"task_completed: {rollout.get('task_completed')}")
        print(f"evaluation_success: {rollout.get('evaluation_success')}")

        if rollout.get('decisions'):
            decision = rollout['decisions'][0]
            print(f"\nFirst decision:")
            print(f"  selected_skill_ids: {len(decision.get('selected_skill_ids', []))} skills")
            print(f"  candidate_skill_ids: {len(decision.get('candidate_skill_ids', []))} candidates")
            print(f"  ranking_mode: {decision.get('ranking_mode', 'N/A')}")
            print(f"  candidate_source: {decision.get('candidate_source', 'N/A')}")
            gate_decision = decision.get('evidence_gate_decision', 'N/A')
            if isinstance(gate_decision, dict):
                gate_decision = gate_decision.get('decision', 'N/A')
            print(f"  evidence_gate_decision: {gate_decision}")
    print()

    # Check for policy logprobs (if available)
    has_logprobs_flag = sum(1 for r in rollouts
                            for d in r.get('decisions', [])
                            if d.get('has_policy_logprobs', False))
    has_actual_logprobs = sum(1 for r in rollouts
                              for d in r.get('decisions', [])
                              if 'policy_logprobs' in d)
    print("=== Policy Logprobs (diagnostic only) ===")
    print(f"  Decisions with has_policy_logprobs=true: {has_logprobs_flag}/{total_decisions}")
    print(f"  Decisions with actual policy_logprobs data: {has_actual_logprobs}/{total_decisions}")
    if has_actual_logprobs == 0:
        print("  ⚠️  No policy logprobs found - will need to add controller logging for online RL")
    print()

    print("=" * 60)
    print("✅ Verification complete!")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Verify current_route_rollouts.jsonl structure'
    )
    parser.add_argument(
        '--rollouts_path',
        required=True,
        help='Path to current_route_rollouts.jsonl'
    )

    args = parser.parse_args()

    success = verify_rollouts(args.rollouts_path)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
