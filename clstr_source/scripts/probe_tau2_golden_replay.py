from __future__ import annotations

import argparse
import json

from clstr.tau2_tool_result_replay import replay_tau2_golden_tool_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau2_repo_root", required=True)
    parser.add_argument("--max_tasks_per_domain", type=int, default=2)
    args = parser.parse_args()
    _results, report = replay_tau2_golden_tool_results(
        args.tau2_repo_root,
        max_tasks_per_domain=args.max_tasks_per_domain,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
