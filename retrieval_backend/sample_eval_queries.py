"""Create a stable hash-selected evaluation subset, optionally stratified."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--per-stratum", type=int, default=100)
    parser.add_argument("--stratum-field", default="variant")
    parser.add_argument("--seed", default="shenzhi-eval-v1")
    args = parser.parse_args()
    buckets: dict[str, list[tuple[str, dict]]] = {}
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split") or "") != args.split:
                continue
            query_id = str(row.get("query_id") or "")
            if not query_id:
                continue
            stratum = str(row.get(args.stratum_field) or "default")
            digest = hashlib.sha256(f"{args.seed}|{query_id}".encode("utf-8")).hexdigest()
            buckets.setdefault(stratum, []).append((digest, row))
    selected = []
    counts = {}
    for stratum, rows in sorted(buckets.items()):
        rows.sort(key=lambda item: (item[0], str(item[1].get("query_id") or "")))
        chosen = rows[: max(1, int(args.per_stratum))]
        counts[stratum] = len(chosen)
        selected.extend((digest, row) for digest, row in chosen)
    selected.sort(key=lambda item: (item[0], str(item[1].get("query_id") or "")))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for _digest, row in selected),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "selection_protocol": "sha256_per_stratum_v1",
                "seed": args.seed,
                "split": args.split,
                "stratum_field": args.stratum_field,
                "per_stratum": args.per_stratum,
                "counts": counts,
                "row_count": len(selected),
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
