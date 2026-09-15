from __future__ import annotations

import json
from pathlib import Path


def write_retrieval_predictions(
    predictions: dict[str | int, list[str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {str(key): value for key, value in predictions.items()}
    output_path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
