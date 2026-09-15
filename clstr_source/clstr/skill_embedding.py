from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from clstr.external_data import write_json, write_jsonl
from clstr.skillnet_aux_rebuild import load_skillnet_skills


VALID_OR_TEST_SPLITS = {"valid", "valid_seen", "valid_unseen", "test", "eval", "dev"}


def _split_for_row(row: dict[str, Any]) -> str:
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    return str(row.get("split") or provenance.get("split") or provenance.get("alfworld_split") or "train").lower()


def _is_train_row(row: dict[str, Any]) -> bool:
    split = _split_for_row(row)
    return split not in VALID_OR_TEST_SPLITS


def _unique_limited(values: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = " ".join(str(value or "").strip().split())
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def action_template(action: str) -> str:
    text = " ".join(str(action or "").strip().lower().split())
    if re.match(r"^go to .+ \d+$", text):
        return "go to <receptacle>"
    if re.match(r"^(open|close|examine|inspect) .+ \d+$", text):
        return f"{text.split()[0]} <receptacle>"
    if re.match(r"^(clean|heat|cool) .+ \d+ with .+ \d+$", text):
        return f"{text.split()[0]} <object> with <receptacle>"
    if re.match(r"^take .+ \d+ from .+ \d+$", text):
        return "take <object> from <receptacle>"
    if re.match(r"^put .+ \d+ (in|on|in/on) .+ \d+$", text):
        return "put <object> in/on <receptacle>"
    text = re.sub(r"\b[a-z]+(?:\s+[a-z]+)*\s+\d+\b", lambda m: _slot_phrase(m.group(0)), text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _slot_phrase(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    base = " ".join(words[:-1])
    if base in {"fridge", "sinkbasin", "microwave", "stoveburner", "toaster", "cabinet", "drawer", "safe"}:
        return "<receptacle>"
    if base in {"desk", "table", "diningtable", "countertop", "sidetable", "coffeetable", "shelf", "sofa", "bed"}:
        return "<receptacle>"
    return "<object>"


def _heuristic_preconditions_effects(skill_id: str, actions: list[str]) -> tuple[list[str], list[str]]:
    text = " ".join([skill_id, *actions]).lower()
    if "go to" in text or "navigator" in skill_id:
        return ["target receptacle or object is available in admissible actions"], ["agent location changes to target"]
    if "open " in text or "opener" in skill_id:
        return ["target receptacle is present and closed"], ["target receptacle becomes open and contents may be visible"]
    if "close " in text or "closer" in skill_id:
        return ["target receptacle is present and open"], ["target receptacle becomes closed"]
    if "take " in text or "picker" in skill_id or "retriever" in skill_id:
        return ["target object has been observed at source receptacle"], ["target object moves into inventory"]
    if "put " in text or "placer" in skill_id or "storer" in skill_id:
        return ["target object is in inventory and destination receptacle is reachable"], ["target object is placed at destination"]
    if "clean " in text:
        return ["target object is in inventory and cleaning receptacle is reachable"], ["target object becomes clean"]
    if "heat " in text:
        return ["target object is in inventory and heating appliance is reachable"], ["target object becomes heated"]
    if "cool " in text:
        return ["target object is in inventory and cooling appliance is reachable"], ["target object becomes cooled"]
    if "look" in text or "examine" in text:
        return ["target object or receptacle is visible or reachable"], ["observation reveals target state or contents"]
    return ["candidate action is legal in current admissible actions"], ["environment returns next observation"]


def _skillnet_body_map(skillnet_root: str | Path | None) -> dict[str, str]:
    if not skillnet_root:
        return {}
    root = Path(skillnet_root)
    if not root.exists():
        return {}
    body_map: dict[str, str] = {}
    for env_skills in load_skillnet_skills(root).values():
        for skill_id, skill in env_skills.items():
            body = str(skill.get("body") or "").strip()
            if body:
                body_map[str(skill_id)] = body
    return body_map


def enrich_skill_rows_for_embedding(
    skill_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    skillnet_root: str | Path | None = None,
    max_examples_per_skill: int = 16,
    max_negative_examples_per_skill: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    body_map = _skillnet_body_map(skillnet_root)
    positives: dict[str, list[str]] = defaultdict(list)
    negatives: dict[str, list[str]] = defaultdict(list)
    skipped = Counter()
    for row in train_rows:
        if not _is_train_row(row):
            skipped["non_train_split"] += 1
            continue
        skill_id = str(row.get("skill_id") or "").strip()
        if not skill_id:
            skipped["missing_skill_id"] += 1
            continue
        expert = str(row.get("expert_action") or row.get("expert_action_t") or row.get("action_text") or "").strip()
        if expert:
            positives[skill_id].append(expert)
        hard_negative = str(row.get("hard_negative_action") or row.get("qwen_action_t") or "").strip()
        if hard_negative and hard_negative != expert:
            negatives[skill_id].append(hard_negative)

    enriched: list[dict[str, Any]] = []
    train_only_action_example_count = 0
    for skill in skill_rows:
        row = dict(skill)
        skill_id = str(row.get("skill_id") or "").strip()
        if not row.get("body") and skill_id in body_map:
            row["body"] = body_map[skill_id]
            row["body_source"] = "skillnet_skill_md"
        positive_examples = _unique_limited(positives.get(skill_id, []), max_examples_per_skill)
        negative_examples = _unique_limited(negatives.get(skill_id, []), max_negative_examples_per_skill)
        templates = _unique_limited([action_template(action) for action in positive_examples], max_examples_per_skill)
        preconditions, effects = _heuristic_preconditions_effects(skill_id, positive_examples)
        row["positive_action_examples"] = positive_examples
        row["negative_action_examples"] = negative_examples
        row["action_templates"] = templates
        row["preconditions"] = preconditions
        row["effects"] = effects
        row["skill_embedding_schema_version"] = "clstr_enriched_v1"
        row["skill_embedding_fields"] = [
            "name",
            "description",
            "body",
            "action_templates",
            "positive_action_examples",
            "negative_action_examples",
            "preconditions",
            "effects",
        ]
        train_only_action_example_count += len(positive_examples)
        enriched.append(row)

    report = {
        "status": "ok",
        "skill_count": len(enriched),
        "skills_with_body": sum(1 for row in enriched if row.get("body")),
        "skills_with_positive_action_examples": sum(1 for row in enriched if row.get("positive_action_examples")),
        "train_only_action_example_count": train_only_action_example_count,
        "skipped_action_rows": dict(sorted(skipped.items())),
        "valid_or_test_used_for_skill_enrichment": False,
        "skillnet_root": str(skillnet_root) if skillnet_root else None,
        "schema_version": "clstr_enriched_v1",
    }
    return sorted(enriched, key=lambda row: str(row.get("skill_id") or "")), report


def build_enriched_skill_embedding_data(
    skills_path: str | Path,
    train_path: str | Path,
    output_dir: str | Path,
    report_output_dir: str | Path,
    skillnet_root: str | Path | None = None,
) -> dict[str, Any]:
    skills = [__import__("json").loads(line) for line in Path(skills_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [__import__("json").loads(line) for line in Path(train_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    enriched, report = enrich_skill_rows_for_embedding(skills, rows, skillnet_root=skillnet_root)
    output_dir = Path(output_dir)
    report_output_dir = Path(report_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir.mkdir(parents=True, exist_ok=True)
    out_skills = output_dir / "skills.jsonl"
    write_jsonl(out_skills, enriched)
    report.update(
        {
            "input_skills_path": str(skills_path),
            "input_train_path": str(train_path),
            "output_skills_path": str(out_skills),
            "data_role": "skill_embedding_enrichment_train_split_only",
        }
    )
    write_json(report_output_dir / "skill_embedding_report.json", report)
    return report
