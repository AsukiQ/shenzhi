from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_APPWORLD_ROOT = Path("/data/home/scyb713/run/xzf/AAAI/autodl-tmp/appworld_root")
DEFAULT_SKILL_POOL = Path("data/appworld_skill_pool/skill_pool.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/appworld_routing")


@dataclass(frozen=True)
class ApiRef:
    app: str
    api: str
    method: str
    path: str

    @property
    def ref(self) -> str:
        return f"{self.app}.{self.api}"


def read_json(path: str | Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_split_task_ids(appworld_root: str | Path, split: str) -> list[str]:
    path = Path(appworld_root) / "data" / "datasets" / f"{split}.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_skill_pool(path: str | Path = DEFAULT_SKILL_POOL) -> list[dict[str, Any]]:
    return read_jsonl(path)


def _path_template_to_regex(template: str) -> re.Pattern[str]:
    escaped = re.escape(template)
    pattern = re.sub(r"\\\{[^/]+?\\\}", r"[^/]+", escaped)
    return re.compile(f"^{pattern}$")


def load_api_catalog(appworld_root: str | Path) -> list[ApiRef]:
    docs_dir = Path(appworld_root) / "data" / "api_docs" / "standard"
    refs: list[ApiRef] = []
    if not docs_dir.exists():
        return refs
    for path in sorted(docs_dir.glob("*.json")):
        if path.stem == "api_docs":
            continue
        data = read_json(path, default={}) or {}
        if not isinstance(data, dict):
            continue
        for api_name, spec in data.items():
            if not isinstance(spec, dict):
                continue
            app = str(spec.get("app_name") or path.stem)
            refs.append(
                ApiRef(
                    app=app,
                    api=str(spec.get("api_name") or api_name),
                    method=str(spec.get("method") or "").upper(),
                    path=str(spec.get("path") or ""),
                )
            )
    return refs


def match_api_call(call: dict[str, Any], catalog: Sequence[ApiRef]) -> ApiRef | None:
    method = str(call.get("method") or "").upper()
    url = str(call.get("url") or call.get("path") or "")
    for ref in catalog:
        if ref.method != method:
            continue
        if ref.path == url or _path_template_to_regex(ref.path).match(url):
            return ref
    parts = [part for part in url.split("/") if part]
    if not parts:
        return None
    return ApiRef(app=parts[0], api=parts[-1].replace("-", "_"), method=method, path=url)


def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        tokens.add(token)
        if token.endswith("s") and len(token) > 3:
            tokens.add(token[:-1])
    return tokens


def _skill_api_refs(skill: dict[str, Any]) -> set[str]:
    text = "\n".join(
        [
            str(skill.get("executor_desc", "")),
            str(skill.get("body", "")),
            str(skill.get("description", "")),
        ]
    )
    refs = {
        f"{match.group(1)}.{match.group(2)}"
        for match in re.finditer(r"apis\.([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)", text)
    }
    return refs


def _skill_apps(skill: dict[str, Any]) -> set[str]:
    apps = {ref.split(".", 1)[0] for ref in _skill_api_refs(skill)}
    text = " ".join(
        [
            str(skill.get("skill_id", "")),
            str(skill.get("name", "")),
            str(skill.get("executor_desc", "")),
        ]
    ).lower()
    for app in ("amazon", "file_system", "gmail", "phone", "simple_note", "splitwise", "spotify", "supervisor", "todoist", "venmo"):
        if app in text or app.replace("_", " ") in text:
            apps.add(app)
    return apps


def build_task_record(appworld_root: str | Path, task_id: str, split: str, catalog: Sequence[ApiRef]) -> dict[str, Any]:
    task_dir = Path(appworld_root) / "data" / "tasks" / task_id
    specs = read_json(task_dir / "specs.json", default={}) or {}
    gt_dir = task_dir / "ground_truth"
    required_apps = read_json(gt_dir / "required_apps.json", default=[]) or []
    if not isinstance(required_apps, list):
        required_apps = []
    calls = read_json(gt_dir / "api_calls.json", default=[]) or []
    if not isinstance(calls, list):
        calls = []
    matched_refs = [ref for call in calls if (ref := match_api_call(call, catalog)) is not None]
    api_refs = sorted({ref.ref for ref in matched_refs})
    api_apps = sorted({ref.app for ref in matched_refs})
    instruction = str(specs.get("instruction", ""))
    return {
        "task_id": task_id,
        "query_id": task_id,
        "split": split,
        "instruction_text": instruction,
        "required_apps": [str(app) for app in required_apps],
        "api_refs": api_refs,
        "api_apps": api_apps,
        "api_call_count": len(calls),
        "query": build_appworld_query_text(
            instruction=instruction,
            required_apps=[str(app) for app in required_apps],
            api_refs=api_refs,
            api_call_count=len(calls),
        ),
        "source": "appworld_ground_truth_api_trace_for_routing_supervision",
        "train_allowed": split == "train",
    }


def build_appworld_query_text(
    *,
    instruction: str,
    required_apps: Sequence[str],
    api_refs: Sequence[str],
    api_call_count: int,
) -> str:
    return "\n".join(
        [
            f"Instruction: {instruction}",
            f"Required apps: {', '.join(required_apps) if required_apps else 'unknown'}",
            f"Observed train-only API refs: {', '.join(api_refs[:32]) if api_refs else 'unknown'}",
            f"Observed API call count: {api_call_count}",
            "Route to the most relevant reusable AppWorld skill.",
        ]
    )


def score_skill_for_task(task: dict[str, Any], skill: dict[str, Any], index: int = 0) -> float:
    required_apps = {str(app) for app in task.get("required_apps", [])}
    api_apps = {str(app) for app in task.get("api_apps", [])}
    task_api_refs = {str(ref) for ref in task.get("api_refs", [])}
    skill_refs = _skill_api_refs(skill)
    skill_apps = _skill_apps(skill)

    score = 0.0
    score += 12.0 * len(task_api_refs & skill_refs)
    score += 4.0 * len((required_apps | api_apps) & skill_apps)

    query_tokens = _tokenize(str(task.get("instruction_text", "")))
    skill_tokens = _tokenize(
        "\n".join(
            [
                str(skill.get("name", "")),
                str(skill.get("description", "")),
                str(skill.get("executor_desc", "")),
                str(skill.get("body", "")),
            ]
        )
    )
    overlap = len(query_tokens & skill_tokens)
    denom = math.sqrt(max(len(query_tokens), 1) * max(len(skill_tokens), 1))
    score += overlap / denom if denom else 0.0
    score += 1.0e-9 * (100000 - index)
    return float(score)


def select_positive_skills(
    task: dict[str, Any],
    skills: Sequence[dict[str, Any]],
    positives_per_task: int = 5,
) -> list[dict[str, Any]]:
    scored = [
        (score_skill_for_task(task, skill, index), index, skill)
        for index, skill in enumerate(skills)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [skill for score, _index, skill in scored if score > 0.0][: max(1, positives_per_task)]
    if not selected and scored:
        selected = [scored[0][2]]
    return selected


def _build_replay_row(task: dict[str, Any], positives: Sequence[dict[str, Any]]) -> dict[str, Any]:
    api_refs = ", ".join(task.get("api_refs", [])[:12]) or "unknown"
    return {
        "task_id": task["task_id"],
        "query": task["query"],
        "instruction_text": task["instruction_text"],
        "split": task["split"],
        "success": True,
        "reward": 1.0,
        "steps": [
            {
                "skill_id": skill.get("skill_id"),
                "skill_name": skill.get("name"),
                "observation": f"matched AppWorld routing evidence: {api_refs}",
            }
            for skill in positives
        ],
    }


def build_appworld_routing_corpus(
    *,
    appworld_root: str | Path = DEFAULT_APPWORLD_ROOT,
    skill_pool_path: str | Path = DEFAULT_SKILL_POOL,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    splits: Sequence[str] = ("train", "dev", "test_normal", "test_challenge"),
    positives_per_task: int = 5,
    max_tasks_per_split: int | None = None,
) -> dict[str, Any]:
    appworld_root = Path(appworld_root)
    output_dir = Path(output_dir)
    skills = load_skill_pool(skill_pool_path)
    catalog = load_api_catalog(appworld_root)
    manifest: dict[str, Any] = {
        "status": "ok",
        "appworld_root": str(appworld_root),
        "skill_pool_path": str(skill_pool_path),
        "output_dir": str(output_dir),
        "skill_count": len(skills),
        "api_catalog_count": len(catalog),
        "positives_per_task": positives_per_task,
        "splits": {},
        "training_split": "train",
        "dev_test_used_for_training": False,
        "supervision_source": "AppWorld train split ground_truth api_calls are used only to build train routing positives.",
    }
    if not skills:
        manifest["status"] = "blocked"
        manifest["blocker"] = "skill_pool_empty"
        write_json(output_dir / "manifest.json", manifest)
        return manifest

    for split in splits:
        task_ids = load_split_task_ids(appworld_root, split)
        if max_tasks_per_split is not None:
            task_ids = task_ids[:max_tasks_per_split]
        task_rows: list[dict[str, Any]] = []
        qrel_rows: list[dict[str, Any]] = []
        replay_rows: list[dict[str, Any]] = []
        positive_counts: list[int] = []
        for task_id in task_ids:
            task = build_task_record(appworld_root, task_id, split, catalog)
            positives = select_positive_skills(task, skills, positives_per_task=positives_per_task)
            positive_ids = [str(skill.get("skill_id")) for skill in positives]
            task["positive_skill_ids"] = positive_ids
            task["positive_skill_names"] = [str(skill.get("name")) for skill in positives]
            task_rows.append(task)
            positive_counts.append(len(positive_ids))
            for skill_id in positive_ids:
                qrel_rows.append({"query_id": task_id, "skill_id": skill_id, "relevance": 1, "split": split})
            if split == "train":
                replay_rows.append(_build_replay_row(task, positives[: max(1, positives_per_task)]))

        task_count = write_jsonl(output_dir / f"{split}_tasks.jsonl", task_rows)
        qrel_count = write_jsonl(output_dir / f"{split}_qrels.jsonl", qrel_rows)
        replay_count = 0
        if split == "train":
            replay_count = write_jsonl(output_dir / "train_replay.jsonl", replay_rows)
        manifest["splits"][split] = {
            "task_count": task_count,
            "qrel_count": qrel_count,
            "replay_count": replay_count,
            "tasks_path": str(output_dir / f"{split}_tasks.jsonl"),
            "qrels_path": str(output_dir / f"{split}_qrels.jsonl"),
            "avg_positive_skills": round(sum(positive_counts) / max(len(positive_counts), 1), 3),
        }
    write_json(output_dir / "manifest.json", manifest)
    return manifest
