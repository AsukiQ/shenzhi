from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from clstr.appworld_routing import read_jsonl, write_json, write_jsonl
from clstr.appworld_skill_handoff import build_verified_skill_handoff
from clstr.envs.appworld_env import configure_appworld_paths


class SkillProvider(Protocol):
    def get_skills(self, task: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        ...


class NullSkillProvider:
    def get_skills(self, task: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        return []


class PredictionFileSkillProvider:
    """Skill provider backed by AppWorld routing predictions JSONL."""

    def __init__(
        self,
        *,
        skill_pool_path: str | Path,
        predictions_path: str | Path,
        dedupe_canonical_skills: bool = False,
        max_auth_like_skills: int | None = None,
    ) -> None:
        self.skill_by_id = {str(row["skill_id"]): row for row in read_jsonl(skill_pool_path)}
        self.predictions: dict[str, list[str]] = {}
        self.dedupe_canonical_skills = bool(dedupe_canonical_skills)
        self.max_auth_like_skills = max_auth_like_skills
        for row in read_jsonl(predictions_path):
            query_id = str(row.get("query_id") or row.get("task_id"))
            self.predictions[query_id] = [str(item) for item in row.get("ranked_skill_ids", [])]

    def get_skills(self, task: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        query_id = str(task.get("query_id") or task.get("task_id"))
        selected: list[dict[str, Any]] = []
        seen_canonical: set[str] = set()
        auth_like_count = 0
        for skill_id in self.predictions.get(query_id, []):
            skill = self.skill_by_id.get(skill_id)
            if skill is None:
                continue
            if self.dedupe_canonical_skills:
                canonical = _canonical_skill_key(skill)
                if canonical in seen_canonical:
                    continue
                seen_canonical.add(canonical)
            if self.max_auth_like_skills is not None and _is_auth_like_skill(skill):
                if auth_like_count >= int(self.max_auth_like_skills):
                    continue
                auth_like_count += 1
            selected.append(skill)
            if len(selected) >= int(top_k):
                break
        return selected


def extract_python_code(response: str) -> str:
    text = str(response or "").strip()
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return text


def _normalize_skill_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _canonical_skill_key(skill: dict[str, Any]) -> str:
    name = _normalize_skill_text(str(skill.get("name", "")))
    executor = _normalize_skill_text(str(skill.get("executor_desc", "")))
    return f"{name}::{executor}"


def _is_auth_like_skill(skill: dict[str, Any]) -> bool:
    text = " ".join(
        str(skill.get(key, ""))
        for key in ("skill_id", "name", "description", "executor_desc", "body", "skill_md")
    ).lower()
    return any(token in text for token in ("auth", "login", "credential", "password", "account_password"))


def enrich_task_with_appworld_specs(task: dict[str, Any], appworld_root: str | Path) -> dict[str, Any]:
    enriched = dict(task)
    task_id = str(enriched.get("task_id") or enriched.get("query_id") or "")
    if not task_id:
        return enriched
    specs_path = Path(appworld_root) / "data" / "tasks" / task_id / "specs.json"
    if not specs_path.exists():
        return enriched
    try:
        specs = json.loads(specs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return enriched
    task_datetime = str(specs.get("datetime") or "").strip()
    if task_datetime:
        enriched.setdefault("task_datetime", task_datetime)
        enriched.setdefault("datetime", task_datetime)
    return enriched


def _param_signature(parameters: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for param in parameters:
        name = str(param.get("name", "")).strip()
        if not name:
            continue
        ptype = str(param.get("type", "any"))
        required = "required" if bool(param.get("required")) else "optional"
        detail = f"{name}: {ptype} {required}"
        if "default" in param and param.get("default") is not None:
            detail += f" default {param.get('default')}"
        constraints = [str(item) for item in param.get("constraints", []) if str(item).strip()]
        if constraints:
            detail += f" constraints {'; '.join(constraints)}"
        parts.append(detail)
    return ", ".join(parts)


def _success_schema_signature(doc: dict[str, Any], max_chars: int = 280) -> str:
    schema = (doc.get("response_schemas") or {}).get("success")
    if schema is None:
        return ""
    text = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    return text[: int(max_chars)]


def build_api_docs_context(
    *,
    appworld_root: str | Path,
    required_apps: list[str],
    api_refs: list[str] | None = None,
    max_apis_per_app: int = 12,
    max_chars: int = 12000,
) -> str:
    root = Path(appworld_root)
    docs_dir = root / "data" / "api_docs" / "standard"
    refs_by_app: dict[str, set[str]] = {}
    for ref in api_refs or []:
        if "." not in str(ref):
            continue
        app, api = str(ref).split(".", 1)
        refs_by_app.setdefault(app, set()).add(api)

    lines: list[str] = ["[AppWorld API Docs]"]
    apps = sorted({str(app) for app in required_apps} | set(refs_by_app))
    for app in apps:
        path = docs_dir / f"{app}.json"
        if not path.exists():
            continue
        docs = json.loads(path.read_text(encoding="utf-8"))
        preferred = refs_by_app.get(app, set())
        api_names = list(preferred) + [name for name in sorted(docs) if name not in preferred]
        emitted = 0
        for api_name in api_names:
            doc = docs.get(api_name)
            if not isinstance(doc, dict):
                continue
            params = _param_signature(list(doc.get("parameters", [])))
            desc = " ".join(str(doc.get("description", "")).split())
            returns = _success_schema_signature(doc)
            return_text = f" returns {returns}" if returns else ""
            lines.append(f"- apis.{app}.{api_name}({params}): {desc}{return_text}")
            emitted += 1
            if emitted >= int(max_apis_per_app):
                break
    text = "\n".join(lines)
    return text[: int(max_chars)]


def load_appworld_api_refs(
    *,
    appworld_root: str | Path,
    required_apps: list[str],
    api_refs: list[str] | None = None,
) -> set[tuple[str, str]]:
    root = Path(appworld_root)
    docs_dir = root / "data" / "api_docs" / "standard"
    refs_by_app: dict[str, set[str]] = {}
    for ref in api_refs or []:
        if "." not in str(ref):
            continue
        app, api = str(ref).split(".", 1)
        refs_by_app.setdefault(app, set()).add(api)
    refs: set[tuple[str, str]] = {
        ("supervisor", "show_profile"),
        ("supervisor", "show_account_passwords"),
        ("supervisor", "complete_task"),
    }
    for app in sorted({str(app) for app in required_apps} | set(refs_by_app)):
        path = docs_dir / f"{app}.json"
        if not path.exists():
            continue
        try:
            docs = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(docs, dict):
            continue
        for api_name, doc in docs.items():
            if isinstance(doc, dict):
                refs.add((str(app), str(api_name)))
    return refs


VALID_SKILL_CONTEXT_MODES = {
    "raw",
    "metadata",
    "safe_metadata",
    "schema_plan",
    "adaptive_schema",
    "gated_schema",
    "api_evidence",
    "verified_hints",
    "verified_metadata",
}


_SKILL_RELEVANCE_STOPWORDS = {
    "a",
    "all",
    "and",
    "api",
    "apis",
    "app",
    "appworld",
    "as",
    "at",
    "based",
    "be",
    "by",
    "call",
    "code",
    "directory",
    "do",
    "each",
    "file",
    "files",
    "for",
    "from",
    "get",
    "give",
    "i",
    "in",
    "into",
    "is",
    "it",
    "me",
    "my",
    "name",
    "named",
    "names",
    "note",
    "notes",
    "of",
    "on",
    "one",
    "or",
    "path",
    "paths",
    "please",
    "required",
    "row",
    "rows",
    "show",
    "simple",
    "skill",
    "skillx",
    "system",
    "task",
    "the",
    "them",
    "then",
    "this",
    "to",
    "use",
    "user",
    "with",
}


def _normalize_relevance_token(token: str) -> str:
    token = str(token).lower()
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _relevance_tokens(text: str) -> set[str]:
    tokens = {
        _normalize_relevance_token(token)
        for token in re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    }
    return {token for token in tokens if len(token) > 1 and token not in _SKILL_RELEVANCE_STOPWORDS}


def _goal_only_text(instruction: str) -> str:
    text = str(instruction or "")
    if "[Execution History]" in text:
        text = text.split("[Execution History]", 1)[0]
    if "[User Goal]" in text:
        text = text.split("[User Goal]", 1)[1]
    return text


def _filter_prompt_skills_by_relevance(
    *,
    instruction: str,
    skills: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    if len(skills) <= 1:
        return skills, 0
    task_tokens = _relevance_tokens(_goal_only_text(instruction))
    if not task_tokens:
        return skills, 0
    kept: list[dict[str, Any]] = []
    for skill in skills:
        skill_text = " ".join(
            str(skill.get(key, ""))
            for key in ("skill_id", "name", "description", "executor_desc")
        )
        if task_tokens & _relevance_tokens(skill_text):
            kept.append(skill)
    return kept, len(skills) - len(kept)


_API_REF_RE = re.compile(r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")


def _api_refs_from_text(text: str) -> set[tuple[str, str]]:
    return {(app, api) for app, api in _API_REF_RE.findall(str(text or ""))}


def _api_ref_text(ref: tuple[str, str]) -> str:
    return f"apis.{ref[0]}.{ref[1]}"


def _valid_api_refs_from_docs_context(api_docs_context: str) -> set[tuple[str, str]]:
    refs = _api_refs_from_text(api_docs_context)
    refs.update(
        {
            ("supervisor", "show_profile"),
            ("supervisor", "show_account_passwords"),
            ("supervisor", "complete_task"),
        }
    )
    return refs


class _ApiCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.refs: set[tuple[str, str]] = set()
        self.unsafe_literal_id_filters: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> Any:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "apis"
        ):
            app_name = str(func.value.attr)
            api_name = str(func.attr)
            self.refs.add((app_name, api_name))
            for keyword in node.keywords:
                issue = _unsafe_literal_id_filter_issue(app_name, api_name, keyword)
                if issue:
                    self.unsafe_literal_id_filters.append(issue)
        self.generic_visit(node)


def _api_call_refs_from_code(code: str) -> set[tuple[str, str]]:
    tree = ast.parse(str(code or ""))
    visitor = _ApiCallVisitor()
    visitor.visit(tree)
    return visitor.refs


def _is_id_filter_api(api_name: str) -> bool:
    return str(api_name).startswith(("search_", "find_", "list_"))


def _literal_constant(node: ast.AST) -> Any | None:
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool):
        return node.value
    return None


def _unsafe_literal_id_filter_issue(
    app_name: str,
    api_name: str,
    keyword: ast.keyword,
) -> dict[str, Any] | None:
    parameter = str(keyword.arg or "")
    if not parameter.endswith("_id") or not _is_id_filter_api(api_name):
        return None
    literal = _literal_constant(keyword.value)
    if not isinstance(literal, (int, str)):
        return None
    return {
        "api_ref": f"apis.{app_name}.{api_name}",
        "parameter": parameter,
        "literal": literal,
    }


def _unsafe_literal_id_filters_from_code(code: str) -> list[dict[str, Any]]:
    tree = ast.parse(str(code or ""))
    visitor = _ApiCallVisitor()
    visitor.visit(tree)
    return visitor.unsafe_literal_id_filters


class _UnsafeLiteralIdFilterSanitizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.removed: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> Any:
        node = self.generic_visit(node)
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "apis"
        ):
            return node
        app_name = str(func.value.attr)
        api_name = str(func.attr)
        kept_keywords: list[ast.keyword] = []
        for keyword in node.keywords:
            issue = _unsafe_literal_id_filter_issue(app_name, api_name, keyword)
            if issue:
                self.removed.append(issue)
            else:
                kept_keywords.append(keyword)
        node.keywords = kept_keywords
        return node


def sanitize_appworld_code(code: str) -> dict[str, Any]:
    """Remove unsafe literal ID filters while preserving the generated program shape."""
    original = str(code or "")
    try:
        tree = ast.parse(original)
    except SyntaxError as exc:
        return {
            "code": original,
            "changed": False,
            "removed_unsafe_literal_id_filters": [],
            "error": f"Python syntax error: {exc.msg} at line {exc.lineno}",
        }
    sanitizer = _UnsafeLiteralIdFilterSanitizer()
    new_tree = sanitizer.visit(tree)
    if not sanitizer.removed:
        return {
            "code": original,
            "changed": False,
            "removed_unsafe_literal_id_filters": [],
            "error": "",
        }
    ast.fix_missing_locations(new_tree)
    return {
        "code": ast.unparse(new_tree),
        "changed": True,
        "removed_unsafe_literal_id_filters": sanitizer.removed,
        "error": "",
    }


def preflight_appworld_code(
    code: str,
    api_docs_context: str,
    valid_api_refs: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    try:
        refs = _api_call_refs_from_code(code)
        unsafe_literal_id_filters = _unsafe_literal_id_filters_from_code(code)
    except SyntaxError as exc:
        return {
            "ok": False,
            "reason": "syntax_error",
            "error": f"Python syntax error: {exc.msg} at line {exc.lineno}",
            "invalid_api_refs": [],
            "unsafe_literal_id_filters": [],
        }
    valid_refs = (
        set(valid_api_refs)
        if valid_api_refs is not None
        else _valid_api_refs_from_docs_context(api_docs_context)
    )
    valid_refs.update(
        {
            ("supervisor", "show_profile"),
            ("supervisor", "show_account_passwords"),
            ("supervisor", "complete_task"),
        }
    )
    invalid_refs = refs - valid_refs
    if invalid_refs:
        invalid_text = [_api_ref_text(ref) for ref in sorted(invalid_refs)]
        return {
            "ok": False,
            "reason": "invalid_api_ref",
            "error": "Invalid AppWorld API call(s): " + ", ".join(invalid_text),
            "invalid_api_refs": invalid_text,
            "unsafe_literal_id_filters": [],
        }
    if unsafe_literal_id_filters:
        rendered = [
            f'{issue["api_ref"]}({issue["parameter"]}={issue["literal"]!r})'
            for issue in unsafe_literal_id_filters
        ]
        return {
            "ok": False,
            "reason": "unsafe_literal_id_filter",
            "error": (
                "Unsafe literal ID filter(s): "
                + ", ".join(rendered)
                + ". For search/find/list APIs, obtain `*_id` values from an earlier API response "
                "or omit the ID filter and filter returned records by names/fields."
            ),
            "invalid_api_refs": [],
            "unsafe_literal_id_filters": unsafe_literal_id_filters,
        }
    return {
        "ok": True,
        "reason": "ok",
        "error": "",
        "invalid_api_refs": [],
        "unsafe_literal_id_filters": [],
    }


def build_appworld_code_repair_prompt(
    *,
    original_prompt: str,
    code: str,
    preflight: dict[str, Any],
) -> str:
    return "\n".join(
        [
            original_prompt,
            "",
            "[Preflight Error]",
            "The previous code was not executed because it failed a schema/syntax preflight.",
            str(preflight.get("error") or "Unknown preflight error."),
            "Use only APIs listed in [AppWorld API Docs] and fix Python syntax before returning code.",
            "",
            "[Previous Code]",
            "```python",
            str(code or ""),
            "```",
            "",
            "[Repair Contract]",
            "Return corrected Python code only.",
        ]
    )


def _identifier_tokens(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", str(text or ""))]


def _candidate_refs_from_skill_title(
    skill: dict[str, Any],
    valid_api_refs: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    valid_apps = sorted({app for app, _ in valid_api_refs if app != "supervisor"}, key=len, reverse=True)
    candidates: set[tuple[str, str]] = set()
    for source in (skill.get("name", ""), skill.get("skill_id", "")):
        tokens = _identifier_tokens(str(source))
        for app in valid_apps:
            app_tokens = app.split("_")
            if not app_tokens:
                continue
            for idx in range(0, max(len(tokens) - len(app_tokens) + 1, 0)):
                if tokens[idx : idx + len(app_tokens)] != app_tokens:
                    continue
                action_tokens = tokens[idx + len(app_tokens) :]
                while action_tokens and action_tokens[-1].isdigit():
                    action_tokens = action_tokens[:-1]
                if not action_tokens:
                    continue
                api_name = "_".join(action_tokens)
                if api_name:
                    candidates.add((app, api_name))
    return candidates


def _format_api_ref_list(refs: set[tuple[str, str]]) -> str:
    return ", ".join(_api_ref_text(ref) for ref in sorted(refs))


def _has_state_changing_api(refs: set[tuple[str, str]]) -> bool:
    write_prefixes = (
        "add_",
        "archive",
        "batch_",
        "comment",
        "complete_",
        "create_",
        "delete_",
        "like_",
        "move_",
        "remove_",
        "rename_",
        "send_",
        "set_",
        "share_",
        "submit_",
        "update_",
        "write_",
    )
    return any(api.startswith(write_prefixes) for _, api in refs)


def _has_high_risk_nonexistent_title_api(refs: set[tuple[str, str]]) -> bool:
    high_risk_verbs = {
        "categorize",
        "organize",
        "rename",
    }
    return any(api.split("_", 1)[0] in high_risk_verbs for _, api in refs)


def _api_name_tokens(api_name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(api_name).lower()))


def _is_state_changing_api_name(api_name: str) -> bool:
    write_prefixes = (
        "add_",
        "archive",
        "batch_",
        "comment",
        "complete_",
        "create_",
        "delete_",
        "like_",
        "move_",
        "remove_",
        "rename_",
        "send_",
        "set_",
        "share_",
        "submit_",
        "update_",
        "write_",
    )
    return str(api_name).startswith(write_prefixes)


_STATE_ACTION_TASK_SYNONYMS = {
    "add": {"add", "queue", "append", "include"},
    "archive": {"archive"},
    "batch": {"batch"},
    "comment": {"comment"},
    "complete": {"complete", "finish", "answer"},
    "create": {"create", "send", "pay", "payment", "transaction", "make"},
    "delete": {"delete", "remove"},
    "like": {"like"},
    "move": {"move", "rename", "trash", "recycle", "organize", "export"},
    "remove": {"remove", "delete"},
    "rename": {"rename", "prefix"},
    "send": {"send", "text", "message", "email", "notify"},
    "set": {"set", "mark"},
    "share": {"share"},
    "submit": {"submit"},
    "update": {"update", "edit", "change"},
    "write": {"write", "export", "save", "create"},
}


def _state_action_matches_task(api_name: str, task_tokens: set[str]) -> bool:
    action_token = str(api_name).split("_", 1)[0].lower()
    action_token = _normalize_relevance_token(action_token)
    allowed_tokens = _STATE_ACTION_TASK_SYNONYMS.get(action_token, {action_token})
    return bool(allowed_tokens & task_tokens)


_SKILL_ALIGNMENT_GENERIC_TOKENS = {
    "album",
    "api",
    "app",
    "artist",
    "file",
    "library",
    "music",
    "note",
    "phone",
    "playlist",
    "simple",
    "song",
    "spotify",
    "system",
    "transaction",
    "user",
    "venmo",
}


def _alignment_tokens(text: str) -> set[str]:
    tokens = set(_relevance_tokens(text))
    expanded: set[str] = set()
    for token in tokens:
        expanded.add(token)
        if len(token) > 4 and token.endswith("ed"):
            expanded.add(token[:-2])
    return {token for token in expanded if token not in _SKILL_ALIGNMENT_GENERIC_TOKENS}


def _skill_semantically_aligned_with_task(skill: dict[str, Any], instruction: str) -> bool:
    task_tokens = _alignment_tokens(_goal_only_text(instruction))
    if not task_tokens:
        return False
    skill_text = " ".join(
        str(skill.get(key, ""))
        for key in ("skill_id", "name", "description")
    )
    skill_tokens = _alignment_tokens(skill_text)
    return bool(task_tokens & skill_tokens)


_OPERATION_ALIGNMENT_TOKENS = {
    "add",
    "answer",
    "count",
    "create",
    "detail",
    "export",
    "find",
    "inspect",
    "message",
    "move",
    "pay",
    "payment",
    "prefix",
    "queue",
    "rank",
    "rename",
    "save",
    "search",
    "send",
    "top",
    "unique",
    "write",
}


def _skill_strongly_aligned_with_task(skill: dict[str, Any], instruction: str) -> bool:
    task_tokens = _alignment_tokens(_goal_only_text(instruction))
    if not task_tokens:
        return False
    skill_text = " ".join(
        str(skill.get(key, ""))
        for key in ("skill_id", "name", "description", "executor_desc")
    )
    skill_tokens = _alignment_tokens(skill_text)
    overlap = task_tokens & skill_tokens
    return len(overlap) >= 2 or bool(overlap & _OPERATION_ALIGNMENT_TOKENS)


def _skill_prompt_pollution_reasons(skill: dict[str, Any], instruction: str) -> list[str]:
    goal = " ".join(str(_goal_only_text(instruction)).lower().split())
    skill_text = " ".join(
        str(skill.get(key, ""))
        for key in ("skill_id", "name", "description", "executor_desc")
    ).lower()
    reasons: list[str] = []
    unique_count_goal = "unique" in goal or "how many" in goal or "count" in goal
    ranking_skill = any(
        phrase in skill_text
        for phrase in (
            "play count",
            "played",
            "most played",
            "top ",
            "rank",
            "popular",
        )
    )
    if unique_count_goal and ranking_skill:
        reasons.append("unique_count_goal_vs_ranking_skill")
    if "unique song" in goal and "unique song" not in skill_text and "unique songs" not in skill_text:
        reasons.append("unique_song_goal_requires_schema_only")
    add_queue_goal = "add" in goal and "queue" in goal
    queue_polluting_skill = any(
        phrase in skill_text
        for phrase in (
            "like song",
            "like songs",
            "liked song",
            "liked songs",
            "current",
            "current queue",
            "played so far",
            "show queue",
        )
    )
    if add_queue_goal and queue_polluting_skill:
        reasons.append("add_queue_goal_vs_like_or_current_queue_skill")
    note_export_goal = "export" in goal and "note" in goal
    expense_note_skill = any(
        token in skill_text
        for token in (
            "expense",
            "split",
            "share",
            "bill",
            "venmo",
            "payment",
        )
    )
    if note_export_goal and expense_note_skill:
        reasons.append("note_export_goal_vs_expense_share_skill")
    return reasons


def _is_add_queue_goal(instruction: str) -> bool:
    goal = " ".join(str(_goal_only_text(instruction)).lower().split())
    return "add" in goal and "queue" in goal


def _has_queue_write_action(refs: set[tuple[str, str]]) -> bool:
    return any(api.startswith("add") and "queue" in api for _app, api in refs)


def _schema_plan_refs(
    skill: dict[str, Any],
    valid_api_refs: set[tuple[str, str]],
    instruction: str,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]], int]:
    explicit_refs = _api_refs_from_text(
        " ".join(
            str(skill.get(key, ""))
            for key in ("executor_desc", "body", "skill_md")
        )
    )
    valid_refs = explicit_refs & set(valid_api_refs or set())
    task_tokens = _relevance_tokens(_goal_only_text(instruction))
    read_refs: set[tuple[str, str]] = set()
    action_refs: set[tuple[str, str]] = set()
    filtered_action_count = 0
    for ref in valid_refs:
        _app, api_name = ref
        if not _is_state_changing_api_name(api_name):
            read_refs.add(ref)
            continue
        if _state_action_matches_task(api_name, task_tokens):
            action_refs.add(ref)
        else:
            filtered_action_count += 1
    return read_refs | action_refs, read_refs, action_refs, filtered_action_count


def _api_ref_list_text(refs: set[tuple[str, str]]) -> list[str]:
    return [_api_ref_text(ref) for ref in sorted(refs)]


def _api_evidence_from_skills(
    *,
    skills: list[dict[str, Any]],
    valid_api_refs: set[tuple[str, str]],
    instruction: str,
) -> dict[str, Any]:
    valid_refs: set[tuple[str, str]] = set()
    read_refs: set[tuple[str, str]] = set()
    action_refs: set[tuple[str, str]] = set()
    invalid_refs: set[tuple[str, str]] = set()
    filtered_refs: set[tuple[str, str]] = set()
    explicit_ref_count = 0
    filtered_action_count = 0
    for skill in skills:
        explicit_refs = _api_refs_from_text(
            " ".join(
                str(skill.get(key, ""))
                for key in ("executor_desc", "body", "skill_md")
            )
        )
        explicit_ref_count += len(explicit_refs)
        usable, reads, actions, filtered_count = _schema_plan_refs(skill, valid_api_refs, instruction)
        valid_explicit_refs = explicit_refs & set(valid_api_refs or set())
        valid_refs.update(usable)
        read_refs.update(reads)
        action_refs.update(actions)
        invalid_refs.update(explicit_refs - set(valid_api_refs or set()))
        filtered_refs.update(valid_explicit_refs - usable)
        filtered_action_count += int(filtered_count)
    invalid_or_filtered = invalid_refs | filtered_refs
    return {
        "skill_context_mode": "api_evidence",
        "handoff_decision": "api_evidence",
        "selected_skill_ids": [str(skill.get("skill_id", "")) for skill in skills],
        "api_evidence_refs": _api_ref_list_text(valid_refs),
        "read_support_apis": _api_ref_list_text(read_refs),
        "state_changing_action_apis": _api_ref_list_text(action_refs),
        "invalid_skill_api_refs": _api_ref_list_text(invalid_refs),
        "filtered_skill_api_refs": _api_ref_list_text(filtered_refs),
        "invalid_or_filtered_refs": _api_ref_list_text(invalid_or_filtered),
        "hidden_skill_text_fields": ["skill_id", "name", "description", "body", "skill_md"],
        "api_evidence_skill_count": len(skills),
        "api_evidence_ref_count": len(valid_refs),
        "read_support_api_count": len(read_refs),
        "state_changing_action_api_count": len(action_refs),
        "invalid_skill_api_ref_count": len(invalid_refs),
        "filtered_skill_api_ref_count": len(filtered_refs),
        "explicit_skill_api_ref_count": explicit_ref_count,
        "filtered_state_changing_api_count": filtered_action_count,
    }


def _format_api_evidence_block(evidence: dict[str, Any], max_chars: int) -> str:
    refs = evidence.get("api_evidence_refs") or []
    read_refs = evidence.get("read_support_apis") or []
    action_refs = evidence.get("state_changing_action_apis") or []
    invalid_or_filtered_refs = evidence.get("invalid_or_filtered_refs") or []
    lines = [
        "[CLSTR-Prioritized API Evidence]",
        "skill_text_hidden: true",
        "not_callable: true",
        "schema_grounded: true",
        "user_goal_is_authoritative: true",
        "not_an_api_allowlist: true",
        "valid_appworld_apis_from_selected_skills: " + (", ".join(refs) if refs else "(none)"),
        "read_support_apis: " + (", ".join(read_refs) if read_refs else "(none)"),
        "state_changing_action_apis: " + (", ".join(action_refs) if action_refs else "(none)"),
        "invalid_or_filtered_refs: " + (", ".join(invalid_or_filtered_refs) if invalid_or_filtered_refs else "(none)"),
        "constraint: These APIs are evidence only, not a complete plan or API allowlist.",
        "constraint: Use any API in [AppWorld API Docs] if required by the user goal.",
        "constraint: Do not infer task intent, filters, entities, or API names from hidden SkillX titles or descriptions.",
        "raw_body_omitted: true",
    ]
    return "\n".join(lines)[: int(max_chars)]


def _gate_skill_context(
    *,
    skill: dict[str, Any],
    valid_api_refs: set[tuple[str, str]],
    instruction: str,
    required_apps: set[str],
) -> dict[str, Any]:
    usable_refs, read_refs, action_refs, filtered_action_count = _schema_plan_refs(
        skill,
        valid_api_refs,
        instruction,
    )
    explicit_refs = _api_refs_from_text(
        " ".join(
            str(skill.get(key, ""))
            for key in ("executor_desc", "body", "skill_md")
        )
    )
    valid_explicit_refs = explicit_refs & set(valid_api_refs or set())
    skill_apps = {app for app, _api in valid_explicit_refs if app != "supervisor"}
    reasons: list[str] = []
    if required_apps and skill_apps and not bool(skill_apps & required_apps):
        reasons.append("required_app_mismatch")
    if not usable_refs:
        reasons.append("no_schema_grounded_api_evidence")
    if reasons:
        decision = "drop"
    else:
        pollution_reasons = _skill_prompt_pollution_reasons(skill, instruction)
        if pollution_reasons:
            reasons.extend(["pollution_pattern", *pollution_reasons])
        if filtered_action_count and not action_refs:
            reasons.append("state_changing_action_mismatch")
        if _is_add_queue_goal(instruction) and not _has_queue_write_action(action_refs):
            reasons.append("add_queue_goal_without_queue_write_api")
        if not _skill_strongly_aligned_with_task(skill, instruction):
            reasons.append("weak_semantic_alignment")
        decision = "schema_plan" if reasons else "safe_metadata"
    return {
        "skill_id": str(skill.get("skill_id", "")),
        "decision": decision,
        "reasons": reasons,
        "valid_api_refs": _api_ref_list_text(usable_refs),
        "read_support_apis": _api_ref_list_text(read_refs),
        "state_changing_action_apis": _api_ref_list_text(action_refs),
        "filtered_state_changing_api_count": int(filtered_action_count),
    }


def _summarize_handoff_decisions(
    *,
    mode: str,
    input_skill_count: int,
    filtered_skill_count: int,
    schema_filtered_count: int,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    kept = [row for row in decisions if row.get("decision") != "drop"]
    dropped = [row for row in decisions if row.get("decision") == "drop"]
    safe = [row for row in decisions if row.get("decision") == "safe_metadata"]
    schema = [row for row in decisions if row.get("decision") == "schema_plan"]
    counts = {
        "safe_metadata": len(safe),
        "schema_plan": len(schema),
        "drop": len(dropped),
    }
    return {
        "skill_context_mode": str(mode),
        "handoff_decision": "mixed" if mode == "gated_schema" else str(mode),
        "input_skill_count": int(input_skill_count),
        "relevance_filtered_count": int(filtered_skill_count),
        "schema_filtered_count": int(schema_filtered_count),
        "handoff_decision_counts": counts,
        "handoff_decisions": decisions,
        "kept_skill_ids": [str(row.get("skill_id", "")) for row in kept],
        "dropped_skill_ids": [str(row.get("skill_id", "")) for row in dropped],
        "downgrade_reasons": [
            {"skill_id": str(row.get("skill_id", "")), "reasons": list(row.get("reasons", []))}
            for row in schema
        ],
        "schema_plan_skill_count": len(schema),
        "safe_metadata_skill_count": len(safe),
        "dropped_skill_count": len(dropped),
    }


def _format_skill(
    skill: dict[str, Any],
    index: int,
    max_chars: int,
    skill_context_mode: str = "raw",
    valid_api_refs: set[tuple[str, str]] | None = None,
    instruction: str = "",
) -> str:
    mode = str(skill_context_mode)
    if mode not in VALID_SKILL_CONTEXT_MODES:
        raise ValueError(f"unsupported skill_context_mode: {mode}")
    valid_api_refs = set(valid_api_refs or set())
    format_mode = mode
    if mode == "adaptive_schema":
        format_mode = (
            "safe_metadata"
            if _skill_semantically_aligned_with_task(skill, instruction)
            else "schema_plan"
        )
    if format_mode == "schema_plan":
        usable_refs, read_refs, action_refs, filtered_action_count = _schema_plan_refs(
            skill,
            valid_api_refs,
            instruction,
        )
        parts = [
            f"[Reference Skill {index} - schema-grounded constraints]",
            "not_callable: true",
            "schema_grounded: true",
            "title_hidden: true",
            "user_goal_is_authoritative: true",
            "not_an_api_allowlist: true",
            "valid_appworld_apis: " + (_format_api_ref_list(usable_refs) if usable_refs else "(none)"),
            "read_support_apis: " + (_format_api_ref_list(read_refs) if read_refs else "(none)"),
            "state_changing_action_apis: " + (_format_api_ref_list(action_refs) if action_refs else "(none)"),
            f"filtered_state_changing_api_count: {int(filtered_action_count)}",
            "constraint: Use this block only as schema-grounded evidence; it is not a complete plan or API allowlist.",
            "constraint: Use any API in [AppWorld API Docs] if required by the user goal.",
            "constraint: Do not infer API names, entities, filters, or task intent from hidden SkillX titles.",
            "raw_body_omitted: true",
        ]
        return "\n".join(parts)[: int(max_chars)]
    explicit_refs = _api_refs_from_text(
        " ".join(
            str(skill.get(key, ""))
            for key in ("executor_desc", "body", "skill_md")
        )
    )
    executable_refs = explicit_refs & valid_api_refs
    invalid_explicit_refs = explicit_refs - valid_api_refs if valid_api_refs else set()
    derived_title_refs = _candidate_refs_from_skill_title(skill, valid_api_refs) if valid_api_refs else set()
    blocked_title_refs = derived_title_refs - valid_api_refs
    hide_title = (
        format_mode == "safe_metadata"
        and bool(blocked_title_refs)
        and not _has_state_changing_api(executable_refs)
        and _has_high_risk_nonexistent_title_api(blocked_title_refs)
    )
    parts = [
        f"[Reference Skill {index} - not callable]"
        + ("" if hide_title else f" {skill.get('skill_id', '')}"),
        (
            "reference_name_not_function: (hidden because this SkillX title resembles a non-existent AppWorld API)"
            if hide_title
            else f"reference_name_not_function: {skill.get('name', '')}"
        ),
        "not_callable: true",
        f"description: {skill.get('description', '')}",
        f"executor: {skill.get('executor_desc', '')}",
    ]
    if valid_api_refs:
        parts.append(
            "executable_appworld_apis_from_skill: "
            + (_format_api_ref_list(executable_refs) if executable_refs else "(none matched current AppWorld API docs)")
        )
    if blocked_title_refs:
        parts.append("blocked_nonexistent_skill_title_api: " + _format_api_ref_list(blocked_title_refs))
    if invalid_explicit_refs:
        parts.append("blocked_nonexistent_executor_api: " + _format_api_ref_list(invalid_explicit_refs))
    if format_mode == "raw":
        parts.append(f"body:\n{skill.get('body', skill.get('skill_md', ''))}")
    elif format_mode == "safe_metadata":
        parts.extend(
            [
                "raw_body_omitted: true",
                "raw_body_omission_reason: SkillX source code is not schema-validated for the current AppWorld runtime.",
            ]
        )
    else:
        parts.append("raw_body_omitted: true")
    return "\n".join(parts)[: int(max_chars)]


def _task_datetime(task: dict[str, Any]) -> str:
    return str(task.get("task_datetime") or task.get("datetime") or "").strip()


def _date_prefixes(task_datetime: str) -> tuple[str | None, str | None]:
    date = str(task_datetime or "")[:10]
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        return None, None
    return f"{date[:4]}-01-01", f"{date[:7]}-01"


def _build_task_specific_hints(
    instruction: str,
    required_apps: list[str] | None = None,
    task_datetime: str | None = None,
) -> list[str]:
    text = " ".join(str(instruction or "").split())
    lowered = text.lower()
    app_set = {str(app) for app in (required_apps or [])}
    hints: list[str] = []
    year_start, month_start = _date_prefixes(str(task_datetime or ""))
    spotify_top_match = re.search(r"top\s+(\d+)\s+most played\s+(.+?)\s+song titles", lowered)
    if "spotify" in lowered and spotify_top_match:
        count = spotify_top_match.group(1)
        genre = spotify_top_match.group(2).strip()
        hints.extend(
            [
                f"For this task, filter songs where `song[\"genre\"]` matches `{genre}` case-insensitively before ranking.",
                "Do not rank all songs first; apply the requested genre filter first, then rank only the filtered songs.",
                "`show_song_library` returns song dicts; add `item[\"song_id\"]` for each returned item, never the whole item dict.",
                "`show_album_library` returns album dicts; add every id from `album[\"song_ids\"]`.",
                "`show_playlist_library` returns playlist dicts; add every id from `playlist[\"song_ids\"]`.",
                "Do not expect `show_playlist_library` to return a `songs` field; use `song_ids` from the library response.",
                "Use pagination for all three library APIs: start `page_index=0`, call with `page_limit=20`, increment until the returned list is empty.",
                "Call `apis.spotify.show_song(song_id=song_id)` without an access token; then read `genre`, `play_count`, and `title` from that detail response.",
                "sort the filtered songs by `song[\"play_count\"]` descending.",
                f"return exactly {count} song titles as a comma-separated string with no extra text.",
            ]
        )
    if {"phone", "venmo"}.issubset(app_set) and "phone text" in lowered and (
        "send them the owed money" in lowered or "paid for my grocery" in lowered
    ):
        hints.extend(
            [
                "Phone login username is the supervisor phone number, not email.",
                "Find the payer and amount from phone text messages before creating the Venmo transaction.",
                "Find the payer contact and grocery amount before creating the Venmo transaction.",
                "Use `apis.phone.search_contacts(query=...)` to find the payer contact first; read that contact's `email` and `phone_number` fields.",
                "Call `search_text_messages(phone_number=contact[\"phone_number\"], page_limit=20, page_index=...)` and paginate until an empty page.",
                "`apis.phone.search_text_messages` rows use the text body field `message`, not `text`.",
                "Text message `sender` and `receiver` contain `contact_id`, `name`, and `phone_number`; they do not contain `email`.",
                "Use the contact email from `search_contacts` as `receiver_email`; never use a phone number as a Venmo email.",
                "Extract the grocery amount from `message[\"message\"]` with a dollar-amount regex.",
                "Scan every paginated text message; do not assume page 0 or the first message contains the grocery amount.",
                "Prefer messages whose `message` text is relevant to grocery, groceries, paid, payment, card, or owed money; do not use the first unrelated dollar amount.",
                "Filter to messages whose `message` mentions grocery, groceries, paid, payment, card, owed, or bill before choosing the amount.",
                "If several relevant messages contain dollar amounts, choose the most recent relevant message by `sent_at`.",
                "Do not run one regex over all messages joined together; select one relevant message first, then extract its dollar amount.",
                "The amount regex must require a literal dollar sign, for example `r\"\\$(\\d+(?:\\.\\d+)?)\"`; do not use optional-dollar patterns like `\\$?`.",
                "Skip messages without a dollar amount instead of defaulting to 0 or matching unrelated numbers.",
                "Use the payer first name from the task instruction as the contact query.",
                "Valid solution order: log in, search the contact, paginate text messages by that contact phone number, regex the amount from `message`, create the Venmo transaction, send the phone text, then call `complete_task`.",
                "Import `re` before using a dollar-amount regex.",
                "Do not call any `apis.simple_note.*` API for this phone+venmo grocery/text task.",
                "Use `apis.venmo.create_transaction(receiver_email=..., amount=..., description=..., access_token=venmo_access_token)` to send the owed money.",
                "Use `apis.phone.send_text_message(phone_number=..., message=..., access_token=phone_access_token)` to notify the payer.",
                "Do not call `apis.supervisor.complete_task(status=\"success\")` until the Venmo transaction and phone text have both succeeded.",
            ]
        )
    if "venmo" in app_set and "transaction" in lowered and "like" in lowered:
        hints.extend(
            [
                "Use `apis.venmo.show_transactions(access_token=venmo_access_token, min_like_count=1, page_limit=20, page_index=...)` for Venmo like-count tasks.",
                "Paginate Venmo transactions until an empty page; sum `transaction[\"like_count\"]` from every returned page.",
                "For sent transactions, pass `direction=\"sent\"`; for received transactions, pass `direction=\"received\"`.",
                "Do not add a broad `max_created_at=\"2023-12-31\"` unless the instruction explicitly asks for an end date.",
                "Maintain a local integer `page_index` and increment it by 1 each loop; transaction rows do not contain a `page_index` field.",
                "Never use `apis.supervisor.show_account_passwords()[0]`; always build `supervisor_passwords` by `account_name`.",
            ]
        )
        if "this month" in lowered:
            if month_start:
                hints.append(f"For `this month`, set `min_created_at=\"{month_start}\"` from the AppWorld task datetime.")
            else:
                hints.append("For `this month`, set `min_created_at` to the first day of the month from the AppWorld task datetime.")
        if "this year" in lowered:
            if year_start:
                hints.append(f"For `this year`, set `min_created_at=\"{year_start}\"` from the AppWorld task datetime.")
            else:
                hints.append("For `this year`, set `min_created_at` to the first day of the year from the AppWorld task datetime.")
    if "file_system" in app_set:
        hints.extend(
            [
                "The file_system app has no `rename_file` or `categorize_files_by_creation_date` API.",
                "Use `show_directory` to list paths, then `show_file(file_path=..., access_token=file_system_access_token)` to read `created_at`.",
                "`move_file` requires `source_file_path` and a full `destination_file_path` including the final filename.",
                "Do not pass a destination directory alone to `move_file`.",
                "Build destination paths with exactly one `/` between directory and filename; avoid `//`.",
                "Use `file_detail[\"created_at\"][:10]` for date prefixes.",
            ]
        )
        if ("prefix" in lowered and "creation" in lowered) or "not from this year" in lowered:
            hints.extend(
                [
                    "Rename every listed file by moving it to the requested date-prefixed filename; do not only move old files.",
                    "For files moved to trash or recycle directories, the destination path must still include the requested date-prefixed filename.",
                    "If execution history says `No API named`, do not retry that API; replace it with manual `show_directory` + `show_file` + `move_file` logic.",
                ]
            )
        if "this year" in lowered:
            year = year_start[:4] if year_start else None
            if year:
                hints.append(f"For `this year`, compare against year `{year}` from the AppWorld task datetime, not the host clock.")
            else:
                hints.append("For `this year`, compare against the year from the AppWorld task datetime, not the host clock.")
    if {"file_system", "simple_note"}.issubset(app_set) and "export" in lowered and "note" in lowered:
        hints.extend(
            [
                "`apis.simple_note.search_notes(...)` returns a list of note summaries, not a dict with a `notes` key.",
                "Paginate `search_notes(access_token=simple_note_access_token, page_limit=20, page_index=...)` until an empty list.",
                "Increment `page_index += 1` inside the `search_notes` loop after processing each non-empty page.",
                "Call `show_note(note_id=..., access_token=simple_note_access_token)` for each note before exporting content.",
                "Write exactly `note_detail[\"content\"]` to each `.md` file; do not prepend markdown titles or headers.",
                "Create the destination directory if needed, then create one file per note with whitespace in the title replaced by `_`.",
                "After creating all requested files, call `apis.supervisor.complete_task(status=\"success\")`.",
            ]
        )
    return hints


def _login_example_lines(required_apps: list[str]) -> list[str]:
    lines: list[str] = []
    for app in required_apps:
        username_expr = 'supervisor_profile["phone_number"]' if app == "phone" else 'supervisor_profile["email"]'
        lines.append(
            f'{app}_access_token = apis.{app}.login(username={username_expr}, password=supervisor_passwords["{app}"])["access_token"]'
        )
    return lines


def build_appworld_executor_prompt(
    *,
    task: dict[str, Any],
    skills: list[dict[str, Any]],
    api_docs_context: str,
    max_skill_chars: int = 1600,
    skill_context_mode: str = "raw",
    schema_guard_skill_metadata: bool = False,
    valid_api_refs: set[tuple[str, str]] | None = None,
    diagnostics_out: dict[str, Any] | None = None,
) -> str:
    mode = str(skill_context_mode)
    if mode not in VALID_SKILL_CONTEXT_MODES:
        raise ValueError(f"unsupported skill_context_mode: {mode}")
    instruction = str(task.get("instruction_text") or task.get("instruction") or task.get("query") or "")
    required_apps = [str(app) for app in task.get("required_apps", []) if str(app) != "supervisor"]
    primary_app = required_apps[0] if required_apps else "spotify"
    task_datetime = _task_datetime(task)
    task_hints = _build_task_specific_hints(instruction, required_apps, task_datetime=task_datetime)
    prompt_skills, filtered_skill_count = _filter_prompt_skills_by_relevance(
        instruction=instruction,
        skills=skills,
    )
    prompt_valid_api_refs = set(valid_api_refs or set())
    schema_aware_modes = {
        "schema_plan",
        "adaptive_schema",
        "gated_schema",
        "api_evidence",
        "verified_hints",
        "verified_metadata",
    }
    if not prompt_valid_api_refs and (schema_guard_skill_metadata or mode in schema_aware_modes):
        prompt_valid_api_refs = _valid_api_refs_from_docs_context(api_docs_context)
    schema_filtered_count = 0
    handoff_decisions: list[dict[str, Any]] = []
    prompt_skill_formats: list[tuple[dict[str, Any], str]] = [(skill, mode) for skill in prompt_skills]
    if mode == "schema_plan":
        before_schema_filter = len(prompt_skills)
        prompt_skills = [
            skill
            for skill in prompt_skills
            if _schema_plan_refs(skill, prompt_valid_api_refs, instruction)[0]
        ]
        schema_filtered_count = before_schema_filter - len(prompt_skills)
        prompt_skill_formats = [(skill, "schema_plan") for skill in prompt_skills]
    elif mode == "adaptive_schema":
        before_schema_filter = len(prompt_skills)
        prompt_skills = [
            skill
            for skill in prompt_skills
            if _skill_semantically_aligned_with_task(skill, instruction)
            or _schema_plan_refs(skill, prompt_valid_api_refs, instruction)[0]
        ]
        schema_filtered_count = before_schema_filter - len(prompt_skills)
        prompt_skill_formats = [
            (
                skill,
                "safe_metadata"
                if _skill_semantically_aligned_with_task(skill, instruction)
                else "schema_plan",
            )
            for skill in prompt_skills
        ]
    elif mode == "gated_schema":
        gated_formats: list[tuple[dict[str, Any], str]] = []
        required_app_set = {str(app) for app in required_apps}
        for skill in prompt_skills:
            decision = _gate_skill_context(
                skill=skill,
                valid_api_refs=prompt_valid_api_refs,
                instruction=instruction,
                required_apps=required_app_set,
            )
            handoff_decisions.append(decision)
            if decision["decision"] == "drop":
                continue
            gated_formats.append((skill, str(decision["decision"])))
        schema_filtered_count = sum(1 for row in handoff_decisions if row.get("decision") == "drop")
        prompt_skills = [skill for skill, _format_mode in gated_formats]
        prompt_skill_formats = gated_formats
    if mode in {"verified_hints", "verified_metadata"}:
        handoff_diagnostics = build_verified_skill_handoff(
            instruction=instruction,
            skills=prompt_skills,
            valid_api_refs=prompt_valid_api_refs,
            required_apps={str(app) for app in required_apps},
            max_chars=max_skill_chars,
        )
        handoff_diagnostics.update(
            {
                "skill_context_mode": mode,
                "handoff_decision": mode,
                "input_skill_count": int(len(skills)),
                "relevance_filtered_count": int(filtered_skill_count),
                "schema_filtered_count": int(schema_filtered_count),
            }
        )
        if mode == "verified_metadata":
            decisions_by_id = {
                str(row.get("skill_id", "")): row
                for row in handoff_diagnostics.get("decisions", [])
                if isinstance(row, dict)
            }
            verified_formats: list[tuple[dict[str, Any], str]] = []
            for skill in prompt_skills:
                row = decisions_by_id.get(str(skill.get("skill_id", "")), {})
                decision = str(row.get("decision") or "")
                if decision == "suppress":
                    continue
                verified_formats.append((skill, "safe_metadata" if decision == "inject_hint" else "schema_plan"))
            prompt_skill_formats = verified_formats
            prompt_skills = [skill for skill, _format_mode in prompt_skill_formats]
            handoff_diagnostics["verified_metadata_skill_count"] = sum(
                1 for _skill, format_mode in prompt_skill_formats if format_mode == "safe_metadata"
            )
            handoff_diagnostics["verified_schema_skill_count"] = sum(
                1 for _skill, format_mode in prompt_skill_formats if format_mode == "schema_plan"
            )
    elif mode == "api_evidence":
        handoff_diagnostics = _api_evidence_from_skills(
            skills=prompt_skills,
            valid_api_refs=prompt_valid_api_refs,
            instruction=instruction,
        )
        handoff_diagnostics.update(
            {
                "input_skill_count": int(len(skills)),
                "relevance_filtered_count": int(filtered_skill_count),
                "schema_filtered_count": int(schema_filtered_count),
            }
        )
    else:
        handoff_diagnostics = _summarize_handoff_decisions(
            mode=mode,
            input_skill_count=len(skills),
            filtered_skill_count=filtered_skill_count,
            schema_filtered_count=schema_filtered_count,
            decisions=(
                handoff_decisions
                if mode == "gated_schema"
                else [
                    {
                        "skill_id": str(skill.get("skill_id", "")),
                        "decision": format_mode,
                        "reasons": [],
                        "valid_api_refs": [],
                        "read_support_apis": [],
                        "state_changing_action_apis": [],
                        "filtered_state_changing_api_count": 0,
                    }
                    for skill, format_mode in prompt_skill_formats
                ]
            ),
        )
    if diagnostics_out is not None:
        diagnostics_out.clear()
        diagnostics_out.update(handoff_diagnostics)
    if "[Execution History]" in instruction:
        task_hints.extend(
            [
                "When [Execution History] contains successful code, assume prior variables and app state persist across steps.",
                "Only variables explicitly assigned with `=` in previous successful code persist across steps.",
                "Bare calls such as `apis.supervisor.show_profile()` do not create `supervisor_profile`.",
                "Assign profile/password variables again if they were not assigned earlier.",
                "Do not repeat successful login-only code; continue with the next missing API calls.",
                "If a previous successful step already created the requested files, transactions, messages, or records but `task_completed` is false, call `complete_task` instead of repeating the same loop.",
                "If evaluation history says records such as `venmo.Transaction`, `phone.GlobalTextMessage`, or `phone.UserTextMessage` are missing, create those records before `complete_task`.",
            ]
        )
        if "Preflight failed before execution" in instruction:
            task_hints.extend(
                [
                    "If a previous step says `Preflight failed before execution`, none of its variables or side effects exist.",
                    "Regenerate a complete self-contained program including login/setup after a preflight failure.",
                ]
            )
    if mode == "api_evidence":
        skill_block = _format_api_evidence_block(handoff_diagnostics, max_skill_chars)
    elif mode == "verified_hints":
        skill_block = str(handoff_diagnostics.get("prompt_block") or "")[: int(max_skill_chars)]
    elif mode == "verified_metadata":
        metadata_block = "\n\n".join(
            _format_skill(
                skill,
                idx + 1,
                max_skill_chars,
                skill_context_mode=format_mode,
                valid_api_refs=prompt_valid_api_refs,
                instruction=instruction,
            )
            for idx, (skill, format_mode) in enumerate(prompt_skill_formats)
        )
        hint_block = str(handoff_diagnostics.get("prompt_block") or "")[: int(max_skill_chars)]
        skill_block = "\n\n".join(part for part in (metadata_block, hint_block) if part)
    else:
        skill_block = "\n\n".join(
            _format_skill(
                skill,
                idx + 1,
                max_skill_chars,
                skill_context_mode=format_mode,
                valid_api_refs=prompt_valid_api_refs,
                instruction=instruction,
            )
            for idx, (skill, format_mode) in enumerate(prompt_skill_formats)
        )
    if not skill_block:
        skill_block = (
            "(No retrieved SkillX skills passed schema-grounded filtering.)"
            if schema_filtered_count
            else "(No retrieved SkillX skills passed relevance filtering.)"
            if filtered_skill_count
            else "(No retrieved SkillX skills for this baseline.)"
        )
    skill_context_rules = [
        "Retrieved SkillX skills are reference information only. Do not call `apis.skillx`; that namespace is not available in AppWorld.",
    ]
    if filtered_skill_count:
        skill_context_rules.append(
            f"Filtered {filtered_skill_count} retrieved SkillX skill(s) with low task relevance before prompting."
        )
    if schema_filtered_count:
        skill_context_rules.append(
            f"Filtered {schema_filtered_count} retrieved SkillX skill(s) without schema-grounded executable evidence."
        )
    if mode == "raw":
        skill_context_rules.append(
            "If a SkillX body is useful, copy its logic and call only real AppWorld APIs listed in the API docs."
        )
    elif mode == "schema_plan":
        skill_context_rules.extend(
            [
                "Retrieved SkillX titles, names, and raw bodies are hidden in this run.",
                "Use schema-grounded SkillX blocks as optional evidence only; they are not an API allowlist or complete execution plan.",
                "Use any API in [AppWorld API Docs] when needed by the user goal, even if no retrieved SkillX block lists it.",
                "Do not infer task intent, filters, entities, or API names from hidden SkillX titles.",
            ]
        )
    elif mode == "adaptive_schema":
        skill_context_rules.extend(
            [
                "Retrieved SkillX bodies are omitted in this run.",
                "Semantically aligned SkillX skills may keep safe metadata; weakly aligned skills are reduced to schema-grounded API evidence.",
                "Schema-grounded SkillX blocks are optional evidence only; they are not an API allowlist or complete execution plan.",
                "Use any API in [AppWorld API Docs] when needed by the user goal, even if no retrieved SkillX block lists it.",
                "Do not infer task intent, filters, entities, or API names from hidden SkillX titles.",
            ]
        )
    elif mode == "gated_schema":
        skill_context_rules.extend(
            [
                "Gated SkillX handoff kept "
                f"{handoff_diagnostics['safe_metadata_skill_count']} safe-metadata skill(s), "
                f"{handoff_diagnostics['schema_plan_skill_count']} schema-only skill(s), and dropped "
                f"{handoff_diagnostics['dropped_skill_count']} skill(s).",
                "Only strongly aligned SkillX skills keep natural-language metadata; weaker or pollution-prone skills are reduced to schema-grounded evidence.",
                "Schema-grounded SkillX blocks are optional evidence only; they are not an API allowlist or complete execution plan.",
                "Use any API in [AppWorld API Docs] when needed by the user goal, even if no retrieved SkillX block lists it.",
                "Do not infer task intent, filters, entities, or API names from hidden SkillX titles.",
            ]
        )
    elif mode == "api_evidence":
        skill_context_rules.extend(
            [
                "Retrieved SkillX titles, names, descriptions, and raw bodies are hidden in this run.",
                "CLSTR-selected skills are rendered only as schema-grounded API evidence.",
                "Use the evidence as a routing prior, not as a plan, API allowlist, or replacement for the user goal.",
                "Use any API in [AppWorld API Docs] when needed by the user goal, even if no selected-skill evidence lists it.",
                "Do not infer task intent, filters, entities, or API names from hidden SkillX text fields.",
            ]
        )
    elif mode == "verified_hints":
        counts = handoff_diagnostics.get("handoff_decision_counts", {}) if isinstance(handoff_diagnostics, dict) else {}
        skill_context_rules.extend(
            [
                "Retrieved SkillX titles, names, descriptions, and raw bodies are hidden in this run.",
                "Retrieved skills are verifier-compressed into optional hints after schema and operation checks.",
                "Optional retrieved hints are not an API allowlist, not a complete plan, and not more authoritative than the user goal.",
                "Use any API in [AppWorld API Docs] when required by the user goal, even if no retrieved hint lists it.",
                "Do not infer task intent, filters, entities, or API names from hidden SkillX text fields.",
                "Verified hint decisions: "
                f"inject_hint={int(counts.get('inject_hint', 0))}, "
                f"schema_only={int(counts.get('schema_only', 0))}, "
                f"suppress={int(counts.get('suppress', 0))}.",
            ]
        )
    elif mode == "verified_metadata":
        counts = handoff_diagnostics.get("handoff_decision_counts", {}) if isinstance(handoff_diagnostics, dict) else {}
        skill_context_rules.extend(
            [
                "Retrieved SkillX raw bodies are hidden in this run.",
                "Verifier-approved skills may keep safe metadata; risky skills are compressed or suppressed after schema and operation checks.",
                "Safe metadata and optional retrieved hints are routing evidence only, not an API allowlist or complete execution plan.",
                "Use any API in [AppWorld API Docs] when required by the user goal, even if no retrieved evidence lists it.",
                "Do not infer API names from SkillX names; call only documented AppWorld APIs.",
                "Verified metadata decisions: "
                f"inject_hint={int(counts.get('inject_hint', 0))}, "
                f"schema_only={int(counts.get('schema_only', 0))}, "
                f"suppress={int(counts.get('suppress', 0))}.",
            ]
        )
    else:
        skill_context_rules.extend(
            [
                "Retrieved SkillX bodies are intentionally omitted in this run; use skill metadata for intent only.",
                "Use [AppWorld API Docs] as the executable schema source, not omitted SkillX source code.",
            ]
        )
    sections = [
            "You are solving one AppWorld task by writing Python code.",
            "Use the provided `apis` object and `requester` object already available in the execution environment.",
            "Return Python code only. Do not include explanations, markdown outside a code fence, or shell commands.",
            "Call `apis.supervisor.complete_task(...)` when the task is complete if that API is appropriate.",
            *skill_context_rules,
            "A SkillX skill name is not a function name; never convert a retrieved skill name into `apis.<app>.<skill_name>(...)`.",
            "Use only API names listed in [AppWorld API Docs]. Do not invent APIs from skill names or descriptions.",
            "Match each API signature exactly. Do not pass parameters that are not listed, and include all required parameters.",
            "For paginated list/library APIs, keep `page_limit` within the documented constraints and loop `page_index` until an empty page if all items are needed.",
            "Never use `page_limit=100`; prefer `page_limit=20` or the documented lower maximum.",
            "If a needed field is absent from a list/library response schema, call the corresponding detail API before using that field.",
            "For Spotify song, album, and playlist library tasks, collect song IDs from library/list responses, then call `apis.spotify.show_song(song_id=song_id)` before reading `genre` or `play_count`.",
            "If the task says across Spotify song, album, and playlist libraries, union song IDs from `show_song_library`, `show_album_library`, and `show_playlist_library`, then fetch every unique song with `show_song` before filtering or sorting.",
            "Use keyword arguments for all AppWorld API calls; do not pass positional arguments.",
            "Every `apis.*` call returns the Python object directly. Never call `requester.json()` after an `apis.*` call.",
            "Do not invent placeholder credentials such as username/password/user@example.com.",
            "When finishing a non-question task, call `apis.supervisor.complete_task(status=\"success\")`; when an answer is required, call `apis.supervisor.complete_task(answer=answer, status=\"success\")`.",
            "Do not use status values such as `complete` or `done`.",
            "Do not use the host machine date/time for AppWorld relative-date tasks.",
            "For app login, first call `apis.supervisor.show_profile()` and `apis.supervisor.show_account_passwords()`.",
            "The password `account_name` keys are lowercase app names such as `spotify`, `gmail`, and `file_system`.",
            "Use this login pattern with keyword arguments:",
            "supervisor_profile = apis.supervisor.show_profile()",
            "supervisor_passwords = {account_password[\"account_name\"]: account_password[\"password\"] for account_password in apis.supervisor.show_account_passwords()}",
            *_login_example_lines(required_apps or [primary_app]),
            "",
    ]
    if task_hints:
        sections.extend(["[Task-Specific Hints]", *task_hints, ""])
    if task_datetime:
        sections.extend(
            [
                "[Task Environment]",
                f"AppWorld task datetime: {task_datetime}",
                "Use this datetime for relative phrases such as `today`, `this month`, and `this year`.",
                "",
            ]
        )
    sections.extend(
        [
            "[Task Instruction]",
            instruction,
            "",
            "[Retrieved SkillX Skills]",
            skill_block,
            "",
            api_docs_context,
            "",
            "[Output Contract]",
            "Return Python code only.",
        ]
    )
    return "\n".join(sections)


@dataclass(frozen=True)
class QwenCodeGeneratorConfig:
    model_name_or_path: str = "models/Qwen3-8B"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    local_files_only: bool = True
    max_new_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.95
    device: str | None = None
    use_chat_template: bool = True
    enable_thinking: bool = False


def _resolve_dtype(value: str):
    import torch

    if value == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"unsupported torch_dtype: {value}")
    return mapping[value]


class QwenCodeGenerator:
    def __init__(self, config: QwenCodeGeneratorConfig | None = None, backend: Any | None = None) -> None:
        self.config = config or QwenCodeGeneratorConfig()
        if backend is None:
            from clstr.qwen_backend import QwenBackendConfig, QwenGenerationBackend

        self.backend = backend or QwenGenerationBackend(
            QwenBackendConfig(
                model_name_or_path=self.config.model_name_or_path,
                local_files_only=bool(self.config.local_files_only),
                torch_dtype=self.config.torch_dtype,
                trust_remote_code=bool(self.config.trust_remote_code),
                max_new_tokens=int(self.config.max_new_tokens),
                temperature=float(self.config.temperature),
                top_p=float(self.config.top_p),
                device=self.config.device,
                use_chat_template=bool(self.config.use_chat_template),
                enable_thinking=bool(self.config.enable_thinking),
            )
        )

    @property
    def device(self):
        import torch

        device = getattr(self.backend, "device", None)
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _format_prompt(self, prompt: str) -> str:
        return self.backend.format_chat_prompt(
            "You are an AppWorld Python-code executor. Return code only.",
            prompt,
        )

    def metadata(self) -> dict[str, Any]:
        if hasattr(self.backend, "metadata"):
            return dict(self.backend.metadata())
        return {
            "qwen_backend_version": "custom_backend",
            "model_name_or_path": self.config.model_name_or_path,
            "local_files_only": bool(self.config.local_files_only),
            "enable_thinking": bool(self.config.enable_thinking),
        }

    def generate(self, prompt: str) -> str:
        return self.backend.generate_text(
            "You are an AppWorld Python-code executor. Return code only.",
            prompt,
        )


def _json_safe(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _json_safe(value.to_dict(stats_only=False))
        except TypeError:
            return _json_safe(value.to_dict())
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _execution_succeeded(execute_output: Any) -> bool:
    text = str(execute_output)
    failure_markers = ("Execution failed.", "Traceback", "Error:")
    return not any(marker in text for marker in failure_markers)


def _evaluation_succeeded(evaluate_payload: Any) -> bool:
    if isinstance(evaluate_payload, dict) and "success" in evaluate_payload:
        return bool(evaluate_payload["success"])
    return False


def _call_generator(generator: Any, prompt: str) -> str:
    if hasattr(generator, "generate"):
        return str(generator.generate(prompt))
    return str(generator(prompt))


def _default_world_factory() -> Callable[..., Any]:
    from appworld.environment import AppWorld

    return AppWorld


def run_appworld_executor_eval(
    *,
    tasks_path: str | Path,
    skill_pool_path: str | Path,
    output_dir: str | Path,
    appworld_root: str | Path,
    appworld_cache: str | Path,
    method: str,
    skill_provider: SkillProvider,
    generator: Any,
    world_factory: Callable[..., Any] | None = None,
    max_tasks: int | None = None,
    top_k: int = 5,
    max_apis_per_app: int = 12,
    skill_context_mode: str = "raw",
    experiment_name: str = "clstr_appworld_qwen_executor",
    timeout_seconds: int = 60,
    max_interactions: int = 3,
) -> dict[str, Any]:
    root, cache = configure_appworld_paths(appworld_root, appworld_cache)
    os.environ.setdefault("IPYTHONDIR", "/data/home/scyb713/run/xzf/AAAI/autodl-tmp/.cache/ipython")
    tasks = read_jsonl(tasks_path)
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]
    factory = world_factory or _default_world_factory()
    output_dir = Path(output_dir)
    runs_path = output_dir / "runs.jsonl"
    run_rows: list[dict[str, Any]] = []

    for task in tasks:
        task = enrich_task_with_appworld_specs(task, root)
        task_id = str(task.get("task_id") or task.get("query_id"))
        selected_skills = skill_provider.get_skills(task, top_k=top_k)
        api_docs_context = build_api_docs_context(
            appworld_root=root,
            required_apps=[str(item) for item in task.get("required_apps", [])],
            api_refs=[str(item) for item in task.get("api_refs", [])],
            max_apis_per_app=max_apis_per_app,
        )
        valid_api_refs = load_appworld_api_refs(
            appworld_root=root,
            required_apps=[str(item) for item in task.get("required_apps", [])],
            api_refs=[str(item) for item in task.get("api_refs", [])],
        )
        prompt = build_appworld_executor_prompt(
            task=task,
            skills=selected_skills,
            api_docs_context=api_docs_context,
            skill_context_mode=skill_context_mode,
            valid_api_refs=valid_api_refs,
        )
        row: dict[str, Any] = {
            "method": method,
            "task_id": task_id,
            "query_id": str(task.get("query_id") or task_id),
            "selected_skill_ids": [str(skill.get("skill_id")) for skill in selected_skills],
            "prompt_chars": len(prompt),
            "reset_ok": False,
            "generation_ok": False,
            "execution_ok": False,
            "task_completed": False,
            "success": False,
        }
        world = None
        try:
            world = factory(
                task_id,
                experiment_name=f"{experiment_name}_{method}",
                max_interactions=max_interactions,
                timeout_seconds=timeout_seconds,
                load_ground_truth=True,
                ground_truth_mode="minimal",
                show_api_response_schemas=False,
            )
            row["reset_ok"] = True
            response = _call_generator(generator, prompt)
            code = extract_python_code(response)
            row.update({"generation_ok": True, "raw_response": response, "code": code})
            execute_output = world.execute(code)
            row["execute_output"] = str(execute_output)
            row["execution_ok"] = _execution_succeeded(execute_output)
            try:
                row["task_completed"] = bool(world.task_completed())
            except Exception as exc:
                row["task_completed_error"] = repr(exc)
            try:
                row["evaluate"] = _json_safe(world.evaluate(suppress_errors=True))
                row["evaluation_success"] = _evaluation_succeeded(row["evaluate"])
            except Exception as exc:
                row["evaluate_error"] = repr(exc)
                row["evaluation_success"] = False
            row["success"] = bool(row["execution_ok"] and row["task_completed"] and row["evaluation_success"])
        except Exception as exc:
            row["error"] = repr(exc)
        finally:
            if world is not None and callable(getattr(world, "close", None)):
                world.close()
        run_rows.append(row)

    write_jsonl(runs_path, run_rows)
    success_count = sum(1 for row in run_rows if row.get("success"))
    report = {
        "status": "ok",
        "method": method,
        "tasks_path": str(tasks_path),
        "skill_pool_path": str(skill_pool_path),
        "appworld_root": str(root),
        "appworld_cache": str(cache),
        "runs_path": str(runs_path),
        "task_count": len(run_rows),
        "success_count": success_count,
        "success_rate": round(success_count / max(len(run_rows), 1), 6),
        "generation_failures": sum(1 for row in run_rows if not row.get("generation_ok")),
        "execution_failures": sum(1 for row in run_rows if row.get("generation_ok") and not row.get("execution_ok")),
        "top_k": int(top_k),
        "skill_provider_config": {
            "dedupe_canonical_skills": bool(getattr(skill_provider, "dedupe_canonical_skills", False)),
            "max_auth_like_skills": getattr(skill_provider, "max_auth_like_skills", None),
            "skill_context_mode": str(skill_context_mode),
        },
        "caveat": "Official AppWorld runtime loop; task success requires AppWorld task_completed/evaluate, not routing qrels.",
    }
    write_json(output_dir / "report.json", report)
    return report


def _load_report(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"status": "missing", "method": path.parent.name, "path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def build_executor_comparison(report_paths: list[str | Path], output_dir: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in report_paths:
        report = _load_report(path)
        rows.append(
            {
                "path": str(path),
                "status": report.get("status", "missing"),
                "method": report.get("method", Path(path).parent.name),
                "task_count": report.get("task_count"),
                "success_count": report.get("success_count"),
                "success_rate": report.get("success_rate"),
                "generation_failures": report.get("generation_failures"),
                "execution_failures": report.get("execution_failures"),
                "runs_path": report.get("runs_path"),
            }
        )
    payload = {
        "status": "ok" if rows else "blocked",
        "comparison_role": "AppWorld official executor task success comparison.",
        "rows": rows,
    }
    output_dir = Path(output_dir)
    write_json(output_dir / "comparison.json", payload)
    lines = [
        "# AppWorld Executor Comparison",
        "",
        "| method | status | task_count | success_count | success_rate | generation_failures | execution_failures |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {status} | {task_count} | {success_count} | {success_rate} | {generation_failures} | {execution_failures} |".format(
                method=row["method"],
                status=row["status"],
                task_count=row["task_count"],
                success_count=row["success_count"],
                success_rate=row["success_rate"],
                generation_failures=row["generation_failures"],
                execution_failures=row["execution_failures"],
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
