from __future__ import annotations

import ast
import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def build_official_react_system_prompt(*, max_interactions: int = 40) -> str:
    return f"""I am your supervisor, and you are an AI Assistant whose job is to complete my day-to-day tasks fully autonomously.

You will interact with AppWorld apps through a multi-step conversation using a Python REPL. Write one Python code block per turn. The environment will execute it and return printed output. Use the output to decide the next step until the task is complete.

Use these AppWorld API documentation helpers before assuming an API signature:
```python
print(apis.api_docs.show_app_descriptions())
print(apis.api_docs.show_api_descriptions(app_name='spotify'))
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))
```

After you know the relevant APIs, prefer a complete script over one trivial API call per turn. When an API supports `page_index` and `page_limit`, iterate pages until an empty page or a page smaller than the limit; process the current page before checking whether its length is smaller than page_limit. For tasks that ask across multiple libraries, folders, apps, or sources, inspect every named source before answering. Keep full record dictionaries until after filtering, ranking, counting, or sorting; do not replace records with only names/titles before later computations.

Never invent credentials. For authenticated apps, get the requester profile from `apis.supervisor.show_profile()`, get app passwords from `apis.supervisor.show_account_passwords()`, call the app `login` API, store the returned `access_token`, and pass it to later authenticated APIs. Do not sign up, reset passwords, verify accounts, delete accounts, or update account profiles unless the user explicitly asks for that account-management action.

You have at most {int(max_interactions)} interactions. Prefer minimal, reversible reads before writes. For answer tasks, call `apis.supervisor.complete_task(answer=result)` after computing the final answer; do not pass the answer as a positional argument. For state-change-only tasks, call `apis.supervisor.complete_task()` only after the requested state change is complete.

Retrieved skill evidence may be provided. It is advisory evidence only. Do not treat retrieved skill evidence as executable APIs, credentials, or a complete plan. Valid executable APIs must come from `apis.api_docs` or the AppWorld runtime.
"""


@dataclass
class OfficialReActExecutorConfig:
    max_interactions: int = 40
    timeout_seconds: int = 120
    max_wrong_completion_retries: int = 0
    completion_precheck_mode: str = "off"
    preflight_max_repairs: int = 0
    include_skill_evidence: bool = True
    max_history_chars: int = 16000
    valid_api_refs: set[tuple[str, str]] | None = None
    api_docs_context: str = ""


@dataclass
class EvidenceResult:
    prompt_text: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _build_official_code_repair_prompt(
    *,
    original_prompt: str,
    code: str,
    failed_check: dict[str, Any],
    stage: str,
) -> str:
    reason = str(failed_check.get("reason") or "unknown")
    error = str(failed_check.get("error") or "The previous code did not satisfy executor checks.")
    return "\n".join(
        [
            str(original_prompt),
            "",
            "[Code Repair Feedback]",
            f"Stage: {stage}",
            f"Reason: {reason}",
            error,
            "",
            "[Previous Code]",
            "```python",
            str(code or ""),
            "```",
            "",
            "[Repair Contract]",
            "Return one corrected Python code block only.",
            "Keep all task constraints from the original task and retrieved evidence.",
            "Do not call `complete_task` until the corrected code satisfies the feedback.",
        ]
    )


def _extract_code(text: str) -> str:
    from clstr.appworld_executor import extract_python_code

    return extract_python_code(text)


_ACCOUNT_MUTATION_APIS = {
    "delete_account",
    "reset_password",
    "send_password_reset_code",
    "send_verification_code",
    "signup",
    "update_account_name",
    "verify_account",
}


def _expanded_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", str(text or "").lower()))
    expanded = set(tokens)
    for token in tokens:
        if len(token) > 3 and token.endswith("ies"):
            expanded.add(token[:-3] + "y")
        if len(token) > 3 and token.endswith("s"):
            expanded.add(token[:-1])
        if len(token) > 4 and token.endswith("ed"):
            expanded.add(token[:-2])
    return expanded


def _expand_query_tokens_for_api_docs(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    synonyms = {
        "called": {"name", "title"},
        "containing": {"add", "include"},
        "make": {"create"},
        "made": {"create"},
    }
    for token in list(tokens):
        expanded.update(synonyms.get(token, set()))
    return expanded


_API_DOC_WRITE_ACTIONS = {
    "add",
    "approve",
    "archive",
    "book",
    "buy",
    "cancel",
    "categorize",
    "change",
    "copy",
    "create",
    "delete",
    "deny",
    "download",
    "follow",
    "like",
    "mark",
    "move",
    "order",
    "organize",
    "pay",
    "remove",
    "rename",
    "schedule",
    "send",
    "set",
    "transfer",
    "unfollow",
    "unlike",
    "update",
    "upload",
}


def _official_param_signature(parameters: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for param in parameters:
        name = str(param.get("name", "")).strip()
        if not name:
            continue
        ptype = str(param.get("type", "any"))
        required = "required" if bool(param.get("required")) else "optional"
        detail = f"{name}: {ptype} {required}"
        constraints = [str(item) for item in param.get("constraints", []) if str(item).strip()]
        if constraints:
            detail += f" constraints {'; '.join(constraints)}"
        parts.append(detail)
    return ", ".join(parts)


def _official_success_schema(doc: dict[str, Any], max_chars: int = 220) -> str:
    schema = (doc.get("response_schemas") or {}).get("success")
    if schema is None:
        return ""
    return json.dumps(schema, ensure_ascii=False, sort_keys=True)[: int(max_chars)]


def _task_api_ref_bonus(task: dict[str, Any]) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = set()
    for ref in task.get("api_refs", []) or []:
        if "." not in str(ref):
            continue
        app, api = str(ref).split(".", 1)
        refs.add((app, api))
    return refs


def build_official_api_docs_context(
    *,
    appworld_root: str | Path,
    task: dict[str, Any],
    max_apis_per_app: int = 14,
    max_chars: int = 12000,
    use_task_api_ref_bonus: bool = False,
) -> str:
    instruction = str(task.get("instruction_text") or task.get("instruction") or task.get("query") or "")
    query_tokens = _expand_query_tokens_for_api_docs(_expanded_tokens(instruction))
    explicit_refs = _task_api_ref_bonus(task) if bool(use_task_api_ref_bonus) else set()
    required_apps = [str(app) for app in task.get("required_apps", []) if str(app) != "supervisor"]
    docs_dir = Path(appworld_root) / "data" / "api_docs" / "standard"
    lines: list[str] = ["[Task-Relevant AppWorld API Docs]"]
    for app in required_apps:
        path = docs_dir / f"{app}.json"
        if not path.exists():
            continue
        try:
            docs = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        scored: list[tuple[int, str, dict[str, Any]]] = []
        for api_name, doc in docs.items():
            if not isinstance(doc, dict):
                continue
            doc_text = " ".join(
                [
                    str(api_name),
                    str(doc.get("description", "")),
                    " ".join(str(param.get("name", "")) for param in doc.get("parameters", []) or []),
                    _official_success_schema(doc),
                ]
            )
            doc_tokens = _expanded_tokens(doc_text)
            api_tokens = _expanded_tokens(str(api_name))
            score = 3 * len(query_tokens & doc_tokens)
            write_actions = api_tokens & _API_DOC_WRITE_ACTIONS
            if write_actions and write_actions & query_tokens:
                score += 18
                if (api_tokens - write_actions) & query_tokens:
                    score += 10
            if (app, str(api_name)) in explicit_refs:
                score += 100
            if "library" in query_tokens and "library" in doc_tokens:
                score += 12
            for focus in ("song", "album", "playlist", "genre", "play", "count", "like", "rating"):
                if focus in query_tokens and focus in doc_tokens:
                    score += 4
            if str(api_name) == "login":
                score += 2
            if str(api_name) in _ACCOUNT_MUTATION_APIS and not _task_allows_account_mutation(task):
                score -= 100
            scored.append((score, str(api_name), doc))
        emitted = 0
        for score, api_name, doc in sorted(scored, key=lambda item: (-item[0], item[1])):
            if score <= 0 and emitted > 0:
                continue
            params = _official_param_signature(list(doc.get("parameters", []) or []))
            desc = " ".join(str(doc.get("description", "")).split())
            returns = _official_success_schema(doc)
            suffix = f" returns {returns}" if returns else ""
            lines.append(f"- apis.{app}.{api_name}({params}): {desc}{suffix}")
            emitted += 1
            if emitted >= int(max_apis_per_app):
                break
    if len(lines) == 1:
        return ""
    text = "\n".join(lines)
    return text[: int(max_chars)]


def build_official_clstr_state_text(
    *,
    task: dict[str, Any],
    history: list[dict[str, Any]],
    api_docs_context: str = "",
    max_chars: int = 8000,
) -> str:
    from clstr.appworld_multistep import build_multistep_state_text

    base_state = build_multistep_state_text(
        task=task,
        steps=history,
        max_chars=max_chars,
        include_selected_skill_ids=False,
    )
    sections = [base_state]
    if str(api_docs_context or "").strip():
        sections.extend(["", "[Available API Inventory]", str(api_docs_context).strip()])
    return "\n".join(sections)[: int(max_chars)]


def _api_call_refs_from_code(code: str) -> set[tuple[str, str]]:
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return set()
    refs: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "apis"
        ):
            refs.add((str(func.value.attr), str(func.attr)))
    return refs


_DEFAULT_OFFICIAL_API_REFS = {
    ("api_docs", "show_app_descriptions"),
    ("api_docs", "show_api_descriptions"),
    ("api_docs", "show_api_doc"),
    ("supervisor", "show_profile"),
    ("supervisor", "show_account_passwords"),
    ("supervisor", "complete_task"),
}


class _CompleteTaskAnswerNormalizer(ast.NodeTransformer):
    def __init__(self, *, task: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.changed = False
        self.reason = "unchanged"
        self.state_change_only = _task_is_state_change_only(task or {})

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "complete_task"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "supervisor"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "apis"
        ):
            return node
        if self.state_change_only:
            original_arg_count = len(node.args)
            original_keyword_count = len(node.keywords)
            node.args = []
            node.keywords = [keyword for keyword in node.keywords if keyword.arg != "answer"]
            if len(node.args) != original_arg_count or len(node.keywords) != original_keyword_count:
                self.changed = True
                self.reason = "state_change_complete_task_answer_removed"
            return node
        has_answer_keyword = any(keyword.arg == "answer" for keyword in node.keywords)
        if len(node.args) == 1 and not has_answer_keyword:
            answer_arg = node.args.pop(0)
            node.keywords.insert(0, ast.keyword(arg="answer", value=answer_arg))
            self.changed = True
            self.reason = "complete_task_positional_answer_to_keyword"
        return node


def _is_api_attr(node: ast.AST, app: str, api: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == api
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == app
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "apis"
    )


def _is_api_call(node: ast.Call, app: str, api: str) -> bool:
    return _is_api_attr(node.func, app, api)


def _task_requests_unfiltered_spotify_playlist_library(task: dict[str, Any]) -> bool:
    normalized = " ".join(re.findall(r"[a-z0-9]+", _instruction_text(task).lower()))
    if not normalized:
        return False
    tokens = set(normalized.split())
    if ({"public", "private"} & tokens) and ({"playlist", "playlists"} & tokens):
        return False
    return (
        "all playlists" in normalized
        or "playlist library" in normalized
        or "playlists library" in normalized
    )


class _SpotifyPlaylistLibraryScopeNormalizer(ast.NodeTransformer):
    def __init__(self, *, task: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.changed = False
        self.reason = "unchanged"
        self.remove_visibility_filter = _task_requests_unfiltered_spotify_playlist_library(task or {})

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if not self.remove_visibility_filter:
            return node
        if not _is_api_call(node, "spotify", "show_playlist_library") and not any(
            _is_api_attr(arg, "spotify", "show_playlist_library") for arg in node.args
        ):
            return node
        original_keyword_count = len(node.keywords)
        node.keywords = [keyword for keyword in node.keywords if keyword.arg != "is_public"]
        if len(node.keywords) != original_keyword_count:
            self.changed = True
            self.reason = "spotify_playlist_library_visibility_filter_removed"
        return node


def _single_break(body: list[ast.stmt]) -> bool:
    return len(body) == 1 and isinstance(body[0], ast.Break)


def _not_name_expr(node: ast.AST) -> tuple[str, ast.AST] | None:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Name):
        return str(node.operand.id), copy.deepcopy(node)
    return None


def _len_lt_expr(node: ast.AST) -> tuple[str, ast.AST] | None:
    if not isinstance(node, ast.Compare):
        return None
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Lt):
        return None
    left = node.left
    if not (
        isinstance(left, ast.Call)
        and isinstance(left.func, ast.Name)
        and left.func.id == "len"
        and len(left.args) == 1
        and isinstance(left.args[0], ast.Name)
    ):
        return None
    return str(left.args[0].id), copy.deepcopy(node)


def _pagination_short_page_break_parts(test: ast.AST) -> tuple[str, ast.AST, ast.AST] | None:
    if not isinstance(test, ast.BoolOp) or not isinstance(test.op, ast.Or):
        return None
    empty_test: tuple[str, ast.AST] | None = None
    short_test: tuple[str, ast.AST] | None = None
    for value in test.values:
        empty_test = empty_test or _not_name_expr(value)
        short_test = short_test or _len_lt_expr(value)
    if empty_test is None or short_test is None:
        return None
    if empty_test[0] != short_test[0]:
        return None
    return empty_test[0], empty_test[1], short_test[1]


def _extend_arg_name(stmt: ast.stmt) -> str | None:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    call = stmt.value
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "extend":
        return None
    if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
        return None
    return str(call.args[0].id)


class _PaginationShortPageNormalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.changed = False
        self.reason = "unchanged"

    def visit_While(self, node: ast.While) -> ast.AST:
        node = self.generic_visit(node)
        new_body: list[ast.stmt] = []
        idx = 0
        while idx < len(node.body):
            current = node.body[idx]
            next_stmt = node.body[idx + 1] if idx + 1 < len(node.body) else None
            if isinstance(current, ast.If) and next_stmt is not None and _single_break(current.body) and not current.orelse:
                parts = _pagination_short_page_break_parts(current.test)
                extended_name = _extend_arg_name(next_stmt)
                if parts is not None and extended_name == parts[0]:
                    page_name, empty_test, short_test = parts
                    empty_guard = ast.copy_location(
                        ast.If(test=empty_test, body=[ast.Break()], orelse=[]),
                        current,
                    )
                    short_guard = ast.copy_location(
                        ast.If(test=short_test, body=[ast.Break()], orelse=[]),
                        current,
                    )
                    new_body.extend([empty_guard, next_stmt, short_guard])
                    self.changed = True
                    self.reason = "pagination_short_page_break_after_extend"
                    idx += 2
                    continue
            new_body.append(current)
            idx += 1
        node.body = new_body
        return node


def _instruction_text(task: dict[str, Any]) -> str:
    return str(task.get("instruction_text") or task.get("instruction") or task.get("query") or "")


def _task_expects_answer(task: dict[str, Any]) -> bool:
    text = _instruction_text(task).strip().lower()
    normalized = " ".join(re.findall(r"[a-z0-9]+", text))
    if "?" in text:
        return True
    if re.search(r"\b(what|which|who|whom|whose|when|where|why)\b", normalized):
        return True
    if re.search(r"\bhow (many|much|long|old|far|often)\b", normalized):
        return True
    answer_phrases = (
        "calculate",
        "compute",
        "count",
        "find out",
        "give me",
        "list",
        "provide",
        "return",
        "show me",
        "summarize",
        "tell me",
    )
    return any(normalized.startswith(phrase) for phrase in answer_phrases)


def _task_is_state_change_only(task: dict[str, Any]) -> bool:
    if _task_expects_answer(task):
        return False
    tokens = set(re.findall(r"[a-z0-9]+", _instruction_text(task).lower()))
    state_change_tokens = {
        "add",
        "approve",
        "archive",
        "book",
        "buy",
        "cancel",
        "categorize",
        "change",
        "copy",
        "create",
        "delete",
        "deny",
        "download",
        "follow",
        "like",
        "make",
        "mark",
        "move",
        "order",
        "organize",
        "pay",
        "remove",
        "remind",
        "rename",
        "schedule",
        "send",
        "set",
        "transfer",
        "unfollow",
        "unlike",
        "update",
        "upload",
    }
    return bool(tokens & state_change_tokens)


def _normalize_official_appworld_code(
    code: str, *, task: dict[str, Any] | None = None
) -> dict[str, Any]:
    original = str(code or "")
    try:
        tree = ast.parse(original)
    except SyntaxError as exc:
        return {
            "code": original,
            "changed": False,
            "reason": "syntax_error",
            "error": f"Python syntax error: {exc.msg} at line {exc.lineno}",
        }
    normalizers = [
        _CompleteTaskAnswerNormalizer(task=task),
        _SpotifyPlaylistLibraryScopeNormalizer(task=task),
        _PaginationShortPageNormalizer(),
    ]
    new_tree = tree
    reasons = []
    for normalizer in normalizers:
        new_tree = normalizer.visit(new_tree)
        if normalizer.changed:
            reasons.append(normalizer.reason)
    if not reasons:
        return {"code": original, "changed": False, "reason": "unchanged", "error": ""}
    ast.fix_missing_locations(new_tree)
    return {
        "code": ast.unparse(new_tree),
        "changed": True,
        "reason": "+".join(reasons),
        "error": "",
    }


def _task_allows_account_mutation(task: dict[str, Any]) -> bool:
    text = str(task.get("instruction_text") or task.get("instruction") or task.get("query") or "").lower()
    normalized = " ".join(re.findall(r"[a-z0-9]+", text))
    allowed_phrases = (
        "create account",
        "delete account",
        "reset password",
        "sign up",
        "signup",
        "update account",
        "verify account",
        "verification code",
    )
    return any(phrase in normalized for phrase in allowed_phrases)


def _task_needs_execution_guard(task: dict[str, Any]) -> bool:
    text = str(task.get("instruction_text") or task.get("instruction") or task.get("query") or "").lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    triggers = {
        "amount",
        "answer",
        "count",
        "date",
        "edm",
        "file",
        "files",
        "folder",
        "genre",
        "library",
        "libraries",
        "message",
        "messages",
        "money",
        "move",
        "much",
        "pay",
        "paid",
        "payment",
        "prefix",
        "ranking",
        "reply",
        "r",
        "send",
        "sort",
        "text",
        "top",
        "trash",
        "venmoed",
    }
    return bool(tokens & triggers) or "r&b" in text


def _task_needs_complete_ledger_guard(task: dict[str, Any], required_apps: list[str]) -> bool:
    text = str(task.get("instruction_text") or task.get("instruction") or task.get("query") or "").lower()
    tokens = set(re.findall(r"[a-z0-9]+", text))
    money_terms = {
        "amount",
        "bill",
        "dinner",
        "expense",
        "money",
        "much",
        "paid",
        "pay",
        "payment",
        "request",
        "transaction",
        "transactions",
        "venmoed",
    }
    aggregate_terms = {
        "all",
        "count",
        "everyone",
        "how",
        "many",
        "much",
        "others",
        "sum",
        "total",
    }
    return bool({"venmo", "splitwise"} & set(required_apps)) and bool(tokens & money_terms) and bool(tokens & aggregate_terms)


def _official_safety_preflight(
    code: str,
    task: dict[str, Any],
    *,
    valid_api_refs: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    refs = _api_call_refs_from_code(code)
    blocked = sorted(
        f"apis.{app}.{api}"
        for app, api in refs
        if api in _ACCOUNT_MUTATION_APIS and not _task_allows_account_mutation(task)
    )
    if blocked:
        return {
            "ok": False,
            "reason": "unrequested_account_mutation",
            "blocked_api_refs": blocked,
            "error": (
                "Safety preflight blocked unrequested account-management API call(s): "
                + ", ".join(blocked)
                + ". Use supervisor profile/password APIs and app login for authentication instead."
            ),
        }
    if valid_api_refs is not None:
        valid_refs = set(valid_api_refs) | set(_DEFAULT_OFFICIAL_API_REFS)
        invalid_refs = sorted(refs - valid_refs)
        if invalid_refs:
            invalid_text = [f"apis.{app}.{api}" for app, api in invalid_refs]
            return {
                "ok": False,
                "reason": "invalid_api_ref",
                "blocked_api_refs": [],
                "invalid_api_refs": invalid_text,
                "error": (
                    "Schema preflight blocked invalid AppWorld API call(s): "
                    + ", ".join(invalid_text)
                    + ". Use `apis.api_docs.show_api_descriptions` and `show_api_doc` to inspect valid APIs."
                ),
            }
    return {"ok": True, "reason": "ok", "blocked_api_refs": [], "invalid_api_refs": [], "error": ""}


def _compact_text(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= int(max_chars):
        return text
    return text[: max(0, int(max_chars) - 15)].rstrip() + "\n...[truncated]"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


_WRITE_API_VERBS = {
    "add",
    "approve",
    "archive",
    "book",
    "buy",
    "cancel",
    "categorize",
    "change",
    "copy",
    "create",
    "delete",
    "deny",
    "download",
    "follow",
    "like",
    "mark",
    "move",
    "order",
    "organize",
    "pay",
    "remove",
    "rename",
    "schedule",
    "send",
    "set",
    "transfer",
    "unfollow",
    "unlike",
    "update",
    "upload",
}


def _api_name_tokens(api_name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(api_name or "").lower()))


def _code_has_write_api_call(code: str) -> bool:
    for app, api in _api_call_refs_from_code(code):
        if app in {"api_docs", "supervisor"}:
            continue
        if _api_name_tokens(api) & _WRITE_API_VERBS:
            return True
    return False


def _code_calls_complete_task(code: str) -> bool:
    return ("supervisor", "complete_task") in _api_call_refs_from_code(code)


def _stmt_text(stmt: ast.AST | list[ast.stmt]) -> str:
    try:
        if isinstance(stmt, list):
            return "\n".join(ast.unparse(item) for item in stmt)
        return ast.unparse(stmt)
    except Exception:
        return ""


def _stmt_refs(stmt: ast.AST | list[ast.stmt]) -> set[tuple[str, str]]:
    return _api_call_refs_from_code(_stmt_text(stmt))


def _stmt_has_complete_task(stmt: ast.AST | list[ast.stmt]) -> bool:
    return ("supervisor", "complete_task") in _stmt_refs(stmt)


def _stmt_has_write_api(stmt: ast.AST | list[ast.stmt]) -> bool:
    for app, api in _stmt_refs(stmt):
        if app in {"api_docs", "supervisor"}:
            continue
        if _api_name_tokens(api) & _WRITE_API_VERBS:
            return True
    return False


def _stmt_has_api(stmt: ast.AST | list[ast.stmt], app: str, api: str) -> bool:
    return (app, api) in _stmt_refs(stmt)


def _requires_payment_message_write_guard(task: dict[str, Any], constraint_hints: list[str]) -> bool:
    text = _instruction_text(task).lower()
    required_apps = {str(app) for app in task.get("required_apps", [])}
    hint_text = " ".join(str(hint).lower() for hint in constraint_hints)
    has_payment = "venmo" in required_apps or "payment" in text or "pay" in text or "money" in text
    has_message = "phone" in required_apps and ("text" in text or "message" in text or "conversation" in text)
    hinted = "payment tasks based on a phone conversation" in hint_text or "after the payment action" in hint_text
    return bool((has_payment and has_message) or hinted)


def _guard_depends_on_optional_value(test: ast.AST) -> bool:
    text = _stmt_text(test).lower()
    if "none" not in text:
        return False
    return any(token in text for token in ("amount", "email", "phone", "contact", "recipient", "receiver"))


def _state_change_completion_guard_violation_reason(
    *,
    code: str,
    task: dict[str, Any],
    constraint_hints: list[str],
) -> str:
    if not _requires_payment_message_write_guard(task, constraint_hints):
        return ""
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        body_has_write = _stmt_has_write_api(node.body)
        orelse_has_write = _stmt_has_write_api(node.orelse)
        if body_has_write and _stmt_has_complete_task(node.orelse) and not orelse_has_write:
            return "state_change_completion_has_missing_write_fallback"
        if orelse_has_write and _stmt_has_complete_task(node.body) and not body_has_write:
            return "state_change_completion_has_missing_write_fallback"
    top_level = list(getattr(tree, "body", []) or [])
    for idx, stmt in enumerate(top_level):
        if not isinstance(stmt, ast.If):
            continue
        if not _guard_depends_on_optional_value(stmt.test):
            continue
        if not _stmt_has_api(stmt.body, "venmo", "create_transaction"):
            continue
        later = top_level[idx + 1 :]
        if _stmt_has_complete_task(later):
            return "state_change_completion_after_guarded_write"
    return ""


_ANSWER_CONSTRAINT_STOPWORDS = {
    "account",
    "across",
    "all",
    "and",
    "answer",
    "app",
    "apps",
    "artist",
    "artists",
    "based",
    "comma",
    "complete",
    "day",
    "days",
    "file",
    "files",
    "folder",
    "folders",
    "from",
    "give",
    "library",
    "libraries",
    "list",
    "many",
    "most",
    "played",
    "playlist",
    "playlists",
    "please",
    "separated",
    "song",
    "songs",
    "spotify",
    "task",
    "the",
    "their",
    "title",
    "titles",
    "top",
    "user",
    "what",
    "which",
    "with",
    "your",
}


def _semantic_tokens(text: str) -> set[str]:
    lowered = str(text or "").lower()
    tokens = set(re.findall(r"[a-z0-9]+(?:&[a-z0-9]+)?", lowered))
    expanded = set(tokens)
    for token in tokens:
        if "&" in token:
            expanded.add(token.replace("&", ""))
            expanded.update(part for part in token.split("&") if part)
    return expanded


_KNOWN_FILTER_CONSTRAINT_TOKENS = {
    "blues",
    "classical",
    "country",
    "edm",
    "hiphop",
    "indie",
    "jazz",
    "metal",
    "pop",
    "punk",
    "r&b",
    "rap",
    "rb",
    "rock",
}


def _quoted_constraint_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in re.findall(r"[\"']([^\"']+)[\"']", str(text or "")):
        tokens.update(token for token in _semantic_tokens(match) if len(token) >= 2)
    return tokens


def _capitalized_entity_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for phrase in re.findall(r"\b(?:[A-Z][a-z0-9]+(?:\s+[A-Z][a-z0-9]+)+)\b", str(text or "")):
        tokens.update(token for token in _semantic_tokens(phrase) if len(token) >= 3)
    return tokens


def _quantity_constraint_tokens(text: str) -> set[str]:
    lowered = str(text or "").lower()
    quantity_words = (
        "top",
        "first",
        "last",
        "latest",
        "oldest",
        "newest",
        "most",
        "least",
        "highest",
        "lowest",
        "limit",
        "exactly",
    )
    constraints: set[str] = set()
    for word in quantity_words:
        for match in re.findall(rf"\b{word}\s+(\d+)\b", lowered):
            constraints.add(match)
    for match in re.findall(r"\b(\d+)\s+(?:songs?|items?|results?|files?|transactions?|albums?|playlists?)\b", lowered):
        constraints.add(match)
    return constraints


def _answer_constraint_tokens(task: dict[str, Any]) -> list[str]:
    instruction = _instruction_text(task)
    tokens = _semantic_tokens(instruction)
    constraints: set[str] = set()
    for token in tokens:
        if token in _KNOWN_FILTER_CONSTRAINT_TOKENS:
            constraints.add(token)
    constraints.update(_quantity_constraint_tokens(instruction))
    constraints.update(_quoted_constraint_tokens(instruction))
    constraints.update(_capitalized_entity_tokens(instruction))
    constraints = {
        token
        for token in constraints
        if token not in _ANSWER_CONSTRAINT_STOPWORDS and (len(token) >= 2 or token.isdigit())
    }
    return sorted(constraints)


def _executable_code_text(code: Any) -> str:
    text = str(code or "")
    try:
        return ast.unparse(ast.parse(text))
    except SyntaxError:
        return text


def _history_code_text(history: list[dict[str, Any]]) -> str:
    return "\n".join(_executable_code_text(step.get("code") or step.get("raw_code") or "") for step in history)


def _history_has_app_api_call(history: list[dict[str, Any]]) -> bool:
    for step in history:
        for app, _api in _api_call_refs_from_code(str(step.get("code") or step.get("raw_code") or "")):
            if app not in {"api_docs", "supervisor"}:
                return True
    return False


def _task_requires_cross_library_genre_filter(task: dict[str, Any], constraint_tokens: list[str]) -> bool:
    tokens = _semantic_tokens(_instruction_text(task))
    has_genre = bool(set(constraint_tokens) & _KNOWN_FILTER_CONSTRAINT_TOKENS)
    has_all_sources = {"song", "album", "playlist"} <= tokens or {"songs", "albums", "playlists"} <= tokens
    has_library_scope = bool({"library", "libraries"} & tokens)
    return bool(has_genre and has_all_sources and has_library_scope)


def _iter_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _has_cross_library_genre_filter_over_union(code: str) -> bool:
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        iter_name = _iter_name(node.iter).lower()
        target_text = ast.unparse(node.target).lower()
        if "all_song_ids" not in iter_name and not ("song_id" in target_text and "all" in iter_name):
            continue
        body_text = "\n".join(ast.unparse(item) for item in node.body).lower()
        if "genre" in body_text:
            return True
    return False


def _constraint_hints_from_diagnostics(evidence_diagnostics: dict[str, Any] | None) -> list[str]:
    diagnostics = evidence_diagnostics if isinstance(evidence_diagnostics, dict) else {}
    hints = diagnostics.get("constraint_hints") or []
    if not isinstance(hints, (list, tuple)):
        return []
    return [str(hint) for hint in hints if str(hint).strip()]


def _task_numeric_literals(task: dict[str, Any]) -> set[str]:
    text = _instruction_text(task)
    values: set[str] = set()
    for match in re.findall(r"(?:\$|usd\s*)?(\d+(?:\.\d+)?)", str(text or "").lower()):
        values.add(match)
        if "." not in match:
            values.add(match + ".0")
    return values


def _violates_inclusion_exclusion_constraint(
    *,
    code: str,
    task: dict[str, Any],
    constraint_hints: list[str],
) -> bool:
    if not any("do not subtract it from records that already omit it" in hint.lower() for hint in constraint_hints):
        return False
    executable = _executable_code_text(code).lower()
    for value in _task_numeric_literals(task):
        if re.search(rf"-\s*{re.escape(value)}\b", executable):
            return True
    return False


def _assignment_texts(tree: ast.AST) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            try:
                assignments[target.id] = ast.unparse(node.value).lower()
            except Exception:
                continue
    return assignments


def _expand_assignment_refs(text: str, assignments: dict[str, str], *, max_rounds: int = 3) -> str:
    expanded = str(text or "").lower()
    for _ in range(max(0, int(max_rounds))):
        changed = False
        additions: list[str] = []
        for name, value in assignments.items():
            if re.search(rf"\b{re.escape(str(name).lower())}\b", expanded) and value not in expanded:
                additions.append(value)
                changed = True
        if not changed:
            break
        expanded = expanded + " " + " ".join(additions)
    return expanded


def _move_file_destination_texts(code: str) -> list[str]:
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return []
    assignments = _assignment_texts(tree)
    destinations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        try:
            call_text = ast.unparse(node.func)
        except Exception:
            call_text = ""
        if call_text != "apis.file_system.move_file":
            continue
        for keyword in node.keywords:
            if keyword.arg != "destination_file_path":
                continue
            try:
                destination = ast.unparse(keyword.value).lower()
            except Exception:
                destination = ""
            if isinstance(keyword.value, ast.Name):
                destination = assignments.get(keyword.value.id, destination)
            destinations.append(_expand_assignment_refs(destination, assignments))
    return destinations


def _violates_file_move_prefix_constraint(
    *,
    code: str,
    constraint_hints: list[str],
) -> bool:
    if not any("renamed/prefixed basename" in hint.lower() for hint in constraint_hints):
        return False
    for destination in _move_file_destination_texts(code):
        if "archive" not in destination:
            continue
        has_original_basename = "split('/')[-1]" in destination or 'split(\"/\")[-1]' in destination
        has_prefixed_basename = any(
            marker in destination
            for marker in (
                "new_file_name",
                "prefixed",
                "renamed",
                "created",
                "created_at",
                "date",
            )
        )
        if has_original_basename and not has_prefixed_basename:
            return True
    return False


def _requires_underscore_date_prefix(task: dict[str, Any], constraint_hints: list[str]) -> bool:
    instruction = _instruction_text(task).lower()
    if "yyyy_mm_dd" in instruction:
        return True
    return any("yyyy_mm_dd" in hint.lower() and "underscore" in hint.lower() for hint in constraint_hints)


def _has_underscore_date_reformat(code: str) -> bool:
    lowered = str(code or "").lower()
    return any(
        marker in lowered
        for marker in (
            ".replace('-', '_')",
            '.replace("-", "_")',
            "strftime('%y_%m_%d')",
            'strftime("%y_%m_%d")',
            "strftime('%Y_%m_%d')".lower(),
            'strftime("%Y_%m_%d")'.lower(),
            "'_'.join",
            '"_".join',
        )
    )


def _uses_raw_created_at_as_prefix(code: str) -> bool:
    lowered = str(code or "").lower()
    return bool(
        re.search(r"f[\"'][^\"']*\{created_at\}_", lowered)
        or re.search(r"created_at\s*\+\s*[\"']_", lowered)
    )


def _violates_date_prefix_format_constraint(
    *,
    code: str,
    task: dict[str, Any],
    constraint_hints: list[str],
) -> bool:
    if not _requires_underscore_date_prefix(task, constraint_hints):
        return False
    if _has_underscore_date_reformat(code):
        return False
    return _uses_raw_created_at_as_prefix(_executable_code_text(code))


def _completion_precheck(
    *,
    code: str,
    task: dict[str, Any],
    history: list[dict[str, Any]],
    mode: str,
    evidence_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not _code_calls_complete_task(code):
        return {"ok": True, "reason": "not_completion", "error": ""}
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode in {"", "off", "none", "false", "0"}:
        return {"ok": True, "reason": "disabled", "error": ""}
    if normalized_mode != "constraint_tokens":
        return {
            "ok": False,
            "reason": "unknown_completion_precheck_mode",
            "mode": normalized_mode,
            "error": f"Unknown completion precheck mode: {normalized_mode}",
        }

    refs = _api_call_refs_from_code(code)
    has_current_app_call = any(app not in {"api_docs", "supervisor"} for app, _api in refs)
    has_any_app_call = bool(has_current_app_call or _history_has_app_api_call(history))
    if _task_expects_answer(task) and not has_any_app_call:
        return {
            "ok": False,
            "reason": "answer_completion_without_app_evidence",
            "error": "Answer completion requires at least one task app API observation before complete_task.",
        }
    if _task_is_state_change_only(task) and not (
        _code_has_write_api_call(code)
        or any(_code_has_write_api_call(str(step.get("code") or step.get("raw_code") or "")) for step in history)
    ):
        return {
            "ok": False,
            "reason": "state_change_completion_without_write_evidence",
            "error": "State-change completion requires a prior or current task write API call.",
        }

    if _task_expects_answer(task):
        constraint_tokens = _answer_constraint_tokens(task)
        executable_text = _history_code_text(history) + "\n" + _executable_code_text(code)
        code_tokens = _semantic_tokens(executable_text)
        missing = [token for token in constraint_tokens if token not in code_tokens]
        if missing:
            return {
                "ok": False,
                "reason": "missing_answer_constraint_tokens",
                "missing_constraint_tokens": missing,
                "constraint_tokens": constraint_tokens,
                "error": (
                    "Answer completion is missing task constraint token(s) in the executed code path: "
                    + ", ".join(missing)
                    + ". Recompute the answer while preserving these constraints."
                ),
            }
        if _task_requires_cross_library_genre_filter(task, constraint_tokens) and not _has_cross_library_genre_filter_over_union(
            executable_text
        ):
            return {
                "ok": False,
                "reason": "cross_library_genre_filter_not_over_union",
                "constraint_tokens": constraint_tokens,
                "error": (
                    "For genre-constrained Spotify tasks across song, album, and playlist libraries, "
                    "build a combined song id set from every named source, then filter/rank songs by genre "
                    "from that combined song id set before completing."
                ),
            }
    constraint_hints = _constraint_hints_from_diagnostics(evidence_diagnostics)
    if _violates_inclusion_exclusion_constraint(code=code, task=task, constraint_hints=constraint_hints):
        return {
            "ok": False,
            "reason": "violates_inclusion_exclusion_constraint",
            "error": (
                "Generated completion appears to subtract a user amount/share even though the retrieved "
                "constraint says records already omit that item. Recompute the aggregate and add the "
                "stated user amount/share when the final answer includes the user."
            ),
        }
    if _violates_file_move_prefix_constraint(code=code, constraint_hints=constraint_hints):
        return {
            "ok": False,
            "reason": "violates_file_move_prefix_constraint",
            "error": (
                "Generated completion appears to move/archive an original basename without the required "
                "renamed/prefixed basename. Build destination_file_path from the final directory plus "
                "the prefixed basename before completing."
            ),
        }
    if _violates_date_prefix_format_constraint(code=code, task=task, constraint_hints=constraint_hints):
        return {
            "ok": False,
            "reason": "violates_date_prefix_format_constraint",
            "error": (
                "Generated completion appears to use a raw YYYY-MM-DD date string as a file-name prefix "
                "where the task requires YYYY_MM_DD_. Convert the date prefix with underscores, for "
                "example `created_at.replace('-', '_')`, before concatenating it with the basename."
            ),
        }
    state_change_guard_reason = _state_change_completion_guard_violation_reason(
        code=code,
        task=task,
        constraint_hints=constraint_hints,
    )
    if state_change_guard_reason:
        return {
            "ok": False,
            "reason": state_change_guard_reason,
            "error": (
                "Generated completion has a path that can call complete_task before the required "
                "state-changing write actions have actually succeeded. For payment/message tasks, "
                "do not complete in missing-value fallback branches; keep reading API records until "
                "amount, recipient, payment transaction, and requested text message are all resolved."
            ),
        }
    return {"ok": True, "reason": "ok", "error": ""}


def _history_needs_idempotent_write_recovery(history: list[dict[str, Any]]) -> bool:
    if not history:
        return False
    step = history[-1]
    output = str(step.get("execute_output") or "").lower()
    if not _code_has_write_api_call(str(step.get("code") or step.get("raw_code") or "")):
        return False
    if not ("execution failed" in output or "response status code" in output):
        return False
    return any(marker in output for marker in ("already", "exists", "duplicate", "satisfied"))


def _execution_error_recovery_hints(history: list[dict[str, Any]]) -> list[str]:
    if not history:
        return []
    step = history[-1]
    output = str(step.get("execute_output") or "")
    lowered = output.lower()
    if not ("execution failed" in lowered or "response status code" in lowered or "typeerror" in lowered):
        return []

    hints: list[str] = []
    if "list indices must be integers or slices" in lowered and "not str" in lowered:
        hints.append(
            "A previous API call returned a list, not a dict. Use the returned list directly "
            "(iterate it or index it with integers); do not access ['results'] unless the API docs "
            "show a dict response with a results field."
        )
    if "invalid min_created_at format" in lowered or "invalid max_created_at format" in lowered:
        hints.append(
            "Convert datetime values to date-only strings before passing date filters; "
            "min_created_at/max_created_at must be exactly YYYY-MM-DD. If you used "
            "`apis.phone.get_current_date_and_time()['date']`, it may be a natural-language date; "
            "do not slice its first four characters. Prefer the Task datetime above when available."
        )
    return hints


def _evaluation_report_payload(report: Any) -> Any:
    if hasattr(report, "to_dict") and callable(getattr(report, "to_dict")):
        try:
            return _json_safe(report.to_dict(stats_only=False))
        except TypeError:
            return _json_safe(report.to_dict())
        except Exception:
            pass
    if hasattr(report, "report") and callable(getattr(report, "report")):
        try:
            return _json_safe(report.report())
        except Exception:
            pass
    return _json_safe(report)


def _evaluation_report_success(report: Any) -> bool:
    if hasattr(report, "success"):
        return bool(getattr(report, "success"))
    if hasattr(report, "to_dict") and callable(getattr(report, "to_dict")):
        try:
            report_dict = report.to_dict(stats_only=True)
        except TypeError:
            report_dict = report.to_dict()
        except Exception:
            report_dict = None
        if isinstance(report_dict, dict) and "success" in report_dict:
            return bool(report_dict.get("success"))
    if hasattr(report, "report") and callable(getattr(report, "report")):
        try:
            report = report.report()
        except Exception:
            return False
    if isinstance(report, dict):
        return bool(report.get("success", report.get("passed", False)))
    text = str(report).lower()
    failed_match = re.search(r"num\s+failed\s+tests\s*:\s*(\d+)", text)
    total_match = re.search(r"num\s+total\s+tests\s*:\s*(\d+)", text)
    if failed_match:
        failed = int(failed_match.group(1))
        total = int(total_match.group(1)) if total_match else 1
        return failed == 0 and total > 0
    return "success" in text and "false" not in text


def _evaluate_world(world: Any) -> dict[str, Any]:
    try:
        try:
            report = world.evaluate(suppress_errors=True)
        except TypeError:
            report = world.evaluate()
        return {
            "success": _evaluation_report_success(report),
            "report": _evaluation_report_payload(report),
            "error": "",
        }
    except Exception as exc:
        return {"success": False, "report": "", "error": repr(exc)}


def _evaluation_success(world: Any) -> bool:
    return bool(_evaluate_world(world).get("success", False))


def _verifier_feedback_for_prompt(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return "Official evaluator failed; detailed trace omitted to avoid leaking reference answers."
    feedback: dict[str, Any] = {}
    for key in ("success", "num_tests", "difficulty"):
        if key in payload:
            feedback[key] = payload[key]
    failures = []
    for failure in payload.get("failures", []) or []:
        if not isinstance(failure, dict):
            continue
        failures.append(
            {
                key: failure.get(key)
                for key in ("requirement", "label")
                if failure.get(key) is not None
            }
        )
    if failures:
        feedback["failures"] = failures
    return feedback or "Official evaluator failed; detailed trace omitted to avoid leaking reference answers."


def _build_user_prompt(
    task: dict[str, Any],
    history: list[dict[str, Any]],
    evidence: str = "",
    *,
    max_history_chars: int = 16000,
    api_docs_context: str = "",
) -> str:
    instruction = str(task.get("instruction_text") or task.get("instruction") or task.get("query") or "")
    required_apps = [str(app) for app in task.get("required_apps", []) if str(app) != "supervisor"]
    parts = [
        f"Task: {instruction}",
        f"Task id: {task.get('task_id', task.get('query_id', ''))}",
    ]
    if task.get("task_datetime"):
        parts.append(
            f"Task datetime: {task['task_datetime']}\n"
            "Use this as the authoritative current time for relative-date constraints. "
            f"In code, set `task_datetime = \"{task['task_datetime']}\"`. "
            "For date API filters, convert it to date-only YYYY-MM-DD with `task_datetime.split('T')[0]`; "
            "do not append time to date-only filter parameters. "
            "An ongoing/current year starts on YYYY-01-01, and an ongoing/current month starts on YYYY-MM-01."
        )
    if required_apps:
        login_lines = [
            "[Authentication Helper]",
            "`apis.supervisor.show_account_passwords()` returns a list of rows, not a dict.",
            "Use this pattern when an app login/access token is required:",
            "```python",
            "profile = apis.supervisor.show_profile()",
            "password_by_app = {row['account_name']: row['password'] for row in apis.supervisor.show_account_passwords()}",
        ]
        for app in required_apps:
            username_expr = "profile['phone_number']" if app == "phone" else "profile['email']"
            login_lines.append(
                f"{app}_access_token = apis.{app}.login(username={username_expr}, password=password_by_app['{app}'])['access_token']"
            )
        login_lines.extend(["```", "Never use placeholder usernames, emails, passwords, or tokens."])
        parts.append("\n".join(login_lines))
    if "phone" in required_apps and any(app in {"venmo", "splitwise"} for app in required_apps):
        parts.append(
            "\n".join(
                [
                    "[Cross-App Identity Mapping]",
                    "When a task targets people by relationship, use `apis.phone.search_contacts(..., relationship=...)` when available.",
                    "Use the contact email to match or query Venmo/Splitwise users and transactions; use email, not phone_number, when comparing against transaction sender/receiver emails.",
                ]
            )
        )
    if "spotify" in required_apps:
        parts.append(
            "\n".join(
                [
                    "[Spotify Library Schema Guard]",
                    "song library rows use `song_id`; album and playlist rows use `song_ids`.",
                    "show_playlist(...)[\"songs\"] rows use `id`; call `show_song(song_id=row[\"id\"])` before reading `play_count` or detail `song_id`.",
                    "If task names song/album/playlist, inspect all named sources.",
                    "For all playlists, do not set `is_public=True`; omit `is_public` unless requested.",
                    "For top-played genre tasks, filter by genre before ranking by play_count.",
                ]
            )
        )
    if _task_needs_execution_guard(task):
        parts.append(
            "\n".join(
                [
                    "[Execution Correctness Guard]",
                    "Never guess task values; derive amounts, recipients, dates, ids, file names, and answers from APIs.",
                    "For a phone text conversation, inspect messages before payment/reply.",
                    "Preserve filters and source scope: filter before ranking/counting/sorting/moving/answering.",
                    "For file moves/renames, use only the basename under the target directory.",
                    "For file moves/renames, the destination path must include the target directory and final filename; build it from the target directory plus the source basename, not from a bare filename or directory alone.",
                ]
            )
        )
    if _task_needs_complete_ledger_guard(task, required_apps):
        parts.append(
            "\n".join(
                [
                    "[Complete Ledger Guard]",
                    "For money or transaction aggregation, prefer complete ledger APIs such as `show_transactions` or payment-request APIs when available.",
                    "Do not rely on social/feed APIs as the only source for totals, counts, or who-paid-whom questions.",
                    "Paginate ledger reads before filtering by person, date, description, status, direction, or amount.",
                ]
            )
        )
    if api_docs_context:
        parts.append(str(api_docs_context))
    if evidence:
        parts.append(
            "\n".join(
                [
                    "[Retrieved Evidence Guard]",
                    "User constraints override retrieved skill evidence. Do not let skill evidence drop or weaken filters such as genre, date, status, entity names, source/library scope, or ranking/count/limit constraints from the task.",
                    "Use retrieved evidence only as optional operational hints after preserving every user-specified constraint.",
                ]
            )
        )
        parts.append(evidence)
    # Small smoke tests deliberately use tight prompt budgets; reserve room for
    # fixed schema/helper sections so history compaction actually bounds prompt size.
    history_budget = int(max_history_chars)
    if history_budget < 1000:
        history_budget = max(120, history_budget // 2)
    per_step_budget = max(120, history_budget // max(len(history), 1))
    code_budget = max(120, per_step_budget // 3)
    output_budget = max(160, per_step_budget - code_budget)
    for step in history:
        parts.append("Assistant code:\n```python\n" + _compact_text(step.get("code", ""), code_budget) + "\n```")
        parts.append("Output:\n```\n" + _compact_text(step.get("execute_output", ""), output_budget) + "\n```")
        if bool(step.get("task_completed")) and not bool(step.get("evaluation_success")):
            verifier_payload = step.get("evaluate", "")
            if step.get("evaluate_error"):
                verifier_payload = step.get("evaluate_error")
            verifier_feedback = _verifier_feedback_for_prompt(verifier_payload)
            parts.append(
                "Official verifier:\n```\n"
                + _compact_text(
                    "The task was marked complete, but the official evaluator did not pass. "
                    "Continue fixing the state or answer, then call complete_task again only after addressing the verifier feedback.\n"
                    + str(verifier_feedback),
                    output_budget,
                )
                + "\n```"
            )
        precheck = step.get("completion_precheck")
        if isinstance(precheck, dict) and precheck.get("ok") is False:
            missing = precheck.get("missing_constraint_tokens") or []
            missing_text = ""
            if missing:
                missing_text = "\nMissing task constraint tokens: " + ", ".join(str(item) for item in missing)
            parts.append(
                "Completion precheck:\n```\n"
                + _compact_text(
                    str(precheck.get("error") or "Completion precheck blocked the previous complete_task call.")
                    + missing_text,
                    output_budget,
                )
                + "\n```"
            )
    if _history_needs_idempotent_write_recovery(history):
        parts.append(
            "\n".join(
                [
                    "[Write Failure Recovery]",
                    "The previous write API failure may mean the requested state is already satisfied.",
                    "Check the current state if useful, or retry safe idempotent write loops with `raise_on_failure=False` when the API supports it.",
                    "Do not abandon the task only because an add/create/like/follow/move operation reports already done, already exists, or already liked.",
                ]
            )
        )
    recovery_hints = _execution_error_recovery_hints(history)
    if recovery_hints:
        parts.append("[API Error Recovery]\n" + "\n".join(f"- {hint}" for hint in recovery_hints))
    parts.append("Write the next Python code block.")
    return "\n\n".join(parts)


def run_official_react_task(
    *,
    task: dict[str, Any],
    world: Any,
    generator: Any,
    config: OfficialReActExecutorConfig,
    evidence_builder: Any | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    system_prompt = build_official_react_system_prompt(max_interactions=config.max_interactions)
    history: list[dict[str, Any]] = []
    row = {"task_id": task.get("task_id") or task.get("query_id"), "steps": []}
    wrong_completion_retries = 0
    for step_idx in range(max(1, int(config.max_interactions))):
        if progress_callback is not None:
            progress_callback(
                "official_executor_step_prepare_start",
                {
                    "task_id": row["task_id"],
                    "step_idx": step_idx,
                    "history_steps": len(history),
                },
            )
        raw_evidence = evidence_builder(task=task, history=history) if evidence_builder else ""
        if isinstance(raw_evidence, EvidenceResult):
            evidence = raw_evidence.prompt_text
            evidence_diagnostics = dict(raw_evidence.diagnostics)
        else:
            evidence = str(raw_evidence or "")
            evidence_diagnostics = {}
        if progress_callback is not None:
            progress_callback(
                "official_executor_step_evidence_done",
                {
                    "task_id": row["task_id"],
                    "step_idx": step_idx,
                    "history_steps": len(history),
                    "evidence_chars": len(str(evidence)),
                },
            )
        user_prompt = _build_user_prompt(
            task,
            history,
            evidence,
            max_history_chars=config.max_history_chars,
            api_docs_context=config.api_docs_context,
        )
        if progress_callback is not None:
            progress_callback(
                "official_executor_step_start",
                {
                    "task_id": row["task_id"],
                    "step_idx": step_idx,
                    "history_steps": len(history),
                    "prompt_chars": len(user_prompt),
                },
            )
        response = generator.generate_text(system_prompt, user_prompt)
        raw_code = _extract_code(response)
        code_normalizer = _normalize_official_appworld_code(raw_code, task=task)
        code = str(code_normalizer.get("code", raw_code))
        preflight = _official_safety_preflight(code, task, valid_api_refs=config.valid_api_refs)
        completion_precheck = (
            _completion_precheck(
                code=code,
                task=task,
                history=history,
                mode=config.completion_precheck_mode,
                evidence_diagnostics=evidence_diagnostics,
            )
            if bool(preflight.get("ok"))
            else {"ok": True, "reason": "skipped_preflight_failed", "error": ""}
        )
        repair_count = 0
        repair_errors: list[dict[str, Any]] = []
        repair_raw_responses: list[str] = []
        repair_code_normalizers: list[dict[str, Any]] = []
        while (
            repair_count < max(0, int(config.preflight_max_repairs))
            and (not bool(preflight.get("ok")) or not bool(completion_precheck.get("ok")))
        ):
            failed_stage = "preflight" if not bool(preflight.get("ok")) else "completion_precheck"
            failed_check = preflight if failed_stage == "preflight" else completion_precheck
            repair_errors.append({"stage": failed_stage, **dict(failed_check)})
            repair_prompt = _build_official_code_repair_prompt(
                original_prompt=user_prompt,
                code=code,
                failed_check=failed_check,
                stage=failed_stage,
            )
            repair_response = generator.generate_text(system_prompt, repair_prompt)
            repair_raw_responses.append(repair_response)
            raw_code = _extract_code(repair_response)
            code_normalizer = _normalize_official_appworld_code(raw_code, task=task)
            repair_code_normalizers.append({key: value for key, value in code_normalizer.items() if key != "code"})
            code = str(code_normalizer.get("code", raw_code))
            preflight = _official_safety_preflight(code, task, valid_api_refs=config.valid_api_refs)
            completion_precheck = (
                _completion_precheck(
                    code=code,
                    task=task,
                    history=history,
                    mode=config.completion_precheck_mode,
                    evidence_diagnostics=evidence_diagnostics,
                )
                if bool(preflight.get("ok"))
                else {"ok": True, "reason": "skipped_preflight_failed", "error": ""}
            )
            repair_count += 1
        if bool(preflight.get("ok")) and bool(completion_precheck.get("ok")):
            execute_output = str(world.execute(code))
            execution_attempted = True
        else:
            if not bool(preflight.get("ok")):
                execute_output = "Safety preflight blocked before execution: " + str(preflight.get("error", ""))
            else:
                execute_output = "Completion precheck blocked before execution: " + str(
                    completion_precheck.get("error", "")
                )
            execution_attempted = False
        from clstr.appworld_executor import _execution_succeeded

        execution_ok = bool(execution_attempted) and bool(_execution_succeeded(execute_output))
        step = {
            "step_idx": step_idx,
            "user_prompt": user_prompt,
            "raw_response": response,
            "raw_code": raw_code,
            "code": code,
            "code_normalizer": dict(code_normalizer),
            "execute_output": execute_output,
            "execution_attempted": execution_attempted,
            "execution_ok": execution_ok,
            "preflight_ok": bool(preflight.get("ok")),
            "preflight": dict(preflight),
            "completion_precheck": dict(completion_precheck),
            "repair_count": int(repair_count),
            "repair_errors": repair_errors,
            "repair_raw_responses": repair_raw_responses,
            "repair_code_normalizers": repair_code_normalizers,
            "skill_evidence": evidence_diagnostics,
            "task_completed": bool(world.task_completed()),
            "evaluation_success": False,
        }
        if step["task_completed"]:
            evaluation = _evaluate_world(world)
            step["evaluation_success"] = bool(evaluation.get("success", False))
            step["evaluate"] = evaluation.get("report", "")
            if evaluation.get("error"):
                step["evaluate_error"] = evaluation.get("error")
        if evidence_builder is not None and callable(getattr(evidence_builder, "observe", None)):
            evidence_builder.observe(code=code, execute_output=execute_output, step=step)
        row["steps"].append(step)
        history.append(step)
        if progress_callback is not None:
            progress_callback(
                "official_executor_step_done",
                {
                    "task_id": row["task_id"],
                    "step_idx": step_idx,
                    "execution_ok": bool(step.get("execution_ok", False)),
                    "execution_attempted": bool(step.get("execution_attempted", False)),
                    "task_completed": bool(step.get("task_completed", False)),
                    "evaluation_success": bool(step.get("evaluation_success", False)),
                    "code_chars": len(str(code)),
                    "output_chars": len(str(execute_output)),
                },
            )
        if step["task_completed"]:
            if step["evaluation_success"]:
                step["stop_reason"] = "evaluation_success"
                break
            if wrong_completion_retries >= max(0, int(config.max_wrong_completion_retries)):
                step["stop_reason"] = "wrong_completion_retry_limit"
                break
            wrong_completion_retries += 1
            step["stop_reason"] = "wrong_completion_retry"
    row["task_completed"] = bool(row["steps"][-1]["task_completed"]) if row["steps"] else False
    row["evaluation_success"] = (
        bool(row["steps"][-1].get("evaluation_success", False)) if row["steps"] else _evaluation_success(world)
    )
    row["success"] = bool(row["task_completed"] and row["evaluation_success"])
    return row
