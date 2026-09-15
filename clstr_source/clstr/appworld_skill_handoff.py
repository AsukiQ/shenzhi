from __future__ import annotations

import re
from typing import Any

from clstr.evidence_gate import gate_verified_skill_handoff


_API_REF_RE = re.compile(r"\bapis\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")

_STOPWORDS = {
    "a",
    "all",
    "and",
    "api",
    "apis",
    "app",
    "appworld",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "have",
    "i",
    "in",
    "into",
    "is",
    "me",
    "my",
    "of",
    "on",
    "or",
    "skill",
    "skillx",
    "task",
    "that",
    "the",
    "them",
    "there",
    "to",
    "use",
    "user",
    "with",
}

_WRITE_PREFIXES = (
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

_STATE_CHANGING_GOAL_TERMS = {
    "add",
    "archive",
    "attach",
    "cancel",
    "change",
    "complete",
    "create",
    "delete",
    "edit",
    "export",
    "follow",
    "like",
    "make",
    "mark",
    "move",
    "pay",
    "post",
    "remove",
    "rename",
    "reply",
    "request",
    "save",
    "send",
    "set",
    "share",
    "submit",
    "text",
    "transfer",
    "update",
    "write",
}

_READ_ONLY_QUESTION_PREFIXES = (
    "how many",
    "how much",
    "what ",
    "which ",
    "who ",
    "when ",
    "where ",
)


def _normalize_token(token: str) -> str:
    token = str(token).lower()
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {
        _normalize_token(token)
        for token in re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
        if len(token) > 1 and token not in _STOPWORDS
    }


def _expand_goal_action_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    synonyms = {
        "called": {"name", "title"},
        "containing": {"add", "include"},
        "made": {"create"},
        "make": {"create"},
    }
    for token in list(tokens):
        expanded.update(synonyms.get(token, set()))
    return expanded


def _has_state_changing_goal_term(goal_text: str) -> bool:
    return bool(_expand_goal_action_tokens(_tokens(goal_text)) & _STATE_CHANGING_GOAL_TERMS)


def _is_read_only_goal(goal_text: str) -> bool:
    text = " ".join(str(goal_text or "").lower().split())
    if not text:
        return False
    if "how many" in text or "how much" in text:
        return True
    looks_like_question = text.endswith("?") or any(text.startswith(prefix) for prefix in _READ_ONLY_QUESTION_PREFIXES)
    return bool(looks_like_question and not _has_state_changing_goal_term(text))


def _is_state_changing_goal(goal_text: str) -> bool:
    return bool(not _is_read_only_goal(goal_text) and _has_state_changing_goal_term(goal_text))


def _goal_only_text(instruction: str) -> str:
    text = str(instruction or "")
    if "[Execution History]" in text:
        text = text.split("[Execution History]", 1)[0]
    if "[User Goal]" in text:
        text = text.split("[User Goal]", 1)[1]
    return text


def _api_refs_from_text(text: str) -> set[tuple[str, str]]:
    return {(app, api) for app, api in _API_REF_RE.findall(str(text or ""))}


def _api_ref_text(ref: tuple[str, str]) -> str:
    return f"apis.{ref[0]}.{ref[1]}"


def _api_ref_list(refs: set[tuple[str, str]]) -> list[str]:
    return [_api_ref_text(ref) for ref in sorted(refs)]


def _refs_from_api_texts(values: Any) -> set[tuple[str, str]]:
    refs: set[tuple[str, str]] = set()
    for value in values or []:
        text = str(value)
        if not text.startswith("apis."):
            continue
        bare = text.replace("apis.", "", 1)
        if "." not in bare:
            continue
        app, api = bare.split(".", 1)
        refs.add((app, api))
    return refs


def _is_auth_or_helper_ref(ref: tuple[str, str]) -> bool:
    app, api = ref
    api_name = str(api).lower()
    return str(app) == "supervisor" or api_name == "login" or "auth" in api_name


def _skill_schema_text(skill: dict[str, Any]) -> str:
    return " ".join(str(skill.get(key, "")) for key in ("executor_desc", "body", "skill_md"))


def _skill_public_text(skill: dict[str, Any]) -> str:
    return " ".join(str(skill.get(key, "")) for key in ("skill_id", "name", "description", "executor_desc"))


def _is_state_changing_api(api_name: str) -> bool:
    return str(api_name).startswith(_WRITE_PREFIXES)


def _state_action_matches_task(api_name: str, task_tokens: set[str]) -> bool:
    action_token = _normalize_token(str(api_name).split("_", 1)[0])
    allowed = _STATE_ACTION_TASK_SYNONYMS.get(action_token, {action_token})
    return bool(allowed & _expand_goal_action_tokens(task_tokens))


def _is_add_queue_goal(goal_text: str) -> bool:
    text = " ".join(str(goal_text).lower().split())
    return "add" in text and "queue" in text


def _is_unique_song_count_goal(goal_text: str) -> bool:
    text = " ".join(str(goal_text).lower().split())
    return ("unique" in text or "how many" in text or "count" in text) and "song" in text


def _is_note_export_goal(goal_text: str) -> bool:
    text = " ".join(str(goal_text).lower().split())
    return "export" in text and "note" in text


def _is_payment_message_goal(goal_text: str) -> bool:
    tokens = _tokens(goal_text)
    payment_terms = {"owed", "paid", "pay", "payment", "money", "grocery", "groceries", "bill"}
    message_terms = {"phone", "text", "message", "conversation"}
    return bool(tokens & payment_terms and tokens & message_terms)


def _has_queue_write_action(refs: set[tuple[str, str]]) -> bool:
    return any("queue" in api and api.startswith("add") for _app, api in refs)


def _pollution_reasons(skill: dict[str, Any], goal_text: str) -> list[str]:
    text = _skill_public_text(skill).lower()
    reasons: list[str] = []
    if _is_unique_song_count_goal(goal_text):
        ranking_terms = ("play count", "played", "most played", "top ", "rank", "popular")
        if any(term in text for term in ranking_terms):
            reasons.append("unique_count_goal_vs_ranking_skill")
        if "unique song" in str(goal_text).lower() and "unique song" not in text and "unique songs" not in text:
            reasons.append("unique_song_goal_requires_verified_workflow")
    if _is_add_queue_goal(goal_text):
        queue_pollution_terms = (
            "like song",
            "like songs",
            "liked song",
            "liked songs",
            "current",
            "current queue",
            "played so far",
            "show queue",
        )
        if any(term in text for term in queue_pollution_terms):
            reasons.append("add_queue_goal_vs_like_or_current_queue_skill")
    if _is_note_export_goal(goal_text):
        expense_terms = ("expense", "split", "share", "bill", "venmo", "payment")
        if any(term in text for term in expense_terms):
            reasons.append("note_export_goal_vs_expense_share_skill")
    return reasons


def _schema_refs(
    *,
    skill: dict[str, Any],
    valid_api_refs: set[tuple[str, str]],
    goal_text: str,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]], set[tuple[str, str]], int, set[tuple[str, str]]]:
    explicit_refs = _api_refs_from_text(_skill_schema_text(skill))
    valid_refs = explicit_refs & set(valid_api_refs or set())
    task_tokens = _tokens(goal_text)
    read_refs: set[tuple[str, str]] = set()
    action_refs: set[tuple[str, str]] = set()
    filtered_count = 0
    for ref in valid_refs:
        _app, api_name = ref
        if not _is_state_changing_api(api_name):
            read_refs.add(ref)
        elif _state_action_matches_task(api_name, task_tokens):
            action_refs.add(ref)
        else:
            filtered_count += 1
    return read_refs | action_refs, read_refs, action_refs, filtered_count, explicit_refs - set(valid_api_refs or set())


def _skill_apps(refs: set[tuple[str, str]]) -> set[str]:
    return {app for app, _api in refs if app != "supervisor"}


def _semantic_overlap(skill: dict[str, Any], goal_text: str) -> set[str]:
    generic = {
        "album",
        "api",
        "artist",
        "file",
        "library",
        "music",
        "note",
        "playlist",
        "simple",
        "song",
        "spotify",
        "system",
        "user",
    }
    return (_tokens(_skill_public_text(skill)) - generic) & (_tokens(goal_text) - generic)


def _reference_score_for(skill: dict[str, Any], reference_scores: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(reference_scores, dict):
        return {}
    raw = reference_scores.get(str(skill.get("skill_id", "")), {})
    return raw if isinstance(raw, dict) else {}


def _verify_skill(
    *,
    skill: dict[str, Any],
    instruction: str,
    valid_api_refs: set[tuple[str, str]],
    required_apps: set[str],
    reference_scores: dict[str, Any] | None,
) -> dict[str, Any]:
    goal_text = _goal_only_text(instruction)
    usable_refs, read_refs, action_refs, filtered_action_count, invalid_refs = _schema_refs(
        skill=skill,
        valid_api_refs=valid_api_refs,
        goal_text=goal_text,
    )
    reference = _reference_score_for(skill, reference_scores)
    ref_score = reference.get("operation_match_score")
    try:
        ref_score_float = float(ref_score) if ref_score is not None else None
    except (TypeError, ValueError):
        ref_score_float = None
    ref_label = str(reference.get("label", "")) if reference.get("label") is not None else ""
    strong_reference_match = ref_score_float is not None and ref_score_float >= 0.75 and ref_label != "unsafe_to_inject"
    skill_apps = _skill_apps(usable_refs)
    reasons: list[str] = []
    if required_apps and skill_apps and not (skill_apps & required_apps):
        reasons.append("required_app_mismatch")
    if not usable_refs:
        reasons.append("no_schema_grounded_api_evidence")
    if _is_read_only_goal(goal_text) and action_refs:
        reasons.append("read_only_goal_with_state_changing_skill")
    if _is_state_changing_goal(goal_text) and usable_refs and not action_refs and not strong_reference_match:
        reasons.append("state_changing_goal_without_state_changing_api")
    reasons.extend(_pollution_reasons(skill, goal_text))
    if filtered_action_count and not action_refs:
        reasons.append("state_changing_action_mismatch")
    if _is_add_queue_goal(goal_text) and not _has_queue_write_action(action_refs):
        reasons.append("add_queue_goal_without_queue_write_api")

    overlap = _semantic_overlap(skill, goal_text)

    if reasons:
        decision = "suppress"
    elif action_refs or overlap or strong_reference_match:
        decision = "inject_hint"
    else:
        decision = "schema_only"

    return {
        "skill_id": str(skill.get("skill_id", "")),
        "decision": decision,
        "reasons": reasons,
        "valid_api_refs": _api_ref_list(usable_refs),
        "read_support_apis": _api_ref_list(read_refs),
        "state_changing_action_apis": _api_ref_list(action_refs),
        "invalid_skill_api_refs": _api_ref_list(invalid_refs),
        "filtered_state_changing_api_count": int(filtered_action_count),
        "semantic_overlap": sorted(overlap),
        "reference_label": ref_label,
        "reference_operation_match_score": ref_score_float,
    }


def _workflow_hints(
    *,
    instruction: str,
    usable_refs: set[tuple[str, str]],
    action_refs: set[tuple[str, str]],
) -> list[str]:
    goal_text = _goal_only_text(instruction)
    refs = {_api_ref_text(ref) for ref in usable_refs | action_refs}
    hints: list[str] = []
    if _is_add_queue_goal(goal_text) and any(ref.endswith(".add_to_queue") or "add" in ref and "queue" in ref for ref in refs):
        hints.append("Search/filter songs by the requested entity and condition.")
        hints.append("Do not invent artist ids; use search results or detail responses and check `song[\"artists\"]` names.")
        hints.append("Never use literal or placeholder `artist_id` values such as 1 or 12345.")
        hints.append("Do not stop after an exploratory search call; in the same code block, add all matching songs to the queue and complete the task.")
        hints.append("Add each matching song to the queue.")
    if _is_unique_song_count_goal(goal_text):
        library_refs = {
            "apis.spotify.show_song_library",
            "apis.spotify.show_album_library",
            "apis.spotify.show_playlist_library",
        }
        if refs & library_refs:
            hints.append("Collect song ids across song library, album library, and playlist library before counting unique songs.")
    if _is_note_export_goal(goal_text) and any(ref.startswith("apis.simple_note.") for ref in refs):
        hints.append("Search notes, fetch full note content, then write/export the requested file.")
    return list(dict.fromkeys(hints))


def _constraint_hints(
    *,
    instruction: str,
    usable_refs: set[tuple[str, str]],
    action_refs: set[tuple[str, str]],
) -> list[str]:
    goal_text = _goal_only_text(instruction)
    normalized_goal = " ".join(goal_text.lower().split())
    refs = usable_refs | action_refs
    hints: list[str] = []

    if _is_read_only_goal(goal_text):
        hints.append("Do not call complete_task with an answer until the computed value satisfies all inclusion/exclusion constraints in the user goal.")
        has_explicit_amount = bool(re.search(r"(?:\$|usd|dollar|amount|share|total)", normalized_goal))
        has_exclusion = " except " in f" {normalized_goal} " or "excluding" in normalized_goal or "not include" in normalized_goal
        has_user_inclusion = "including me" in normalized_goal or "including the user" in normalized_goal or "my share" in normalized_goal
        if has_explicit_amount and has_exclusion and has_user_inclusion:
            hints.append("For aggregate answers where records omit the user's item but the final answer includes the user, add the explicitly stated user amount/share to the aggregate; do not subtract it from records that already omit it.")

    if ("file_system", "move_file") in refs:
        file_goal_terms = _tokens(goal_text)
        if {"move", "archive", "rename", "prefix", "organize"} & file_goal_terms:
            hints.append("For file move/rename operations, destination path must include the target directory and final file name.")
        if "prefix" in file_goal_terms or "rename" in file_goal_terms:
            hints.append("When adding prefixes, derive file names from the basename, not the full source path.")
        if "yyyy_mm_dd" in normalized_goal:
            hints.append("When adding a quoted date prefix like YYYY_MM_DD_, format the date with underscores before concatenating it with the basename.")
        if ("prefix" in file_goal_terms or "rename" in file_goal_terms) and ({"move", "archive"} & file_goal_terms):
            hints.append("When a file must be both renamed/prefixed and moved, combine the final directory with the renamed/prefixed basename in one destination path.")

    if action_refs:
        hints.append("Do not complete the task until all requested state-changing actions have been applied and checked against the user constraints.")
    if _is_payment_message_goal(goal_text) and ("venmo", "create_transaction") in refs:
        hints.append("For payment tasks based on a phone conversation, read the conversation before sending money; derive the amount and recipient from API records, not guesses.")
        if ("phone", "send_text_message") in refs:
            hints.append("After the payment action, send the requested phone text message to the same resolved contact.")
        if re.search(r"[\"'][^\"']+[\"']", goal_text):
            hints.append("Preserve quoted notes/messages exactly when passing payment notes or message text.")
    return list(dict.fromkeys(hints))


def _count_decisions(decisions: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"inject_hint": 0, "schema_only": 0, "suppress": 0}
    for row in decisions:
        decision = str(row.get("decision", ""))
        if decision in counts:
            counts[decision] += 1
    return counts


def _format_legacy_hints_prompt(
    *,
    useful_apis: list[str],
    state_changing_action_apis: list[str],
    constraint_hints: list[str],
    workflow_hints: list[str],
    max_chars: int,
) -> str:
    lines = [
        "[Optional Retrieved Hints]",
        "These hints are extracted from retrieved examples and may be incomplete.",
        "Use the user goal and [AppWorld API Docs] as authoritative.",
        "Do not infer API names from retrieved hints.",
        "Ignore these hints if they conflict with the task or API docs.",
        "",
        "Potentially useful APIs:",
    ]
    useful = list(dict.fromkeys(str(api) for api in useful_apis if str(api).strip()))
    lines.extend(f"- {api}" for api in useful) if useful else lines.append("- (none)")
    lines.extend(["", "State-changing APIs with operation match:"])
    actions = list(dict.fromkeys(str(api) for api in state_changing_action_apis if str(api).strip()))
    lines.extend(f"- {api}" for api in actions) if actions else lines.append("- (none)")
    lines.extend(["", "Constraint checks:"])
    constraints = list(dict.fromkeys(str(hint) for hint in constraint_hints if str(hint).strip()))
    lines.extend(f"- {hint}" for hint in constraints) if constraints else lines.append("- (none)")
    lines.extend(["", "Possible workflow hints:"])
    hints = list(dict.fromkeys(str(hint) for hint in workflow_hints if str(hint).strip()))
    lines.extend(f"- {hint}" for hint in hints) if hints else lines.append("- (none)")
    lines.extend(["", "raw_skill_text_omitted: true"])
    return "\n".join(lines)[: int(max_chars)]


def build_verified_skill_handoff(
    *,
    instruction: str,
    skills: list[dict[str, Any]],
    valid_api_refs: set[tuple[str, str]],
    required_apps: set[str] | list[str] | tuple[str, ...] | None = None,
    reference_scores: dict[str, Any] | None = None,
    controller_diagnostics: dict[str, Any] | None = None,
    max_chars: int = 1600,
    prompt_style: str = "structured_evidence",
    visible_skill_limit: int | None = None,
) -> dict[str, Any]:
    style = str(prompt_style or "structured_evidence")
    if style not in {"structured_evidence", "legacy_hints"}:
        raise ValueError(f"unsupported verified handoff prompt_style: {style}")
    required_app_set = {str(app) for app in (required_apps or []) if str(app) != "supervisor"}
    valid_refs = set(valid_api_refs or set())
    decisions = [
        _verify_skill(
            skill=skill,
            instruction=instruction,
            valid_api_refs=valid_refs,
            required_apps=required_app_set,
            reference_scores=reference_scores,
        )
        for skill in skills
    ]

    if visible_skill_limit is None:
        visible_count = len(decisions)
    else:
        visible_count = max(0, min(int(visible_skill_limit), len(decisions)))
    visible_decisions = list(decisions[:visible_count])
    visible_skill_ids = [str(row.get("skill_id", "")) for row in visible_decisions]
    rescued_action_decisions = [
        row
        for row in decisions[visible_count:]
        if row.get("decision") != "suppress" and row.get("state_changing_action_apis")
    ]
    rescued_action_skill_ids = [str(row.get("skill_id", "")) for row in rescued_action_decisions]
    evidence_decisions = visible_decisions + rescued_action_decisions

    useful_refs: set[tuple[str, str]] = set()
    action_refs: set[tuple[str, str]] = set()
    suppressed_refs: set[tuple[str, str]] = set()
    for row in evidence_decisions:
        row_refs = _refs_from_api_texts(row.get("valid_api_refs", []))
        row_actions = _refs_from_api_texts(row.get("state_changing_action_apis", []))
        if row.get("decision") == "suppress":
            suppressed_refs.update(row_refs)
        else:
            useful_refs.update(row_refs)
            action_refs.update(row_actions)

    fallback_source = ""
    goal_text = _goal_only_text(instruction)
    if not useful_refs and _is_add_queue_goal(goal_text) and ("spotify", "add_to_queue") in valid_refs:
        fallback_source = "task_schema"
        for ref in (
            ("spotify", "search_songs"),
            ("spotify", "show_song"),
            ("spotify", "add_to_queue"),
        ):
            if ref in valid_refs:
                useful_refs.add(ref)
        action_refs.add(("spotify", "add_to_queue"))
    if not useful_refs and _is_payment_message_goal(goal_text) and ("venmo", "create_transaction") in valid_refs:
        fallback_source = "task_schema_payment_message"
        for ref in (
            ("phone", "search_contacts"),
            ("phone", "search_text_messages"),
            ("venmo", "create_transaction"),
            ("phone", "send_text_message"),
        ):
            if ref in valid_refs:
                useful_refs.add(ref)
        for ref in (("venmo", "create_transaction"), ("phone", "send_text_message")):
            if ref in valid_refs:
                action_refs.add(ref)
    if useful_refs and not action_refs and all(_is_auth_or_helper_ref(ref) for ref in useful_refs):
        fallback_source = fallback_source or "auth_only_evidence_suppressed"
        useful_refs.clear()

    workflow_hints = _workflow_hints(
        instruction=instruction,
        usable_refs=useful_refs,
        action_refs=action_refs,
    )
    constraint_hints = _constraint_hints(
        instruction=instruction,
        usable_refs=useful_refs,
        action_refs=action_refs,
    )
    gate_decisions = decisions
    if fallback_source == "task_schema_payment_message":
        gate_decisions = decisions + [
            {
                "skill_id": "schema_fallback/payment_message",
                "decision": "schema_only",
                "reasons": [],
                "valid_api_refs": _api_ref_list(useful_refs),
                "read_support_apis": _api_ref_list(useful_refs - action_refs),
                "state_changing_action_apis": _api_ref_list(action_refs),
                "invalid_skill_api_refs": [],
                "filtered_state_changing_api_count": 0,
                "semantic_overlap": [],
                "reference_label": "",
                "reference_operation_match_score": 0.0,
            }
        ]
    gate = gate_verified_skill_handoff(
        selected_skill_ids=[str(skill.get("skill_id", "")) for skill in skills],
        handoff_decisions=gate_decisions,
        useful_apis=_api_ref_list(useful_refs),
        state_changing_action_apis=_api_ref_list(action_refs),
        workflow_hints=workflow_hints,
        constraint_hints=constraint_hints,
        controller_diagnostics=controller_diagnostics,
        max_chars=max_chars,
    )
    prompt_block = gate.prompt_block
    if style == "legacy_hints" and gate.decision.decision != "no_evidence_fallback":
        visible_hints = workflow_hints if gate.decision.decision == "workflow_hint_evidence" else []
        prompt_block = _format_legacy_hints_prompt(
            useful_apis=_api_ref_list(useful_refs),
            state_changing_action_apis=_api_ref_list(action_refs),
            constraint_hints=constraint_hints,
            workflow_hints=visible_hints,
            max_chars=max_chars,
        )
    counts = _count_decisions(decisions)
    return {
        "skill_context_mode": "verified_hints",
        "handoff_decision": "verified_hints",
        "prompt_style": style,
        "input_skill_count": len(skills),
        "selected_skill_ids": [str(skill.get("skill_id", "")) for skill in skills],
        "visible_skill_limit": visible_skill_limit,
        "visible_skill_ids": visible_skill_ids,
        "rescued_action_skill_ids": rescued_action_skill_ids,
        "hidden_skill_text_fields": ["skill_id", "name", "description", "body", "skill_md"],
        "decisions": decisions,
        "handoff_decisions": decisions,
        "handoff_decision_counts": counts,
        "kept_skill_ids": [row["skill_id"] for row in decisions if row.get("decision") != "suppress"],
        "suppressed_skill_ids": [row["skill_id"] for row in decisions if row.get("decision") == "suppress"],
        "useful_apis": _api_ref_list(useful_refs),
        "state_changing_action_apis": _api_ref_list(action_refs),
        "suppressed_schema_refs": _api_ref_list(suppressed_refs),
        "workflow_hints": workflow_hints,
        "constraint_hints": constraint_hints,
        "fallback_source": fallback_source,
        "gate_decision": gate.decision.decision,
        "gate_confidence": gate.decision.confidence,
        "gate_reasons": gate.decision.reasons,
        "evidence_gate": gate.decision.to_dict(),
        "exact_base_executor_fallback": gate.decision.exact_base_executor_fallback,
        "prompt_visible_evidence_text": prompt_block,
        "prompt_block": prompt_block,
    }
