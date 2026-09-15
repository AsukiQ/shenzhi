from __future__ import annotations

from pathlib import Path

from clstr.exact_candidate_cache import load_or_build_exact_candidate_rankings


def _source_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"hidden_size": 2}\n', encoding="utf-8")
    skills = tmp_path / "selected_skills.jsonl"
    train = tmp_path / "train_queries.jsonl"
    eval_path = tmp_path / "eval_queries.jsonl"
    skills.write_text('{"skill_id":"skill/a"}\n{"skill_id":"skill/b"}\n', encoding="utf-8")
    train.write_text('{"query_id":"q0"}\n', encoding="utf-8")
    eval_path.write_text('{"query_id":"q1"}\n', encoding="utf-8")
    return checkpoint, skills, train, eval_path


def test_exact_candidate_cache_hits_and_content_changes_rebuild(tmp_path: Path) -> None:
    checkpoint, skills, train, eval_path = _source_files(tmp_path)
    calls = []

    def builder():
        calls.append(1)
        return {"q0": ["skill/a"], "q1": ["skill/b"]}

    kwargs = {
        "candidate_cache_dir": tmp_path / "cache",
        "checkpoint_path": checkpoint,
        "selected_skills_path": skills,
        "train_queries_path": train,
        "eval_queries_path": eval_path,
        "pooling": "cls",
        "query_text_mode": "raw",
        "tokenizer_padding_side": "right",
        "torch_dtype": "bfloat16",
        "max_length": 512,
        "top_k": 1,
        "max_train_queries": None,
        "max_eval_queries": None,
        "skill_ids": ["skill/a", "skill/b"],
        "query_ids": ["q0", "q1"],
        "builder": builder,
    }
    first, first_report = load_or_build_exact_candidate_rankings(**kwargs)
    second, second_report = load_or_build_exact_candidate_rankings(**kwargs)

    assert first == second
    assert first_report["hit"] is False
    assert second_report["hit"] is True
    assert len(calls) == 1

    train.write_text('{"query_id":"q0","revision":2}\n', encoding="utf-8")
    _third, third_report = load_or_build_exact_candidate_rankings(**kwargs)
    assert third_report["hit"] is False
    assert third_report["identity"] != first_report["identity"]
    assert len(calls) == 2


def test_exact_candidate_cache_rejects_rankings_outside_query_catalog(tmp_path: Path) -> None:
    checkpoint, skills, train, eval_path = _source_files(tmp_path)

    try:
        load_or_build_exact_candidate_rankings(
            candidate_cache_dir=None,
            checkpoint_path=checkpoint,
            selected_skills_path=skills,
            train_queries_path=train,
            eval_queries_path=eval_path,
            pooling="cls",
            query_text_mode="raw",
            tokenizer_padding_side="right",
            torch_dtype="bfloat16",
            max_length=512,
            top_k=1,
            max_train_queries=None,
            max_eval_queries=None,
            skill_ids=["skill/a", "skill/b"],
            query_ids=["q0", "q1"],
            candidate_skill_ids_by_query=[["skill/a"], ["skill/b"]],
            builder=lambda: {"q0": ["skill/b"], "q1": ["skill/b"]},
        )
    except ValueError as exc:
        assert "legal catalog" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected an illegal candidate ranking to be rejected")
