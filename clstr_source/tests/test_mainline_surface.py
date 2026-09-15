from __future__ import annotations

from pathlib import Path


LEGACY_PATHS = (
    "clstr/stage" + "3_unified_" + "hrpo.py",
    "clstr/qwen_stage" + "3_unified_" + "hrpo.py",
    "clstr/stage" + "3_quality_gate.py",
    "clstr/online_" + "hrpo.py",
    "clstr/appworld_act_" + "hrpo.py",
    "clstr/rollout.py",
    "clstr/train.py",
    "clstr/infer.py",
)


def test_legacy_stage3_hrpo_and_train_surfaces_are_deleted():
    existing = [path for path in LEGACY_PATHS if Path(path).exists()]

    assert existing == []


def test_active_training_surface_contains_only_stage0_stage1_stage2_stage4():
    forbidden = (
        "Stage" + "3",
        "HR" + "PO",
        "stage" + "3_unified_" + "hrpo",
        "qwen_stage" + "3_unified_" + "hrpo",
        "stage" + "3_joint",
        "from clstr.roll" + "out import",
        "from clstr." + "train import",
        "run_" + "clstr_qwen" + "3_online_" + "hrpo",
        "run_" + "clstr_v4_1b_alfworld_online_" + "hrpo",
        "run_appworld_clstr_" + "hrpo_train",
    )
    active_roots = [Path("clstr"), Path("scripts"), Path("README.md")]
    hits = []
    for root in active_roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits.extend((str(path), token) for token in forbidden if token in text)

    assert hits == []


def test_readme_documents_only_the_four_stage_causal_training_path():
    text = Path("README.md").read_text(encoding="utf-8")

    assert "Stage0 → Stage1 → Stage2 → Stage4" in text
    assert "Stage3" not in text
    assert "HRPO" not in text
