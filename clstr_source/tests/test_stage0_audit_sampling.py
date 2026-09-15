from __future__ import annotations

from clstr.stage0_audit_sampling import select_audit_rows, select_rows_round_robin


def test_select_audit_rows_round_robins_benchmark_label_buckets():
    rows = []
    for idx in range(6):
        rows.append(
            {
                "benchmark": "alfworld",
                "task_id": f"alf-{idx}",
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
            }
        )
    for benchmark in ("toolbench_g3", "traject_bench"):
        rows.append(
            {
                "benchmark": benchmark,
                "task_id": benchmark,
                "skill_id": "skill/a",
                "next_skill_id": "skill/b",
                "loss_mask": {"routing": True, "L_trans_skill_ce": True},
            }
        )

    selected = select_audit_rows(rows, max_rows=3)

    assert len(selected) == 3
    assert {row["benchmark"] for row in selected} == {"alfworld", "toolbench_g3", "traject_bench"}


def test_select_rows_round_robin_accepts_custom_bucket_key():
    rows = [
        {"domain": "Education", "trajectory_type": "parallel", "id": "e1"},
        {"domain": "Education", "trajectory_type": "parallel", "id": "e2"},
        {"domain": "Travel", "trajectory_type": "parallel", "id": "t1"},
        {"domain": "Travel", "trajectory_type": "sequential", "id": "t2"},
    ]

    selected = select_rows_round_robin(
        rows,
        max_rows=3,
        key_fn=lambda row: (str(row["domain"]), str(row["trajectory_type"])),
    )

    assert [row["id"] for row in selected] == ["e1", "t1", "t2"]
