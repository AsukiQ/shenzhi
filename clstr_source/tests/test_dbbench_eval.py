import json

from scripts.run_dbbench_clstr_eval import build_dbbench_harness_audit


def test_dbbench_harness_audit_marks_sft_dataset_as_not_closed_loop_harness(tmp_path):
    report = build_dbbench_harness_audit(output_path=tmp_path / "audit.json")

    assert report["status"] == "blocked_no_closed_loop_eval_harness"
    assert report["sft_dataset_is_eval_harness"] is False
    assert report["not_closed_loop_success"] is True
    assert "u-10bei/dbbench_sft_dataset_react_v4" in report["known_sft_dataset"]
    assert "official_closed_loop_harness" in report["missing_items"]
    assert "closed_loop_eval_command" in report["missing_items"]
    assert report["reproducible_commands"]["audit"].startswith("python")
    assert "<official-dbbench-package-or-repo>" in report["reproducible_commands"]["install_harness"]
    assert "<official-dbbench-data-download-command>" in report["reproducible_commands"]["download_data"]
    saved = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert saved["status"] == "blocked_no_closed_loop_eval_harness"
