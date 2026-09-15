from clstr.appworld_smoke import run_appworld_smoke


def test_run_appworld_smoke_reports_missing_module(tmp_path):
    report = run_appworld_smoke(
        output_dir=tmp_path / "appworld_smoke",
        module_name="definitely_missing_appworld_module_for_test",
    )

    assert report["status"] == "blocked"
    assert report["blocker"] == "appworld_module_missing"
    assert (tmp_path / "appworld_smoke" / "blocker_report.json").exists()


def test_run_appworld_smoke_records_importable_module_without_reset(tmp_path):
    report = run_appworld_smoke(
        output_dir=tmp_path / "appworld_smoke",
        module_name="json",
    )

    assert report["status"] == "ok"
    assert report["import_ok"] is True
    assert report["reset_attempted"] is False
    assert (tmp_path / "appworld_smoke" / "report.json").exists()


def test_run_appworld_smoke_reports_missing_data_dir_when_root_is_known(tmp_path):
    appworld_root = tmp_path / "appworld_root"
    appworld_cache = tmp_path / "appworld_cache"
    appworld_root.mkdir()
    appworld_cache.mkdir()

    report = run_appworld_smoke(
        output_dir=tmp_path / "appworld_smoke",
        module_name="json",
        appworld_root=appworld_root,
        appworld_cache=appworld_cache,
    )

    assert report["status"] == "partial"
    assert report["blocker"] == "appworld_data_missing"
    assert report["appworld_root"] == str(appworld_root)
    assert report["appworld_cache"] == str(appworld_cache)
    assert report["data_dir_exists"] is False
    assert (tmp_path / "appworld_smoke" / "report.json").exists()
    assert (tmp_path / "appworld_smoke" / "blocker_report.json").exists()


def test_run_appworld_smoke_records_dataset_split_summary(tmp_path):
    appworld_root = tmp_path / "appworld_root"
    datasets_dir = appworld_root / "data" / "datasets"
    datasets_dir.mkdir(parents=True)
    (datasets_dir / "train.txt").write_text("task_a_1\ntask_b_1\n", encoding="utf-8")
    (datasets_dir / "dev.txt").write_text("task_c_1\n", encoding="utf-8")

    report = run_appworld_smoke(
        output_dir=tmp_path / "appworld_smoke",
        module_name="json",
        appworld_root=appworld_root,
    )

    assert report["status"] == "ok"
    assert report["task_splits"]["train"]["count"] == 2
    assert report["task_splits"]["train"]["sample_task_ids"] == ["task_a_1", "task_b_1"]
    assert report["task_splits"]["dev"]["count"] == 1
