from pathlib import Path

from clstr.skillret_official import _default_hf_pooling


def test_bge_m3_local_model_uses_cls_pooling(tmp_path: Path) -> None:
    model_dir = tmp_path / "bge-m3"
    pooling_dir = model_dir / "1_Pooling"
    pooling_dir.mkdir(parents=True)
    (pooling_dir / "config.json").write_text(
        '{"pooling_mode_cls_token": true, "pooling_mode_mean_tokens": false}\n',
        encoding="utf-8",
    )

    assert _default_hf_pooling(str(model_dir)) == "cls"


def test_non_bge_models_keep_last_token_default(tmp_path: Path) -> None:
    model_dir = tmp_path / "Qwen3-Embedding-0.6B"
    model_dir.mkdir()

    assert _default_hf_pooling(str(model_dir)) == "last_token"
