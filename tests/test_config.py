import json

import pytest

from upscale import config


def test_manifest_upscalers_and_default():
    m = config.manifest()
    assert m["default_upscaler"] in m["upscalers"]
    assert set(config.upscaler_names()) >= {"7b", "7b-sharp", "3b"}
    for f in list(m["upscalers"].values()) + m["shared"]:
        assert len(f["sha256"]) == 64 and f["bytes"] > 0 and f["url"].startswith("https://")


def test_default_upscaler_from_local_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert config.default_upscaler() == config.manifest()["default_upscaler"]
    (tmp_path / "config.local.json").write_text(json.dumps({"upscaler": "3b"}))
    assert config.default_upscaler() == "3b"
    assert config.upscaler_file() == config.manifest()["upscalers"]["3b"]["name"]
    (tmp_path / "config.local.json").write_text(json.dumps({"upscaler": "nonsense"}))
    assert config.default_upscaler() == config.manifest()["default_upscaler"]


def test_unknown_upscaler_rejected():
    with pytest.raises(ValueError):
        config.upscaler_file("99b")


def test_local_image_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCAL_IMAGE_DIR", str(tmp_path))
    assert config.local_image_dir() == tmp_path
