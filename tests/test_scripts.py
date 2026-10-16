import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_safety_check_patterns():
    safe = load("check_repo_safe")
    probs = []
    # Built in pieces so this file doesn't trip the check itself.
    safe.scan_text("photos in D:" + "\\Photos\\Family and C:" + "\\Users\\someone\\x", "f", probs)
    labels = " ".join(probs)
    assert "absolute drive path" in labels and "Windows user path" in labels
    for p in ["D:" + "\\Family Photos\\bunica.jpg", "E:" + "/Photos/Family/x.jpg"]:
        probs = []
        safe.scan_text(p, "f", probs)
        assert probs, p
    probs = []
    safe.scan_text('opencv-python==5.0.0.93  "D:\\\\path\\\\to\\\\local-image"  -ReuseModels D:\\old\\seedvr2\\models',
                   "f", probs)
    assert probs == []


def test_download_choice_keeps_installed_and_config(monkeypatch, tmp_path):
    dl = load("download_models")
    monkeypatch.setattr(dl, "ROOT", tmp_path)
    (tmp_path / "models.json").write_text((ROOT / "models.json").read_text(encoding="utf-8"), encoding="utf-8")
    fetched = []
    monkeypatch.setattr(dl, "fetch", lambda f, r: fetched.append(f["name"]) or True)
    monkeypatch.setattr("sys.argv", ["download_models.py"])
    ups = json.loads((ROOT / "models.json").read_text(encoding="utf-8"))["upscalers"]

    def seedvr2_fetched():
        return [n for n in fetched if n.startswith("seedvr2_ema")]

    assert dl.main() == 0 and seedvr2_fetched() == [ups["7b"]["name"]]          # fresh install: the default
    fetched.clear()
    (tmp_path / "config.local.json").write_text(json.dumps({"upscaler": "3b"}), encoding="utf-8-sig")
    assert dl.main() == 0 and seedvr2_fetched() == [ups["3b"]["name"]]          # the configured choice
    fetched.clear()
    (tmp_path / "models" / "seedvr2").mkdir(parents=True)
    (tmp_path / "models" / "seedvr2" / ups["7b-sharp"]["name"]).write_text("")
    assert dl.main() == 0 and seedvr2_fetched() == [ups["7b-sharp"]["name"]]    # re-run: what's installed
    assert json.loads((tmp_path / "config.local.json").read_text(encoding="utf-8-sig")) == {"upscaler": "3b"}


def test_save_default_never_clobbers_invalid_config(monkeypatch, tmp_path):
    dl = load("download_models")
    monkeypatch.setattr(dl, "ROOT", tmp_path)
    bad = '{"local_image_dir": "x\\no\\escape"}'  # a Windows path pasted without doubling the backslashes
    (tmp_path / "config.local.json").write_text(bad, encoding="utf-8")
    dl.save_default("3b")
    assert (tmp_path / "config.local.json").read_text(encoding="utf-8") == bad
    (tmp_path / "config.local.json").write_text(json.dumps({"local_image_dir": "x"}), encoding="utf-8-sig")
    dl.save_default("3b")
    assert json.loads((tmp_path / "config.local.json").read_text(encoding="utf-8")) == {
        "local_image_dir": "x", "upscaler": "3b"}
