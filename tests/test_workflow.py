"""The workflow end to end, with local-image and SeedVR2 replaced by fakes (no GPU needed)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from upscale import config, images, qwen, seedvr2, workflow


def quiet(*a):
    pass


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(config, "INPUT", tmp_path / "input")
    photo = tmp_path / "input" / "family photo.jpg"
    rng = np.random.default_rng(5)
    images.save(photo, cv2.GaussianBlur(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8), (0, 0), 3))
    calls = {"stop": 0, "edit": []}

    def fake_edit(prompt, imgs, out_dir, scale, count, seed, timeout=3600):
        calls["edit"].append((prompt, [Path(p).name for p in imgs], scale, count, seed))
        out_dir.mkdir(parents=True, exist_ok=True)
        made = []
        for k in range(count):
            p = out_dir / f"q_{seed}_{k}.png"
            images.save(p, np.full((1008, 1344, 3), 40 * (k + 1), np.uint8))
            made.append(p)
        return made

    def fake_stop():
        calls["stop"] += 1

    monkeypatch.setattr(qwen, "edit", fake_edit)
    monkeypatch.setattr(qwen, "stop", fake_stop)
    monkeypatch.setattr(qwen, "gpu_used_mb", lambda: 1000)
    monkeypatch.setattr(workflow.faces, "detect", lambda img, **k: [])
    return photo, calls


def test_variants_layout_numbering_and_stop(env):
    photo, calls = env
    workflow.variants(photo, seeds=[1, 2], log=quiet)
    d = config.OUTPUT / "family photo"
    vs = sorted(p.name for p in (d / "variants").glob("variant_*_400x300.png"))
    assert vs == [f"variant_{n}_400x300.png" for n in range(1, 5)]
    assert (d / "variants" / "variant_0_original_400x300.jpg").exists()
    assert (d / "family photo_0_original_400x300.jpg").exists()
    assert (d / "variants" / "contact_sheet.jpg").exists()
    assert calls["stop"] == 1 and [c[4] for c in calls["edit"]] == [1, 2]
    info = json.loads((d / "work" / "variants.json").read_text(encoding="utf-8"))
    assert [v["n"] for v in info["variants"]] == [1, 2, 3, 4] and info["photo_sha256"]


def test_server_stopped_even_when_an_edit_fails(env, monkeypatch):
    photo, calls = env

    def boom(*a, **k):
        raise qwen.QwenError("fail")

    monkeypatch.setattr(qwen, "edit", boom)
    with pytest.raises(qwen.QwenError):
        workflow.variants(photo, seeds=[1, 2], log=quiet)
    assert calls["stop"] == 1


def test_rerun_after_deleting_work_never_overwrites(env):
    photo, _ = env
    workflow.variants(photo, seeds=[1, 2], log=quiet)
    d = config.OUTPUT / "family photo"
    for p in (d / "work").glob("variant_*_native.png"):
        p.unlink()
    workflow.variants(photo, seeds=[3, 4], log=quiet)
    nums = sorted(int(p.name.split("_")[1]) for p in (d / "variants").glob("variant_*_400x300.png"))
    assert nums == list(range(1, 9))


def test_same_name_different_photo_refused(env, tmp_path):
    photo, _ = env
    workflow.variants(photo, seeds=[1, 2], log=quiet)
    other = tmp_path / "elsewhere" / "family photo.jpg"
    images.save(other, np.zeros((300, 400, 3), np.uint8))
    with pytest.raises(workflow.WorkflowError, match="different photo"):
        workflow.variants(other, seeds=[1, 2], log=quiet)


def test_finish_names_sizes_and_model(env, monkeypatch):
    photo, _ = env
    workflow.variants(photo, seeds=[1, 2], log=quiet)
    seen = {}

    def fake_upscale(img, scale, log_path, model=None):
        seen.update(scale=scale, model=model)
        h, w = img.shape[:2]
        return cv2.resize(img, (w * scale, h * scale)), {"tiles": 2, "tile_in": 1008, "seconds": 1,
                                                          "gpu_peak_mb": 1}

    monkeypatch.setattr(seedvr2, "upscale", fake_upscale)
    workflow.finish(photo, 2, model="3b", log=quiet)
    d = config.OUTPUT / "family photo"
    tw, th = images.final_size(400, 300)
    assert (d / f"family photo_2_final_v2_3b_{tw}x{th}.png").exists()
    assert (d / f"family photo_3_final_v2_3b_{tw}x{th}_sharpened.png").exists()
    assert (d / f"family photo_1_original_{tw}x{th}.png").exists()
    big = next(d.glob("family photo_4_upscaled_v2_3b_*.png"))
    bw, bh = (int(v) for v in big.stem.split("_")[-1].split("x"))
    assert bw == 2688 and abs(bw / bh - 400 / 300) < 0.002   # the original's proportions, not Qwen's
    assert seen == {"scale": 2, "model": "3b"}
    with pytest.raises(workflow.WorkflowError, match="original"):
        workflow.finish(photo, 0, log=quiet)
    with pytest.raises(workflow.WorkflowError, match="not found"):
        workflow.finish(photo, 9, log=quiet)


def test_mix_uses_face_numbers_and_rejects_bad_ones(env, monkeypatch):
    photo, _ = env
    workflow.variants(photo, seeds=[1, 2], log=quiet)
    def two_faces(img, **k):  # in the coordinates of the image it is given, like the real detector
        h, w = img.shape[:2]
        return [(w // 6, h // 3, w // 8, h // 6), (w * 2 // 3, h // 3, w // 8, h // 6)]

    monkeypatch.setattr(workflow.faces, "detect", two_faces)
    got = {}

    def fake_mix(base, donors, found):
        got["donors"] = sorted(donors)
        return base, []

    monkeypatch.setattr(workflow.faces, "mix_heads", fake_mix)
    assert workflow.mix(photo, 1, {2: 3}, log=quiet) == 5 and got["donors"] == [2]
    with pytest.raises(workflow.WorkflowError, match="not among"):
        workflow.mix(photo, 1, {3: 2}, log=quiet)
    assert workflow.mix(photo, 1, None, log=quiet) is None


def test_qwen_edit_command_line(monkeypatch, tmp_path):
    gen = tmp_path / "li" / "scripts" / "generate.py"
    gen.parent.mkdir(parents=True)
    gen.write_text("")
    monkeypatch.setenv("LOCAL_IMAGE_DIR", str(tmp_path / "li"))
    seen = {}

    class Done:
        returncode = 0

    def fake_run(cmd, *a, **k):
        seen["cmd"] = cmd
        out = Path(cmd[cmd.index("--out") + 1])
        for i in range(2):
            (out / f"o{i}.png").write_bytes(b"x")
        return Done()

    monkeypatch.setattr(qwen.subprocess, "run", fake_run)
    made = qwen.edit("restore", [tmp_path / "a.jpg", tmp_path / "ref.jpg"], tmp_path / "out", 1024, 2, 7)
    c = seen["cmd"]
    assert c.count("--edit") == 2 and c[c.index("--count") + 1] == "2" and "--rgb" in c and "--timeout" in c
    assert len(made) == 2
    monkeypatch.setattr(qwen.subprocess, "run", lambda cmd, *a, **k: Done())
    with pytest.raises(qwen.QwenError):
        qwen.edit("restore", [tmp_path / "a.jpg"], tmp_path / "out2", 1024, 2, 7)
