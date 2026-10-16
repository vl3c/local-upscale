import cv2
import numpy as np
import pytest

from upscale import config, seedvr2


@pytest.mark.parametrize("W,H,scale", [(2368, 1760, 2), (1120, 960, 2), (900, 600, 2), (5000, 3800, 3),
                                       (9000, 7000, 4), (500, 300, 2)])
def test_tile_plan_fits_and_covers(W, H, scale):
    t, ov, xs, ys = seedvr2.tile_plan(W, H, scale)
    assert t * scale <= seedvr2.MAX_TILE_OUT and t <= min(W, H)
    assert xs[0] == 0 and xs[-1] + t == W and ys[0] == 0 and ys[-1] + t == H


@pytest.mark.parametrize("W,H,scale", [(1500, 1100, 2), (1100, 1040, 2), (600, 300, 2), (1400, 900, 3)])
def test_blend_is_exact_for_consistent_tiles(W, H, scale):
    """If every tile is an exact upscale of its crop, the blend must reproduce the upscaled image."""
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (H, W, 3), dtype=np.uint8), (0, 0), 2)
    t, ov, xs, ys = seedvr2.tile_plan(W, H, scale, tile=512, overlap=64)
    up = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    tiles = [up[y * scale:(y + t) * scale, x * scale:(x + t) * scale] for y in ys for x in xs]
    out = seedvr2.blend(H, W, scale, t, xs, ys, tiles)
    assert np.abs(out.astype(int) - up.astype(int)).max() <= 1


def test_ascii_workdir_avoids_non_ascii(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE", tmp_path / "Crăciun" / ".cache")
    d = seedvr2.ascii_workdir()
    assert str(d).isascii() and d.is_dir()


def test_missing_weights_message(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ENGINE", tmp_path)
    (tmp_path / "inference_cli.py").write_text("")
    monkeypatch.setattr(config, "SEEDVR2_MODELS", tmp_path / "models")
    with pytest.raises(seedvr2.SeedVR2Error, match="not installed"):
        seedvr2._check_install("x.safetensors")
