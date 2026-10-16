import numpy as np
import pytest

from upscale import images


@pytest.mark.parametrize("w,h,expected", [(2592, 1944, 2048), (563, 480, 1024), (1283, 1652, 1456), (1944, 1784, 1856)])
def test_qwen_scale(w, h, expected):
    s = images.qwen_scale(w, h)
    assert s == expected and s % 16 == 0 and 1024 <= s <= 2048


def test_final_size_keeps_large_originals():
    assert images.final_size(2592, 1944) == (2592, 1944)
    assert images.final_size(2048, 1000) == (2048, 1000)


def test_final_size_small_originals_go_to_2k_long_side():
    w, h = images.final_size(563, 480)
    assert max(w, h) == 2048 and abs(w / h - 563 / 480) < 0.002
    assert images.final_size(583, 748)[1] == 2048


def test_sr_factor_bounds():
    assert images.sr_factor((2592, 1944), (2368, 1760)) == 2   # minimum 2x (supersampling)
    assert images.sr_factor((2048, 1746), (1120, 960)) == 2
    assert images.sr_factor((4000, 3000), (1000, 750)) == 4
    assert images.sr_factor((9000, 9000), (1000, 1000)) == 4   # capped


@pytest.mark.parametrize("size,tile,ov", [(2368, 1024, 128), (1024, 1024, 128), (1100, 1024, 128),
                                          (5000, 682, 128), (1040, 1024, 128), (300, 300, 75)])
def test_tile_positions_cover_evenly(size, tile, ov):
    pos = images.tile_positions(size, tile, ov)
    assert pos[0] == 0 and pos[-1] + tile == size
    gaps = [b - a for a, b in zip(pos, pos[1:])]
    assert all(tile - g >= ov for g in gaps)          # every overlap is at least ov
    assert not gaps or max(gaps) - min(gaps) <= 1      # evenly spaced, no thin last tile


def test_tile_positions_minimal_count():
    assert images.tile_positions(1100, 1024, 128) == [0, 76]
    assert len(images.tile_positions(2368, 1024, 128)) == 3


def test_ramp_partitions_unity_across_an_overlap():
    n, ov = 100, 20
    r = images.ramp(n, ov)
    assert r[ov:-ov].min() == 1.0
    assert np.allclose(r[-ov:] + r[:ov], 1.0)  # the next tile's rising ramp completes this one's falling ramp


def test_resize_and_unsharp_shapes():
    im = (np.random.rand(40, 60, 3) * 255).astype(np.uint8)
    assert images.resize(im, 30, 20).shape == (20, 30, 3)
    assert images.resize(im, 120, 80).shape == (80, 120, 3)
    assert images.unsharp(im).shape == im.shape


def test_save_read_roundtrip_with_unicode_path(tmp_path):
    im = (np.random.rand(16, 16, 3) * 255).astype(np.uint8)
    p = images.save(tmp_path / "ăîșț photo.png", im)
    assert np.array_equal(images.read(p), im)


def test_grid_pads():
    tiles = [np.full((10, 10, 3), i, np.uint8) for i in range(4)]
    g = images.grid(tiles, 3)
    assert g.shape == (20, 30, 3) and g[15, 25].sum() == 0
