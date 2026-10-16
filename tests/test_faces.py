import cv2
import numpy as np

from upscale import faces


def test_color_match_moves_statistics():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 120, (64, 64, 3), dtype=np.uint8)
    ref = rng.integers(100, 255, (64, 64, 3), dtype=np.uint8)
    out = faces.color_match(src, ref)
    lab = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    assert abs(lab(out)[..., 0].mean() - lab(ref)[..., 0].mean()) < 3


def test_head_box_stays_inside_image():
    for face in [(0, 0, 50, 60), (950, 700, 60, 80), (500, 400, 100, 120)]:
        x0, y0, side = faces.head_box(face, 1000, 800)
        assert x0 >= 0 and y0 >= 0 and x0 + side <= 1000 and y0 + side <= 800


def test_mix_replaces_only_the_head_and_never_below_the_chin():
    rng = np.random.default_rng(2)
    tex = lambda: cv2.GaussianBlur(rng.integers(0, 255, (600, 600, 3), dtype=np.uint8), (0, 0), 3)
    base, donor = tex(), tex()
    face = (250, 200, 100, 120)  # x, y, w, h -> chin at y=320
    out, notes = faces.mix_heads(base, {1: donor}, [face])
    c = lambda a, b: np.corrcoef(a.astype(np.float32).ravel(), b.astype(np.float32).ravel())[0, 1]
    centre = (slice(230, 290), slice(270, 330))
    assert c(out[centre], donor[centre]) > 0.9        # the face comes from the donor
    assert np.array_equal(out[340:, :], base[340:, :])  # nothing below the chin is replaced
    assert np.array_equal(out[:, :100], base[:, :100])  # far from the head: untouched
    assert len(notes) == 1


def test_detect_without_model_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(faces.config, "FACE_MODEL", tmp_path / "missing.onnx")
    assert faces.detect(np.zeros((100, 100, 3), np.uint8)) == []
    assert faces.strip([np.zeros((100, 100, 3), np.uint8)], ["x"]) is None


def test_numbered_draws_on_a_copy():
    im = np.zeros((300, 300, 3), np.uint8)
    out = faces.numbered(im, [(50, 50, 80, 80)])
    assert im.sum() == 0 and out.sum() > 0


def test_head_mask_is_zero_on_every_box_edge():
    for face in [(100, 100, 80, 110), (40, 20, 120, 100), (0, 0, 60, 60)]:
        x0, y0, side = faces.head_box(face, 600, 600)
        m = faces.head_mask(face, x0, y0, side)
        assert max(m[0].max(), m[-1].max(), m[:, 0].max(), m[:, -1].max()) == 0
        x, y, w, h = face
        assert m[y - y0 + h // 2, x - x0 + w // 2] > 0.99  # the face centre is fully replaced


def test_square_crop_pads_instead_of_stretching():
    im = np.full((100, 100, 3), 200, np.uint8)
    c = faces.square_crop(im, 5, 50, 30)
    assert c.shape == (60, 60, 3) and c[30, 0].sum() == 0 and c[30, 59].sum() == 600


def test_quiet_opencv_hides_warnings_and_restores_the_level():
    log = cv2.utils.logging
    before = log.getLogLevel()
    with faces._quiet_opencv():
        assert log.getLogLevel() == log.LOG_LEVEL_ERROR
    assert log.getLogLevel() == before
