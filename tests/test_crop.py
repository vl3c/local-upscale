import cv2
import numpy as np

from upscale import crop


def synthetic_print(W=1600, H=2000, quad=((90, 60), (1520, 40), (1540, 1950), (70, 1960))):
    """A textured photo lying slightly skewed on a green table, with a thin white paper border
    and sensor noise, like a phone shot of a print."""
    rng = np.random.default_rng(3)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = (40, 120, 50)
    photo = cv2.GaussianBlur(rng.integers(60, 200, (H, W, 3), dtype=np.uint8), (0, 0), 8)
    cv2.circle(photo, (800, 900), 300, (50, 50, 50), -1)
    paper = np.zeros((H, W), np.uint8)
    cv2.fillPoly(paper, [np.int32(quad)], 255)
    inner = cv2.erode(paper, np.ones((25, 25), np.uint8))
    img[paper > 0] = (235, 235, 235)
    img[inner > 0] = photo[inner > 0]
    img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img, np.float32(quad)


def test_order_corners():
    pts = [(10, 300), (300, 10), (310, 290), (5, 8)]
    assert crop.order_corners(pts).tolist() == [[5, 8], [300, 10], [310, 290], [10, 300]]


def test_find_print_on_a_table():
    img, quad = synthetic_print()
    found = crop.find_print(img)
    assert found is not None
    assert np.abs(found - quad).max() < 25


def test_plain_picture_is_not_a_print():
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, (800, 1000, 3), dtype=np.uint8), (0, 0), 6)
    assert crop.find_print(img) is None


def test_straighten_size_and_inset():
    img, quad = synthetic_print()
    flat = crop.straighten(img, quad, inset=0.02)
    w = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3]))
    assert abs(flat.shape[1] - w * 0.96) < 4
    green = flat[..., 1].astype(np.int16) - flat[..., 0].astype(np.int16) > 50
    assert green.mean() < 0.01  # no table left inside the crop
