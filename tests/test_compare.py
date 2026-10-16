"""compare.html: which images it lists, in what order, under which labels and keys."""
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from upscale import compare, config, images

spec = importlib.util.spec_from_file_location("cli", Path(__file__).resolve().parents[1] / "scripts" / "upscale.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def img(path, value=0, size=(30, 40)):
    images.save(path, np.full((*size, 3), value, np.uint8))


def page_items(page):
    return json.loads(re.search(r"var ITEMS = (\[.*?\]);", page.read_text(encoding="utf-8")).group(1))


def test_photo_folder_order_labels_and_keys(tmp_path):
    d = tmp_path / "family photo"
    img(d / "family photo_0_original_40x30.jpg", 0)
    img(d / "variants" / "variant_0_original_40x30.jpg", 0)       # same bytes as the top-level original
    for n in (10, 2, 1):
        img(d / "variants" / f"variant_{n}_40x30.png", n)
    img(d / "variants" / "contact_sheet.jpg", 99)                   # overviews are left out
    img(d / "variants" / "faces.jpg", 98)
    img(d / "family photo_1_original_2048x1536.png", 50)
    img(d / "family photo_2_final_v2_7b_2048x1536.png", 51)
    img(d / "family photo_3_final_v2_7b_2048x1536_sharpened.png", 52)
    img(d / "work" / "variant_1_native.png", 1)                     # work/ is never listed
    page = compare.write(d)
    assert page == d / "compare.html"
    items = page_items(page)
    assert [i["label"] for i in items] == ["Original", "Variant 1", "Variant 2", "Variant 10",
                                          "Original resized", "Final v2 7b", "Final v2 7b sharpened"]
    assert [i["key"] for i in items] == list("1234567")
    assert [i["original"] for i in items] == [True] + [False] * 6
    assert items[0]["src"] == "family%20photo_0_original_40x30.jpg"
    assert items[1]["src"] == "variants/variant_1_40x30.png"


def test_ad_hoc_folder_borrows_the_parents_original(tmp_path):
    d = tmp_path / "photo" / "younger"
    img(tmp_path / "photo" / "photo_0_original_40x30.jpg")
    img(d / "younger_05y_b_40x30.png", 2)
    img(d / "younger_05y_a_40x30.png", 1)
    items = page_items(compare.write(d))
    assert [i["label"] for i in items] == ["Original", "05y a", "05y b"]
    img(d / "younger_05y_a_80x60.png", 3)                          # same name but the size: numbered
    labels = [i["label"] for i in page_items(compare.write(d))]
    assert labels == ["Original", "05y a", "05y a 2", "05y b"] and not any("x" in t for t in labels)
    assert items[0]["src"] == "../photo_0_original_40x30.jpg"


def test_page_is_offline_and_escapes_names(tmp_path):
    d = tmp_path / "fotografii vechi"
    img(d / "fată {DATA} 1.png")
    text = compare.write(d, title="a<b> {TITLE}").read_text(encoding="utf-8")
    assert "<title>a&lt;b&gt; {TITLE}: compare</title>" in text    # placeholders are filled only once
    assert not re.search(r"(src|href)=[\"']?(https?:)?//|@import|url\(", text)  # nothing from the network
    item = page_items(d / "compare.html")[0]
    assert item["src"] == "fat%C4%83%20%7BDATA%7D%201.png" and item["label"] == "Fată {DATA} 1"


def test_more_images_than_keys_and_empty_folder(tmp_path):
    d = tmp_path / "many"
    for n in range(25):
        img(d / f"x_{n:02d}.png", n)
    keys = [i["key"] for i in page_items(compare.write(d))]
    assert keys[:10] == list("1234567890") and keys[10:20] == list("qwertyuiop") and keys[20:] == [""] * 5
    assert compare.write(tmp_path / "empty") is None


def test_cli_compare_photo_or_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "OUTPUT", tmp_path / "output")
    img(tmp_path / "output" / "pic" / "variants" / "variant_1_40x30.png")
    assert cli.main(["compare", str(tmp_path / "input" / "pic.jpg")]) == 0
    assert (tmp_path / "output" / "pic" / "compare.html").exists()
    assert cli.main(["compare", str(tmp_path / "output" / "pic" / "variants")]) == 0
    assert cli.main(["compare", str(tmp_path / "input" / "nothing.jpg")]) == 1


def test_photo_called_original_and_tiff_original(tmp_path):
    d = tmp_path / "original"
    img(d / "original_0_original_40x30.tif", 0)
    img(d / "variants" / "variant_0_original_40x30.tif", 0)
    img(d / "variants" / "variant_1_40x30.png", 1)
    img(d / "original_2_final_v1_7b_2048x1536.png", 2)
    items = page_items(compare.write(d))
    assert [i["label"] for i in items] == ["Original", "Variant 1", "Final v1 7b"]
    assert [i["original"] for i in items] == [True, False, False]
    assert items[0]["src"] == "work/compare_original.png"           # browsers can't show TIFF
    assert images.read(d / "work" / "compare_original.png").shape == (30, 40, 3)


def test_no_original_and_relative_folder(tmp_path, monkeypatch):
    d = tmp_path / "photo" / "younger"
    img(d / "younger_a_40x30.png")
    monkeypatch.chdir(d)
    items = page_items(compare.write(Path(".")))
    assert [i["label"] for i in items] == ["A"] and not items[0]["original"]
    assert "<title>younger: compare</title>" in (d / "compare.html").read_text(encoding="utf-8")
    img(tmp_path / "photo" / "photo_0_original_40x30.jpg")
    assert cli.main(["compare", "."]) == 0
    assert page_items(d / "compare.html")[0]["src"] == "../photo_0_original_40x30.jpg"
