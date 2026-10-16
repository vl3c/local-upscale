import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("cli", Path(__file__).resolve().parents[1] / "scripts" / "upscale.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_parser_commands():
    p = cli.build_parser()
    a = p.parse_args(["variants", "input/x.jpg", "--ref", "a.jpg", "b.jpg", "--colorize", "--seeds", "1", "2"])
    assert a.cmd == "variants" and len(a.ref) == 2 and a.colorize and a.seeds == [1, 2]
    a = p.parse_args(["finish", "x.jpg", "--variant", "3", "--full", "--no-sharpened"])
    assert a.variant == 3 and a.full and a.no_sharpened
    a = p.parse_args(["mix", "x.jpg", "--base", "3", "--face", "1=2", "2=4"])
    assert cli.parse_picks(a.face) == {1: 2, 2: 4}
    a = p.parse_args(["crop", "x.jpg", "--corners", "1,2,3,4,5,6,7,8", "--inset", "0.01"])
    assert a.corners == [1, 2, 3, 4, 5, 6, 7, 8] and a.inset == 0.01
    assert cli.parse_picks(["1:3"]) == {1: 3}
    with pytest.raises(SystemExit):
        cli.parse_picks(["1-3"])


def test_bad_corners_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["crop", "x.jpg", "--corners", "1,2,3"])


def test_prompts_pick():
    from upscale import prompts
    assert prompts.pick(False, False) == prompts.RESTORE
    assert prompts.pick(True, False) == prompts.COLORIZE
    assert "Never copy" in prompts.pick(False, True)
    assert prompts.pick(True, True).startswith(prompts.COLORIZE)
