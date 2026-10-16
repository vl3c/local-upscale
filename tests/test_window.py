import importlib.util
import subprocess
import sys
from pathlib import Path

from upscale import seedvr2, window

ROOT = Path(__file__).resolve().parents[1]


def window_cmdline(cmd="variants", *extra, log="L.log"):
    job = ["--in-window", cmd, "input/Crăciun 1987.jpg", *extra]
    return window.command(job[1:], cmd, Path(job[2]), Path(log))


def test_command_runs_logtee_then_the_job_in_window_mode():
    c = window_cmdline("finish", "--variant", "3")
    assert c[1] == str(window.LOGTEE) and c[c.index("--log") + 1] == "L.log"
    assert c[c.index("--") + 1:] == [sys.executable, str(window.CLI), "--in-window", "finish",
                                     "input/Crăciun 1987.jpg", "--variant", "3"]
    assert "close this window to stop it" in c[c.index("--title") + 1]


def test_jobs_finds_this_installs_windows_once():
    variants = window_cmdline("variants")
    kept = window_cmdline("variants", "--keep-server")
    other = [sys.executable, "elsewhere/scripts/logtee.py", "--log", "x", "--", sys.executable, "main.py"]
    procs = [
        {"pid": 10, "ppid": 1, "cmdline": variants},     # the venv launcher
        {"pid": 11, "ppid": 10, "cmdline": variants},    # the interpreter it starts
        {"pid": 12, "ppid": 11, "cmdline": [sys.executable, str(window.CLI), "--in-window", "variants", "x.jpg"]},
        {"pid": 20, "ppid": 1, "cmdline": kept},
        {"pid": 21, "ppid": 20, "cmdline": ["python.exe", "x"]},
        {"pid": 25, "ppid": 1, "cmdline": kept},          # held open after its job failed: no job below
        {"pid": 26, "ppid": 25, "cmdline": kept},
        {"pid": 30, "ppid": 1, "cmdline": other},
        {"pid": 31, "ppid": 30, "cmdline": ["python.exe", "main.py"]},
        {"pid": 40, "ppid": 1, "cmdline": None},          # access denied
    ]
    found = {j["pid"]: j for j in window.jobs(procs)}
    assert set(found) == {10, 20}
    assert found[10]["what"] == "variants Crăciun 1987.jpg" and found[10]["log"] == "L.log"
    assert found[10]["stops_local_image"] and not found[20]["stops_local_image"]


def test_follow_echoes_the_log_and_returns_the_exit_code(tmp_path):
    log = tmp_path / "job.log"
    log.write_bytes("=== started ===\n12:00:00 Qwen edit: seed 1\n12:00:01 variant 1 – ok\r\n"
                    "=== exited with code 3 at 12:00:02 ===\n".encode("utf-8"))

    class Running:
        def poll(self):
            return None

    seen = []
    assert window.follow(Running(), log, seen.append, poll=0) == 3
    assert seen == ["12:00:00 Qwen edit: seed 1", "12:00:01 variant 1 – ok"]


def test_follow_reports_a_closed_window(tmp_path):
    log = tmp_path / "job.log"
    log.write_text("=== started ===\n12:00:00 Qwen edit: seed 1\n", encoding="utf-8")

    class Gone:
        def poll(self):
            return 1

    seen = []
    assert window.follow(Gone(), log, seen.append, poll=0) is None and seen == ["12:00:00 Qwen edit: seed 1"]


def test_on_close_can_be_cancelled():
    cancel = window.on_close(print)
    assert print in window._on_close
    cancel()
    assert print not in window._on_close


def test_new_log_is_named_by_command_and_photo_and_pruned(monkeypatch, tmp_path):
    monkeypatch.setattr(window, "LOGS", tmp_path)
    for i in range(window.KEEP_LOGS + 5):
        (tmp_path / f"2026{i:04d}_finish_x.log").write_text("")
    log = window.new_log("finish", Path("input/a b:c.jpg"))
    assert log.parent == tmp_path and log.name.endswith("_finish_a_b_c.log")
    assert len(list(tmp_path.glob("*.log"))) == window.KEEP_LOGS - 1


def test_cli_hands_gpu_commands_to_the_window(monkeypatch):
    spec = importlib.util.spec_from_file_location("cli_w", ROOT / "scripts" / "upscale.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = []
    monkeypatch.setattr(window, "run", lambda argv, cmd, photo: seen.append((argv, cmd, photo)) or 7)
    monkeypatch.setattr(cli.os, "name", "nt")
    assert cli.main(["finish", "x.jpg", "--variant", "2"]) == 7
    assert seen == [(["finish", "x.jpg", "--variant", "2"], "finish", Path("x.jpg"))]
    assert cli.build_parser().parse_args(["--in-window", "finish", "x.jpg", "--variant", "2"]).in_window


def run_logtee(tmp_path, code, *opts):
    log = tmp_path / "job.log"
    job = [sys.executable, "-c", f"import sys; print('working'); print('Traceback (most recent call last):'); sys.exit({code})"]
    r = subprocess.run([sys.executable, str(window.LOGTEE), "--log", str(log), "--show", "working", *opts, "--", *job],
                       capture_output=True, text=True, timeout=60)
    return r, log.read_text(encoding="utf-8")


def test_logtee_logs_everything_and_returns_the_code(tmp_path):
    r, log = run_logtee(tmp_path, 3)
    assert r.returncode == 3 and "working" in r.stdout and "Traceback" in r.stdout
    assert "=== exited with code 3" in log.splitlines()[-1]


def test_logtee_holds_a_failed_window_then_exits_cleanly(tmp_path):
    r, log = run_logtee(tmp_path, 3, "--hold-on-error", "0.2")
    assert r.returncode == 0 and "closes in" in r.stdout and "exited with code 3" in log
    (tmp_path / "job.log.stop").write_text("stop")  # upscale.py stop is stopping it: no hold
    r, log = run_logtee(tmp_path, 3, "--hold-on-error", "30")
    assert r.returncode == 0 and "closes in" not in r.stdout


def test_seedvr2_progress_lines():
    assert seedvr2.progress_line("[20:20:30.077]  ━━━━━━━━ Phase 2: DiT upscaling ━━━━━━━━") == \
        "  SeedVR2: Phase 2: DiT upscaling"
    assert seedvr2.progress_line("[20:21:29.811] 🎬 Upscaling batch 2/4") == "  SeedVR2: upscaling tile 2/4"
    assert seedvr2.progress_line("[20:20:32.502] 🎨 Encoding batch 1/4") is None


def test_describe_reads_the_title_and_spelled_out_options():
    j = window.describe(window_cmdline("variants", "--colorize"))
    assert j["what"] == "variants Crăciun 1987.jpg" and j["stops_local_image"]
    assert not window.describe(window_cmdline("variants", "--keep-server"))["stops_local_image"]
    assert not window.describe(window_cmdline("finish", "--variant", "2"))["stops_local_image"]


def test_follow_stopped_ctrl_c_and_split_lines(tmp_path):
    log = tmp_path / "job.log"
    log.write_text(f"12:00:00 working\n12:00:05 {window.STOPPED}\n=== exited with code 15 at x ===\n",
                   encoding="utf-8")

    class Running:
        def poll(self):
            return None

    assert window.follow(Running(), log, lambda s: None, poll=0) == 1    # stop.cmd: always 1

    log.write_bytes(b"12:00:00 half a li")
    chunks = [b"ne\n=== exited with code 3221225786 at x ===\n"]

    class Growing:
        def poll(self):
            if chunks:
                with open(log, "ab") as f:
                    f.write(chunks.pop())
            return None

    seen = []
    code = window.follow(Growing(), log, seen.append, poll=0)
    assert seen == ["12:00:00 half a line"] and code in window.CTRL_C_EXIT


def test_run_refuses_a_missing_photo_without_opening_a_window(monkeypatch, tmp_path):
    monkeypatch.setattr(window.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    seen = []
    assert window.run(["finish", "nope.jpg", "--variant", "1"], "finish", tmp_path / "nope.jpg", seen.append) == 1
    assert "no such photo" in seen[0]


class FakeProc:
    def __init__(self, pid, cmdline, name="python.exe", kids=()):
        self.pid, self._cmd, self._name, self.kids, self.killed = pid, cmdline, name, list(kids), False

    def children(self, recursive=False):
        return self.kids

    def cmdline(self):
        return self._cmd

    def name(self):
        return self._name

    def kill(self):
        self.killed = True

    def is_running(self):
        return not self.killed


def test_stop_kills_the_job_not_the_window_and_stops_local_image(monkeypatch, tmp_path):
    log = tmp_path / "job.log"
    log.write_text("", encoding="utf-8")
    win = window_cmdline("variants", log=str(log))
    logtee_child = FakeProc(11, win)
    host = FakeProc(12, ["conhost.exe"], name="conhost.exe")
    job = FakeProc(13, ["python.exe", "upscale.py", "--in-window", "variants"])
    engine = FakeProc(14, ["python.exe", "generate.py"])
    top = FakeProc(10, win, kids=[logtee_child, host, job, engine])

    class FakePsutil:
        NoSuchProcess = ProcessLookupError

        @staticmethod
        def Process(pid):
            assert pid == 10
            return top

        @staticmethod
        def wait_procs(procs, timeout=None):
            return procs, []

    monkeypatch.setitem(sys.modules, "psutil", FakePsutil)
    monkeypatch.setattr(window, "jobs", lambda: [dict(window.describe(win), pid=10)])
    stopped = []
    from upscale import qwen
    monkeypatch.setattr(qwen, "stop", lambda detached=False: stopped.append(detached))
    out = []
    assert window.stop(out=out.append) == 0
    assert job.killed and engine.killed and not logtee_child.killed and not host.killed
    assert stopped == [False] and not Path(str(log) + ".stop").exists()
    assert log.read_text(encoding="utf-8").strip().endswith(window.STOPPED)
    monkeypatch.setattr(window, "jobs", lambda: [])
    assert window.stop(out=out.append) is None and window.stop(only=10, out=out.append) is None
