"""GPU jobs in their own visible window, the way local-image runs its server.

`variants` and `finish` hold the GPU for minutes, so upscale.py doesn't run them out of sight: it
opens a new console window titled "local-upscale <command> <photo> - close this window to stop it"
and runs the job there through scripts/logtee.py. The window shows the job's progress and any
errors; everything goes to logs/<time>_<command>_<photo>.log. The command you ran waits, echoes the
log as it grows, and exits with the job's exit code, so scripts and agents see what they always did.

Stopping it:
- close the window: Windows ends every process attached to it (the job, the SeedVR2 engine,
  local-image's client), which frees the GPU and RAM; `variants` also stops local-image's server,
  which runs in its own window, as it does when it finishes (see on_close);
- or run stop.cmd (`upscale.py stop`), which stops it and closes the window;
- or press Ctrl+C in the command you ran.
If the job fails on its own, its window stays up for a minute so the error can be read.

One GPU job at a time per install: a second `variants` or `finish` refuses to start while one runs.
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from . import config

WINDOWED = ("variants", "finish")
LOGS = config.ROOT / "logs"
LOGTEE = config.ROOT / "scripts" / "logtee.py"
CLI = config.ROOT / "scripts" / "upscale.py"
HOLD_SECONDS = 60
KEEP_LOGS = 50
EXIT = re.compile(r"=== exited with code (-?\d+)")
CONSOLE_HOSTS = ("conhost.exe", "openconsole.exe")
STOPPED = "Stopped by upscale.py stop (stop.cmd or Ctrl+C)."


# --- the window ---------------------------------------------------------------------------------

def title(cmd: str, photo: Path) -> str:
    return f"local-upscale {cmd} {Path(photo).name} - close this window to stop it"


def new_log(cmd: str, photo: Path) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    old = sorted(LOGS.glob("*_*.log"))
    for p in old[:max(0, len(old) - KEEP_LOGS + 1)]:
        p.unlink(missing_ok=True)
    stem = re.sub(r'[<>:"/\\|?*\s]+', "_", Path(photo).stem)[:60]
    return LOGS / f"{time.strftime('%Y%m%d-%H%M%S')}_{cmd}_{stem}.log"


def command(argv: list[str], cmd: str, photo: Path, log: Path) -> list[str]:
    """What runs in the window: logtee showing every line, then upscale.py itself in --in-window mode."""
    return [sys.executable, str(LOGTEE), "--log", str(log), "--show", ".", "--title", title(cmd, photo),
            "--hold-on-error", str(HOLD_SECONDS), "--",
            sys.executable, str(CLI), "--in-window", *argv]


def run(argv: list[str], cmd: str, photo: Path, out=print) -> int:
    """Run `upscale.py <argv>` in a new visible console window; echo its log here until it ends.
    Returns the job's exit code (1 if its window was closed first, 130 after Ctrl+C here)."""
    busy = jobs()
    if busy:
        out(f"error: a local-upscale job is already running ({busy[0]['what']}): look for its window, "
            "or run stop.cmd to stop it. The GPU only holds one at a time.")
        return 1
    log = new_log(cmd, photo)
    out(f'Running in its own window ("{title(cmd, photo)}").\n'
        f"Close that window, or run stop.cmd, to stop it and free the GPU. Log: {log}")
    # No handles of ours go with it (close_fds, no redirection): the window would otherwise hold
    # our stdout open, and whoever reads our output (a pipe, an agent) would wait on the window.
    proc = subprocess.Popen(command(argv, cmd, photo, log), creationflags=subprocess.CREATE_NEW_CONSOLE)
    try:
        code = follow(proc, log, out)
    except KeyboardInterrupt:
        out("Ctrl+C: stopping the job.")
        stop(only=proc.pid, out=out)
        return 130
    if code is None:
        out("The job's window was closed before it finished, so the job stopped.")
        return 1
    return code


def follow(proc, log: Path, out=print, poll: float = 0.3) -> int | None:
    """Print the log's lines as they arrive until logtee records the exit code (returned; 1 if
    stop() ended the job), or the window goes away without one (None: it was closed)."""
    pos, pending, stopped = 0, b"", False
    while True:
        alive = proc.poll() is None  # before reading, so nothing written before it exited is missed
        if log.exists():
            with open(log, "rb") as f:
                f.seek(pos)
                data = f.read()
            pos += len(data)
            *lines, pending = (pending + data).split(b"\n")
            for raw in lines:
                line = raw.decode("utf-8", "replace").rstrip("\r")
                m = EXIT.match(line)
                if m:
                    return 1 if stopped else int(m.group(1))
                if line and not line.startswith("=== "):
                    stopped = stopped or line.endswith(STOPPED)
                    out(line)
        if not alive:
            return None
        time.sleep(poll)


# --- in the window: cleanup when it is closed ----------------------------------------------------

_on_close: list = []
_handler = None
CLOSE_EVENTS = (2, 5, 6)  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT


def on_close(action):
    """Run `action` (quick: Windows allows ~5 s) if this process's console window is closed.
    Returns a function that cancels it. Closing the window kills everything attached to it, but
    not what runs in another window, such as local-image's server."""
    global _handler
    if os.name == "nt" and _handler is None:
        @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
        def handler(event):
            if event in CLOSE_EVENTS:
                for a in list(_on_close):
                    try:
                        a()
                    except Exception:
                        pass
            return 0  # let Windows (or Python, for Ctrl+C) carry on as usual
        _handler = handler  # keep a reference, or the callback is collected
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_handler, True)
    _on_close.append(action)
    return lambda: action in _on_close and _on_close.remove(action)


def detached(cmd: list[str]) -> None:
    """Start cmd outside this console, so it outlives the window being closed."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    subprocess.Popen(cmd, creationflags=flags, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


# --- finding and stopping jobs ------------------------------------------------------------------

def _same(a: str, b: Path) -> bool:
    """Same file, through junctions (~/.claude/skills/local-upscale links here) and case."""
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except (OSError, ValueError):
        return False


def is_window(cmdline: list[str]) -> bool:
    """A logtee process of this install running an upscale.py job."""
    return (any(a.lower().endswith("logtee.py") and _same(a, LOGTEE) for a in cmdline)
            and any(a.lower().endswith("upscale.py") and _same(a, CLI) for a in cmdline))


def describe(cmdline: list[str]) -> dict:
    """The job a window's command line runs: its command, photo, log, and whether it stops
    local-image's server when it ends."""
    job = cmdline[cmdline.index("--in-window") + 1:] if "--in-window" in cmdline else []
    cmd = next((a for a in job if a in WINDOWED), "?")
    rest = job[job.index(cmd) + 1:] if cmd in job else []
    photo = rest[0] if rest else "?"
    log = cmdline[cmdline.index("--log") + 1] if "--log" in cmdline else None
    return {"cmd": cmd, "what": f"{cmd} {Path(photo).name}", "log": log,
            "stops_local_image": cmd == "variants" and "--keep-server" not in job}


def jobs(procs=None) -> list[dict]:
    """Job windows of this install whose job is still running: the topmost logtee process of each
    (the venv's python.exe is a launcher, so each appears twice), with describe()'s fields and its
    pid. A window held open after its job failed doesn't count."""
    if procs is None:
        try:
            import psutil
        except ImportError:  # only the CI environment lacks it
            return []
        procs = [p.info for p in psutil.process_iter(["pid", "ppid", "name", "cmdline"])]
    mine = {p["pid"]: p for p in procs if p.get("cmdline") and is_window(p["cmdline"])}
    children: dict = {}
    for p in procs:
        if (p.get("name") or "").lower() not in CONSOLE_HOSTS:  # the window itself, not the job
            children.setdefault(p.get("ppid"), []).append(p["pid"])

    def busy(pid, depth=0):  # anything below the logtee chain: the job itself
        return depth < 4 and any(c not in mine or busy(c, depth + 1) for c in children.get(pid, []))

    return [dict(describe(p["cmdline"]), pid=pid) for pid, p in mine.items()
            if p.get("ppid") not in mine and busy(pid)]


def stop(only: int | None = None, out=print) -> int:
    """Stop this install's running jobs (or only the one whose window process is `only`), close
    their windows, and stop local-image's server if a stopped job would have stopped it.
    Exit status: 0 stopped or nothing running, 1 something survived."""
    import psutil
    found = [j for j in jobs() if only is None or j["pid"] == only]
    if not found:
        out("  No local-upscale job is running.")
        return 0
    windows, victims, markers = [], [], []
    for j in found:
        try:
            w = psutil.Process(j["pid"])
            kids = w.children(recursive=True)
        except psutil.NoSuchProcess:
            continue
        windows.append(w)
        # The job and everything it started (SeedVR2, local-image's client); not logtee itself,
        # which then sees the job end, finds the marker and closes the window cleanly.
        victims += [k for k in kids if not _safe(lambda: is_window(k.cmdline())
                                                 or k.name().lower() in CONSOLE_HOSTS)]
        if j["log"]:
            m = Path(j["log"] + ".stop")
            m.write_text("stop", encoding="ascii")
            markers.append(m)
            try:  # so the command that started it says why it ended
                with open(j["log"], "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} {STOPPED}\n")
            except OSError:
                pass
        out(f"  stopping {j['what']} (window PID {j['pid']})")
    try:
        for k in victims:
            try:
                k.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(victims, timeout=10)
        _, open_ = psutil.wait_procs(windows, timeout=10)
        if any(j["stops_local_image"] for j in found):
            from . import qwen
            out("  stopping local-image's server, as the job would have")
            try:
                qwen.stop()
            except qwen.LocalImageMissing:
                pass
    finally:
        for m in markers:
            m.unlink(missing_ok=True)
    survived = [k for k in victims if k.is_running()]
    if survived:
        out("  Some processes survived; try again or use Task Manager.")
        return 1
    out("  Stopped." + (" Its window is still open; close it by hand." if open_ else " Its window closed."))
    return 0


def _safe(f):
    try:
        return f()
    except Exception:
        return False
