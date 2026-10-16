"""GPU jobs in their own visible window, the way local-image runs its server.

`variants` and `finish` hold the GPU for minutes, so upscale.py doesn't run them out of sight: it
opens a new console window titled "local-upscale <command> <photo> - close this window to stop it"
and runs the job there through scripts/logtee.py. The window shows the job's progress and any
errors; everything goes to logs/<time>_<pid>_<command>_<photo>.log. The command you ran waits,
echoes the log as it grows, and exits with the job's exit code, so scripts and agents see what
they always did.

Stopping it:
- close the window: Windows ends every process attached to it (the job, the SeedVR2 engine,
  local-image's client), which frees the GPU and RAM; `variants` also stops local-image's server,
  which runs in its own window, as it does when it finishes (see on_close);
- or run stop.cmd (`upscale.py stop`), which stops it and closes the window;
- or press Ctrl+C in the window, or in the command you ran.
If the job fails on its own, its window stays up for a minute so the error can be read.

One GPU job at a time per install: a second `variants` or `finish` refuses to start while one runs.
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import threading
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
CTRL_C_EXIT = (0xC000013A, -0x3FFFFEC6)  # STATUS_CONTROL_C_EXIT, unsigned and signed


# --- the window ---------------------------------------------------------------------------------

def title(cmd: str, photo: Path) -> str:
    return f"local-upscale {cmd} {Path(photo).name} - close this window to stop it"


def new_log(cmd: str, photo: Path) -> Path:
    LOGS.mkdir(parents=True, exist_ok=True)
    old = sorted(LOGS.glob("*_*.log"))
    for p in old[:max(0, len(old) - KEEP_LOGS + 1)]:
        p.unlink(missing_ok=True)
    stem = re.sub(r'[<>:"/\\|?*\s]+', "_", Path(photo).stem)[:60]
    return LOGS / f"{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}_{cmd}_{stem}.log"


def command(argv: list[str], cmd: str, photo: Path, log: Path) -> list[str]:
    """What runs in the window: logtee showing every line, then upscale.py itself in --in-window mode."""
    return [sys.executable, str(LOGTEE), "--log", str(log), "--show", ".", "--title", title(cmd, photo),
            "--hold-on-error", str(HOLD_SECONDS), "--",
            sys.executable, str(CLI), "--in-window", *argv]


def run(argv: list[str], cmd: str, photo: Path, out=print) -> int:
    """Run `upscale.py <argv>` in a new visible console window; echo its log here until it ends.
    Returns the job's exit code (1 if its window was closed first or stop.cmd ended it, 130 after
    Ctrl+C). upscale.py checks the input files first, so a typo doesn't open a window."""
    log = new_log(cmd, photo)
    full = command(argv, cmd, photo, log)
    proc = None
    try:
        with launch_lock():  # two starts at once must not both pass the one-job check
            busy = jobs()
            if busy:
                out(f"error: a local-upscale job is already running ({busy[0]['what']}): look for its "
                    "window, or run stop.cmd to stop it. The GPU only holds one at a time.")
                return 1
            out(f'Running in its own window ("{title(cmd, photo)}").\n'
                f"Close that window, or run stop.cmd, to stop it and free the GPU. Log: {log}")
            proc = start(full)
            _wait_started(proc, log)  # until jobs() can see it
        code = follow(proc, log, out)
    except KeyboardInterrupt:
        proc = proc or _window_for(log)  # Ctrl+C after the window opened but before its pid came back
        if not proc:
            out("Ctrl+C: nothing was started.")
            return 130
        out("Ctrl+C: stopping the job.")
        if stop(only=proc.pid, out=out) is None and proc.poll() is None:  # not seen as a job yet
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            _closed(full, out)
        return 130
    if code is None:
        out("The job's window was closed before it finished, so the job stopped.")
        _closed(full, out)
        return 1
    if code in CTRL_C_EXIT:
        out("The job was stopped with Ctrl+C in its window.")
        _closed(full, out)  # a second Ctrl+C there can cut its own cleanup short
        return 130
    return code


START = """import subprocess, sys
flags = subprocess.CREATE_NEW_CONSOLE
try:
    p = subprocess.Popen(sys.argv[1:], creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB)
except OSError:  # a job object that doesn't allow breaking away
    p = subprocess.Popen(sys.argv[1:], creationflags=flags)
print(p.pid)
"""


class Window:
    """The job window's top process, watched by pid (Popen-like: pid, poll())."""

    def __init__(self, pid: int):
        import psutil
        self.pid = pid
        try:
            self._p = psutil.Process(pid)
        except psutil.NoSuchProcess:
            self._p = None

    def poll(self):
        try:
            return None if self._p and self._p.is_running() else 1  # is_running also catches pid reuse
        except Exception:  # AccessDenied and the like: don't call a live job gone
            import psutil
            return None if psutil.pid_exists(self.pid) else 1


def start(full: list[str]):
    """Open the window outside our process tree, as local-image's start.cmd does with `start`: a
    short-lived Python starts it and exits. So stopping the command that started it (an agent
    ending its background task kills that command's process tree or job object) leaves the job
    on screen for the user, rather than killing it unseen with local-image's server still holding
    the GPU. No handles of ours go with it (no redirection, close_fds): the window would otherwise
    hold our stdout open, and whoever reads our output (a pipe, an agent) would wait on it."""
    try:
        import psutil  # noqa: F401  (Window needs it)
    except ImportError:
        return subprocess.Popen(full, creationflags=subprocess.CREATE_NEW_CONSOLE)
    # CREATE_NO_WINDOW: its own hidden console, so a Ctrl+C here can't interrupt it halfway.
    r = subprocess.run([sys.executable, "-c", START, *full], capture_output=True, encoding="utf-8",
                       errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
    pid = r.stdout.strip()
    if r.returncode != 0 or not pid.isdigit():
        raise OSError(f"could not open the job window: {r.stderr.strip()[-500:]}")
    return Window(int(pid))


def _window_for(log: Path):
    """The window writing `log`, if one is up (its log name is unique to the command that made it)."""
    try:
        import psutil
    except ImportError:
        return None
    procs = [p.info for p in psutil.process_iter(["pid", "ppid", "cmdline"])]
    mine = {p["pid"]: p for p in procs if p.get("cmdline") and is_window(p["cmdline"])
            and describe(p["cmdline"])["log"] == str(log)}
    top = [pid for pid, p in mine.items() if p.get("ppid") not in mine]
    return Window(top[0]) if top else None


def _closed(full: list[str], out) -> None:
    """The window went away without the job finishing. A variants job may not have managed to stop
    local-image's server (in its own window) on the way out; we are outside that console, so do it."""
    if describe(full)["stops_local_image"]:
        from . import qwen
        out("Stopping local-image's server, as the job would have.")
        try:
            qwen.stop()
        except qwen.LocalImageMissing:
            pass


def _wait_started(proc, log: Path, seconds: float = 15) -> None:
    """Until the job's first line is in the log (logtee is running it), or the window is gone."""
    end = time.monotonic() + seconds
    while time.monotonic() < end and proc.poll() is None:
        try:
            if log.stat().st_size > 0 and b"\n" in log.read_bytes().split(b"\n", 1)[1]:
                return
        except (OSError, IndexError):
            pass
        time.sleep(0.1)


class launch_lock:
    """An exclusive lock on logs/launch.lock while a job is checked for and started."""

    def __enter__(self):
        LOGS.mkdir(parents=True, exist_ok=True)
        self.f = open(LOGS / "launch.lock", "a+b")
        try:
            import msvcrt
        except ImportError:
            return self
        end = time.monotonic() + 30
        while True:
            try:
                self.f.seek(0)
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
                self.locked = msvcrt
                return self
            except OSError:
                if time.monotonic() > end:
                    return self  # a stuck holder: go ahead rather than never start
                time.sleep(0.2)

    def __exit__(self, *exc):
        m = getattr(self, "locked", None)
        if m:
            self.f.seek(0)
            m.locking(self.f.fileno(), m.LK_UNLCK, 1)
        self.f.close()


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
closing = threading.Event()   # set once the window is being closed
_done = threading.Event()     # set once the close actions have run
CLOSE_EVENTS = (2, 5, 6)  # CTRL_CLOSE_EVENT, CTRL_LOGOFF_EVENT, CTRL_SHUTDOWN_EVENT
CLOSE_GRACE = 0.5


def _run_close_actions() -> None:
    closing.set()
    for a in list(_on_close):
        try:
            a()
        except Exception:
            pass
    _done.set()


def on_close(action):
    """Run `action` (quick: Windows allows ~5 s) if this process's console window is closed.
    Returns a function that cancels it. Closing the window kills everything attached to it, but
    not what runs in another window, such as local-image's server.

    The close also kills our console children (local-image's client), so the main thread can
    reach its cleanup while the handler thread is still starting. The cleanup should therefore
    give a close CLOSE_GRACE seconds to show up (`closing.wait(CLOSE_GRACE)`) and cancel only
    after doing its own stop, so a close at any point finds the action in place. Once the window
    is closing, cancel leaves the action and waits (up to 4 s) for it to have run."""
    global _handler
    if os.name == "nt" and _handler is None:
        @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
        def handler(event):
            if event in CLOSE_EVENTS:
                _run_close_actions()
            return 0  # let Windows (or Python, for Ctrl+C) carry on as usual
        _handler = handler  # keep a reference, or the callback is collected
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_handler, True)
    _on_close.append(action)

    def cancel():
        if closing.is_set():
            _done.wait(4)
        elif action in _on_close:
            _on_close.remove(action)
    return cancel


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
    def after(flag):
        return cmdline[cmdline.index(flag) + 1] if flag in cmdline[:-1] else None

    job = cmdline[cmdline.index("--in-window") + 1:] if "--in-window" in cmdline else []
    cmd = next((a for a in job if a in WINDOWED), "?")
    t = after("--title") or ""  # title() put the command and the photo's name there
    what = t[len("local-upscale "):t.rfind(" - ")] if t.startswith("local-upscale ") and " - " in t else cmd
    # The parser takes no abbreviations (allow_abbrev=False), so --keep-server is spelled out.
    return {"cmd": cmd, "what": what, "log": after("--log"),
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


def stop(only: int | None = None, out=print) -> int | None:
    """Stop this install's running jobs (or only the one whose window process is `only`), close
    their windows, and stop local-image's server if a stopped job would have stopped it.
    Returns 0 stopped, 1 something survived, None nothing was running."""
    import psutil
    found = [j for j in jobs() if only is None or j["pid"] == only]
    if not found:
        if only is None:
            out("  No local-upscale job is running.")
        return None
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
