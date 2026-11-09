"""Run-control helpers for long experiments: status heartbeat, owner lock, STOP marker, disk guard, schedules, counters.

Pure Python (no torch / numpy / emulator) so it is cheap to test.  Nothing here kills a process it did not start:
``RunLock`` only distinguishes a LIVE owner from a stale file and refuses to take a live lock.
"""
from __future__ import annotations

import contextlib
import ctypes
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


class RunHalted(RuntimeError):
    """The run stopped itself visibly (low disk, source changed, ...). ``reason`` is the status ``halt_reason``."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


class LockHeld(RuntimeError):
    """A live process already owns the run lock."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def pid_alive(pid: int) -> bool:
    """True when ``pid`` is a running process.  Windows: OpenProcess + exit code; POSIX: signal 0."""
    pid = int(pid)
    if pid <= 0:
        return False
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class RunLock:
    """``RUNNING.lock`` holding ``{"pid", "started_utc"}``.  A stale file (dead owner) is replaced and the takeover is
    recorded in ``taken_over_from``; a live owner raises ``LockHeld``.  ``release`` only removes a lock this instance owns."""

    def __init__(self, path, pid: int | None = None):
        self.path = Path(path)
        self.pid = int(pid if pid is not None else os.getpid())
        self.taken_over_from: dict | None = None
        self._held = False

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                owner = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                owner = {}
            owner_pid = int(owner.get("pid", 0) or 0)
            if owner_pid and owner_pid != self.pid and pid_alive(owner_pid):
                raise LockHeld(f"{self.path} is owned by live process {owner_pid}")
            self.taken_over_from = owner or {"unreadable": True}
            self.path.unlink(missing_ok=True)
        payload = json.dumps({"pid": self.pid, "started_utc": utc_now()})
        with self.path.open("x", encoding="utf-8") as handle:   # exclusive create: two starters cannot both win
            handle.write(payload)
        self._held = True
        return self

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        try:
            owner = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if int(owner.get("pid", -1)) == self.pid:
            self.path.unlink(missing_ok=True)


def stop_requested(path) -> bool:
    return bool(path) and Path(path).exists()


def free_gib(path) -> float:
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / 1024 ** 3


def check_disk(path, min_gib: float) -> None:
    """Raise ``RunHalted('low_disk')`` when less than ``min_gib`` GiB is free (0 disables the guard)."""
    if min_gib and free_gib(path) < float(min_gib):
        raise RunHalted("low_disk", f"{free_gib(path):.1f} GiB free < {min_gib} GiB guard at {path}")


@contextlib.contextmanager
def prevent_sleep():
    """Process-scoped Windows sleep prevention, released on exit.  No global power setting is touched."""
    active = False
    if sys.platform == "win32":
        try:
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            active = True
        except Exception:
            active = False
    try:
        yield active
    finally:
        if active:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:
                pass


class StatusWriter:
    """Atomic ``status.json`` with a monotonic heartbeat.  ``set`` merges fields; ``beat`` rewrites it."""

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.data: dict = {}
        self.seq = 0
        self._t0 = time.monotonic()

    def set(self, **fields) -> None:
        self.data.update(fields)

    def beat(self, **fields) -> None:
        self.data.update(fields)
        self.seq += 1
        self.data.update(heartbeat_seq=self.seq, heartbeat_monotonic_s=round(time.monotonic() - self._t0, 3),
                         timestamp_utc=utc_now(), pid=os.getpid())
        if self.path is not None:
            atomic_write_json(self.path, self.data)


def code_fingerprint(root) -> dict:
    """SHA256 of every ``src/*.py`` and ``tools/*.py`` under ``root`` (the code that defines an experiment's identity)."""
    import hashlib
    root = Path(root)
    return {str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("src", "tools") for p in sorted((root / folder).glob("*.py"))}


def fingerprint_digest(fp: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()


# Fields that may legitimately differ when a run is continued: budget, control paths and bookkeeping.  Everything else
# (lr, PPO, rewards, masks, entropy/novelty schedules, seeds, environment settings) is part of the experiment identity.
CONFIG_IDENTITY_EXEMPT = ("updates", "stop_path", "trace_path", "save_path", "run_id", "control_dir", "min_free_gib",
                          "snapshot_every", "generation_every", "lr_schedule_updates")


def config_identity(cfg) -> str:
    import hashlib
    d = {k: v for k, v in cfg.to_dict().items() if k not in CONFIG_IDENTITY_EXEMPT}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def write_run_manifest(control, cfg, code_fp: dict, *, resume_kind: str, init_parent_sha256=None, generation=None,
                       lock_takeover=None) -> Path:
    """``<control>/manifest.json``: what code/ROM/state/dependencies/config produced this run.  An existing manifest is
    kept as ``manifest-before-<UTC>.json`` (never overwritten silently)."""
    import hashlib
    import importlib.metadata
    import shutil
    import subprocess
    control = Path(control)
    root = Path(__file__).resolve().parent.parent
    control.mkdir(parents=True, exist_ok=True)

    def digest(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "src", "tools", "tests"], cwd=root,
                                             stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        commit, dirty = None, None
    deps = {}
    for k in ("torch", "numpy", "pyboy"):
        try:
            deps[k] = importlib.metadata.version(k)
        except Exception:
            deps[k] = None
    manifest = {
        "created_utc": utc_now(), "resume_kind": resume_kind, "init_parent_sha256": init_parent_sha256,
        "continued_from_generation": generation, "lock_takeover": lock_takeover,
        "code_commit": commit, "code_tree_dirty_in_src_tools_tests": dirty, "source_hashes": code_fp,
        "code_fingerprint_sha256": fingerprint_digest(code_fp),
        "rom_hashes": {p.name: digest(p) for p in (root / "rom").glob("*") if p.suffix in (".gbc", ".sym")},
        "initial_state_hashes": {str(p.relative_to(root)).replace("\\", "/"): digest(p) for p in (root / "states").glob("*.state")},
        "dependencies": deps, "python": sys.version, "config_identity_sha256": config_identity(cfg),
        "config": cfg.to_dict(),
        "development_seeds": [10101, 10102, 20001, 20002], "reserved_heldout_seeds": [50001, 50002],
        "assistance": {"supply_injection": cfg.supply_injection, "escape_assist": cfg.escape_assist,
                       "auto_dialogue": cfg.auto_dialogue, "path_progress": cfg.path_progress,
                       "goal_potential": cfg.goal_potential,
                       "assisted_start_balls": cfg.assisted_start_balls},   # > 0 = ASSISTED run: never mix with organic results
        "note": "Ordinary emulator options (RTC lock, wOptions via training_options) are environment setup, not progress writes.",
    }
    path = control / "manifest.json"
    if path.exists():
        shutil.copy2(path, control / ("manifest-before-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".json"))
    atomic_write_json(path, manifest)
    return path


def lr_for_update(base_lr: float, end_frac: float, update: int, horizon: int) -> float:
    """Linear LR decay as a function of the GLOBAL update index over a declared ``horizon`` (never per reporting
    interval, so it cannot restart as a sawtooth).  ``end_frac == 1`` is a constant LR."""
    if end_frac >= 1.0:
        return float(base_lr)
    frac = min(1.0, max(0.0, float(update) / float(max(1, horizon))))
    return float(base_lr) * (1.0 - (1.0 - float(end_frac)) * frac)


class Counters:
    """Truthful accounting: decisions and emulator frames are different things, episodes end for different reasons, and
    a restart that abandons non-terminal trajectories is recorded rather than hidden.  JSON-able via ``state``."""

    FIELDS = ("train_decisions", "train_frames", "reset_frames", "episodes_ended", "external_restarts",
              "abandoned_decisions", "abandoned_episodes_in_progress")

    def __init__(self):
        self.train_decisions = 0
        self.train_frames = 0           # sum of per-step emulator frame deltas (turns, warps, dialogue, battle included)
        self.reset_frames = 0           # frames spent inside episode resets (start jitter), counted separately
        self.episodes_ended = {}        # end_reason -> count
        self.external_restarts = 0      # process restarts that did not continue worker state exactly
        self.abandoned_decisions = 0    # decisions whose non-terminal trajectories were discarded by a restart
        self.abandoned_episodes_in_progress = 0

    def add_step(self, info: dict | None, done: bool) -> None:
        info = info or {}
        t = info.get("transition") or {}
        self.train_decisions += 1
        self.train_frames += max(0, int(t.get("frame_after", 0)) - int(t.get("frame_before", 0)))
        if done:
            self.reset_frames += int(info.get("reset_frames", 0))
            reason = str(info.get("end_reason") or "unknown")
            self.episodes_ended[reason] = self.episodes_ended.get(reason, 0) + 1

    def state(self) -> dict:
        return {k: (dict(getattr(self, k)) if isinstance(getattr(self, k), dict) else getattr(self, k)) for k in self.FIELDS}

    def load(self, state: dict) -> None:
        for k in self.FIELDS:
            if k in state:
                setattr(self, k, dict(state[k]) if isinstance(state[k], dict) else int(state[k]))
