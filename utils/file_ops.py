# Keeper - file operations: collecting shots, moving rejects and duplicates,
# and the undo log.
#
# Ground rules:
#   - originals are never deleted or overwritten, only moved into subfolders
#     of the processed folder;
#   - every move is written to the undo log BEFORE it happens (write-ahead), and
#     the log itself is replaced atomically, so even a killed run can be undone;
#   - undo only ever moves files between the folder and its own output
#     subfolders - paths read from the log are validated, never trusted.

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import config
from core.sharpness import is_kept

LOG_VERSION = 2

# The log holds two kinds of runs. Entries without a "kind" predate the
# correction feature and are moves; the Hungarian kind names come from
# versions before 2026-10.
KIND_MOVE = "move"
KIND_CORRECTION = "correction"
_LEGACY_KINDS = {"mozgatas": KIND_MOVE, "javitas": KIND_CORRECTION}

STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"

# Output folders an undo may move files back from (current and legacy names).
MOVE_FOLDERS = frozenset(
    [config.FOLDER_REJECTS, config.FOLDER_DUPLICATES]
    + [old for old, new in config.LEGACY_FOLDERS.items() if new != config.FOLDER_CORRECTED]
)


@dataclass
class ImageItem:
    """
    One SHOT, not one file: a RAW+JPEG pair (DSC01234.ARW + DSC01234.JPG) is
    measured once and always moved together.
    """

    primary: Path      # the file that is measured
    files: list        # every file that moves with the shot (primary included)

    @property
    def is_pair(self):
        return len(self.files) > 1


def _files_with_extensions(folder, extensions):
    folder = Path(folder)
    if not folder.is_dir():
        return []

    return [
        entry
        for entry in sorted(folder.iterdir())
        if entry.is_file() and entry.suffix.lower() in extensions
    ]


def collect_images(folder):
    """
    The shots of a folder, sorted by name. Not recursive on purpose: the
    output subfolders are skipped automatically and the user processes exactly
    one folder. RAW and JPEG files with the same stem (case-insensitive) form
    one ImageItem; the JPEG is measured because it decodes fastest.
    """
    files = _files_with_extensions(folder, config.IMAGE_EXTENSIONS)

    if not config.PAIR_RAW_WITH_JPEG:
        return [ImageItem(primary=f, files=[f]) for f in files]

    groups = {}
    for f in files:
        groups.setdefault(f.stem.lower(), []).append(f)

    items = []
    for members in groups.values():
        jpegs = [f for f in members if f.suffix.lower() in config.JPEG_EXTENSIONS]
        items.append(ImageItem(primary=jpegs[0] if jpegs else members[0], files=members))

    return items


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


def long_path(path):
    """
    The path as a string the OS accepts even beyond MAX_PATH: on Windows long
    absolute paths get the extended-length prefix.
    """
    text = os.path.abspath(str(path))
    prefix = "\\\\?\\"
    if os.name != "nt" or len(text) < config.LONG_PATH_LIMIT or text.startswith(prefix):
        return text
    if text.startswith("\\\\"):
        return prefix + "UNC\\" + text[2:]
    return prefix + text


def _same_dir(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def is_plain_name(name):
    """True for a bare file name: no directory part, drive, stream or '..'."""
    if not isinstance(name, str) or name in ("", ".", ".."):
        return False
    if any(ch in name for ch in "/\\:\0"):
        return False
    return Path(name).name == name


def safe_subfolder(folder, name):
    """
    folder/name - or None if that subfolder is a link or junction leading
    somewhere else. A crafted folder must not be able to redirect an undo.
    """
    sub = Path(folder) / name
    try:
        expected = os.path.join(os.path.realpath(folder), name)
        return sub if _same_dir(sub, expected) else None
    except OSError:
        return None


def _move_file(src, dst):
    """
    Rename without overwriting and without a copy fallback. Source and target
    are on the same volume, so this is atomic; a locked source simply fails
    and nothing is left behind (shutil.move would copy, then fail to delete).
    """
    if os.path.lexists(long_path(dst)):
        raise FileExistsError("the target already exists: %s" % Path(dst).name)
    os.rename(long_path(src), long_path(dst))


def unique_destination(folder, name, reserved=None):
    """
    A free target path in folder: name, name_1, name_2 ... The real name is
    logged, so undo stays exact. reserved (lower-case names) lets a caller plan
    several moves before any of them happens.
    """
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, counter = name, 0
    while True:
        target = Path(folder) / candidate
        taken = reserved is not None and candidate.lower() in reserved
        if not taken and not os.path.lexists(long_path(target)):
            if reserved is not None:
                reserved.add(candidate.lower())
            return target
        counter += 1
        candidate = "%s_%d%s" % (stem, counter, suffix)


# ---------------------------------------------------------------------------
# the undo log
# ---------------------------------------------------------------------------


def log_path(folder):
    return Path(folder) / config.LOG_FILENAME


def read_log(folder):
    """
    The undo log; an empty one if the file does not exist. A damaged log
    raises - it is never overwritten, that would lose the undo information.
    """
    path = log_path(folder)
    if not path.exists():
        return {"runs": []}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise ValueError("the undo log is damaged")

    return data


def write_log(folder, data):
    """
    Atomic write: temp file, flush + fsync, then os.replace. A log without
    runs is removed instead of leaving clutter in the user's folder.
    """
    path = log_path(folder)

    if not data.get("runs"):
        if path.exists():
            path.unlink()
        return

    data["version"] = LOG_VERSION
    temp = path.with_name(path.name + ".tmp")
    try:
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def run_kind(run):
    kind = run.get("kind", KIND_MOVE)
    return _LEGACY_KINDS.get(kind, kind)


def _now():
    return datetime.now().isoformat(timespec="seconds")


def last_run_kind(folder):
    """Kind of the last run, or None (also for a damaged log)."""
    try:
        runs = read_log(folder)["runs"]
    except Exception:
        return None
    return run_kind(runs[-1]) if runs else None


def has_undoable_run(folder):
    """True if the log has a run to undo. False for a damaged log."""
    try:
        return bool(read_log(folder)["runs"])
    except Exception:
        return False


def last_run_summary(folder):
    """
    What the undo button would undo, for the confirmation dialog:
    {"kind", "count", "timestamp", "interrupted"} or None.
    """
    try:
        runs = read_log(folder)["runs"]
    except Exception:
        return None
    if not runs:
        return None

    run = runs[-1]
    kind = run_kind(run)
    entries = run.get("moves" if kind == KIND_MOVE else "created") or []
    return {
        "kind": kind,
        "count": len(entries),
        "timestamp": str(run.get("timestamp", "")),
        "interrupted": run.get("status") == STATUS_IN_PROGRESS,
    }


# ---------------------------------------------------------------------------
# moving
# ---------------------------------------------------------------------------


@dataclass
class MoveReport:
    """What was moved and what failed."""

    moved: list = field(default_factory=list)    # (source, target) pairs
    failed: list = field(default_factory=list)   # (path, reason) pairs
    # set when nothing was moved at all (read-only folder, damaged log ...)
    error: str = None


def destination_folder(result, threshold):
    """
    Where a shot goes, or None if it stays. Decided in ONE place, because the
    cards, the counters and the actual move must agree:
      1. an unreadable shot is never touched
      2. a manual "keep" beats everything
      3. below the threshold it is a reject (a technical failure outranks
         being a duplicate)
      4. finally the duplicate mark
    """
    if result.score is None or result.override is True:
        return None
    if not is_kept(result, threshold):
        return config.FOLDER_REJECTS
    if result.duplicate:
        return config.FOLDER_DUPLICATES
    return None


def move_to_rejects(folder, results, threshold):
    """
    Moves rejects to rejected/ and surplus series members to duplicates/, in
    ONE log run so a single undo brings back both. Unreadable shots and shots
    that are not directly inside folder are never touched.
    """
    folder = Path(folder)
    report = MoveReport()

    planned = []
    for result in results:
        target = destination_folder(result, threshold)
        if target is None:
            continue
        # results always belong to the folder they were measured in; anything
        # else (e.g. a stale result list after switching folders) is refused
        if not all(_same_dir(Path(f).parent, folder) for f in result.files):
            report.failed.append((result.path, "not in the selected folder, skipped"))
            continue
        planned.append((result, target))

    if not planned:
        return report

    # read the log BEFORE touching anything: what cannot be logged must not
    # be moved
    try:
        data = read_log(folder)
    except Exception as exc:
        report.error = "The undo log is damaged, nothing was moved: %s" % exc
        return report

    # only create the folders that will receive files
    for name in sorted({target for _, target in planned}):
        try:
            (folder / name).mkdir(exist_ok=True)
        except OSError as exc:
            report.error = "The %s folder cannot be created: %s" % (name, exc)
            return report

    run = {
        "timestamp": _now(),
        "kind": KIND_MOVE,
        "status": STATUS_IN_PROGRESS,
        "threshold": threshold,
        "moves": [],
    }
    data["runs"].append(run)
    reserved = {}

    try:
        for start in range(0, len(planned), config.LOG_BATCH):
            batch = []
            for result, target in planned[start:start + config.LOG_BATCH]:
                names = reserved.setdefault(target, set())
                entries = [
                    {
                        "from": Path(src).name,
                        "to": "%s/%s" % (
                            target,
                            unique_destination(folder / target, Path(src).name, names).name,
                        ),
                        "score": round(result.score, 1),
                    }
                    for src in result.files
                ]
                run["moves"].extend(entries)
                batch.append((result, target, entries))

            # write-ahead: the planned moves are on disk before any of them
            write_log(folder, data)

            for result, target, entries in batch:
                _move_shot(folder, result, target, entries, run, report)
    except OSError as exc:
        # the log could not be written: stop. Everything moved so far is in
        # the last successfully written log, so it can still be undone.
        report.error = "The undo log could not be written, the run stopped: %s" % exc
        return report

    run["status"] = STATUS_DONE
    if not run["moves"]:
        data["runs"].remove(run)
    write_log(folder, data)

    # a folder that received nothing (every move failed) is not left behind
    remove_empty_folders(folder, sorted({target for _, target in planned}))
    return report


def _move_shot(folder, result, target, entries, run, report):
    """
    Moves every file of one shot. If any file fails, the files already moved
    go back, so a RAW+JPEG pair is never split.
    """
    done = []
    failure = None
    for src, entry in zip(result.files, entries):
        try:
            _move_file(src, folder / entry["to"])
        except OSError as exc:
            failure = (Path(src), str(exc))
            break
        done.append((Path(src), folder / entry["to"], entry))

    if failure is None:
        report.moved.extend((src, dst) for src, dst, _ in done)
        return

    stuck = []
    for src, dst, entry in reversed(done):
        try:
            _move_file(dst, src)
        except OSError:
            # it stays logged, so Undo can still bring it back
            stuck.append(entry)
            report.failed.append((dst, "left in the %s folder; Undo moves it back" % target))

    dropped = [e for e in entries if not any(e is s for s in stuck)]
    run["moves"] = [e for e in run["moves"] if not any(e is d for d in dropped)]
    report.failed.append(failure)


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------


@dataclass
class UndoReport:
    """What the undo handled, what it skipped and why."""

    restored: list = field(default_factory=list)  # restored (or removed) paths
    skipped: list = field(default_factory=list)   # (path, reason) pairs
    # entries kept in the log because they may succeed later (name taken,
    # file locked); the run stays undoable while this is > 0
    remaining: int = 0


def _logged_move(folder, entry):
    """
    (stored file, original place) for a log entry, or None if the entry does
    not describe a move inside this folder. "from" must be a bare file name
    and "to" must be <output folder>/<bare file name>.
    """
    if not isinstance(entry, dict):
        return None
    source, stored = entry.get("from"), entry.get("to")
    if not is_plain_name(source) or not isinstance(stored, str):
        return None

    parts = stored.split("/")
    if len(parts) != 2 or parts[0] not in MOVE_FOLDERS or not is_plain_name(parts[1]):
        return None

    sub = safe_subfolder(folder, parts[0])
    if sub is None:
        return None
    return sub / parts[1], Path(folder) / source


def remove_empty_folders(folder, names):
    for name in names:
        sub = Path(folder) / name
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()


def undo_last_run(folder):
    """
    Moves the files of the last (move) run back. Existing files are never
    overwritten. Entries that may still succeed later stay in the log; the
    run disappears only when nothing restorable is left.
    """
    folder = Path(folder)
    data = read_log(folder)
    report = UndoReport()

    if not data["runs"]:
        return report

    run = data["runs"][-1]
    if run_kind(run) != KIND_MOVE:
        raise ValueError("the last run was not a move; save_ops undoes corrections")

    keep = []
    # backwards, so _1/_2 suffixes resolve in reverse order
    for entry in reversed(run.get("moves") or []):
        paths = _logged_move(folder, entry)
        if paths is None:
            shown = entry.get("to") if isinstance(entry, dict) else entry
            report.skipped.append((str(shown), "unsafe path in the undo log, ignored"))
            continue
        stored, original = paths

        if not os.path.lexists(long_path(stored)):
            if os.path.lexists(long_path(original)):
                continue  # logged ahead but never moved (interrupted run)
            report.skipped.append((stored, "no longer in the %s folder" % stored.parent.name))
            continue

        if os.path.lexists(long_path(original)):
            report.skipped.append((original, "a file with the same name is already there"))
            keep.append(entry)
            continue

        try:
            _move_file(stored, original)
        except OSError as exc:
            report.skipped.append((stored, str(exc)))
            keep.append(entry)
            continue

        report.restored.append(original)

    if keep:
        run["moves"] = list(reversed(keep))
        run["status"] = STATUS_DONE
    else:
        data["runs"].pop()
    write_log(folder, data)

    remove_empty_folders(folder, sorted(MOVE_FOLDERS))
    report.remaining = len(keep)
    return report
