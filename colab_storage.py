"""Drive persistence helpers, also embedded in the standalone Colab notebook."""
import json
import errno
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINT_MARKER = "checkpoint_complete.json"


def retry_io(operation, attempts=3):
    """Retry transient mounted-Drive failures, without hiding persistent errors."""
    for attempt in range(attempts):
        try:
            return operation()
        except OSError as error:
            if error.errno in (errno.ENOSPC, errno.EDQUOT, errno.EACCES, errno.EPERM) or attempt + 1 == attempts:
                raise
            delay = 2 ** attempt
            print(f'I/O tạm thời thất bại; thử lại sau {delay}s ({attempt + 1}/{attempts}).')
            time.sleep(delay)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(value, ensure_ascii=False, indent=2)
    def write():
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(path)
    retry_io(write)


def seal_checkpoint(directory, run):
    """Commit a snapshot only after weights and training state have been written."""
    directory = Path(directory)
    files = {p.relative_to(directory).as_posix(): p.stat().st_size
             for p in directory.rglob("*") if p.is_file()
             and p.name != CHECKPOINT_MARKER and not p.name.endswith(".tmp")}
    atomic_json(directory / CHECKPOINT_MARKER, {
        "format": 2, "step": run["completed_steps"],
        "fingerprint": run["fingerprint"], "files": files,
    })


def valid_checkpoint(directory, fingerprint=None):
    directory = Path(directory)
    try:
        marker = json.loads((directory / CHECKPOINT_MARKER).read_text(encoding="utf-8"))
        files = marker["files"]
        if marker.get("format") != 2 or not isinstance(files, dict):
            return False
        required = {"run.json", "training_state.pt", "adapter/adapter_config.json"}
        if not required <= files.keys() or not any(
                name.startswith("adapter/") and name.endswith((".safetensors", ".bin"))
                for name in files):
            return False
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if run["config"]["train_projector"] and "extra_trainable.pt" not in files:
            return False
        if marker["step"] != run["completed_steps"] or marker["fingerprint"] != run["fingerprint"]:
            return False
        if fingerprint is not None and marker["fingerprint"] != fingerprint:
            return False
        for name, size in files.items():
            path = (directory / name).resolve()
            path.relative_to(directory.resolve())
            if not path.is_file() or path.stat().st_size != size or size == 0:
                return False
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


class DriveStore:
    """Copy completed files atomically; publish snapshot markers last."""

    def __init__(self, output_root, saved_root, enabled=True):
        self.output = Path(output_root).resolve()
        self.saved = Path(saved_root).resolve()
        self.enabled = enabled
        if self.output == self.saved or self.output in self.saved.parents or self.saved in self.output.parents:
            raise ValueError("Local output and Drive backup must be separate directories.")

    @staticmethod
    def _files(source):
        files = [source] if source.is_file() else [p for p in source.rglob("*") if p.is_file()]
        return sorted((p for p in files if not p.name.endswith((".tmp", ".uploading", ".partial"))),
                      key=lambda p: (p.name == CHECKPOINT_MARKER,
                                     p.name.endswith(("run.json", ".manifest.json")), str(p)))

    def backup(self, source):
        if not self.enabled:
            return
        return retry_io(lambda: self._backup_once(source))

    def _backup_once(self, source):
        source = Path(source).resolve()
        relative = source.relative_to(self.output)
        if not source.exists():
            raise FileNotFoundError(source)
        destination = self.saved / relative
        if (source / CHECKPOINT_MARKER).is_file():
            # Reusing an already-published snapshot must not remove its commit marker.
            if valid_checkpoint(destination) and (
                    (source / CHECKPOINT_MARKER).read_bytes() == (destination / CHECKPOINT_MARKER).read_bytes()
                    and (source / 'run.json').read_bytes() == (destination / 'run.json').read_bytes()):
                return
            (destination / CHECKPOINT_MARKER).unlink(missing_ok=True)
        for file in self._files(source):
            dest = destination if source.is_file() else destination / file.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.stat().st_size == file.stat().st_size and (
                    dest.stat().st_mtime_ns == file.stat().st_mtime_ns):
                continue
            temporary = dest.with_name(dest.name + ".uploading")
            def copy_file():
                shutil.copy2(file, temporary)
                temporary.replace(dest)
            copy_file()
        if (source / CHECKPOINT_MARKER).is_file() and not valid_checkpoint(destination):
            raise IOError(f"Drive checkpoint verification failed: {destination}")
        atomic_json(self.saved / "backup_status.json", {
            "last_success_utc": datetime.now(timezone.utc).isoformat(),
            "last_path": relative.as_posix(),
        })

    def restore(self):
        return retry_io(self._restore_once)

    def _restore_once(self):
        if not self.saved.is_dir():
            return 0
        snapshots = {p.parent: valid_checkpoint(p.parent)
                     for p in self.saved.rglob(CHECKPOINT_MARKER)}
        repair = {p: not valid_checkpoint(self.output / p.relative_to(self.saved))
                  for p, valid in snapshots.items() if valid}
        restored = 0
        for source in self._files(self.saved):
            if any(parent in snapshots and not snapshots[parent] for parent in source.parents):
                continue
            # An interrupted new snapshot may not have its marker yet.
            parents = [p for p in source.parents if p == self.saved or self.saved in p.parents]
            snapshot = next((p for p in parents if p.name == "final" or re.fullmatch(r"step-\d+", p.name)), None)
            if snapshot is not None and snapshot not in snapshots:
                record = snapshot / "run.json"
                try:
                    legacy = json.loads(record.read_text(encoding="utf-8")).get("checkpoint_format", 1) < 2
                except (OSError, ValueError):
                    legacy = False
                if not legacy:
                    continue
            destination = self.output / source.relative_to(self.saved)
            if not destination.exists() or (snapshot is not None and repair.get(snapshot, False)):
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(destination.name + ".tmp")
                def copy_file():
                    shutil.copy2(source, temporary)
                    temporary.replace(destination)
                retry_io(copy_file)
                restored += 1
        return restored

    def prune_snapshots(self, directory, keep=1):
        """Remove only older committed step snapshots inside this exact run."""
        directory = Path(directory).resolve()
        relative = directory.relative_to(self.output)
        snapshots = sorted((p for p in directory.glob("step-*")
                            if re.fullmatch(r"step-\d+", p.name) and valid_checkpoint(p)),
                           key=lambda p: int(p.name.split("-")[1]))
        for old in snapshots[:-keep]:
            for base in (directory, self.saved / relative) if self.enabled else (directory,):
                target = base / old.name
                # Verify absolute target and parent before any recursive delete.
                if target.is_symlink() or target.resolve().parent != base.resolve():
                    raise ValueError(f"Unsafe checkpoint path: {target}")
                if target.exists():
                    try:
                        retry_io(lambda: shutil.rmtree(target))
                    except OSError:
                        # Retention cleanup is optional; the new checkpoint is already safe.
                        print(f'Chưa dọn được checkpoint cũ, giữ lại để không ngắt train: {target}')
