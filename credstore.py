"""The teammate key, stored link-safe and private (mode 0600) in this plugin's own data folder.

Hermes gives a plugin a state file (hermes_cli/plugins_state.py PluginState: temp file + replace, mode 0600). The key does not
go there: it goes in ``credential.json`` beside it. Every access walks from the trusted root (the Hermes home, opened and
checked to be a real directory owned by this user) through EVERY path component with dir_fd + O_DIRECTORY | O_NOFOLLOW, creating
missing ones with mkdir(dir_fd=...): a link at ANY component below the home makes the open fail. Writes create a temp file
O_EXCL | O_NOFOLLOW | 0600 in the pinned folder, fsync, then rename inside that folder; reads open the file O_NOFOLLOW and require
a regular file owned by this user with no group or other bits. Every other key of the plugin's state stays in Hermes's own
state file. The same writer keeps the per-thread session generations (thread-epochs.json).
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any, Optional

CREDENTIAL_FILE = "credential.json"
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


class UnsafePath(Exception):
    pass


def _race_point(step: str) -> None:
    """A no-op the race tests replace, to swap a folder for a link between a check and its use."""


def _trusted_root() -> Path:
    """The anchor every path is walked from: the Hermes home this plugin runs in."""
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home())


def _open_root(root: Path) -> int:
    """The Hermes home, opened as a directory and checked to be a real directory owned by this user. It is the trusted
    anchor (Hermes's own data root); every component below it is opened with O_NOFOLLOW relative to a descriptor."""
    try:
        fd = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        raise UnsafePath(f"the Hermes home {root} cannot be opened ({exc.strerror})") from exc
    st = os.fstat(fd)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
        os.close(fd)
        raise UnsafePath(f"the Hermes home {root} is not a folder owned by this user")
    return fd


def _open_dir(folder: Path, *, create: bool = False, root: Optional[Path] = None) -> Optional[int]:
    """A descriptor for ``folder``, reached by walking EVERY component below the trusted root with dir_fd + O_DIRECTORY |
    O_NOFOLLOW (created with mkdir(dir_fd=...) when asked): a link anywhere between the Hermes home and the folder makes the
    open fail. None when the folder does not exist and ``create`` is False."""
    root = Path(root) if root is not None else _trusted_root()
    try:
        rel = Path(os.path.abspath(folder)).relative_to(Path(os.path.abspath(root)))
    except ValueError:
        raise UnsafePath(f"{folder} is not inside the Hermes home {root}") from None
    fd = _open_root(root)
    try:
        for part in rel.parts:
            if part in ("", ".", ".."):
                raise UnsafePath(f"{folder} has a {part!r} component")
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            _race_point(f"open:{part}")
            try:
                nfd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if create:
                    raise
                os.close(fd)
                return None
            except OSError as exc:
                raise UnsafePath(f"{folder}: {part} is a link or not a folder ({exc.strerror})") from exc
            os.close(fd)
            fd = nfd
        st = os.fstat(fd)
        if st.st_uid != os.getuid():
            raise UnsafePath(f"{folder} is not owned by this user")
        return fd
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def write_secret(folder: Path, name: str, data: dict, *, root: Optional[Path] = None, durable_dir: bool = False) -> None:
    """``durable_dir``: the folder's fsync after the rename must succeed too (the thread ledger, rule D); off for the
    credential file, whose callers already tolerate a lost write (pairing again recovers it)."""
    dfd = _open_dir(folder, create=True, root=root)
    tmp = f".{name}.{os.getpid()}.{os.urandom(6).hex()}"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), 0o600, dir_fd=dfd)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                os.fchmod(f.fileno(), 0o600)
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            os.unlink(tmp, dir_fd=dfd)
            raise
        _race_point(f"replace:{name}")
        os.replace(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
        try:
            os.fsync(dfd)   # the rename itself survives a crash
        except OSError:
            if durable_dir:
                raise
    finally:
        try:
            os.unlink(tmp, dir_fd=dfd)
        except FileNotFoundError:
            pass
        os.close(dfd)


def read_secret(folder: Path, name: str, *, root: Optional[Path] = None, require_dict: bool = False) -> Optional[dict]:
    dfd = _open_dir(folder, root=root)
    if dfd is None:
        return None
    try:
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0), dir_fd=dfd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise UnsafePath(f"{folder / name} is a link ({exc.strerror})") from exc
        with os.fdopen(fd, "r", encoding="utf-8") as f:
            st = os.fstat(f.fileno())
            if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
                raise UnsafePath(f"{folder / name} is not a private file of this user (mode {oct(st.st_mode & 0o777)})")
            data = json.load(f)
        if require_dict and not isinstance(data, dict):
            raise ValueError(f"{name} does not hold a JSON object")
        return data if isinstance(data, dict) else None
    finally:
        os.close(dfd)


def delete_secret(folder: Path, name: str, *, root: Optional[Path] = None) -> None:
    dfd = _open_dir(folder, root=root)
    if dfd is None:
        return
    try:
        os.unlink(name, dir_fd=dfd)
    except FileNotFoundError:
        pass
    finally:
        os.close(dfd)


EPOCHS_FILE = "thread-epochs.json"


class CredentialState:
    """ctx.state with the ``credential`` key moved to ``credential.json`` (link-safe, 0600). Other keys pass through to
    Hermes's own state. A credential found in Hermes's state (an older install) is moved over on first read."""

    def __init__(self, state: Any, folder: Optional[Path] = None, root: Optional[Path] = None):
        self._state = state
        path = getattr(state, "path", None)
        self.folder = Path(folder) if folder is not None else (Path(path).parent if path else None)
        self.root = root

    def get(self, key: str, default: Any = None) -> Any:
        if key != "credential" or self.folder is None:
            return self._state.get(key, default) if self._state is not None else default
        got = read_secret(self.folder, CREDENTIAL_FILE, root=self.root)
        if got is None and self._state is not None:
            legacy = self._state.get("credential")
            if isinstance(legacy, dict):
                write_secret(self.folder, CREDENTIAL_FILE, legacy, root=self.root)
                self._state.set("credential", None)
                return legacy
        return got if got is not None else default

    def set(self, key: str, value: Any) -> None:
        if key != "credential" or self.folder is None:
            if self._state is not None:
                self._state.set(key, value)
            return
        if value is None:
            delete_secret(self.folder, CREDENTIAL_FILE, root=self.root)
        else:
            write_secret(self.folder, CREDENTIAL_FILE, value, root=self.root)

    def read_json(self, name: str, require_dict: bool = False) -> Optional[dict]:
        return read_secret(self.folder, name, root=self.root, require_dict=require_dict) if self.folder is not None else None

    def write_json(self, name: str, data: dict, durable_dir: bool = False) -> None:
        if self.folder is not None:
            write_secret(self.folder, name, data, root=self.root, durable_dir=durable_dir)
