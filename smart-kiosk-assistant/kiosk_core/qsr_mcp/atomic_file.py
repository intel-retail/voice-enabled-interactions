"""Crash-safe JSON persistence for the QSR MCP service's small state files.

``board_state`` and ``menu_requests`` each own a single JSON file that is
rewritten in full on every change. A plain ``Path.write_text`` truncates the
file before writing, so a crash, a full disk, or a container stop part-way
through leaves a half-written document behind. Both modules treat an
unparseable file as "no state", so the corruption is silent: an agent-set
board mode or a queue of pending operator approvals simply disappears on the
next start.

Writing to a temporary file in the same directory and then ``os.replace``-ing
it over the target makes the swap atomic on POSIX, so a reader sees either the
complete previous document or the complete new one, never a partial write.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: Path, data: Any) -> None:
    """Serialise ``data`` as JSON and atomically replace ``path``.

    The temporary file is created in the destination directory so the final
    ``os.replace`` stays on one filesystem, which is what makes it atomic.

    Args:
        path: Destination file. Its parent directory must already exist.
        data: Any JSON-serialisable object.

    Raises:
        OSError: If the write or the replace fails. Callers decide whether a
            persistence failure is fatal; the original file is left intact.
    """
    directory = path.parent
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
            handle.flush()
            # Without fsync the rename can be durable while the bytes are not,
            # which reintroduces the truncated-file case after a power loss.
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # Never leave the stray temp file behind on failure; the destination
        # is untouched because the replace is what publishes the new content.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
