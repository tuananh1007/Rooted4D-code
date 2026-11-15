r"""Where things live, resolved once.

The workspace is organised into three top-level areas -- ``Docs`` for written
material, ``Codes`` for the implementation, ``Manuscript`` for the paper -- with
data and results alongside them. This module resolves the workspace root so that
nothing else has to count ``parents[...]`` levels, which is the part that breaks
silently when a directory moves.

An explicit ``ROOTED4D_ROOT`` wins if set, which is what the cluster job uses,
since there the code tree and the data tree are different filesystems joined by
a symlink.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["code_root", "data_root", "docs_root", "results_root", "workspace_root"]

_MARKERS = ("GLOBAL_SPLIT_LOCK.json", "Codes", "Docs")


def workspace_root() -> Path:
    """The directory holding Codes, Docs, Manuscript, data and results."""

    override = os.environ.get("ROOTED4D_ROOT")
    if override:
        return Path(override).resolve()
    # rooted4d/paths.py -> rooted4d -> Codes -> workspace
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if all((candidate / marker).exists() for marker in _MARKERS):
            return candidate
    # Fall back to the historical flat layout rather than raising: a checkout
    # that predates the reorganisation should still import.
    return here.parents[2]


def code_root() -> Path:
    return workspace_root() / "Codes"


def docs_root() -> Path:
    return workspace_root() / "Docs"


def data_root() -> Path:
    override = os.environ.get("ROOTED4D_DATA")
    return Path(override).resolve() if override else workspace_root() / "data"


def results_root() -> Path:
    override = os.environ.get("ROOTED4D_RESULTS")
    return Path(override).resolve() if override else workspace_root() / "results"
