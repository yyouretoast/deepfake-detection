"""Filesystem utilities and cross-platform path handling."""

import os
import pathlib


def patch_pathlib_mkdir() -> None:
    """Fixes Windows Python 3.10 pathlib.Path.mkdir issue with exist_ok."""

    def _patched_mkdir(self: pathlib.Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        if parents:
            os.makedirs(str(self), mode=mode, exist_ok=exist_ok)
        else:
            try:
                os.mkdir(str(self), mode=mode)
            except FileExistsError:
                if not exist_ok:
                    raise

    pathlib.Path.mkdir = _patched_mkdir
