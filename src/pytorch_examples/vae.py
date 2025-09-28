"""Helpers for executing the official PyTorch VAE example within this repository.

The goal is to let us reuse the upstream implementation with (almost) no
modification. We run the original script via :mod:`runpy` in the same process
so that it behaves as if it had been executed from its own directory.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Sequence

DEFAULT_DATA_SUBDIR = Path("data")


class OfficialExampleNotFoundError(FileNotFoundError):
    """Raised when the expected PyTorch example script cannot be located."""


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _example_script_path() -> Path:
    repo_root = _repository_root()
    script_path = repo_root / "examples" / "vae" / "main.py"
    if not script_path.exists():
        raise OfficialExampleNotFoundError(
            "Could not find PyTorch example script at expected location: "
            f"{script_path}. Ensure the examples/ directory is present."
        )
    return script_path


def _ensure_example_data_dir(data_root: Path, *, example_data_dir: Path) -> None:
    if example_data_dir.exists():
        return

    example_data_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        example_data_dir.symlink_to(data_root, target_is_directory=True)
        return
    except OSError:
        # Fall back to creating a directory (the script will re-download if needed).
        example_data_dir.mkdir(parents=True, exist_ok=True)


def run_vae_example(
    argv: Sequence[str] | None = None,
    *,
    data_root: Path | str | None = None,
    ensure_data_link: bool = True,
) -> None:
    """Execute the upstream ``examples/vae/main.py`` script in-process.

    Parameters
    ----------
    argv:
        CLI arguments forwarded to the official script (e.g. ``["--epochs", "5"]``).
    data_root:
        Optional override for the dataset root. By default we reuse ``./data`` at the
        repository root, symlinked into ``examples/data`` to match the original
        relative path layout.
    ensure_data_link:
        When true (default), automatically create a symlink from
        ``examples/data`` to ``data_root`` when possible. If symlinks are not
        supported, a real directory is created instead, allowing the script to
        download the dataset locally.
    """

    script_path = _example_script_path()
    argv = list(argv or [])

    repo_root = _repository_root()
    example_dir = script_path.parent
    example_data_dir = example_dir.parent / "data"

    if data_root is None:
        data_root = repo_root / DEFAULT_DATA_SUBDIR
    data_root = Path(data_root).resolve()

    if ensure_data_link and data_root.exists():
        _ensure_example_data_dir(data_root, example_data_dir=example_data_dir)

    # Ensure outputs directory exists to avoid errors when saving reconstructions.
    (example_dir / "results").mkdir(exist_ok=True)

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(script_path)] + argv
        os.chdir(example_dir)
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)