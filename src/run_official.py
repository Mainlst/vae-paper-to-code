"""Command-line entry point for executing the upstream PyTorch VAE example."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .pytorch_examples import run_vae_example


def _parse_args(argv: Sequence[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the official PyTorch VAE example (examples/vae/main.py) "
            "from this repository while keeping the upstream code untouched."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=(
            "Dataset directory to expose to the example. Defaults to ./data at the "
            "repository root when available."
        ),
    )
    parser.add_argument(
        "--no-data-link",
        action="store_true",
        help=(
            "Disable automatic creation of a symlink from examples/data to the "
            "chosen data root."
        ),
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args, forwarded = _parse_args(argv)
    run_vae_example(
        forwarded,
        data_root=args.data_root,
        ensure_data_link=not args.no_data_link,
    )


if __name__ == "__main__":
    main()
