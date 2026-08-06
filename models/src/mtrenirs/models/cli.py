from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .model import MtReNIRS
from .recording import Recording


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mtrenirs-embed")
    subcommands = parser.add_subparsers(dest="command", required=True)
    embed = subcommands.add_parser("embed")
    embed.add_argument("input", help="Canonical NPZ recording")
    embed.add_argument("--checkpoint", default="mtrenirs-10s")
    embed.add_argument("--output", default="embeddings.npz")
    embed.add_argument("--onset", type=float, default=0.0)
    embed.add_argument("--duration", type=float)
    embed.add_argument(
        "--pool", default="none", choices=("none", "flatten", "temporal_mean")
    )
    embed.add_argument(
        "--remainder", default="drop", choices=("drop", "pad", "overlap")
    )
    arguments = parser.parse_args(argv)
    recording = Recording.from_npz(arguments.input)
    model = MtReNIRS.from_pretrained(arguments.checkpoint)
    result = model.embed_segment(
        recording,
        onset=arguments.onset,
        duration=arguments.duration,
        pool=arguments.pool,
        remainder=arguments.remainder,
    )
    output = Path(arguments.output)
    np.savez_compressed(
        output,
        embeddings=result.embeddings,
        window_onsets_seconds=result.window_onsets_seconds,
        window_duration_seconds=np.asarray(result.window_duration_seconds),
        pooling=np.asarray(result.pooling),
    )
    print(f"wrote {output}: {result.embeddings.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
