from __future__ import annotations

import argparse
from pathlib import Path

from .alignment import align
from .io import load_json, load_npz, load_positions_tsv, load_snirf


def _load_input(path: Path, input_format: str, unit: str):
    selected = "snirf" if input_format == "auto" and path.suffix.lower() == ".snirf" else input_format
    if selected == "snirf":
        return load_snirf(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz(path)
    if suffix == ".json":
        return load_json(path)
    if suffix == ".tsv":
        return load_positions_tsv(path, unit=unit)
    raise ValueError("position input must use .tsv, .json, or .npz")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="montage-align")
    parser.add_argument("input", help="TSV, JSON, NPZ, or SNIRF input")
    parser.add_argument(
        "--input-format",
        "--mode",
        dest="input_format",
        default="auto",
        choices=("auto", "positions", "snirf"),
        help="input mode; auto detects SNIRF by extension",
    )
    parser.add_argument("--unit", default="m", choices=("m", "cm", "mm"))
    parser.add_argument("--output-dir", default="montage-alignment-output")
    arguments = parser.parse_args(argv)
    source = Path(arguments.input)
    montage = _load_input(source, arguments.input_format, arguments.unit)
    result = align(montage)
    output = Path(arguments.output_dir)
    result.save_npz(output / "mapping.npz")
    result.save_json(output / "mapping.json")
    result.save_tsv(output / "mapping.tsv")
    result.plot(output / "montage_before_after.png")
    print(f"mapped {result.valid.sum()}/{len(result.valid)} channels -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
