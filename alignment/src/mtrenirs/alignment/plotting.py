# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from .alignment import _to_2d, load_reference
from .models import AlignmentResult


def _draw_head(axis, radius: float) -> None:
    color = "#64706b"
    axis.add_patch(plt.Circle((0, 0), radius, fill=False, lw=1.2, ec=color))
    axis.plot(
        [-0.10 * radius, 0, 0.10 * radius],
        [0.99 * radius, 1.14 * radius, 0.99 * radius],
        color=color,
        lw=1.2,
    )
    for side in (-1, 1):
        axis.add_patch(
            patches.Ellipse(
                (side * radius, 0),
                0.10 * radius,
                0.28 * radius,
                fill=False,
                lw=1.2,
                ec=color,
            )
        )


def _format_head(axis, radius: float) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(-1.22 * radius, 1.22 * radius)
    axis.set_ylim(-1.24 * radius, 1.24 * radius)
    axis.axis("off")


def plot_alignment(
    result: AlignmentResult,
    path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = load_reference(result.reference)
    ref_2d = np.asarray(reference["positions_2d"])
    assigned = _to_2d(result.ch_pos_1005)
    source = result.source_positions
    source_2d = _to_2d(source) if source.shape[1] == 3 else source
    radius = float(np.nanmax(np.linalg.norm(ref_2d, axis=1)))

    figure, axis = plt.subplots(figsize=(6.2, 6.5), constrained_layout=True)
    _draw_head(axis, radius)
    axis.scatter(ref_2d[:, 0], ref_2d[:, 1], s=8, color="#d0d5d2", zorder=1)
    _format_head(axis, radius)

    finite_assigned = result.valid & np.isfinite(assigned).all(axis=1)
    unique_slots = np.unique(result.slot_ids[finite_assigned])
    unique_points = ref_2d[unique_slots]
    physical_indices: list[int] = []
    seen_physical: set[str] = set()
    for index, name in enumerate(result.names):
        physical_name = _physical_name(name)
        if physical_name not in seen_physical:
            seen_physical.add(physical_name)
            physical_indices.append(index)

    for index in physical_indices:
        if not finite_assigned[index] or not np.isfinite(source_2d[index]).all():
            continue
        slot_point = ref_2d[int(result.slot_ids[index])]
        axis.plot(
            [source_2d[index, 0], slot_point[0]],
            [source_2d[index, 1], slot_point[1]],
            color="#202624",
            alpha=0.38,
            linewidth=0.65,
            zorder=2,
        )

    finite_physical = [
        index for index in physical_indices if np.isfinite(source_2d[index]).all()
    ]
    axis.scatter(
        source_2d[finite_physical, 0],
        source_2d[finite_physical, 1],
        s=40,
        color="#f2b51d",
        edgecolors="#795b00",
        linewidths=0.5,
        zorder=4,
        label="S-D pair midpoint",
    )
    label_size = 4.0 if len(unique_slots) > 100 else 5.5
    for slot_id, point in zip(unique_slots, unique_points):
        axis.annotate(
            str(reference["labels"][int(slot_id)]),
            point,
            xytext=(0, -4),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=label_size,
            color="#27332f",
            zorder=5,
        )
    for slot_id in result.collisions:
        point = ref_2d[int(slot_id)]
        axis.scatter(
            point[0],
            point[1],
            s=115,
            facecolors="none",
            edgecolors="#d32f2f",
            linewidths=1.8,
            zorder=6,
        )

    sd_pairs = len({_physical_name(name) for name in result.names})
    summary = (
        f"{result.method}  |  {sd_pairs} S-D pairs  →  "
        f"{len(unique_slots)} slots  |  collisions: {len(result.collisions)}"
    )
    figure.suptitle(f"{title}\n{summary}" if title else summary, fontsize=12, fontweight="bold")
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.055), ncol=2, frameon=False, fontsize=9)
    figure.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return path


def _physical_name(name: str) -> str:
    parts = str(name).split()
    if len(parts) > 1:
        suffix = parts[-1].lower()
        if suffix in {"hbo", "hbr", "hbt"} or (suffix.isdigit() and len(suffix) >= 3):
            return " ".join(parts[:-1])
    return str(name)
