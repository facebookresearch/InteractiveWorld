"""Offline readability measurement for generated label artwork."""

from __future__ import annotations

from typing import Any


def ink_metrics(png: str) -> dict[str, Any]:
    """Measure glyph height, width, and line count as fractions of a PNG canvas."""
    import numpy as np
    from PIL import Image

    image = np.asarray(Image.open(png).convert("L"), dtype=np.int16)
    height, width = image.shape
    border = max(1, width // 40)
    dark = image[:, border : width - border] < 96
    rows = np.nonzero(dark.any(axis=1))[0]
    if rows.size == 0:
        return {"cap_frac": 0.0, "wide_frac": 0.0, "n_lines": 0}

    runs: list[tuple[int, int]] = []
    start = int(rows[0])
    for previous, current in zip(rows[:-1], rows[1:]):
        if current > previous + 1:
            runs.append((start, int(previous)))
            start = int(current)
    runs.append((start, int(rows[-1])))
    lines = [
        (first, last)
        for first, last in runs
        if (last - first + 1) < 0.30 * height and dark[first : last + 1].mean() < 0.85
    ] or runs
    cap_heights = sorted(last - first + 1 for first, last in lines)
    cap_height = cap_heights[len(cap_heights) // 2]
    widest = max(
        int(np.nonzero(dark[first : last + 1].any(axis=0))[0].ptp() + 1)
        for first, last in lines
    )
    return {
        "cap_frac": cap_height / float(height),
        "wide_frac": widest / float(width),
        "n_lines": len(lines),
    }
