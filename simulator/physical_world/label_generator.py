#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
"""Generate deterministic PNG labels into the TaskPack runtime cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from simulator.taskpack_support.bundled_assets import resolve_label_asset

FONT_BOLD = os.environ.get(
    "INTERACTIVEWORLD_FONT_BOLD",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
)
FONT_REG = os.environ.get(
    "INTERACTIVEWORLD_FONT_REG",
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
)

INK = (24, 24, 24, 255)
PAPER = (250, 250, 248, 255)
WHITE = (255, 255, 255, 255)
FRAME = (86, 92, 102, 255)
GREY = (132, 138, 146, 255)
RED = (198, 30, 30, 255)
AMBER = (238, 148, 18, 255)
GREEN = (36, 132, 66, 255)
BLUE = (38, 92, 168, 255)
FOAM = (34, 38, 44, 255)

_NAMED: dict[str, tuple[int, int, int, int]] = {
    "ink": INK,
    "paper": PAPER,
    "white": WHITE,
    "frame": FRAME,
    "grey": GREY,
    "gray": GREY,
    "red": RED,
    "amber": AMBER,
    "orange": AMBER,
    "green": GREEN,
    "blue": BLUE,
    "foam": FOAM,
    "black": (0, 0, 0, 255),
    "yellow": (250, 204, 20, 255),
}

Box = tuple[float, float, float, float]
Rgba = tuple[int, int, int, int]


@dataclass(frozen=True)
class Canvas:
    """Pixel canvas plus the safe box that artwork must stay inside."""

    w: int
    h: int
    mx: int
    my: int

    @property
    def box(self) -> Box:
        return (
            float(self.mx),
            float(self.my),
            float(self.w - self.mx),
            float(self.h - self.my),
        )


_DEFAULT_MOUNT: dict[str, str] = {
    "capacity_label": "graspable",
    "checklist": "wall",
    "handwritten_note": "graspable",
    "instruction_sheet": "wall",
    "mold_patch": "graspable",
    "order_ticket": "wall",
    "pictogram": "graspable",
    "price_tag": "graspable",
    "recipe_card": "wall",
    "service_card": "wall",
    "waste_stream_label": "container",
    "short_string": "graspable",
    "number": "container",
    "date": "graspable",
    "range_card": "wall",
    "placard": "container",
    "silhouette": "wall",
    "nutrition_report": "graspable",
    "document_scan": "graspable",
    # manual_PANEL, not manual_board: a 6x3 permutation table measurably reads
    # BETTER on the narrow 0.58 m portrait quad than on the 1.19 m landscape one.
    # Same physical letter height either way, but the wide panel spreads the
    # three data columns 0.40 m apart against a 0.07 m row pitch, and readers
    # then lose the row. See the design record section 3.
    "protocol_sheet": "manual_panel",
}

_KIND_ALIAS: dict[str, str] = {
    "capacity": "capacity_label",
    "check_list": "checklist",
    "handwritten": "handwritten_note",
    "note": "handwritten_note",
    "instructions": "instruction_sheet",
    "manual_sheet": "instruction_sheet",
    "packing_slip": "order_ticket",
    "price": "price_tag",
    "recipe": "recipe_card",
    "service_notice": "service_card",
    "waste_label": "waste_stream_label",
    "short string": "short_string",
    "string": "short_string",
    "text": "short_string",
    "numeral": "number",
    "big_numeral": "number",
    "date_code": "date",
    "numeric_range_card": "range_card",
    "card": "range_card",
    "sign": "placard",
    "icon": "pictogram",
    "shadow": "silhouette",
    "protocol": "protocol_sheet",
    "protocol_table": "protocol_sheet",
    "plan_sheet": "protocol_sheet",
    "posted_plans": "protocol_sheet",
    "manual": "protocol_sheet",
    "daily_nutrition_report": "nutrition_report",
    "diet_report": "nutrition_report",
    "scan": "document_scan",
    "scanned_document": "document_scan",
}


def _mount_canvas(mount: str) -> Canvas:
    """Canvas geometry per mount site. ``graspable`` keeps a wide side margin
    because the decal is wrapped around a 110 deg cylinder section.

    ``manual_board`` / ``manual_panel`` match the free-standing rule board that
    ``task_world_builder.mount_manual`` builds: a 1.32 x 0.92 m body whose artwork
    quads are ``(1.32 - 0.10)/n_panels - 0.03`` wide by ``0.92 - 0.06`` high.
    One panel is 1.19 x 0.86 m (1.384:1), two panels are 0.58 x 0.86 m each
    (0.674:1). Authoring at the WRONG aspect squashes the type horizontally --
    a 900x600 ``wall`` card on a 0.674:1 quad loses ~half its letter width.
    """
    table = {
        "graspable": Canvas(620, 760, 106, 48),
        "container": Canvas(900, 320, 40, 28),
        "container_interior": Canvas(900, 320, 40, 28),
        "wall": Canvas(900, 600, 46, 40),
        "floor": Canvas(700, 700, 56, 56),
        "manual_board": Canvas(1660, 1200, 40, 30),
        "manual_panel": Canvas(810, 1201, 26, 30),
    }
    return table.get(mount.strip().lower(), table["container"])


def _kind_mount_override(kind: str, mount: str) -> Canvas | None:
    """Row-bearing / icon kinds need a squarer canvas than a 2.8:1 door strip."""
    table = {
        ("capacity_label", "graspable"): Canvas(620, 760, 106, 48),
        ("price_tag", "graspable"): Canvas(760, 520, 92, 40),
        ("range_card", "container"): Canvas(820, 520, 44, 34),
        ("range_card", "container_interior"): Canvas(820, 520, 44, 34),
        ("silhouette", "container"): Canvas(820, 520, 44, 34),
        ("pictogram", "container"): Canvas(760, 520, 40, 32),
        ("pictogram", "wall"): Canvas(760, 620, 44, 40),
        ("silhouette", "graspable"): Canvas(620, 620, 60, 50),
        ("nutrition_report", "graspable"): Canvas(840, 1120, 36, 36),
    }
    for document_kind in (
        "checklist",
        "handwritten_note",
        "instruction_sheet",
        "order_ticket",
        "recipe_card",
        "service_card",
        "waste_stream_label",
    ):
        table[(document_kind, "graspable")] = Canvas(700, 900, 48, 44)
        table[(document_kind, "container")] = Canvas(820, 620, 44, 36)
        table[(document_kind, "container_interior")] = Canvas(820, 620, 44, 36)
    return table.get((kind, mount.strip().lower()))


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=256)
def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """The DejaVu face, searched rather than assumed.

    FONT_BOLD/FONT_REG are a SYSTEM path that exists on the devvm and on no MAST node. Hardcoding it
    turned every spec-driven build on the cluster into `OSError: cannot open resource` 0.3 s in --
    44 of 44 cells of one probe run, after the designer had produced 44 valid specs. A vendored copy
    now ships in the code tar beside this module; the system path stays first so a devvm run is
    byte-identical to before.
    """
    vendored = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    for path in (
        FONT_BOLD if bold else FONT_REG,
        os.path.join(vendored, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    ):
        try:
            return ImageFont.truetype(path, max(6, int(size)))
        except OSError:
            continue
    # A bitmap default renders an unreadable placard, so this is a LOUD last resort: the cue is the
    # thing the model has to read, and a silently unreadable one scores 0 while looking fine.
    print(
        f"[label_generator] WARNING no DejaVu face found (tried the system path and {vendored}); "
        f"falling back to PIL's bitmap default -- placards will not be legible.",
        flush=True,
    )
    return ImageFont.load_default()


def _text_wh(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    left, top, right, bottom = font.getbbox(text)
    return right - left, bottom - top


def _colour(value: object, default: Rgba) -> Rgba:
    if value is None or value == "":
        return default
    if isinstance(value, (tuple, list)) and len(value) in (3, 4):
        vals = [int(v) for v in value]
        return (vals[0], vals[1], vals[2], vals[3] if len(vals) == 4 else 255)
    text = str(value).strip().lower()
    if text in _NAMED:
        return _NAMED[text]
    if re.fullmatch(r"#?[0-9a-f]{6}", text):
        hexs = text.lstrip("#")
        return (int(hexs[0:2], 16), int(hexs[2:4], 16), int(hexs[4:6], 16), 255)
    raise ValueError(f"unrecognised colour {value!r}")


def _s(fields: dict[str, object], key: str, default: str = "") -> str:
    value = fields.get(key)
    return default if value is None else str(value)


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if not current or _text_wh(trial, font)[0] <= max_w:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _fit(
    text: str,
    box_w: float,
    box_h: float,
    max_size: int,
    min_size: int = 15,
    max_lines: int = 3,
    gap: float = 0.22,
) -> tuple[list[str], ImageFont.FreeTypeFont]:
    """Largest font at which ``text`` wraps into <= ``max_lines`` inside the box."""
    for size in range(int(max_size), int(min_size) - 1, -2):
        font = _font(size)
        lines = _wrap(text, font, box_w)
        if len(lines) > max_lines:
            continue
        line_h = max(_text_wh(ln or "X", font)[1] for ln in lines)
        total = len(lines) * line_h + (len(lines) - 1) * gap * line_h
        if total <= box_h and max(_text_wh(ln, font)[0] for ln in lines) <= box_w:
            return lines, font
    font = _font(min_size)
    lines = _wrap(text, font, box_w)[:max_lines]
    return lines, font


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    font: ImageFont.FreeTypeFont,
    box: Box,
    fill: Rgba,
    gap: float = 0.22,
) -> float:
    """Centre a text block in ``box``; returns the block's bottom y."""
    placements, bottom = _line_layout(lines, font, box, gap)
    for line, origin, _ink_box in placements:
        draw.text(origin, line, font=font, fill=fill)
    return bottom


def _line_layout(
    lines: Sequence[str],
    font: ImageFont.FreeTypeFont,
    box: Box,
    gap: float = 0.22,
) -> tuple[list[tuple[str, tuple[float, float], Box]], float]:
    """Return draw origins and exact ink rectangles for a centred text block."""
    x0, y0, x1, y1 = box
    line_h = max(_text_wh(ln or "X", font)[1] for ln in lines)
    step = line_h * (1.0 + gap)
    total = len(lines) * line_h + (len(lines) - 1) * gap * line_h
    y = y0 + max(0.0, (y1 - y0 - total) * 0.5)
    placements = []
    for line in lines:
        left, top, right, bottom = font.getbbox(line)
        origin = (
            0.5 * (x0 + x1) - 0.5 * (right - left) - left,
            y - top,
        )
        placements.append(
            (
                line,
                origin,
                (
                    origin[0] + left,
                    origin[1] + top,
                    origin[0] + right,
                    origin[1] + bottom,
                ),
            )
        )
        y += step
    return placements, y - (step - line_h)


def _pt(box: Box, u: float, v: float) -> tuple[float, float]:
    x0, y0, x1, y1 = box
    return (x0 + u * (x1 - x0), y0 + v * (y1 - y0))


def _poly(
    draw: ImageDraw.ImageDraw,
    box: Box,
    uv: Iterable[tuple[float, float]],
    fill: Rgba | None = None,
    outline: Rgba | None = None,
    width: int = 1,
) -> None:
    draw.polygon(
        [_pt(box, u, v) for u, v in uv], fill=fill, outline=outline, width=width
    )


def _uline(
    draw: ImageDraw.ImageDraw,
    box: Box,
    uv: Sequence[tuple[float, float]],
    fill: Rgba,
    width_u: float = 0.06,
) -> None:
    span = min(box[2] - box[0], box[3] - box[1])
    draw.line(
        [_pt(box, u, v) for u, v in uv],
        fill=fill,
        width=max(2, int(width_u * span)),
        joint="curve",
    )


def _ellipse(draw: ImageDraw.ImageDraw, box: Box, uv: Box, **kw: object) -> None:
    x0, y0 = _pt(box, uv[0], uv[1])
    x1, y1 = _pt(box, uv[2], uv[3])
    draw.ellipse((x0, y0, x1, y1), **kw)  # pyre-ignore[6]


def _glyph(
    draw: ImageDraw.ImageDraw, box: Box, text: str, fill: Rgba, frac: float = 0.92
) -> None:
    """Fit a short glyph string into ``box`` using the bold face."""
    x0, y0, x1, y1 = box
    lines, font = _fit(
        text,
        (x1 - x0) * frac,
        (y1 - y0) * frac,
        int((y1 - y0)),
        min_size=8,
        max_lines=1,
    )
    _draw_lines(draw, lines, font, box, fill)


# --------------------------------------------------------------------------- #
# pictograms -- APPROXIMATIONS of the published symbols, drawn from primitives
# --------------------------------------------------------------------------- #
_FLAME_UV = [
    (0.50, 0.02),
    (0.60, 0.22),
    (0.70, 0.15),
    (0.72, 0.36),
    (0.84, 0.54),
    (0.80, 0.76),
    (0.64, 0.94),
    (0.36, 0.94),
    (0.20, 0.76),
    (0.16, 0.54),
    (0.28, 0.34),
    (0.34, 0.42),
    (0.40, 0.18),
]


def _icon_flame(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """GHS flame: a solid flame standing on a surface line."""
    body = (box[0], box[1], box[2], box[1] + 0.88 * (box[3] - box[1]))
    _poly(draw, body, _FLAME_UV, fill=ink)
    draw.rectangle((_pt(box, 0.04, 0.92), _pt(box, 0.96, 0.99)), fill=ink)


def _icon_flame_circle(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Oxidiser: flame over a circle (the flame keeps a square aspect)."""
    side = 0.58 * (box[3] - box[1])
    cx = 0.5 * (box[0] + box[2])
    _icon_flame(draw, (cx - 0.5 * side, box[1], cx + 0.5 * side, box[1] + side), ink)
    _ellipse(
        draw,
        box,
        (0.20, 0.64, 0.80, 1.00),
        outline=ink,
        width=max(3, int(0.055 * (box[2] - box[0]))),
    )


def _icon_bomb(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Exploding bomb: a ball under a burst of fragments."""
    _ellipse(draw, box, (0.24, 0.50, 0.74, 1.00), fill=ink)
    burst = []
    for i in range(14):
        rad = 0.34 if i % 2 == 0 else 0.14
        ang = math.radians(i * (360.0 / 14.0) - 90.0)
        burst.append((0.56 + rad * math.cos(ang), 0.32 + rad * 0.86 * math.sin(ang)))
    _poly(draw, box, burst, fill=ink)
    for u, v in ((0.10, 0.14), (0.94, 0.20), (0.86, 0.62)):
        _poly(draw, box, [(u, v), (u + 0.09, v + 0.03), (u + 0.03, v + 0.10)], fill=ink)


def _icon_corrosion(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Corrosion: two vessels pouring, eating into a slab and a hand."""
    _poly(draw, box, [(0.04, 0.78), (0.44, 0.78), (0.44, 0.92), (0.04, 0.92)], fill=ink)
    _poly(draw, box, [(0.17, 0.79), (0.31, 0.79), (0.24, 0.66)], fill=PAPER)
    _poly(draw, box, [(0.56, 0.80), (0.98, 0.80), (0.98, 0.94), (0.56, 0.94)], fill=ink)
    for u in (0.60, 0.71, 0.82):
        _poly(draw, box, [(u, 0.81), (u + 0.08, 0.81), (u + 0.05, 0.68)], fill=ink)
    _poly(draw, box, [(0.04, 0.18), (0.26, 0.04), (0.36, 0.22), (0.16, 0.36)], fill=ink)
    _poly(draw, box, [(0.96, 0.18), (0.74, 0.04), (0.64, 0.22), (0.84, 0.36)], fill=ink)
    _uline(draw, box, [(0.24, 0.34), (0.24, 0.62)], ink, 0.038)
    _uline(draw, box, [(0.76, 0.34), (0.76, 0.64)], ink, 0.038)


def _icon_gas(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Gas cylinder, laid at a slight tilt."""
    draw.rounded_rectangle(
        (_pt(box, 0.30, 0.16), _pt(box, 0.70, 0.94)),
        radius=int(0.16 * (box[2] - box[0])),
        fill=ink,
    )
    draw.rectangle((_pt(box, 0.44, 0.05), _pt(box, 0.56, 0.20)), fill=ink)
    for v in (0.36, 0.52, 0.68):
        _uline(draw, box, [(0.34, v), (0.66, v)], PAPER, 0.035)


def _icon_skull(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Skull and crossbones."""
    for a, b in (((0.10, 0.42), (0.90, 0.96)), ((0.90, 0.42), (0.10, 0.96))):
        _uline(draw, box, [a, b], ink, 0.09)
    _ellipse(draw, box, (0.22, 0.08, 0.78, 0.66), fill=ink)
    _poly(draw, box, [(0.34, 0.60), (0.66, 0.60), (0.62, 0.82), (0.38, 0.82)], fill=ink)
    _ellipse(draw, box, (0.31, 0.26, 0.45, 0.44), fill=PAPER)
    _ellipse(draw, box, (0.55, 0.26, 0.69, 0.44), fill=PAPER)
    _poly(draw, box, [(0.46, 0.46), (0.54, 0.46), (0.50, 0.58)], fill=PAPER)
    for u in (0.42, 0.50, 0.58):
        _uline(draw, box, [(u, 0.64), (u, 0.80)], PAPER, 0.022)


def _icon_health(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Health hazard: torso with a starburst on the chest."""
    _ellipse(draw, box, (0.38, 0.04, 0.62, 0.28), fill=ink)
    _poly(draw, box, [(0.20, 0.34), (0.80, 0.34), (0.86, 0.96), (0.14, 0.96)], fill=ink)
    star = []
    for i in range(16):
        rad = 0.26 if i % 2 == 0 else 0.11
        ang = math.radians(i * 22.5 - 90.0)
        star.append((0.50 + rad * math.cos(ang), 0.62 + rad * math.sin(ang)))
    _poly(draw, box, star, fill=PAPER)


def _icon_environment(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Environmental hazard: dead tree and a fish above water."""
    _uline(draw, box, [(0.22, 0.72), (0.22, 0.14)], ink, 0.05)
    _uline(draw, box, [(0.22, 0.34), (0.06, 0.18)], ink, 0.04)
    _uline(draw, box, [(0.22, 0.44), (0.40, 0.26)], ink, 0.04)
    _ellipse(draw, box, (0.44, 0.44, 0.84, 0.68), fill=ink)
    _poly(draw, box, [(0.84, 0.46), (0.98, 0.38), (0.98, 0.74), (0.84, 0.66)], fill=ink)
    _ellipse(draw, box, (0.52, 0.50, 0.58, 0.56), fill=PAPER)
    _uline(
        draw,
        box,
        [(0.04, 0.84), (0.24, 0.78), (0.46, 0.90), (0.70, 0.78), (0.96, 0.86)],
        ink,
        0.05,
    )


def _icon_biohazard(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Biohazard trefoil, approximated with three rings plus a hub."""
    field = _colour("yellow", AMBER)
    _ellipse(draw, box, (0.0, 0.0, 1.0, 1.0), fill=field)
    for k in range(3):
        ang = math.radians(-90.0 + k * 120.0)
        cu, cv = 0.5 + 0.26 * math.cos(ang), 0.5 + 0.26 * math.sin(ang)
        _ellipse(draw, box, (cu - 0.25, cv - 0.25, cu + 0.25, cv + 0.25), fill=ink)
        _ellipse(draw, box, (cu - 0.14, cv - 0.14, cu + 0.14, cv + 0.14), fill=field)
    _ellipse(draw, box, (0.32, 0.32, 0.68, 0.68), fill=field)
    _ellipse(draw, box, (0.36, 0.36, 0.64, 0.64), fill=ink)
    _ellipse(draw, box, (0.44, 0.44, 0.56, 0.56), fill=field)


def _icon_weee(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """WEEE crossed-out wheeled bin over the solid bar."""
    _poly(draw, box, [(0.26, 0.16), (0.74, 0.16), (0.80, 0.72), (0.20, 0.72)], fill=ink)
    draw.rectangle((_pt(box, 0.22, 0.06), _pt(box, 0.78, 0.15)), fill=ink)
    _ellipse(draw, box, (0.26, 0.72, 0.40, 0.86), fill=ink)
    _ellipse(draw, box, (0.60, 0.72, 0.74, 0.86), fill=ink)
    draw.rectangle((_pt(box, 0.10, 0.90), _pt(box, 0.90, 0.99)), fill=ink)
    _uline(draw, box, [(0.06, 0.06), (0.94, 0.80)], ink, 0.075)
    _uline(draw, box, [(0.94, 0.06), (0.06, 0.80)], ink, 0.075)


def _icon_recycle(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = GREEN) -> None:
    """Mobius loop, approximated as three chevrons on a triangle."""
    for k in range(3):
        ang = math.radians(-90.0 + k * 120.0)
        nxt = math.radians(-90.0 + (k + 1) * 120.0)
        a = (0.5 + 0.34 * math.cos(ang), 0.5 + 0.34 * math.sin(ang))
        b = (0.5 + 0.34 * math.cos(nxt), 0.5 + 0.34 * math.sin(nxt))
        mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
        _uline(draw, box, [a, mid], ink, 0.10)
        head = [
            (b[0], b[1]),
            (mid[0] + 0.12 * (b[1] - a[1]), mid[1] - 0.12 * (b[0] - a[0])),
            (mid[0] - 0.12 * (b[1] - a[1]), mid[1] + 0.12 * (b[0] - a[0])),
        ]
        _poly(draw, box, head, fill=ink)


def _care_tub(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba) -> None:
    _poly(
        draw,
        box,
        [(0.08, 0.34), (0.92, 0.34), (0.80, 0.86), (0.20, 0.86)],
        outline=ink,
        width=max(3, int(0.05 * (box[2] - box[0]))),
    )
    _uline(
        draw,
        box,
        [(0.14, 0.34), (0.30, 0.20), (0.50, 0.30), (0.70, 0.20), (0.88, 0.32)],
        ink,
        0.035,
    )


def _icon_care_wash(
    draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK, temp: str = "40"
) -> None:
    """ISO 3758 wash tub with a temperature numeral."""
    _care_tub(draw, box, ink)
    _glyph(
        draw,
        (
            _pt(box, 0.26, 0.42)[0],
            _pt(box, 0.26, 0.42)[1],
            _pt(box, 0.74, 0.82)[0],
            _pt(box, 0.74, 0.82)[1],
        ),
        temp,
        ink,
    )


def _icon_care_hand(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Hand-wash tub."""
    _care_tub(draw, box, ink)
    _poly(
        draw,
        box,
        [
            (0.34, 0.80),
            (0.34, 0.58),
            (0.40, 0.46),
            (0.46, 0.58),
            (0.46, 0.44),
            (0.52, 0.42),
            (0.54, 0.58),
            (0.60, 0.46),
            (0.66, 0.56),
            (0.66, 0.80),
        ],
        fill=ink,
    )


def _icon_care_bleach(
    draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK, crossed: bool = True
) -> None:
    """ISO 3758 bleach triangle (crossed = do not bleach)."""
    _poly(
        draw,
        box,
        [(0.50, 0.10), (0.92, 0.86), (0.08, 0.86)],
        outline=ink,
        width=max(3, int(0.05 * (box[2] - box[0]))),
    )
    if crossed:
        _uline(draw, box, [(0.22, 0.26), (0.78, 0.82)], ink, 0.055)
        _uline(draw, box, [(0.78, 0.26), (0.22, 0.82)], ink, 0.055)


def _icon_care_iron(
    draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK, dots: int = 1
) -> None:
    """ISO 3758 iron with heat dots."""
    _poly(
        draw,
        box,
        [(0.10, 0.74), (0.90, 0.74), (0.78, 0.44), (0.36, 0.28), (0.16, 0.36)],
        outline=ink,
        width=max(3, int(0.05 * (box[2] - box[0]))),
    )
    draw.rectangle((_pt(box, 0.06, 0.78), _pt(box, 0.94, 0.86)), fill=ink)
    for i in range(max(1, dots)):
        cu = 0.50 + (i - (dots - 1) * 0.5) * 0.16
        _ellipse(draw, box, (cu - 0.05, 0.52, cu + 0.05, 0.62), fill=ink)


def _icon_care_dryclean(
    draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK, letter: str = "P"
) -> None:
    """ISO 3758 professional-care circle."""
    _ellipse(
        draw,
        box,
        (0.08, 0.08, 0.92, 0.92),
        outline=ink,
        width=max(3, int(0.05 * (box[2] - box[0]))),
    )
    _glyph(
        draw,
        (
            _pt(box, 0.26, 0.24)[0],
            _pt(box, 0.26, 0.24)[1],
            _pt(box, 0.74, 0.76)[0],
            _pt(box, 0.74, 0.76)[1],
        ),
        letter,
        ink,
    )


def _icon_care_tumble(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """ISO 3758 tumble-dry square."""
    width = max(3, int(0.05 * (box[2] - box[0])))
    draw.rectangle(
        (_pt(box, 0.08, 0.08), _pt(box, 0.92, 0.92)), outline=ink, width=width
    )
    _ellipse(draw, box, (0.22, 0.22, 0.78, 0.78), outline=ink, width=width)
    _ellipse(draw, box, (0.46, 0.44, 0.54, 0.52), fill=ink)


def _icon_nuts(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Allergen: nut (a peanut shell)."""
    _ellipse(draw, box, (0.14, 0.14, 0.62, 0.62), fill=ink)
    _ellipse(draw, box, (0.40, 0.40, 0.88, 0.88), fill=ink)
    _ellipse(draw, box, (0.26, 0.26, 0.44, 0.44), fill=PAPER)
    _ellipse(draw, box, (0.60, 0.60, 0.76, 0.76), fill=PAPER)


def _grain(
    draw: ImageDraw.ImageDraw, box: Box, cu: float, cv: float, ang: float, ink: Rgba
) -> None:
    """One wheat grain: an ellipse rotated by ``ang`` degrees, as a polygon."""
    rad, pts = math.radians(ang), []
    for i in range(16):
        t = 2.0 * math.pi * i / 16.0
        x, y = 0.17 * math.cos(t), 0.062 * math.sin(t)
        pts.append(
            (
                cu + x * math.cos(rad) - y * math.sin(rad),
                cv + x * math.sin(rad) + y * math.cos(rad),
            )
        )
    _poly(draw, box, pts, fill=ink)


def _icon_gluten(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Allergen: wheat ear."""
    _uline(draw, box, [(0.50, 0.99), (0.50, 0.30)], ink, 0.04)
    for i in range(4):
        cv = 0.34 + i * 0.17
        _grain(draw, box, 0.34, cv, 48.0, ink)
        _grain(draw, box, 0.66, cv, -48.0, ink)
    _grain(draw, box, 0.50, 0.20, 90.0, ink)


def _icon_milk(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Allergen: milk carton."""
    _poly(
        draw,
        box,
        [(0.24, 0.30), (0.50, 0.08), (0.76, 0.30), (0.76, 0.94), (0.24, 0.94)],
        fill=ink,
    )
    draw.rectangle((_pt(box, 0.32, 0.50), _pt(box, 0.68, 0.74)), fill=PAPER)


def _icon_egg(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Allergen: egg."""
    _ellipse(draw, box, (0.22, 0.10, 0.78, 0.94), fill=ink)
    _ellipse(draw, box, (0.40, 0.44, 0.62, 0.68), fill=PAPER)


def _icon_soy(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Allergen: soy pod."""
    _ellipse(draw, box, (0.10, 0.36, 0.90, 0.68), fill=ink)
    for cu in (0.28, 0.50, 0.72):
        _ellipse(draw, box, (cu - 0.09, 0.42, cu + 0.09, 0.62), fill=PAPER)


# --------------------------------------------------------------------------- #
# battery chemistry marks -- separated by GROSS SHAPE, not by fidelity
# --------------------------------------------------------------------------- #
# The item decal is ~36 x 54 px on screen at the delivered 512, of which the icon
# square gets ~33 x 33. At that size only the silhouette and the mean darkness
# survive; interior detail does not. So these five are drawn as five different
# SHAPE FAMILIES -- disc / wide block / tall solid bar / tall hollow bar / ringed
# -- because an icon set that reads as "five battery outlines" is worth no more
# than five question marks. Alkaline vs NiMH is the hard pair: they are the same
# physical object, so they are split on the one real distinction a reader can
# name, primary (solid) vs rechargeable (hollow, carrying the loop mark).
def _icon_battery_coin(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Coin cell (CR2032): a disc with a ``+`` face. Shape family = ROUND."""
    _ellipse(draw, box, (0.04, 0.04, 0.96, 0.96), fill=ink)
    _ellipse(
        draw,
        box,
        (0.15, 0.15, 0.85, 0.85),
        outline=PAPER,
        width=max(2, int(0.035 * (box[2] - box[0]))),
    )
    draw.rectangle((_pt(box, 0.32, 0.45), _pt(box, 0.68, 0.55)), fill=PAPER)
    draw.rectangle((_pt(box, 0.45, 0.32), _pt(box, 0.55, 0.68)), fill=PAPER)


def _icon_battery_pouch(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Li-ion pouch/prismatic cell, laid flat. Shape family = WIDE BLOCK."""
    draw.rounded_rectangle(
        (_pt(box, 0.03, 0.28), _pt(box, 0.84, 0.80)),
        radius=int(0.07 * (box[2] - box[0])),
        fill=ink,
    )
    draw.rectangle((_pt(box, 0.84, 0.40), _pt(box, 0.97, 0.52)), fill=ink)
    draw.rectangle((_pt(box, 0.84, 0.58), _pt(box, 0.97, 0.70)), fill=ink)
    for v in (0.42, 0.66):
        draw.rectangle((_pt(box, 0.12, v), _pt(box, 0.72, v + 0.05)), fill=PAPER)


def _cell_body(box: Box) -> tuple[Box, Box]:
    """The cylindrical-cell silhouette: (body, button-top) in absolute coords."""
    body = (
        _pt(box, 0.26, 0.14)[0],
        _pt(box, 0.26, 0.14)[1],
        _pt(box, 0.74, 0.98)[0],
        _pt(box, 0.74, 0.98)[1],
    )
    top = (
        _pt(box, 0.41, 0.02)[0],
        _pt(box, 0.41, 0.02)[1],
        _pt(box, 0.59, 0.16)[0],
        _pt(box, 0.59, 0.16)[1],
    )
    return body, top


def _icon_battery_cell(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Alkaline primary cell: SOLID tall cylinder. Shape family = TALL SOLID BAR."""
    body, top = _cell_body(box)
    draw.rectangle(top, fill=ink)
    draw.rounded_rectangle(body, radius=int(0.06 * (box[2] - box[0])), fill=ink)
    draw.rectangle((_pt(box, 0.26, 0.30), _pt(box, 0.74, 0.41)), fill=PAPER)


def _icon_battery_rechargeable(
    draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK
) -> None:
    """NiMH rechargeable: HOLLOW tall cylinder carrying the loop mark.

    Same silhouette as the alkaline cell on purpose -- they are the same object
    in the world -- but inverted in fill, so the two differ in mean darkness by
    far more than any interior detail could at 33 px.
    """
    body, top = _cell_body(box)
    w = max(3, int(0.055 * (box[2] - box[0])))
    draw.rectangle(top, fill=ink)
    draw.rounded_rectangle(
        body, radius=int(0.06 * (box[2] - box[0])), fill=PAPER, outline=ink, width=w
    )
    inner = (
        _pt(box, 0.30, 0.42)[0],
        _pt(box, 0.30, 0.42)[1],
        _pt(box, 0.70, 0.86)[0],
        _pt(box, 0.70, 0.86)[1],
    )
    _icon_recycle(draw, inner, ink)


def _icon_battery_none(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = RED) -> None:
    """General waste: a cell struck through. Shape family = RINGED (and RED)."""
    inner = (
        _pt(box, 0.30, 0.22)[0],
        _pt(box, 0.30, 0.22)[1],
        _pt(box, 0.70, 0.78)[0],
        _pt(box, 0.70, 0.78)[1],
    )
    _icon_battery_cell(draw, inner, INK)
    _icon_no_symbol(draw, box, ink)


def _icon_call_number(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Library call number: a book spine carrying a shelf tag."""
    w = max(3, int(0.05 * (box[2] - box[0])))
    draw.rectangle(
        (_pt(box, 0.30, 0.02), _pt(box, 0.70, 0.98)), outline=ink, width=w, fill=PAPER
    )
    draw.rectangle((_pt(box, 0.30, 0.02), _pt(box, 0.70, 0.16)), fill=ink)
    draw.rectangle((_pt(box, 0.36, 0.60), _pt(box, 0.64, 0.92)), fill=ink)
    for v in (0.68, 0.78):
        draw.rectangle((_pt(box, 0.41, v), _pt(box, 0.59, v + 0.05)), fill=PAPER)


def _icon_no_symbol(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = RED) -> None:
    """Prohibition ring (allergen-free / restricted articles)."""
    width = max(4, int(0.10 * (box[2] - box[0])))
    _ellipse(draw, box, (0.06, 0.06, 0.94, 0.94), outline=ink, width=width)
    _uline(draw, box, [(0.20, 0.20), (0.80, 0.80)], ink, 0.10)


def _icon_restricted(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = RED) -> None:
    """Restricted articles: prohibition ring over a hazard triangle."""
    inner = (
        _pt(box, 0.18, 0.18)[0],
        _pt(box, 0.18, 0.18)[1],
        _pt(box, 0.82, 0.82)[0],
        _pt(box, 0.82, 0.82)[1],
    )
    _poly(draw, inner, [(0.50, 0.06), (0.96, 0.90), (0.04, 0.90)], fill=INK)
    _glyph(
        draw,
        (
            _pt(inner, 0.36, 0.34)[0],
            _pt(inner, 0.36, 0.34)[1],
            _pt(inner, 0.64, 0.86)[0],
            _pt(inner, 0.64, 0.86)[1],
        ),
        "!",
        PAPER,
    )
    _icon_no_symbol(draw, box, ink)


def _icon_exclamation(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    _glyph(draw, box, "!", ink)


def _icon_fallback(draw: ImageDraw.ImageDraw, box: Box, ink: Rgba = INK) -> None:
    """Unknown icon key: a bold marker so the miss is visible, never a crash."""
    width = max(4, int(0.06 * (box[2] - box[0])))
    draw.rounded_rectangle(
        (_pt(box, 0.06, 0.18), _pt(box, 0.94, 0.82)),
        radius=int(0.10 * (box[2] - box[0])),
        outline=ink,
        width=width,
    )
    _glyph(
        draw,
        (
            _pt(box, 0.14, 0.28)[0],
            _pt(box, 0.14, 0.28)[1],
            _pt(box, 0.86, 0.72)[0],
            _pt(box, 0.86, 0.72)[1],
        ),
        "?",
        ink,
    )


def _plain_icons() -> dict[str, Callable[[ImageDraw.ImageDraw, Box], None]]:
    """Icons drawn bare (no GHS diamond)."""
    return {
        "biohazard": _icon_biohazard,
        "weee": _icon_weee,
        "recycle": _icon_recycle,
        "flame": _icon_flame,
        "care_wash": _icon_care_wash,
        "care_wash30": lambda d, b: _icon_care_wash(d, b, INK, "30"),
        "care_wash40": lambda d, b: _icon_care_wash(d, b, INK, "40"),
        "care_wash60": lambda d, b: _icon_care_wash(d, b, INK, "60"),
        "care_hand": _icon_care_hand,
        "care_bleach": _icon_care_bleach,
        "care_bleach_ok": lambda d, b: _icon_care_bleach(d, b, INK, False),
        "care_iron": _icon_care_iron,
        "care_iron2": lambda d, b: _icon_care_iron(d, b, INK, 2),
        "care_dryclean": _icon_care_dryclean,
        "care_dryclean_f": lambda d, b: _icon_care_dryclean(d, b, INK, "F"),
        "care_tumble": _icon_care_tumble,
        "allergen_nuts": _icon_nuts,
        "allergen_gluten": _icon_gluten,
        "allergen_milk": _icon_milk,
        "allergen_egg": _icon_egg,
        "allergen_soy": _icon_soy,
        "allergen_none": _icon_no_symbol,
        "no_symbol": _icon_no_symbol,
        "restricted": _icon_restricted,
        "battery_coin": _icon_battery_coin,
        "battery_liion": _icon_battery_pouch,
        "battery_alkaline": _icon_battery_cell,
        "battery_nimh": _icon_battery_rechargeable,
        "no_batteries": _icon_battery_none,
        "call_number": _icon_call_number,
        "fallback": _icon_fallback,
    }


def _ghs_icons() -> dict[str, Callable[[ImageDraw.ImageDraw, Box], None]]:
    """Symbols that render inside the red GHS diamond."""
    return {
        "ghs_flame": _icon_flame,
        "ghs_flame_circle": _icon_flame_circle,
        "ghs_bomb": _icon_bomb,
        "ghs_corrosion": _icon_corrosion,
        "ghs_gas": _icon_gas,
        "ghs_skull": _icon_skull,
        "ghs_exclamation": _icon_exclamation,
        "ghs_health": _icon_health,
        "ghs_environment": _icon_environment,
    }


_GHS_ORDER = [
    "ghs_flame",
    "ghs_flame_circle",
    "ghs_bomb",
    "ghs_corrosion",
    "ghs_gas",
    "ghs_skull",
    "ghs_exclamation",
    "ghs_health",
    "ghs_environment",
]
_CARE_ORDER = [
    "care_wash40",
    "care_bleach",
    "care_iron",
    "care_dryclean",
    "care_tumble",
]
_ALLERGEN_ORDER = [
    "allergen_nuts",
    "allergen_gluten",
    "allergen_milk",
    "allergen_soy",
    "allergen_egg",
    "allergen_none",
]
_BIN_ORDER = ["biohazard", "recycle", "weee", "no_symbol"]


def _diamond(draw: ImageDraw.ImageDraw, box: Box, accent: Rgba) -> Box:
    """Draw the GHS diamond; return the inner square available for the symbol."""
    x0, y0, x1, y1 = box
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    rad = 0.5 * min(x1 - x0, y1 - y0) - 4.0
    pts = [(cx, cy - rad), (cx + rad, cy), (cx, cy + rad), (cx - rad, cy)]
    draw.polygon(pts, fill=WHITE)
    draw.line([*pts, pts[0]], fill=accent, width=max(5, int(rad * 0.11)), joint="curve")
    half = rad * 0.50
    return (cx - half, cy - half * 0.95, cx + half, cy + half * 1.00)


class UnknownIconError(KeyError):
    """An icon key that no table defines. See ``_draw_icon``."""


def _draw_icon(draw: ImageDraw.ImageDraw, box: Box, icon: str, accent: Rgba) -> None:
    """Draw ``icon`` into ``box``, or refuse.

    An unrecognised key used to print a WARN and draw ``_icon_fallback`` -- a ``?`` occupying the
    whole icon square. ``label_generator`` skips a file that already exists, so that warning fired once
    into a build log and the ``?`` then sat in the baked PNG forever. On a row whose decal IS the
    discriminator (``T1-OFFC-01``'s five ``battery_*`` keys, ``T3-HDFU-01``'s ``call_number``) the
    result is an unanswerable episode that renders, scores and looks fine. A missing cue must fail
    at build time, not at scoring time. ``"fallback"`` is still a legal key for a caller that wants
    the marker on purpose.
    """
    key = icon.strip().lower()
    ghs = _ghs_icons()
    if key in ghs:
        ghs[key](draw, _diamond(draw, box, accent))
        return
    plain = _plain_icons()
    if key not in plain:
        raise UnknownIconError(
            f"unknown icon {icon!r}: no such key in _plain_icons() or _ghs_icons(). "
            f"Author the pictogram, or pass one of {sorted(set(plain) | set(ghs))}. "
            f"Pass icon='fallback' only if a '?' marker is genuinely what the label should say."
        )
    plain[key](draw, box)


# --------------------------------------------------------------------------- #
# panels + per-kind renderers
# --------------------------------------------------------------------------- #
def _panel(draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]) -> None:
    fill = _colour(fields.get("bg"), PAPER)
    border = _colour(fields.get("border"), INK)
    bw = int(fields.get("border_w") or max(6, min(cv.w, cv.h) // 46))  # pyre-ignore[6]
    draw.rectangle((0, 0, cv.w - 1, cv.h - 1), fill=border)
    draw.rectangle((bw, bw, cv.w - 1 - bw, cv.h - 1 - bw), fill=fill)


def _title_bar(
    draw: ImageDraw.ImageDraw, cv: Canvas, title: str, accent: Rgba
) -> float:
    """Header band across the top of the sign; returns the y below it."""
    bar_h = max(46.0, 0.20 * cv.h)
    draw.rectangle((0, 0, cv.w, bar_h), fill=accent)
    lines, font = _fit(
        title.upper(), cv.w - 2.0 * cv.mx, bar_h * 0.70, int(bar_h * 0.72), max_lines=1
    )
    _draw_lines(draw, lines, font, (0.0, 0.0, float(cv.w), bar_h), WHITE)
    return bar_h + cv.my * 0.6


def _render_pictogram(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    _panel(draw, cv, fields)
    x0, y0, x1, y1 = cv.box
    caption, sub = _s(fields, "text"), _s(fields, "sub")
    accent = _colour(fields.get("accent"), RED)
    cap_h = 0.20 * (y1 - y0) if caption else 0.0
    sub_h = 0.13 * (y1 - y0) if sub else 0.0
    side = min(x1 - x0, (y1 - y0) - cap_h - sub_h)
    cx = 0.5 * (x0 + x1)
    icon_box = (cx - side * 0.5, y0, cx + side * 0.5, y0 + side)
    _draw_icon(draw, icon_box, _s(fields, "icon", "fallback"), accent)
    if caption:
        lines, font = _fit(
            caption.upper(), x1 - x0, cap_h * 0.92, int(cap_h * 0.62), max_lines=2
        )
        _draw_lines(draw, lines, font, (x0, y0 + side, x1, y1 - sub_h), INK)
    if sub:
        lines, font = _fit(
            sub.upper(), x1 - x0, sub_h * 0.86, int(sub_h * 0.62), max_lines=1
        )
        _draw_lines(draw, lines, font, (x0, y1 - sub_h, x1, y1), accent)


def _short_string_primary_layout(
    cv: Canvas, fields: dict[str, object]
) -> tuple[list[str], ImageFont.FreeTypeFont, Box]:
    x0, y0, x1, y1 = cv.box
    title, sub = _s(fields, "title"), _s(fields, "sub")
    text = _s(fields, "text", "LABEL")
    if str(fields.get("upper", "1")).lower() not in ("0", "false", "no"):
        text = text.upper()
    top = max(46.0, 0.20 * cv.h) + cv.my * 0.6 if title else y0
    sub_h = 0.20 * (y1 - y0) if sub else 0.0
    lines, font = _fit(text, x1 - x0, (y1 - sub_h) - top, int(0.34 * cv.h), max_lines=3)
    return lines, font, (x0, top, x1, y1 - sub_h)


def _render_short_string(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    _panel(draw, cv, fields)
    x0, _y0, x1, y1 = cv.box
    title, sub = _s(fields, "title"), _s(fields, "sub")
    if title:
        _title_bar(draw, cv, title, _colour(fields.get("accent"), FRAME))
    lines, font, primary_box = _short_string_primary_layout(cv, fields)
    _draw_lines(draw, lines, font, primary_box, _colour(fields.get("fg"), INK))
    if sub:
        sub_h = 0.20 * (y1 - cv.box[1])
        lines, font = _fit(
            sub.upper(), x1 - x0, sub_h * 0.80, int(sub_h * 0.56), max_lines=2
        )
        _draw_lines(
            draw,
            lines,
            font,
            (x0, y1 - sub_h, x1, y1),
            _colour(fields.get("accent"), GREY),
        )


def _mold_noise(seed: int, index: int) -> float:
    value = (seed ^ (index * 0x45D9F3B)) & 0xFFFFFFFF
    value = ((value >> 16) ^ value) * 0x45D9F3B & 0xFFFFFFFF
    value = ((value >> 16) ^ value) * 0x45D9F3B & 0xFFFFFFFF
    return float((value >> 16) ^ value) / float(0xFFFFFFFF)


def _mold_value_noise(seed: int, x: float, y: float, cell: float) -> float:
    """Smooth deterministic value noise for an organic, non-vector boundary."""
    gx, gy = math.floor(x / cell), math.floor(y / cell)
    tx, ty = x / cell - gx, y / cell - gy
    tx, ty = tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty)

    def lattice(ix: int, iy: int) -> float:
        index = ((ix * 73856093) ^ (iy * 19349663)) & 0xFFFFFFFF
        return _mold_noise(seed, index)

    top = lattice(gx, gy) * (1.0 - tx) + lattice(gx + 1, gy) * tx
    bottom = lattice(gx, gy + 1) * (1.0 - tx) + lattice(gx + 1, gy + 1) * tx
    return top * (1.0 - ty) + bottom * ty


MoldColony = tuple[float, float, float, float]


def _mold_strength(
    x: float,
    y: float,
    coarse: float,
    medium: float,
    colonies: Sequence[MoldColony],
) -> float:
    strength = 0.0
    for cx, cy, rx, ry in colonies:
        distance = math.hypot((x - cx) / rx, (y - cy) / ry)
        distance += 0.22 * (coarse - 0.5) + 0.09 * (medium - 0.5)
        falloff = max(0.0, min(1.0, (1.04 - distance) / 0.52))
        falloff = falloff * falloff * (3.0 - 2.0 * falloff)
        strength = max(strength, falloff)
    return strength


def _mold_pixel(
    cv: Canvas, seed: int, px: int, py: int, colonies: Sequence[MoldColony]
) -> Rgba:
    x, y = (px + 0.5) / cv.w, (py + 0.5) / cv.h
    coarse = _mold_value_noise(seed + 101, px, py, 44.0)
    medium = _mold_value_noise(seed + 211, px, py, 14.0)
    fine = _mold_value_noise(seed + 307, px, py, 4.8)
    strength = _mold_strength(x, y, coarse, medium, colonies)
    if strength <= 0.025:
        return (0, 0, 0, 0)
    porosity = 0.48 + 0.34 * medium + 0.18 * fine
    alpha = round(145.0 * strength * porosity)
    if alpha < 8:
        return (0, 0, 0, 0)
    shade = max(0.0, min(1.0, 0.72 * strength + 0.28 * fine))
    grain = round(9.0 * (medium - 0.5))
    return (
        max(36, round(98.0 - 50.0 * shade) + grain),
        max(52, round(116.0 - 36.0 * shade) + grain),
        max(46, round(90.0 - 25.0 * shade) + grain),
        min(150, alpha),
    )


def _mold_base_image(
    cv: Canvas, seed: int, colonies: Sequence[MoldColony]
) -> Image.Image:
    pixels: list[Rgba] = [(0, 0, 0, 0)] * (cv.w * cv.h)
    edge = 1.24
    left = max(0, math.floor(min(cx - edge * rx for cx, _, rx, _ in colonies) * cv.w))
    right = min(
        cv.w, math.ceil(max(cx + edge * rx for cx, _, rx, _ in colonies) * cv.w)
    )
    top = max(0, math.floor(min(cy - edge * ry for _, cy, _, ry in colonies) * cv.h))
    bottom = min(
        cv.h, math.ceil(max(cy + edge * ry for _, cy, _, ry in colonies) * cv.h)
    )
    for py in range(top, bottom):
        offset = py * cv.w
        for px in range(left, right):
            pixels[offset + px] = _mold_pixel(cv, seed, px, py, colonies)
    patch = Image.new("RGBA", (cv.w, cv.h), (0, 0, 0, 0))
    patch.putdata(pixels)
    return patch


def _mold_spores_image(
    cv: Canvas, seed: int, colonies: Sequence[MoldColony]
) -> Image.Image:
    spores = Image.new("RGBA", (cv.w, cv.h), (0, 0, 0, 0))
    spore_draw = ImageDraw.Draw(spores)
    scale = min(cv.w, cv.h)
    palette: tuple[Rgba, ...] = (
        (40, 68, 52, 92),
        (58, 82, 62, 84),
        (77, 96, 72, 76),
        (100, 112, 86, 68),
    )
    for index in range(240):
        cx, cy, rx, ry = colonies[index % len(colonies)]
        angle = 2.0 * math.pi * _mold_noise(seed + 401, index)
        radius = _mold_noise(seed + 503, index) ** 0.58
        x = (cx + math.cos(angle) * rx * radius) * cv.w
        y = (cy + math.sin(angle) * ry * radius) * cv.h
        dot = scale * (0.00045 + 0.0025 * _mold_noise(seed + 601, index) ** 2)
        color = palette[
            int(_mold_noise(seed + 701, index) * len(palette)) % len(palette)
        ]
        spore_draw.ellipse((x - dot, y - dot, x + dot, y + dot), fill=color)
    spores = spores.filter(ImageFilter.GaussianBlur(radius=max(0.25, 0.0006 * scale)))
    return spores


def _mold_patch_image(cv: Canvas, fields: dict[str, object]) -> Image.Image:
    """Render a matte, translucent fungal bloom with no printable-sign features."""
    seed = int(fields.get("seed") or 87017)
    colonies: tuple[MoldColony, ...] = (
        (0.465, 0.490, 0.060, 0.052),
        (0.530, 0.520, 0.052, 0.044),
        (0.515, 0.444, 0.043, 0.037),
        (0.570, 0.472, 0.031, 0.028),
        (0.430, 0.540, 0.028, 0.024),
    )
    return Image.alpha_composite(
        _mold_base_image(cv, seed, colonies),
        _mold_spores_image(cv, seed, colonies),
    )


def _render_number(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    _panel(draw, cv, fields)
    x0, y0, x1, y1 = cv.box
    numeral = _s(fields, "number") or _s(fields, "text", "1")
    title, sub = _s(fields, "title"), _s(fields, "sub")
    top, bottom = y0, y1
    if title:
        cap_h = 0.20 * (y1 - y0)
        lines, font = _fit(
            title.upper(), x1 - x0, cap_h * 0.82, int(cap_h * 0.64), max_lines=1
        )
        _draw_lines(draw, lines, font, (x0, y0, x1, y0 + cap_h), GREY)
        top = y0 + cap_h
    if sub:
        sub_h = 0.16 * (y1 - y0)
        lines, font = _fit(
            sub.upper(), x1 - x0, sub_h * 0.82, int(sub_h * 0.60), max_lines=1
        )
        _draw_lines(draw, lines, font, (x0, y1 - sub_h, x1, y1), GREY)
        bottom = y1 - sub_h
    lines, font = _fit(
        numeral, x1 - x0, bottom - top, int(bottom - top), min_size=24, max_lines=1
    )
    _draw_lines(
        draw, lines, font, (x0, top, x1, bottom), _colour(fields.get("fg"), INK)
    )


def _render_date(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    _panel(draw, cv, fields)
    x0, y0, x1, y1 = cv.box
    title = _s(fields, "title", "BEST BEFORE")
    text = _s(fields, "text") or _s(fields, "date", "2026-01-01")
    sub = _s(fields, "sub")
    cap_h = 0.18 * (y1 - y0)
    lines, font = _fit(
        title.upper(), x1 - x0, cap_h * 0.72, int(cap_h * 0.60), max_lines=1
    )
    _draw_lines(draw, lines, font, (x0, y0, x1, y0 + cap_h * 0.86), GREY)
    rule = y0 + cap_h
    draw.rectangle(
        (x0 + 0.10 * (x1 - x0), rule, x1 - 0.10 * (x1 - x0), rule + 8),
        fill=_colour(fields.get("accent"), RED),
    )
    sub_h = 0.20 * (y1 - y0) if sub else 0.0
    lines, font = _fit(
        text.upper(), x1 - x0, (y1 - sub_h) - (rule + 20), int(0.30 * cv.h), max_lines=2
    )
    _draw_lines(
        draw,
        lines,
        font,
        (x0, rule + 20, x1, y1 - sub_h),
        _colour(fields.get("fg"), INK),
    )
    if sub:
        lines, font = _fit(
            sub.upper(), x1 - x0, sub_h * 0.72, int(sub_h * 0.50), max_lines=2
        )
        _draw_lines(draw, lines, font, (x0, y1 - sub_h, x1, y1), GREY)


def _rows_of(fields: dict[str, object]) -> list[tuple[str, str]]:
    """Normalise rows into pairs, accepting ``A|B``, ``A: B``, ``A=B``, or pairs."""
    raw = fields.get("rows") or []
    if isinstance(raw, str):
        raw = [r for r in re.split(r"[;\n]", raw) if r.strip()]
    out: list[tuple[str, str]] = []
    for row in raw:  # pyre-ignore[16]
        if isinstance(row, (tuple, list)) and len(row) == 2:
            out.append((str(row[0]).strip(), str(row[1]).strip()))
            continue
        parts = re.split(r"\s*(?:\||[:=])\s*", str(row), maxsplit=1)
        out.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
    return out


def _render_range_card(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    _panel(draw, cv, fields)
    x0, _, x1, y1 = cv.box
    accent = _colour(fields.get("accent"), FRAME)
    top = _title_bar(draw, cv, _s(fields, "title", "CARD"), accent)
    rows = _rows_of(fields)
    if not rows:
        body = _s(fields, "text", "--")
        lines, font = _fit(
            body.upper(), x1 - x0, y1 - top, int(0.20 * cv.h), max_lines=4
        )
        _draw_lines(draw, lines, font, (x0, top, x1, y1), INK)
        return
    shown = rows[:6]
    if len(shown) < len(rows):
        print(
            f"[label_generator][WARN] range_card: {len(rows) - len(shown)} row(s) dropped (max 6)"
        )
    slot = (y1 - top) / len(shown)
    font = _row_font(shown, x1 - x0, slot)
    for i, (left, right) in enumerate(shown):
        _draw_row(
            draw,
            (x0, top + i * slot, x1, top + (i + 1) * slot),
            left,
            right,
            font,
            accent,
        )


def _row_font(
    rows: Sequence[tuple[str, str]], width: float, slot: float
) -> ImageFont.FreeTypeFont:
    widest = max(len(f"{a}  {b}") for a, b in rows)
    for size in range(int(slot * 0.70), 13, -2):
        font = _font(size)
        if _text_wh("M" * widest, font)[0] <= width * 1.05:
            return font
    return _font(14)


def _draw_row(
    draw: ImageDraw.ImageDraw,
    box: Box,
    left: str,
    right: str,
    font: ImageFont.FreeTypeFont,
    accent: Rgba,
) -> None:
    x0, y0, x1, y1 = box
    baseline = y0 + 0.5 * (y1 - y0) - 0.5 * _text_wh(left or "X", font)[1]
    draw.text((x0, baseline), left.upper(), font=font, fill=INK)
    if right:
        w = _text_wh(right.upper(), font)[0]
        draw.text((x1 - w, baseline), right.upper(), font=font, fill=accent)
        lead_x = x0 + _text_wh(left.upper(), font)[0] + 14
        mid = y0 + 0.5 * (y1 - y0) + 0.30 * _text_wh("X", font)[1]
        while lead_x < x1 - w - 14:
            draw.rectangle((lead_x, mid, lead_x + 4, mid + 4), fill=GREY)
            lead_x += 16
    draw.rectangle((x0, y1 - 3, x1, y1 - 1), fill=(220, 220, 216, 255))


def _semantic_document_rows(fields: dict[str, object]) -> list[tuple[str, str]]:
    rows = _rows_of(fields)
    if not rows:
        raw_text = fields.get("text")
        if isinstance(raw_text, Sequence) and not isinstance(raw_text, (str, bytes)):
            rows = _rows_of({"rows": raw_text})
        elif str(raw_text or "").strip():
            rows = [(str(raw_text).strip(), "")]
    if not rows:
        raise ValueError("semantic document renderers require at least one body row")
    if len(rows) > 6:
        raise ValueError("semantic document renderers support at most six body rows")
    return rows


def _fit_exact(
    text: str,
    box_w: float,
    box_h: float,
    max_size: int,
    *,
    min_size: int = 14,
    max_lines: int = 3,
    bold: bool = True,
) -> tuple[list[str], ImageFont.FreeTypeFont]:
    for size in range(max_size, min_size - 1, -1):
        font = _font(size, bold=bold)
        lines = _wrap(text, font, box_w)
        if len(lines) > max_lines:
            continue
        line_h = max(_text_wh(line or "X", font)[1] for line in lines)
        total_h = line_h * (len(lines) + 0.18 * max(0, len(lines) - 1))
        if total_h <= box_h and all(_text_wh(line, font)[0] <= box_w for line in lines):
            return lines, font
    raise ValueError(f"document text does not fit without truncation: {text!r}")


def _draw_exact_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Box,
    fill: Rgba,
    *,
    max_size: int,
    max_lines: int = 3,
    bold: bool = True,
    centred: bool = False,
) -> int:
    x0, y0, x1, y1 = box
    lines, font = _fit_exact(
        text,
        x1 - x0,
        y1 - y0,
        max_size,
        max_lines=max_lines,
        bold=bold,
    )
    line_h = max(_text_wh(line or "X", font)[1] for line in lines)
    step = line_h * 1.18
    y = y0 + max(0.0, (y1 - y0 - line_h - step * (len(lines) - 1)) * 0.5)
    for line in lines:
        left, top, right, _bottom = font.getbbox(line)
        x = x0 - left if not centred else 0.5 * (x0 + x1 - right + left) - left
        draw.text((x, y - top), line, font=font, fill=fill)
        y += step
    return font.size


def _document_heading(
    draw: ImageDraw.ImageDraw,
    cv: Canvas,
    fields: dict[str, object],
    eyebrow: str,
    accent: Rgba,
) -> float:
    height = 0.22 * cv.h
    draw.rectangle((0, 0, cv.w, height), fill=accent)
    _draw_exact_text(
        draw,
        eyebrow,
        (float(cv.mx), 10.0, float(cv.w - cv.mx), height * 0.30),
        WHITE,
        max_size=max(14, int(0.028 * cv.h)),
        max_lines=1,
    )
    _draw_exact_text(
        draw,
        _s(fields, "title", "DOCUMENT"),
        (float(cv.mx), height * 0.28, float(cv.w - cv.mx), height - 10.0),
        WHITE,
        max_size=max(24, int(0.060 * cv.h)),
        max_lines=2,
    )
    return height + 0.025 * cv.h


def _document_body_bottom(
    draw: ImageDraw.ImageDraw,
    cv: Canvas,
    fields: dict[str, object],
    accent: Rgba,
) -> float:
    sub = _s(fields, "sub").strip()
    if not sub:
        return float(cv.h - cv.my)
    top = cv.h - cv.my - 0.075 * cv.h
    draw.rectangle((cv.mx, top, cv.w - cv.mx, cv.h - cv.my), fill=accent)
    _draw_exact_text(
        draw,
        sub,
        (cv.mx + 12.0, top + 4.0, cv.w - cv.mx - 12.0, cv.h - cv.my - 4.0),
        WHITE,
        max_size=max(14, int(0.027 * cv.h)),
        max_lines=2,
        centred=True,
    )
    return top - 8.0


def _draw_document_pair(
    draw: ImageDraw.ImageDraw,
    box: Box,
    left: str,
    right: str,
    accent: Rgba,
) -> None:
    x0, y0, x1, y1 = box
    max_size = max(16, min(36, int((y1 - y0) * 0.36)))
    if not right:
        _draw_exact_text(
            draw, left, box, INK, max_size=max_size, max_lines=3, bold=False
        )
        return
    split = x0 + 0.28 * (x1 - x0)
    _draw_exact_text(
        draw, left, (x0, y0, split - 10, y1), accent, max_size=max_size, max_lines=2
    )
    _draw_exact_text(
        draw,
        right,
        (split + 10, y0, x1, y1),
        INK,
        max_size=max_size,
        max_lines=3,
        bold=False,
    )


def _draw_document_value(draw: ImageDraw.ImageDraw, box: Box, text: str) -> None:
    max_size = max(16, min(36, int((box[3] - box[1]) * 0.36)))
    _draw_exact_text(draw, text, box, INK, max_size=max_size, max_lines=3, bold=False)


def _render_checklist(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    accent = _colour(fields.get("accent"), (38, 92, 76, 255))
    _panel(draw, cv, {**fields, "bg": fields.get("bg", WHITE), "border": "frame"})
    top = _document_heading(draw, cv, fields, "OPERATIONAL CHECKLIST", accent)
    bottom = _document_body_bottom(draw, cv, fields, accent)
    rows = _semantic_document_rows(fields)
    slot = (bottom - top) / len(rows)
    for index, (left, right) in enumerate(rows):
        y0, y1 = top + index * slot, top + (index + 1) * slot
        if index % 2:
            draw.rectangle((cv.mx, y0, cv.w - cv.mx, y1), fill=(242, 246, 244, 255))
        side = min(34.0, slot * 0.34)
        cy = 0.5 * (y0 + y1)
        draw.rectangle(
            (cv.mx + 8, cy - side / 2, cv.mx + 8 + side, cy + side / 2),
            outline=accent,
            width=max(3, int(side * 0.10)),
        )
        _draw_document_pair(
            draw,
            (cv.mx + side + 28, y0 + 5, cv.w - cv.mx - 8, y1 - 5),
            left,
            right,
            accent,
        )
        draw.line((cv.mx, y1, cv.w - cv.mx, y1), fill=(205, 214, 210, 255), width=2)


def _render_recipe_card(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    accent = _colour(fields.get("accent"), (166, 55, 42, 255))
    cream = _colour(fields.get("bg"), (255, 248, 229, 255))
    _panel(draw, cv, {**fields, "bg": cream, "border": accent, "border_w": 12})
    draw.rectangle((22, 22, cv.w - 23, cv.h - 23), outline=accent, width=3)
    top = _document_heading(draw, cv, fields, "KITCHEN RECIPE", accent)
    bottom = _document_body_bottom(draw, cv, fields, accent)
    rows = _semantic_document_rows(fields)
    slot = (bottom - top) / len(rows)
    for index, (left, right) in enumerate(rows):
        y0, y1 = top + index * slot, top + (index + 1) * slot
        if left.isdigit() and right:
            marker = (
                cv.mx + 4,
                y0 + slot * 0.22,
                cv.mx + 4 + slot * 0.56,
                y1 - slot * 0.22,
            )
            draw.ellipse(marker, fill=(244, 222, 190, 255), outline=accent, width=3)
            _draw_exact_text(
                draw, left, marker, accent, max_size=26, max_lines=1, centred=True
            )
            _draw_document_value(
                draw,
                (cv.mx + slot * 0.68, y0 + 4, cv.w - cv.mx - 8, y1 - 4),
                right,
            )
        else:
            _draw_document_pair(
                draw,
                (cv.mx + 12, y0 + 4, cv.w - cv.mx - 8, y1 - 4),
                left,
                right,
                accent,
            )


def _render_instruction_sheet(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    accent = _colour(fields.get("accent"), BLUE)
    _panel(draw, cv, {**fields, "bg": fields.get("bg", WHITE), "border": accent})
    top = _document_heading(draw, cv, fields, "OPERATING INSTRUCTIONS", accent)
    bottom = _document_body_bottom(draw, cv, fields, accent)
    rows = _semantic_document_rows(fields)
    slot = (bottom - top) / len(rows)
    rail_x = cv.mx + 26
    draw.rectangle((rail_x, top, rail_x + 7, bottom), fill=(188, 211, 239, 255))
    for index, (left, right) in enumerate(rows):
        y0, y1 = top + index * slot, top + (index + 1) * slot
        badge = (rail_x - 13, 0.5 * (y0 + y1) - 13, rail_x + 20, 0.5 * (y0 + y1) + 20)
        draw.ellipse(badge, fill=accent)
        _draw_exact_text(
            draw,
            left if left.isdigit() and right else str(index + 1),
            badge,
            WHITE,
            max_size=21,
            max_lines=1,
            centred=True,
        )
        content_box = (rail_x + 38, y0 + 5, cv.w - cv.mx, y1 - 5)
        if left.isdigit() and right:
            _draw_document_value(draw, content_box, right)
        else:
            _draw_document_pair(draw, content_box, left, right, accent)


def _render_handwritten_note(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    ink = _colour(fields.get("accent"), (39, 75, 145, 255))
    paper = _colour(fields.get("bg"), (255, 251, 214, 255))
    _panel(draw, cv, {**fields, "bg": paper, "border": (207, 191, 137, 255)})
    left = cv.mx + 0.09 * cv.w
    top = cv.my + 0.05 * cv.h
    sub = _s(fields, "sub").strip()
    bottom = cv.h - cv.my - (0.09 * cv.h if sub else 0.0)
    draw.line((left, cv.my, left, bottom), fill=(220, 118, 118, 255), width=3)
    for y in range(int(top + 0.16 * cv.h), int(bottom), max(24, int(0.075 * cv.h))):
        draw.line((cv.mx, y, cv.w - cv.mx, y), fill=(178, 207, 224, 255), width=2)
    _draw_exact_text(
        draw,
        _s(fields, "title", "NOTE"),
        (left + 10, top, cv.w - cv.mx - 14, top + 0.15 * cv.h),
        ink,
        max_size=max(24, int(0.055 * cv.h)),
        max_lines=2,
        bold=False,
    )
    rows = _semantic_document_rows(fields)
    row_top = top + 0.17 * cv.h
    slot = (bottom - row_top) / len(rows)
    for index, (key, value) in enumerate(rows):
        text = f"{key}: {value}" if value else key
        indent = 8 if index % 2 else 0
        _draw_exact_text(
            draw,
            f"- {text}",
            (
                left + 12 + indent,
                row_top + index * slot,
                cv.w - cv.mx - 10,
                row_top + (index + 1) * slot,
            ),
            ink,
            max_size=max(19, int(slot * 0.32)),
            max_lines=2,
            bold=False,
        )
    if sub:
        _draw_exact_text(
            draw,
            sub,
            (left + 12, bottom, cv.w - cv.mx - 10, cv.h - cv.my),
            ink,
            max_size=max(17, int(0.030 * cv.h)),
            max_lines=2,
            bold=False,
            centred=True,
        )


def _render_order_ticket(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    accent = _colour(fields.get("accent"), (50, 58, 64, 255))
    _panel(draw, cv, {**fields, "bg": fields.get("bg", (255, 253, 244, 255))})
    top = _document_heading(draw, cv, fields, "ORDER TICKET / PACKING SLIP", accent)
    bottom = _document_body_bottom(draw, cv, fields, accent)
    rows = _semantic_document_rows(fields)
    for x in range(cv.mx, cv.w - cv.mx, 22):
        draw.rectangle((x, top, x + 10, top + 3), fill=GREY)
    top += 12
    slot = (bottom - top) / len(rows)
    for index, (left, right) in enumerate(rows):
        y0, y1 = top + index * slot, top + (index + 1) * slot
        _draw_document_pair(
            draw, (cv.mx + 8, y0 + 5, cv.w - cv.mx - 8, y1 - 5), left, right, accent
        )
        draw.line((cv.mx, y1, cv.w - cv.mx, y1), fill=GREY, width=2)


def _render_service_card(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    accent = _colour(fields.get("accent"), (190, 118, 25, 255))
    _panel(draw, cv, {**fields, "bg": fields.get("bg", (248, 251, 252, 255))})
    top = _document_heading(draw, cv, fields, "SERVICE NOTICE", accent)
    bottom = _document_body_bottom(draw, cv, fields, accent)
    rows = _semantic_document_rows(fields)
    slot = (bottom - top) / len(rows)
    for index, (left, right) in enumerate(rows):
        y0, y1 = top + index * slot, top + (index + 1) * slot
        box = (cv.mx + 6, y0 + 6, cv.w - cv.mx - 6, y1 - 6)
        draw.rounded_rectangle(box, radius=12, fill=(235, 242, 244, 255))
        draw.rectangle((box[0], box[1], box[0] + 9, box[3]), fill=accent)
        _draw_document_pair(
            draw,
            (box[0] + 25, box[1] + 3, box[2] - 12, box[3] - 3),
            left,
            right,
            accent,
        )


def _render_waste_stream_label(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    title = _s(fields, "title", "WASTE").upper()
    is_sharps = "SHARP" in title
    accepts_glass = "GLASS" in title and "GENERAL" not in title
    default_accent = RED if is_sharps else GREEN if accepts_glass else FRAME
    accent = _colour(fields.get("accent"), default_accent)
    _panel(draw, cv, {**fields, "bg": WHITE, "border": accent, "border_w": 14})
    icon_box = (cv.mx + 10, cv.my + 12, cv.mx + 0.25 * cv.w, cv.h - cv.my - 12)
    x0, y0, x1, y1 = icon_box
    if is_sharps:
        inset = 0.07 * (x1 - x0)
        warning = (
            (0.5 * (x0 + x1), y0 + inset),
            (x1 - inset, y1 - inset),
            (x0 + inset, y1 - inset),
        )
        draw.polygon(warning, outline=RED, width=12)
        cx, cy = 0.5 * (x0 + x1), 0.58 * y1 + 0.42 * y0
        shard = 0.14 * (x1 - x0)
        draw.line(
            (cx - shard, cy - shard, cx, cy, cx + shard, cy - shard), fill=RED, width=9
        )
        draw.line((cx, cy, cx - 0.6 * shard, cy + shard), fill=RED, width=9)
        draw.line((cx, cy, cx + 0.7 * shard, cy + shard), fill=RED, width=9)
    else:
        neck = 0.32 * (x1 - x0)
        bottle = (
            x0 + neck,
            y0 + 0.08 * (y1 - y0),
            x1 - neck,
            y0 + 0.25 * (y1 - y0),
            x1 - 0.08 * (x1 - x0),
            y1,
            x0 + 0.08 * (x1 - x0),
            y1,
        )
        draw.polygon(bottle, outline=accent, width=10)
        if not accepts_glass:
            draw.line((x0, y0, x1, y1), fill=RED, width=16)
    text_left = cv.mx + 0.31 * cv.w
    _draw_exact_text(
        draw,
        title,
        (text_left, cv.my + 8, cv.w - cv.mx - 8, 0.48 * cv.h),
        accent,
        max_size=max(26, int(0.115 * cv.h)),
        max_lines=2,
    )
    rows = [": ".join(filter(None, row)) for row in _semantic_document_rows(fields)]
    _draw_exact_text(
        draw,
        "\n".join(rows),
        (text_left, 0.50 * cv.h, cv.w - cv.mx - 8, cv.h - cv.my - 8),
        RED if is_sharps or not accepts_glass else INK,
        max_size=max(20, int(0.065 * cv.h)),
        max_lines=max(2, len(rows) * 2),
    )


def _render_price_tag(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    accent = _colour(fields.get("accent"), RED)
    _panel(
        draw,
        cv,
        {**fields, "bg": fields.get("bg", (255, 244, 104, 255)), "border": accent},
    )
    x0, y0, x1, y1 = cv.box
    text = _s(fields, "text")
    description, separator, amount = text.partition("$")
    if not separator:
        _draw_exact_text(
            draw, text, cv.box, INK, max_size=int(0.18 * cv.h), centred=True
        )
        return
    _draw_exact_text(
        draw,
        description.strip(),
        (x0, y0, x1, y0 + 0.34 * (y1 - y0)),
        INK,
        max_size=int(0.10 * cv.h),
        max_lines=2,
        centred=True,
    )
    _draw_exact_text(
        draw,
        f"${amount.strip()}",
        (x0, y0 + 0.30 * (y1 - y0), x1, y1),
        accent,
        max_size=int(0.38 * cv.h),
        max_lines=1,
        centred=True,
    )


def _render_capacity_label(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    accent = _colour(fields.get("accent"), BLUE)
    _panel(draw, cv, {**fields, "bg": fields.get("bg", WHITE), "border": accent})
    x0, y0, x1, y1 = cv.box
    draw.rectangle((x0, y0, x1, y0 + 0.10 * (y1 - y0)), fill=accent)
    text = _s(fields, "text")
    match = re.fullmatch(r"(.*?)(\d+(?:\.\d+)?\s*(?:mL|L))", text.strip(), re.I)
    if match is None:
        _draw_exact_text(
            draw, text, cv.box, INK, max_size=int(0.16 * cv.h), centred=True
        )
        return
    product, amount = match.groups()
    _draw_exact_text(
        draw,
        product.strip(),
        (x0, y0 + 0.13 * (y1 - y0), x1, y0 + 0.43 * (y1 - y0)),
        FRAME,
        max_size=int(0.075 * cv.h),
        max_lines=2,
        centred=True,
    )
    _draw_exact_text(
        draw,
        amount,
        (x0, y0 + 0.43 * (y1 - y0), x1, y1),
        accent,
        max_size=int(0.22 * cv.h),
        max_lines=1,
        centred=True,
    )


def _nutrition_rows(fields: dict[str, object]) -> list[tuple[str, str, str, str]]:
    raw = fields.get("rows") or []
    if isinstance(raw, str):
        raw = [row for row in raw.splitlines() if row.strip()]
    rows: list[tuple[str, str, str, str]] = []
    for raw_row in raw:  # pyre-ignore[16]
        values = (
            [str(value).strip() for value in raw_row]
            if isinstance(raw_row, (tuple, list))
            else [value.strip() for value in str(raw_row).split("|")]
        )
        if len(values) != 4:
            raise ValueError(
                "nutrition_report rows need four columns: "
                "nutrient|consumed|daily target|remaining"
            )
        rows.append((values[0], values[1], values[2], values[3]))
    if not rows:
        raise ValueError("nutrition_report needs at least one nutrient row")
    if len(rows) > 9:
        raise ValueError("nutrition_report supports at most nine nutrient rows")
    return rows


def _report_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Box,
    *,
    fill: Rgba,
    bold: bool,
    max_size: int,
) -> None:
    lines, font = _fit(
        text,
        box[2] - box[0],
        box[3] - box[1],
        max_size,
        min_size=12,
        max_lines=1,
    )
    if not bold:
        font = _font(font.size, bold=False)
    _draw_lines(draw, lines, font, box, fill, gap=0.0)


def _render_nutrition_report(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    """A compact diet-log printout, sized for a letter/A4-like paper sheet."""
    _panel(draw, cv, {**fields, "bg": fields.get("bg", WHITE), "border": "frame"})
    navy = _colour(fields.get("accent"), (36, 67, 92, 255))
    soft = _colour(fields.get("card"), (235, 242, 246, 255))
    good = _colour(fields.get("remaining"), (31, 122, 78, 255))
    x0, y0, x1, y1 = cv.box

    brand_h = 0.055 * cv.h
    draw.rectangle((x0, y0, x1, y0 + brand_h), fill=navy)
    _report_text(
        draw,
        _s(fields, "provider", "PERSONAL NUTRITION PLAN"),
        (x0 + 14, y0, x1 - 14, y0 + brand_h),
        fill=WHITE,
        bold=True,
        max_size=int(0.027 * cv.h),
    )

    title_top = y0 + brand_h + 12
    title_h = 0.064 * cv.h
    _report_text(
        draw,
        _s(fields, "title", "Daily Nutrition Summary"),
        (x0, title_top, x1, title_top + title_h),
        fill=INK,
        bold=True,
        max_size=int(0.043 * cv.h),
    )
    details_top = title_top + title_h
    detail_font = _font(max(14, int(0.018 * cv.h)), bold=False)
    draw.text(
        (x0 + 8, details_top),
        _s(fields, "date", "September 2, 2026"),
        font=detail_font,
        fill=FRAME,
    )
    plan = _s(fields, "plan", "2,000 kcal daily plan")
    plan_w = _text_wh(plan, detail_font)[0]
    draw.text((x1 - plan_w - 8, details_top), plan, font=detail_font, fill=FRAME)

    rows = _nutrition_rows(fields)
    table_top = details_top + 0.045 * cv.h
    footer_h = 0.100 * cv.h
    table_bottom = y1 - footer_h
    head_h = 0.050 * cv.h
    columns = (
        x0,
        x0 + 0.34 * (x1 - x0),
        x0 + 0.57 * (x1 - x0),
        x0 + 0.79 * (x1 - x0),
        x1,
    )
    draw.rectangle((x0, table_top, x1, table_top + head_h), fill=soft)
    for index, heading in enumerate(
        ("NUTRIENT", "CONSUMED", "DAILY TARGET", "REMAINING")
    ):
        _report_text(
            draw,
            heading,
            (columns[index] + 7, table_top, columns[index + 1] - 7, table_top + head_h),
            fill=navy,
            bold=True,
            max_size=int(0.019 * cv.h),
        )

    row_h = (table_bottom - table_top - head_h) / len(rows)
    for row_index, row in enumerate(rows):
        top = table_top + head_h + row_index * row_h
        bottom = top + row_h
        if row_index % 2:
            draw.rectangle((x0, top, x1, bottom), fill=(248, 249, 250, 255))
        for column_index, value in enumerate(row):
            _report_text(
                draw,
                value,
                (
                    columns[column_index] + 7,
                    top,
                    columns[column_index + 1] - 7,
                    bottom,
                ),
                fill=good if column_index == 3 else INK,
                bold=column_index in {0, 3},
                max_size=int(0.040 * cv.h),
            )
        draw.rectangle((x0, bottom - 2, x1, bottom), fill=(220, 224, 228, 255))
    for column in columns[1:-1]:
        draw.rectangle(
            (column - 1, table_top, column + 1, table_bottom), fill=(213, 220, 225, 255)
        )

    note = _s(
        fields,
        "note",
        "Daily targets are personalized. Total sugars is a clinician-set ceiling.",
    )
    _report_text(
        draw,
        note,
        (x0 + 8, table_bottom + 8, x1 - 8, y1),
        fill=FRAME,
        bold=False,
        max_size=int(0.017 * cv.h),
    )


def _render_placard(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    """Wall/door sign: coloured title bar over a big body string."""
    fields = {**fields, "bg": fields.get("bg", FRAME)}
    _panel(draw, cv, fields)
    x0, _, x1, y1 = cv.box
    accent = _colour(fields.get("accent"), FRAME)
    top = _title_bar(draw, cv, _s(fields, "title", "SIGN"), accent)
    card = (x0 * 0.6, top, cv.w - x0 * 0.6, y1 + cv.my * 0.5)
    draw.rectangle(card, fill=_colour(fields.get("card"), PAPER))
    sub = _s(fields, "sub")
    sub_h = 0.22 * (card[3] - card[1]) if sub else 0.0
    inner = (card[0] + 16, card[1] + 12, card[2] - 16, card[3] - 12 - sub_h)
    lines, font = _fit(
        _s(fields, "text", "A").upper(),
        inner[2] - inner[0],
        inner[3] - inner[1],
        int(0.62 * (card[3] - card[1])),
        max_lines=2,
    )
    _draw_lines(draw, lines, font, inner, _colour(fields.get("fg"), INK))
    if sub:
        lines, font = _fit(
            sub.upper(),
            inner[2] - inner[0],
            sub_h * 0.76,
            int(sub_h * 0.56),
            max_lines=2,
        )
        _draw_lines(
            draw,
            lines,
            font,
            (inner[0], card[3] - 12 - sub_h, inner[2], card[3] - 12),
            accent,
        )


_SHAPES: dict[str, list[tuple[float, float]]] = {
    "wrench": [
        (0.06, 0.36),
        (0.16, 0.26),
        (0.26, 0.30),
        (0.26, 0.20),
        (0.14, 0.16),
        (0.20, 0.06),
        (0.34, 0.06),
        (0.42, 0.18),
        (0.40, 0.34),
        (0.92, 0.72),
        (0.86, 0.88),
        (0.30, 0.54),
        (0.16, 0.56),
        (0.06, 0.48),
    ],
    "hammer": [
        (0.10, 0.10),
        (0.46, 0.10),
        (0.50, 0.22),
        (0.40, 0.30),
        (0.92, 0.84),
        (0.82, 0.94),
        (0.30, 0.40),
        (0.18, 0.34),
        (0.10, 0.24),
    ],
    "screwdriver": [
        (0.10, 0.14),
        (0.24, 0.06),
        (0.34, 0.20),
        (0.44, 0.32),
        (0.90, 0.80),
        (0.82, 0.92),
        (0.34, 0.44),
        (0.20, 0.34),
    ],
    "pliers": [
        (0.10, 0.08),
        (0.22, 0.06),
        (0.48, 0.42),
        (0.62, 0.42),
        (0.90, 0.06),
        (0.96, 0.16),
        (0.66, 0.56),
        (0.66, 0.94),
        (0.44, 0.94),
        (0.44, 0.56),
        (0.08, 0.18),
    ],
    "mallet": [
        (0.06, 0.16),
        (0.44, 0.16),
        (0.44, 0.40),
        (0.92, 0.40),
        (0.92, 0.64),
        (0.44, 0.64),
        (0.44, 0.88),
        (0.06, 0.88),
    ],
    "tape": [
        (0.06, 0.34),
        (0.52, 0.34),
        (0.62, 0.46),
        (0.62, 0.60),
        (0.96, 0.60),
        (0.96, 0.74),
        (0.58, 0.74),
        (0.58, 0.84),
        (0.06, 0.84),
    ],
}


def _render_silhouette(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    """5S shadow board: a dark tool outline cut into a light foam panel."""
    board = _colour(fields.get("bg"), (196, 199, 203, 255))
    shadow = _colour(fields.get("fg"), FOAM)
    _panel(draw, cv, {**fields, "bg": board, "border": fields.get("border", FOAM)})
    x0, y0, x1, y1 = cv.box
    caption = _s(fields, "text")
    cap_h = 0.18 * (y1 - y0) if caption else 0.0
    shape = _s(fields, "shape", "wrench").strip().lower()
    if shape not in _SHAPES:
        print(
            f"[label_generator][WARN] unknown silhouette shape {shape!r}; using 'wrench'"
        )
    uv = _SHAPES.get(shape, _SHAPES["wrench"])
    area = (x0, y0, x1, y1 - cap_h)
    _poly(draw, area, uv, fill=shadow)
    if caption:
        lines, font = _fit(
            caption.upper(), x1 - x0, cap_h * 0.78, int(cap_h * 0.58), max_lines=1
        )
        _draw_lines(draw, lines, font, (x0, y1 - cap_h, x1, y1), shadow)


# --------------------------------------------------------------------------- #
# protocol_sheet -- the posted-procedure MANUAL for the T6 prologue_intent rows
# --------------------------------------------------------------------------- #
# Six candidate plans have to be posted somewhere or the row is undecodable, but
# six free-text rules are far past the ~300 px/0.9 m readability budget. The sheet
# is therefore a MATRIX: the only per-plan content is one placard letter per
# class, so every decisive glyph is a single high-contrast character in its own
# ruled column, and the class tokens are printed once in the header instead of
# once per plan. Column rules + zebra banding exist because a permutation table
# read one row out of register is silently wrong, not obviously wrong.
ZEBRA: Rgba = (232, 234, 238, 255)


def _plans_of(fields: dict[str, object]) -> list[list[str]]:
    """Normalise ``plans`` into rows of [label, cell, cell, ...].

    Accepts a list of sequences, or strings ``'1|B|D|F'`` / ``'1: B D F'``.
    """
    raw = fields.get("plans") or []
    if isinstance(raw, str):
        raw = [r for r in re.split(r"[;\n]", raw) if r.strip()]
    out: list[list[str]] = []
    for row in raw:  # pyre-ignore[16]
        if isinstance(row, (tuple, list)):
            out.append([str(c).strip() for c in row])
            continue
        text = str(row)
        if "|" in text:
            out.append([c.strip() for c in text.split("|")])
            continue
        head, _, tail = text.partition(":")
        out.append([head.strip()] + tail.split())
    return out


def _cols_of(fields: dict[str, object], n_cells: int) -> list[str]:
    raw = fields.get("cols") or []
    if isinstance(raw, str):
        raw = [c for c in re.split(r"[;,|\n]", raw) if c.strip()]
    cols = [str(c).strip().upper() for c in raw]  # pyre-ignore[16]
    if cols and len(cols) != n_cells:
        print(
            f"[label_generator][WARN] protocol_sheet: {len(cols)} column header(s) for "
            f"{n_cells} cell(s) per plan; the table will not line up"
        )
    return cols or [""] * n_cells


def _grid_x(box: Box, n_cells: int, gutter_frac: float) -> list[float]:
    """Left edge of every column plus the right edge, gutter column first."""
    x0, _, x1, _ = box
    gw = (x1 - x0) * gutter_frac
    cw = (x1 - x0 - gw) / max(1, n_cells)
    return [x0] + [x0 + gw + i * cw for i in range(n_cells + 1)]


def _sheet_bands(cv: Canvas, fields: dict[str, object]) -> tuple[float, float]:
    """(y below the title+sub block, y above the footer banner). The title term
    mirrors ``_title_bar``'s own geometry, which owns the band height."""
    top = max(46.0, 0.20 * cv.h) + cv.my * 0.6
    sub_h = 0.070 * cv.h if _s(fields, "sub") else 0.0
    note_h = 0.150 * cv.h if _s(fields, "note") else 0.0
    return top + sub_h, cv.h - cv.my * 0.5 - note_h


def _draw_head_row(
    draw: ImageDraw.ImageDraw,
    xs: Sequence[float],
    band: tuple[float, float],
    heads: Sequence[str],
    accent: Rgba,
) -> None:
    y0, y1 = band
    draw.rectangle((xs[0], y0, xs[-1], y1), fill=accent)
    for i, head in enumerate(heads):
        if not head:
            continue
        lines, font = _fit(
            head, (xs[i + 1] - xs[i]) * 0.90, (y1 - y0) * 0.68, int((y1 - y0) * 0.62)
        )
        _draw_lines(draw, lines, font, (xs[i], y0, xs[i + 1], y1), WHITE)


def _draw_plan_row(
    draw: ImageDraw.ImageDraw,
    xs: Sequence[float],
    band: tuple[float, float],
    cells: Sequence[str],
    fonts: tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont],
    accent: Rgba,
    zebra: bool,
) -> None:
    y0, y1 = band
    gut, cell_font = fonts
    if zebra:
        draw.rectangle((xs[1], y0, xs[-1], y1), fill=ZEBRA)
    draw.rectangle((xs[0], y0, xs[1], y1), fill=accent)
    _draw_lines(draw, [cells[0]], gut, (xs[0], y0, xs[1], y1), WHITE)
    for i, cell in enumerate(cells[1:]):
        _draw_lines(draw, [cell], cell_font, (xs[i + 1], y0, xs[i + 2], y1), INK)
    for x in xs[1:-1]:
        draw.rectangle((x - 1, y0, x + 1, y1), fill=GREY)
    draw.rectangle((xs[0], y1 - 2, xs[-1], y1), fill=FRAME)


def _sheet_fonts(
    plans: Sequence[Sequence[str]], xs: Sequence[float], slot: float
) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    """One font for the whole grid so a permutation table cannot look ragged."""
    gut_w = (xs[1] - xs[0]) * 0.80
    cell_w = min(xs[i + 2] - xs[i + 1] for i in range(len(xs) - 2)) * 0.84
    widest_gut = max((p[0] for p in plans), key=len, default="1")
    widest_cell = max(
        (c for p in plans for c in p[1:]), key=lambda s: len(s), default="A"
    )
    return (
        _fit(widest_gut, gut_w, slot * 0.74, int(slot * 0.70))[1],
        _fit(widest_cell, cell_w, slot * 0.80, int(slot * 0.76))[1],
    )


def _draw_footer(
    draw: ImageDraw.ImageDraw, cv: Canvas, note: str, accent: Rgba
) -> None:
    """Full-width banner for the fallback rule. It is DECISIVE on the ambiguous
    arm, so it gets banner treatment rather than fine print."""
    y0 = cv.h - cv.my * 0.5 - 0.150 * cv.h
    draw.rectangle(
        (cv.mx * 0.5, y0, cv.w - cv.mx * 0.5, cv.h - cv.my * 0.5), fill=accent
    )
    box = (float(cv.mx), y0 + 0.012 * cv.h, float(cv.w - cv.mx), cv.h - cv.my)
    lines, font = _fit(
        note.upper(),
        box[2] - box[0],
        box[3] - box[1],
        int(0.075 * cv.h),
        min_size=18,
        max_lines=2,
    )
    _draw_lines(draw, lines, font, box, WHITE)


def _render_protocol_sheet(
    draw: ImageDraw.ImageDraw, cv: Canvas, fields: dict[str, object]
) -> None:
    """Posted-procedure sheet: N candidate plans x M classes, one letter a cell."""
    _panel(draw, cv, fields)
    accent = _colour(fields.get("accent"), FRAME)
    plans = _plans_of(fields)
    if not plans:
        raise ValueError("protocol_sheet needs plans=[['1','B','D','F'], ...]")
    n_cells = max(len(p) for p in plans) - 1
    plans = [p + [""] * (n_cells + 1 - len(p)) for p in plans]
    _title_bar(draw, cv, _s(fields, "title", "POSTED PLANS"), accent)
    top, bottom = _sheet_bands(cv, fields)
    if _s(fields, "sub"):
        band = (top - 0.070 * cv.h, top)
        lines, font = _fit(
            _s(fields, "sub").upper(),
            cv.w - 2.0 * cv.mx,
            (band[1] - band[0]) * 0.80,
            int(0.052 * cv.h),
            max_lines=1,  # a wrapped subline runs into the column headers
        )
        _draw_lines(
            draw,
            lines,
            font,
            (float(cv.mx), band[0], float(cv.w - cv.mx), band[1]),
            GREY,
        )
    grid = (float(cv.mx * 0.6), top, float(cv.w - cv.mx * 0.6), bottom)
    xs = _grid_x(
        grid, n_cells, float(fields.get("gutter_frac") or 0.20)
    )  # pyre-ignore[6]
    slot = (bottom - top) / (len(plans) + 1)
    _draw_head_row(
        draw,
        xs,
        (top, top + slot),
        [_s(fields, "gutter", "PLAN").upper()] + _cols_of(fields, n_cells),
        accent,
    )
    fonts = _sheet_fonts(plans, xs, slot)
    for i, cells in enumerate(plans):
        y = top + (i + 1) * slot
        _draw_plan_row(draw, xs, (y, y + slot), cells, fonts, accent, i % 2 == 1)
    if _s(fields, "note"):
        _draw_footer(draw, cv, _s(fields, "note"), _colour(fields.get("note_bg"), RED))


def _renderers() -> dict[
    str, Callable[[ImageDraw.ImageDraw, Canvas, dict[str, object]], None]
]:
    return {
        "capacity_label": _render_capacity_label,
        "checklist": _render_checklist,
        "handwritten_note": _render_handwritten_note,
        "pictogram": _render_pictogram,
        "instruction_sheet": _render_instruction_sheet,
        "order_ticket": _render_order_ticket,
        "price_tag": _render_price_tag,
        "recipe_card": _render_recipe_card,
        "service_card": _render_service_card,
        "waste_stream_label": _render_waste_stream_label,
        "short_string": _render_short_string,
        "number": _render_number,
        "date": _render_date,
        "range_card": _render_range_card,
        "placard": _render_placard,
        "silhouette": _render_silhouette,
        "nutrition_report": _render_nutrition_report,
        "protocol_sheet": _render_protocol_sheet,
    }


def resolve_document_source(fields: dict[str, object]) -> Path:
    source = fields.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("document_scan requires a non-empty relative source path")
    return resolve_label_asset(source)


def _document_dimensions(fields: dict[str, object]) -> tuple[int, int]:
    try:
        with Image.open(resolve_document_source(fields)) as source:
            return int(source.width), int(source.height)
    except OSError as exc:
        raise ValueError(
            f"document_scan source is not a readable image: {exc}"
        ) from exc


def _document_image(cv: Canvas, fields: dict[str, object]) -> Image.Image:
    try:
        with Image.open(resolve_document_source(fields)) as source:
            scan = source.convert("RGB")
    except OSError as exc:
        raise ValueError(
            f"document_scan source is not a readable image: {exc}"
        ) from exc
    if scan.size == (cv.w, cv.h):
        return scan
    scan.thumbnail((cv.w, cv.h), Image.Resampling.LANCZOS)
    page = Image.new("RGB", (cv.w, cv.h), (250, 250, 248))
    page.paste(scan, ((cv.w - scan.width) // 2, (cv.h - scan.height) // 2))
    return page


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def canon_kind(kind: str) -> str:
    key = kind.strip().lower().replace("-", "_").replace(" ", "_")
    key = _KIND_ALIAS.get(key, _KIND_ALIAS.get(kind.strip().lower(), key))
    known = {*_renderers(), "document_scan", "mold_patch"}
    if key not in known:
        raise ValueError(f"unknown label kind {kind!r}; known kinds: {sorted(known)}")
    return key


def _canvas_for(kind: str, fields: dict[str, object]) -> Canvas:
    if kind == "document_scan":
        source_w, source_h = _document_dimensions(fields)
        w = int(fields.get("width") or source_w)  # pyre-ignore[6]
        h = int(fields.get("height") or source_h)  # pyre-ignore[6]
        return Canvas(w, h, 0, 0)
    mount = _s(fields, "mount") or _DEFAULT_MOUNT[kind]
    cv = _kind_mount_override(kind, mount) or _mount_canvas(mount)
    w = int(fields.get("width") or cv.w)  # pyre-ignore[6]
    h = int(fields.get("height") or cv.h)  # pyre-ignore[6]
    if (w, h) == (cv.w, cv.h):
        return cv
    return Canvas(
        w, h, max(16, round(cv.mx / cv.w * w)), max(14, round(cv.my / cv.h * h))
    )


def label_generator(
    kind: str, out_png: str | Path, *, force: bool = False, **fields: object
) -> Path:
    """Render one label PNG.

    Args:
        kind: a renderer returned by ``canon_kind``. Structured document kinds
            include checklist / handwritten_note / instruction_sheet /
            order_ticket / recipe_card / service_card / waste_stream_label;
            compact merchandise kinds include price_tag and capacity_label;
            mold_patch is a transparent physical-surface decal.
        out_png: destination path. An existing file is reused when its recipe
            matches; a legacy file without a recipe is retained and backfilled.
            A changed recipe refreshes the image, and ``force`` always overwrites it.
        fields: ``mount`` (graspable|container|container_interior|wall|floor),
            ``text``, ``sub``, ``title``, ``icon``, ``shape``, ``number``,
            ``rows``, ``bg``/``fg``/``accent``/``border``/``card`` colours,
            ``width``/``height`` overrides.

    Returns:
        The output path (existing or freshly written).
    """
    out = Path(out_png)
    key = canon_kind(kind)
    if out.exists() and not force:
        recipe_path = _recipe_path(out)
        if not recipe_path.exists():
            _write_recipe(out, key, fields)
            print(f"[label_generator] SKIP (exists): {out}")
            return out
        try:
            existing_recipe = json.loads(recipe_path.read_text())
        except (OSError, ValueError, TypeError):
            existing_recipe = None
        if existing_recipe == _recipe_record(key, fields):
            print(f"[label_generator] SKIP (exists): {out}")
            return out
        print(f"[label_generator] REFRESH (recipe changed): {out}")
    cv = _canvas_for(key, fields)
    if key == "document_scan":
        img = _document_image(cv, fields)
    elif key == "mold_patch":
        img = _mold_patch_image(cv, fields)
    else:
        img = Image.new("RGBA", (cv.w, cv.h), _colour(fields.get("bg"), PAPER))
        _renderers()[key](ImageDraw.Draw(img), cv, fields)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, format="PNG", optimize=False)  # no pnginfo/timestamp -> byte-stable
    _write_recipe(out, key, fields)
    print(f"[label_generator] {key:12s} {cv.w}x{cv.h} -> {out}")
    return out


# --------------------------------------------------------------------------- #
# re-typesetting for a mount whose SHAPE is not the canvas's
# --------------------------------------------------------------------------- #
# A decal is drawn by stretching the whole PNG across the quad it is mounted on, so a canvas
# authored at one aspect and mounted on a quad of another is SHEARED by the ratio. Measured on
# ``GEN-s101``: the graspable canvas is 620x760 (0.82) while the side band on a 75 mm bottle is
# 46.6 x 168.4 mm (0.28), i.e. every glyph compressed to 34% of its width -- 5.9 px wide and 23 px
# tall on screen, a vertical smear no reader can resolve.
#
# The fix is to typeset for the shape the mount actually offers rather than to shrink the mount to
# the artwork: the layout engine fills whatever box it is given (it re-breaks the lines), so a tall
# narrow band gets "EXP" over "2025" in big glyphs instead of one squeezed line. Shrinking the quad
# instead would preserve the glyph shape but throw away most of the label's area.
#
# ``label_generator`` already accepts ``width``/``height`` overrides; what was missing is a way for the
# MOUNTING code -- the only place that knows the quad's metres -- to re-issue a call whose fields it
# never saw. Hence the recipe sidecar: every render records what drew it, so any later consumer can
# re-draw the same label at a different shape without the authoring call being in scope.
def _recipe_path(png: Path | str) -> Path:
    return Path(str(png) + ".recipe.json")


def _short_string_readability(fields: dict[str, object]) -> dict[str, object]:
    """Return the rendered primary-text scale without rendering another image."""
    cv = _canvas_for("short_string", fields)
    lines, font, box = _short_string_primary_layout(cv, fields)
    placements, _bottom = _line_layout(lines, font, box)
    ink_boxes = [placement[2] for placement in placements]
    glyph_heights = [_text_wh(line or "X", font)[1] for line in lines]
    glyph_height_px = min(glyph_heights)
    left = min(rect[0] for rect in ink_boxes)
    top = min(rect[1] for rect in ink_boxes)
    right = max(rect[2] for rect in ink_boxes)
    bottom = max(rect[3] for rect in ink_boxes)
    return {
        "schema": 1,
        "kind": "text",
        "primary_glyph_height_px": glyph_height_px,
        "primary_glyph_height_ratio": round(glyph_height_px / cv.h, 8),
        "primary_line_count": len(lines),
        "primary_text_bbox_norm_xyxy": [
            round(left / cv.w, 8),
            round(top / cv.h, 8),
            round(right / cv.w, 8),
            round(bottom / cv.h, 8),
        ],
    }


def short_string_readability(
    *,
    mount: str,
    text: str,
    title: str = "",
    sub: str = "",
    aspect: float | None = None,
) -> dict[str, object]:
    """Measure the exact layout used by a generated short-string label."""
    fields: dict[str, object] = {
        "mount": mount,
        "title": title,
        "text": text,
        "sub": sub,
    }
    if aspect is not None:
        width, height = canvas_px_for("short_string", fields, aspect)
        fields["width"], fields["height"] = width, height
    return _short_string_readability(fields)


def _readability_recipe(
    kind: str, fields: dict[str, object]
) -> dict[str, object] | None:
    if kind == "short_string":
        return _short_string_readability(fields)
    if kind == "nutrition_report":
        return {
            "schema": 1,
            "kind": "text",
            "primary_glyph_height_ratio": 0.030,
            "primary_line_count": len(_nutrition_rows(fields)) + 3,
            "primary_text_bbox_norm_xyxy": [0.04, 0.06, 0.96, 0.94],
        }
    if kind == "document_scan":
        bbox = fields.get("primary_text_bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(
                "document_scan primary_text_bbox_norm_xyxy must contain four values"
            )
        return {
            "schema": 1,
            "kind": "text",
            "primary_glyph_height_ratio": float(
                fields.get("primary_glyph_height_ratio") or 0.018
            ),
            "primary_line_count": int(fields.get("primary_line_count") or 1),
            "primary_text_bbox_norm_xyxy": [float(value) for value in bbox],
        }
    return None


def _recipe_record(kind: str, fields: dict[str, object]) -> dict[str, object]:
    safe = {key: value for key, value in fields.items() if _json_safe(value)}
    recipe: dict[str, object] = {"kind": kind, "fields": safe}
    if kind == "mold_patch":
        recipe["renderer_version"] = 2
    if kind == "document_scan":
        with resolve_document_source(fields).open("rb") as source:
            recipe["source_sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
    readability = _readability_recipe(kind, fields)
    if readability is not None:
        recipe["readability"] = readability
    return json.loads(json.dumps(recipe, sort_keys=True))


def _write_recipe(out: Path, kind: str, fields: dict[str, object]) -> None:
    """Record (kind, fields) beside the PNG so ``retypeset`` can re-draw it at another aspect.

    Written even when the PNG itself is skipped: the cache holds thousands of labels authored
    before this existed, and a build that reuses one still has to be able to re-shape it.
    """
    try:
        recipe = _recipe_record(kind, fields)
        _recipe_path(out).write_text(json.dumps(recipe, sort_keys=True))
    except OSError as exc:  # a read-only cache must not fail the build
        print(f"[label_generator] recipe sidecar not written for {out}: {exc}")


def _json_safe(value: object) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True


def canvas_px_for(
    kind: str, fields: dict[str, object], aspect: float
) -> tuple[int, int]:
    """Pixel canvas of the given width/height ratio, at this mount's own resolution.

    The SHORT side is held at the mount's default short side, so a re-shaped label carries the same
    texel density as the artwork it replaces -- only the proportions change.
    """
    key = canon_kind(kind)
    cv = _kind_mount_override(
        key, _s(fields, "mount") or _DEFAULT_MOUNT[key]
    ) or _mount_canvas(_s(fields, "mount") or _DEFAULT_MOUNT[key])
    short = min(cv.w, cv.h)
    aspect = max(1e-3, float(aspect))
    w, h = (
        (round(short * aspect), short)
        if aspect >= 1.0
        else (short, round(short / aspect))
    )
    scale = min(1.0, _MAX_CANVAS_PX / max(w, h))
    return max(64, round(w * scale)), max(64, round(h * scale))


_MAX_CANVAS_PX = 4096


def document_texture_aspect(png: str | Path) -> float | None:
    """Return a scanned document's authored aspect, or None for other labels."""
    try:
        recipe = json.loads(_recipe_path(png).read_text())
        if recipe.get("kind") != "document_scan":
            return None
        with Image.open(png) as image:
            if image.height <= 0:
                return None
            return float(image.width) / float(image.height)
    except (OSError, TypeError, ValueError):
        return None


def retypeset(
    src_png: str | Path, out_png: str | Path, aspect: float, *, force: bool = False
) -> Path | None:
    """Re-draw ``src_png``'s label into a canvas of width/height ratio ``aspect``.

    Returns the new path, or None when the source carries no recipe (a hand-authored PNG, or one
    written by a tool that predates the sidecar) -- the caller then mounts the original and accepts
    the shear rather than losing the label.
    """
    try:
        rec = json.loads(_recipe_path(src_png).read_text())
    except (OSError, ValueError):
        return None
    kind = str(rec.get("kind") or "")
    fields = dict(rec.get("fields") or {})
    if not kind:
        return None
    if canon_kind(kind) == "document_scan":
        return Path(src_png)
    w, h = canvas_px_for(kind, fields, aspect)
    fields["width"], fields["height"] = w, h
    return label_generator(kind, out_png, force=force, **fields)


# --------------------------------------------------------------------------- #
# manifest mode
# --------------------------------------------------------------------------- #
_TASK_PREFIX = re.compile(r"^t\d+_[a-z0-9]+_\d+__")


def _clean_stem(png_name: str) -> str:
    return _TASK_PREFIX.sub("", Path(png_name).stem).strip("_")


def _phrase(png_name: str) -> str:
    words = re.sub(r"[_\-]+", " ", _clean_stem(png_name)).split()
    return " ".join(words[:6]).upper() or "LABEL"


def _tail_index(png_name: str) -> int:
    match = re.search(r"(\d+)$", _clean_stem(png_name))
    return int(match.group(1)) if match else 1


def _infer_icon(png_name: str) -> str:
    stem = _clean_stem(png_name).lower()
    idx = _tail_index(png_name) - 1
    if "ghs" in stem or "hazard" in stem:
        return _GHS_ORDER[idx % len(_GHS_ORDER)] if "ghs" in stem else "ghs_exclamation"
    if "wash" in stem:
        return "care_wash40"
    if "hand" in stem:
        return "care_hand"
    if "prof" in stem or "dryclean" in stem:
        return "care_dryclean"
    if "care" in stem or "iso" in stem:
        return _CARE_ORDER[idx % len(_CARE_ORDER)]
    if "allergen" in stem:
        for token, icon in (
            ("nut", "allergen_nuts"),
            ("gluten", "allergen_gluten"),
            ("milk", "allergen_milk"),
            ("soy", "allergen_soy"),
            ("egg", "allergen_egg"),
            ("none", "allergen_none"),
        ):
            if token in stem:
                return icon
        return _ALLERGEN_ORDER[idx % len(_ALLERGEN_ORDER)]
    if "biohaz" in stem or "bio" in stem:
        return "biohazard"
    if "weee" in stem or "crossed_bin" in stem:
        return "weee"
    if "restricted" in stem:
        return "restricted"
    if "recycl" in stem:
        return "recycle"
    if "bin" in stem:
        return _BIN_ORDER[idx % len(_BIN_ORDER)]
    return "fallback"


def _synth_date(png_name: str, fmt: str = "%d %b %Y") -> str:
    """Deterministic stand-in date (stable hash -> day offset), NOT task truth."""
    digest = hashlib.sha256(_clean_stem(png_name).encode("utf-8")).hexdigest()
    return (date(2026, 1, 1) + timedelta(days=int(digest[:8], 16) % 540)).strftime(fmt)


def fields_from_manifest_entry(
    entry: dict[str, object],
) -> tuple[str, dict[str, object]]:
    """Map a ``g0_label_manifest.json`` row to (kind, fields).

    Text content is inferred from ``png_name`` because ``text_or_icon`` usually
    describes the whole family ("3 par cards par_1.png / par_2.png / ..."), not
    the single entry. Inferred numerals, dates and range-card bodies are
    PLACEHOLDERS -- pass explicit fields when the task's answer depends on them.
    """
    name = str(entry.get("png_name", "label.png"))
    kind = canon_kind(str(entry.get("kind", "short_string")))
    fields: dict[str, object] = {"mount": str(entry.get("mounted_on", "container"))}
    phrase = _phrase(name)
    if kind == "pictogram":
        fields.update({"icon": _infer_icon(name), "text": phrase})
    elif kind == "number":
        fields.update(
            {"number": str(_tail_index(name)), "title": re.sub(r"\s*\d+$", "", phrase)}
        )
    elif kind == "date":
        fields.update(
            {"title": "BEST BEFORE", "text": _synth_date(name), "sub": phrase}
        )
        print(
            f"[label_generator][WARN] {name}: date content synthesised; pass --text for the real code"
        )
    elif kind == "range_card":
        fields.update(
            {"title": phrase, "text": str(entry.get("text_or_icon", ""))[:60]}
        )
        print(
            f"[label_generator][WARN] {name}: range_card body is a placeholder; pass --row for real values"
        )
    elif kind == "silhouette":
        shapes = list(_SHAPES)
        fields.update(
            {"shape": shapes[(_tail_index(name) - 1) % len(shapes)], "text": phrase}
        )
    else:
        fields.update({"text": phrase})
    return kind, fields


def make_from_manifest(
    manifest: str | Path,
    only: str,
    out_dir: str | Path | None = None,
    force: bool = False,
    **overrides: object,
) -> Path:
    """Render the single manifest row whose ``png_name`` matches ``only``."""
    data = json.loads(Path(manifest).read_text())
    want = only if only.endswith(".png") else f"{only}.png"
    rows = [r for r in data.get("labels", []) if str(r.get("png_name")) == want]
    if not rows:
        raise KeyError(f"{want!r} is not in {manifest}")
    kind, fields = fields_from_manifest_entry(rows[0])
    fields.update({k: v for k, v in overrides.items() if v not in (None, (), [])})
    root = Path(out_dir) if out_dir else Path(str(data.get("label_dir", ".")))
    return label_generator(kind, root / want, force=force, **fields)


# --------------------------------------------------------------------------- #
# probe contact sheet
# --------------------------------------------------------------------------- #
def _probe_specs() -> list[tuple[str, dict[str, object]]]:
    return [
        (
            "pictogram",
            {"icon": "ghs_flame", "text": "flammable", "sub": "keep from heat"},
        ),
        ("short_string", {"text": "wash 40", "sub": "cotton", "mount": "graspable"}),
        ("number", {"number": "3", "title": "unit", "mount": "container"}),
        ("date", {"title": "best before", "text": "14 MAR 2026", "sub": "lot 4471"}),
        (
            "range_card",
            {
                "title": "par levels",
                "mount": "wall",
                "rows": ["towels: 12", "sheets: 8", "soap: 24", "cups: 40"],
            },
        ),
        ("placard", {"title": "cabinet", "text": "D", "mount": "container"}),
        ("silhouette", {"shape": "wrench", "text": "10 mm spanner", "mount": "wall"}),
        (
            "protocol_sheet",
            {
                "mount": "manual_board",
                "title": "turnaround plans",
                "sub": "one plan is in force today",
                "cols": "plate, towel, mat",
                "plans": [
                    "1|B|D|F",
                    "2|B|F|D",
                    "3|D|B|F",
                    "4|D|F|B",
                    "5|F|B|D",
                    "6|F|D|B",
                ],
                "note": "plan not determined -> hold bay",
            },
        ),
    ]


def manual_designs() -> dict[str, dict[str, object]]:
    """The three posted-procedure sheets the T6 ``prologue_intent`` rows need.

    The LETTERS ARE PLACEHOLDERS. Under ``KEY-ANON``/``SEED-PERM`` the builder
    permutes which unit carries which placard letter per seed, so it must pass
    its own ``plans`` (and write the PNG under a seed-carrying file name --
    ``label_generator`` never overwrites, so a fixed name would freeze seed 0's table
    onto every later seed and silently decouple the sheet from the placards).

    Every sheet holds N plans over the SAME three placard letters, so the answer
    alphabet stays at |plans| and a near-miss cannot pass. No sheet names a
    container key, so nothing in ``redact_in_visible_text`` can leak.

    The row ORDER below is a scramble, not the lexicographic enumeration of the
    permutations: printed in lexicographic order the table can be reconstructed
    from its header alone, which hands a reader the plan it never read. The
    builder must re-scramble per seed.
    """
    return {
        "T6-BEN-01": {
            "title": "shelving plans",
            "sub": "one plan is in force this month",
            "gutter": "plan",
            "cols": "500s, 700s, 900s",
            "plans": ["1|D|F|B", "2|B|D|F", "3|F|B|D", "4|D|B|F", "5|F|D|B", "6|B|F|D"],
            "note": "the call number on the spine decides the shelf",
        },
        "T6-HGS-01": {
            "title": "turnaround plans",
            "sub": "one plan in force today",
            "gutter": "plan",
            "cols": "plate, towel, mat",
            "plans": ["1|C|E|A", "2|E|A|C", "3|A|C|E", "4|E|C|A", "5|C|A|E", "6|A|E|C"],
            "note": "plan not determined -> hold bay",
        },
        "T6-RDIN-02": {
            "title": "prep protocols",
            "sub": "today's protocol is posted",
            "gutter": "prot",
            "cols": "nut, gluten, dairy",
            "plans": ["1|F|A|C", "2|C|F|A", "3|A|C|F", "4|F|C|A", "5|A|F|C", "6|C|A|F"],
            "note": "protocol not determined -> hold cabinet",
        },
    }


def build_manual_sheet(
    out_png: str | Path,
    work_dir: str | Path = "/tmp/manual_probe",
    mount: str = "manual_panel",
) -> Path:
    """Render every ``manual_designs()`` sheet and tile them into a contact sheet."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    made = [
        (
            task,
            label_generator(
                "protocol_sheet",
                work / f"{task.lower().replace('-', '_')}__{mount}.png",
                force=True,
                mount=mount,
                **design,
            ),
        )
        for task, design in manual_designs().items()
    ]
    return _tile(made, out_png, cols=2, cell=760)


def build_probe_sheet(
    out_png: str | Path, work_dir: str | Path = "/tmp/label_image_metrics"
) -> Path:
    """Render one sample of every kind, then tile them into a contact sheet."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    made: list[tuple[str, Path]] = []
    for kind, fields in _probe_specs():
        made.append(
            (kind, label_generator(kind, work / f"{kind}.png", force=True, **fields))
        )
    return _tile(made, out_png, cols=4, cell=420)


def _tile(
    made: Sequence[tuple[str, Path]], out_png: str | Path, cols: int, cell: int
) -> Path:
    pad, cap = 16, 40
    rows = (len(made) + cols - 1) // cols
    sheet = Image.new(
        "RGBA", (cols * (cell + pad) + pad, rows * (cell + cap + pad) + pad), WHITE
    )
    draw = ImageDraw.Draw(sheet)
    for i, (kind, path) in enumerate(made):
        img = Image.open(path).convert("RGBA")
        scale = min(cell / img.width, cell / img.height)
        thumb = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.LANCZOS,
        )
        ox = pad + (i % cols) * (cell + pad) + (cell - thumb.width) // 2
        oy = pad + (i // cols) * (cell + cap + pad) + (cell - thumb.height) // 2
        sheet.paste(thumb, (ox, oy), thumb)
        label = f"{kind}  {img.width}x{img.height}"
        box = (
            float(pad + (i % cols) * (cell + pad)),
            float(pad + (i // cols) * (cell + cap + pad) + cell),
            float(pad + (i % cols) * (cell + pad) + cell),
            float(pad + (i // cols) * (cell + cap + pad) + cell + cap),
        )
        lines, font = _fit(label, box[2] - box[0], cap * 0.8, 30, max_lines=1)
        _draw_lines(draw, lines, font, box, INK)
    out = Path(out_png)
    sheet.save(out, format="PNG", optimize=False)
    print(f"[label_generator] probe sheet {sheet.width}x{sheet.height} -> {out}")
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Deterministic benchmark label-PNG generator."
    )
    ap.add_argument(
        "--kind",
        help=f"one of {sorted({*_renderers(), 'document_scan', 'mold_patch'})}",
    )
    ap.add_argument("--out", help="destination PNG")
    ap.add_argument(
        "--mount",
        help="graspable|container|container_interior|wall|floor|manual_board|manual_panel",
    )
    ap.add_argument("--text", help="main string (or the date for --kind date)")
    ap.add_argument("--sub", help="secondary line")
    ap.add_argument("--title", help="title-bar / caption text")
    ap.add_argument(
        "--icon", help=f"pictogram key, e.g. {sorted(_ghs_icons())[:3]} ..."
    )
    ap.add_argument("--shape", help=f"silhouette shape, one of {sorted(_SHAPES)}")
    ap.add_argument("--number", help="numeral for --kind number")
    ap.add_argument(
        "--row",
        action="append",
        default=[],
        help="range_card row 'LEFT: RIGHT' (repeatable)",
    )
    ap.add_argument(
        "--plan",
        action="append",
        default=[],
        help="protocol_sheet row '1|B|D|F' or '1: B D F' (repeatable)",
    )
    ap.add_argument("--cols", help="protocol_sheet column headers, comma separated")
    ap.add_argument("--gutter", help="protocol_sheet left-column header (default PLAN)")
    ap.add_argument("--note", help="protocol_sheet footer banner (the fallback rule)")
    ap.add_argument("--bg")
    ap.add_argument("--fg")
    ap.add_argument("--accent")
    ap.add_argument("--border")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--force", action="store_true", help="overwrite an existing PNG")
    ap.add_argument("--from-manifest", help="g0_label_manifest.json")
    ap.add_argument("--only", help="png_name to render from the manifest")
    ap.add_argument("--out-dir", help="destination directory for --from-manifest")
    ap.add_argument("--list-icons", action="store_true")
    ap.add_argument(
        "--probe-sheet", help="render one sample per kind and tile them into this PNG"
    )
    ap.add_argument(
        "--manual-sheet",
        help="render every manual_designs() protocol sheet and tile them into this PNG",
    )
    return ap


def _cli_fields(args: argparse.Namespace) -> dict[str, object]:
    named = (
        "mount",
        "text",
        "sub",
        "title",
        "icon",
        "shape",
        "number",
        "bg",
        "fg",
        "accent",
        "border",
        "width",
        "height",
        "cols",
        "gutter",
        "note",
    )
    fields: dict[str, object] = {
        k: getattr(args, k) for k in named if getattr(args, k) is not None
    }
    if args.row:
        fields["rows"] = args.row
    if args.plan:
        fields["plans"] = args.plan
    return fields


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.list_icons:
        print("GHS (drawn in the red diamond):", " ".join(sorted(_ghs_icons())))
        print("plain:", " ".join(sorted(_plain_icons())))
        print("silhouette shapes:", " ".join(sorted(_SHAPES)))
        return 0
    if args.probe_sheet:
        build_probe_sheet(args.probe_sheet)
        return 0
    if args.manual_sheet:
        build_manual_sheet(args.manual_sheet, mount=args.mount or "manual_panel")
        return 0
    if args.from_manifest:
        if not args.only:
            print(
                "[label_generator] --from-manifest needs --only <png_name>",
                file=sys.stderr,
            )
            return 2
        make_from_manifest(
            args.from_manifest, args.only, args.out_dir, args.force, **_cli_fields(args)
        )
        return 0
    if not (args.kind and args.out):
        print(
            "[label_generator] need --kind and --out (or --from-manifest/--probe-sheet)",
            file=sys.stderr,
        )
        return 2
    label_generator(args.kind, args.out, force=args.force, **_cli_fields(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
