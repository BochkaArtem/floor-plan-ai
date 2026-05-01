"""Procedural floor plan generator.

Produces:
    * a rasterised RGB image of a floor plan,
    * a list of polygon annotations (walls, windows, doors, rooms),
    * (via :func:`planned_to_outputs`) a semantic 4-class mask suitable for
      training the segmentation U-Net.

Used both as a fast-path baseline generator for the ``/generate`` endpoint
and as a source of synthetic data for COCO export and model training.

The :func:`generate_layout` entrypoint is kept for backward compatibility;
new code should use :func:`generate_semantic_layout` which goes through the
semantic planner and supports varied boundary shapes + room types.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .coco import Polygon
from .layout_planner import (
    ROOM_TYPE_COLORS,
    PlannedLayout,
    plan_layout,
)

ROOM_PALETTE = [
    (255, 224, 178),
    (197, 225, 165),
    (179, 229, 252),
    (248, 187, 208),
    (215, 204, 200),
    (255, 245, 157),
    (255, 205, 210),
    (197, 202, 233),
]


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def area(self) -> float:
        return self.w * self.h

    def as_polygon(self) -> list[tuple[float, float]]:
        return [(self.x, self.y), (self.x2, self.y), (self.x2, self.y2), (self.x, self.y2)]


def split_rect(rect: Rect, num_rooms: int, rng: random.Random, min_side: float = 80.0) -> list[Rect]:
    """Recursively split a rectangle into ``num_rooms`` non-overlapping rectangles."""
    if num_rooms <= 1:
        return [rect]

    # Decide split orientation: prefer the longer side.
    horizontal = rect.w >= rect.h
    if rect.w < min_side * 2 and rect.h >= min_side * 2:
        horizontal = False
    elif rect.h < min_side * 2 and rect.w >= min_side * 2:
        horizontal = True
    elif rect.w < min_side * 2 and rect.h < min_side * 2:
        return [rect]  # cannot split further

    # Roughly proportional split, biased by num_rooms.
    left_count = max(1, num_rooms // 2)
    right_count = num_rooms - left_count
    ratio = left_count / num_rooms
    jitter = rng.uniform(-0.1, 0.1)
    ratio = max(0.3, min(0.7, ratio + jitter))

    if horizontal:
        cut = rect.w * ratio
        left = Rect(rect.x, rect.y, cut, rect.h)
        right = Rect(rect.x + cut, rect.y, rect.w - cut, rect.h)
    else:
        cut = rect.h * ratio
        left = Rect(rect.x, rect.y, rect.w, cut)
        right = Rect(rect.x, rect.y + cut, rect.w, rect.h - cut)

    return split_rect(left, left_count, rng, min_side) + split_rect(right, right_count, rng, min_side)


def _rect_overlap_segment(a: Rect, b: Rect) -> tuple[str, float, float, float] | None:
    """If rectangles ``a`` and ``b`` share part of an edge, return that segment.

    Returns ``(orientation, fixed, lo, hi)`` where orientation is "v" (vertical
    shared edge at x=fixed, y in [lo, hi]) or "h" (horizontal shared edge at
    y=fixed, x in [lo, hi]). Returns ``None`` if they don't share an edge.
    """
    eps = 0.5
    # Vertical shared edge.
    if abs(a.x2 - b.x) < eps:
        lo = max(a.y, b.y)
        hi = min(a.y2, b.y2)
        if hi - lo > 30:
            return ("v", a.x2, lo, hi)
    if abs(b.x2 - a.x) < eps:
        lo = max(a.y, b.y)
        hi = min(a.y2, b.y2)
        if hi - lo > 30:
            return ("v", a.x, lo, hi)
    # Horizontal shared edge.
    if abs(a.y2 - b.y) < eps:
        lo = max(a.x, b.x)
        hi = min(a.x2, b.x2)
        if hi - lo > 30:
            return ("h", a.y2, lo, hi)
    if abs(b.y2 - a.y) < eps:
        lo = max(a.x, b.x)
        hi = min(a.x2, b.x2)
        if hi - lo > 30:
            return ("h", a.y, lo, hi)
    return None


def _door_polygon(seg: tuple[str, float, float, float], size: float = 30.0) -> Polygon:
    orientation, fixed, lo, hi = seg
    centre = (lo + hi) / 2.0
    half = size / 2.0
    thickness = 6.0
    if orientation == "v":
        pts = [
            (fixed - thickness, centre - half),
            (fixed + thickness, centre - half),
            (fixed + thickness, centre + half),
            (fixed - thickness, centre + half),
        ]
    else:
        pts = [
            (centre - half, fixed - thickness),
            (centre + half, fixed - thickness),
            (centre + half, fixed + thickness),
            (centre - half, fixed + thickness),
        ]
    return Polygon(points=pts, category="door")


def _window_polygons(rect: Rect, boundary: Rect, rng: random.Random) -> list[Polygon]:
    """Place 1-2 windows on outer walls of ``rect`` (i.e. walls coinciding with the boundary)."""
    eps = 0.5
    candidates: list[tuple[str, float, float, float]] = []
    if abs(rect.x - boundary.x) < eps:
        candidates.append(("v", rect.x, rect.y + 10, rect.y2 - 10))
    if abs(rect.x2 - boundary.x2) < eps:
        candidates.append(("v", rect.x2, rect.y + 10, rect.y2 - 10))
    if abs(rect.y - boundary.y) < eps:
        candidates.append(("h", rect.y, rect.x + 10, rect.x2 - 10))
    if abs(rect.y2 - boundary.y2) < eps:
        candidates.append(("h", rect.y2, rect.x + 10, rect.x2 - 10))

    if not candidates:
        return []

    polys: list[Polygon] = []
    n = rng.randint(1, min(2, len(candidates)))
    for orientation, fixed, lo, hi in rng.sample(candidates, k=n):
        if hi - lo < 40:
            continue
        size = rng.uniform(40, min(80, hi - lo - 10))
        centre = rng.uniform(lo + size / 2.0 + 5, hi - size / 2.0 - 5)
        thickness = 5.0
        if orientation == "v":
            pts = [
                (fixed - thickness, centre - size / 2),
                (fixed + thickness, centre - size / 2),
                (fixed + thickness, centre + size / 2),
                (fixed - thickness, centre + size / 2),
            ]
        else:
            pts = [
                (centre - size / 2, fixed - thickness),
                (centre + size / 2, fixed - thickness),
                (centre + size / 2, fixed + thickness),
                (centre - size / 2, fixed + thickness),
            ]
        polys.append(Polygon(points=pts, category="window"))
    return polys


def _wall_polygons(rooms: list[Rect], thickness: float = 6.0) -> list[Polygon]:
    """Build wall polygons as thin rectangles around every room edge.

    The simple approach is sufficient for visualisation; downstream consumers
    treat overlapping wall polygons as a union.
    """
    polys: list[Polygon] = []
    t = thickness / 2.0
    for r in rooms:
        # Top, bottom, left, right walls as oriented rectangles.
        polys.append(
            Polygon(
                points=[(r.x - t, r.y - t), (r.x2 + t, r.y - t), (r.x2 + t, r.y + t), (r.x - t, r.y + t)],
                category="wall",
            )
        )
        polys.append(
            Polygon(
                points=[
                    (r.x - t, r.y2 - t),
                    (r.x2 + t, r.y2 - t),
                    (r.x2 + t, r.y2 + t),
                    (r.x - t, r.y2 + t),
                ],
                category="wall",
            )
        )
        polys.append(
            Polygon(
                points=[(r.x - t, r.y - t), (r.x + t, r.y - t), (r.x + t, r.y2 + t), (r.x - t, r.y2 + t)],
                category="wall",
            )
        )
        polys.append(
            Polygon(
                points=[
                    (r.x2 - t, r.y - t),
                    (r.x2 + t, r.y - t),
                    (r.x2 + t, r.y2 + t),
                    (r.x2 - t, r.y2 + t),
                ],
                category="wall",
            )
        )
    return polys


def generate_layout(
    width: int = 640,
    height: int = 480,
    num_rooms: int = 4,
    seed: int | None = None,
    margin: int = 20,
) -> tuple[np.ndarray, list[Polygon]]:
    """Generate a synthetic apartment floor plan.

    Returns
    -------
    (image, polygons)
        ``image`` is an RGB ``np.uint8`` array, ``polygons`` is the COCO-style
        annotation in the same coordinate system.
    """
    rng = random.Random(seed)
    boundary = Rect(margin, margin, width - 2 * margin, height - 2 * margin)
    rooms = split_rect(boundary, max(1, num_rooms), rng)

    polygons: list[Polygon] = []
    for rect in rooms:
        polygons.append(Polygon(points=rect.as_polygon(), category="room"))

    # Walls.
    polygons.extend(_wall_polygons(rooms))

    # Doors between adjacent rooms (one per shared edge).
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            seg = _rect_overlap_segment(rooms[i], rooms[j])
            if seg is not None:
                polygons.append(_door_polygon(seg))

    # Windows on outer walls.
    for rect in rooms:
        polygons.extend(_window_polygons(rect, boundary, rng))

    image = render_layout(width, height, rooms, polygons)
    return image, polygons


def render_layout(
    width: int,
    height: int,
    rooms: list[Rect],
    polygons: list[Polygon],
) -> np.ndarray:
    """Render the floor plan as an RGB image."""
    img = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Filled rooms first (so walls/doors paint over them).
    for idx, rect in enumerate(rooms):
        color = ROOM_PALETTE[idx % len(ROOM_PALETTE)]
        draw.rectangle([rect.x, rect.y, rect.x2, rect.y2], fill=color, outline=None)

    # Walls.
    for poly in polygons:
        if poly.category != "wall":
            continue
        draw.polygon(poly.points, fill=(40, 40, 40))

    # Doors as orange gaps with an arc.
    for poly in polygons:
        if poly.category != "door":
            continue
        bbox = poly.bbox()
        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], fill=(245, 245, 245))
        draw.rectangle([x, y, x + w, y + h], outline=(220, 100, 60), width=2)

    # Windows as blue strokes.
    for poly in polygons:
        if poly.category != "window":
            continue
        draw.polygon(poly.points, fill=(60, 140, 220))

    return np.array(img)


# ---------------------------------------------------------------------------
# Semantic (planner-driven) generation --------------------------------------
# ---------------------------------------------------------------------------


# 4-class index mask used to train the segmentation U-Net.
# Background = 0, wall = 1, window = 2, door = 3, room = 4.
# (We keep "room" separate from background so empty corridor space contrasts.)
MASK_BACKGROUND = 0
MASK_WALL = 1
MASK_WINDOW = 2
MASK_DOOR = 3
MASK_ROOM = 4


def _wall_strokes_from_planned(layout: PlannedLayout, thickness: float = 6.0) -> list[Polygon]:
    """Build wall polygons for a planned layout: every room edge becomes a wall.

    Walls are doubled where rooms share an edge (visually identical but it
    keeps the data simple — downstream rasterisation merges them).
    """
    polys: list[Polygon] = []
    t = thickness / 2.0
    for room in layout.rooms:
        r = room.rect
        polys.extend(
            [
                Polygon(
                    points=[
                        (r.x - t, r.y - t),
                        (r.x2 + t, r.y - t),
                        (r.x2 + t, r.y + t),
                        (r.x - t, r.y + t),
                    ],
                    category="wall",
                ),
                Polygon(
                    points=[
                        (r.x - t, r.y2 - t),
                        (r.x2 + t, r.y2 - t),
                        (r.x2 + t, r.y2 + t),
                        (r.x - t, r.y2 + t),
                    ],
                    category="wall",
                ),
                Polygon(
                    points=[
                        (r.x - t, r.y - t),
                        (r.x + t, r.y - t),
                        (r.x + t, r.y2 + t),
                        (r.x - t, r.y2 + t),
                    ],
                    category="wall",
                ),
                Polygon(
                    points=[
                        (r.x2 - t, r.y - t),
                        (r.x2 + t, r.y - t),
                        (r.x2 + t, r.y2 + t),
                        (r.x2 - t, r.y2 + t),
                    ],
                    category="wall",
                ),
            ]
        )
    return polys


def _door_polygon_from_spec(spec, thickness: float = 6.0) -> Polygon:
    """Door represented as a thin oriented rectangle straddling a wall."""
    if spec.orientation == "v":
        pts = [
            (spec.fixed - thickness, spec.lo),
            (spec.fixed + thickness, spec.lo),
            (spec.fixed + thickness, spec.hi),
            (spec.fixed - thickness, spec.hi),
        ]
    else:
        pts = [
            (spec.lo, spec.fixed - thickness),
            (spec.hi, spec.fixed - thickness),
            (spec.hi, spec.fixed + thickness),
            (spec.lo, spec.fixed + thickness),
        ]
    return Polygon(points=pts, category="door")


def _window_polygon_from_spec(spec, thickness: float = 5.0) -> Polygon:
    if spec.orientation == "v":
        pts = [
            (spec.fixed - thickness, spec.lo),
            (spec.fixed + thickness, spec.lo),
            (spec.fixed + thickness, spec.hi),
            (spec.fixed - thickness, spec.hi),
        ]
    else:
        pts = [
            (spec.lo, spec.fixed - thickness),
            (spec.hi, spec.fixed - thickness),
            (spec.hi, spec.fixed + thickness),
            (spec.lo, spec.fixed + thickness),
        ]
    return Polygon(points=pts, category="window")


def planned_to_polygons(layout: PlannedLayout) -> list[Polygon]:
    """Convert a :class:`PlannedLayout` to COCO-style polygons.

    Categories are the canonical 4-class set (wall/window/door/room). Room
    polygons additionally carry a ``subcategory`` indicating their semantic
    type (kitchen/living/etc) so consumers can render labels.
    """
    polys: list[Polygon] = []
    for room in layout.rooms:
        polys.append(
            Polygon(
                points=room.rect.as_polygon(),
                category="room",
                subcategory=room.type,
            )
        )
    polys.extend(_wall_strokes_from_planned(layout))
    polys.extend(_door_polygon_from_spec(d) for d in layout.doors)
    polys.extend(_window_polygon_from_spec(w) for w in layout.windows)
    return polys


def _try_label_font(size: int) -> ImageFont.ImageFont:
    """Best-effort attempt to load a TrueType font; fall back to default bitmap font."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_planned(layout: PlannedLayout, draw_labels: bool = True) -> np.ndarray:
    """Render a :class:`PlannedLayout` to an RGB image.

    Each room is filled with a colour determined by its semantic type and
    optionally labelled with its type name.
    """
    img = Image.new("RGB", (layout.width, layout.height), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)

    # Rooms first (so walls/doors paint over them).
    for room in layout.rooms:
        color = ROOM_TYPE_COLORS.get(room.type, ROOM_TYPE_COLORS["room"])
        r = room.rect
        draw.rectangle([r.x, r.y, r.x2, r.y2], fill=color, outline=None)

    # Walls.
    for poly in _wall_strokes_from_planned(layout):
        draw.polygon(poly.points, fill=(40, 40, 40))

    # Doors as orange gaps.
    for d in layout.doors:
        spec = d
        if spec.orientation == "v":
            x = spec.fixed - 5
            draw.rectangle([x, spec.lo, x + 10, spec.hi], fill=(245, 245, 245))
            draw.rectangle([x, spec.lo, x + 10, spec.hi], outline=(220, 100, 60), width=2)
        else:
            y = spec.fixed - 5
            draw.rectangle([spec.lo, y, spec.hi, y + 10], fill=(245, 245, 245))
            draw.rectangle([spec.lo, y, spec.hi, y + 10], outline=(220, 100, 60), width=2)

    # Windows as blue strokes.
    for w in layout.windows:
        if w.orientation == "v":
            draw.rectangle(
                [w.fixed - 5, w.lo, w.fixed + 5, w.hi],
                fill=(60, 140, 220),
            )
        else:
            draw.rectangle(
                [w.lo, w.fixed - 5, w.hi, w.fixed + 5],
                fill=(60, 140, 220),
            )

    # Labels last.
    if draw_labels:
        font = _try_label_font(14)
        for room in layout.rooms:
            label = room.type
            cx, cy = room.rect.centroid()
            try:
                bbox = draw.textbbox((cx, cy), label, font=font, anchor="mm")
                draw.rectangle(
                    [bbox[0] - 3, bbox[1] - 2, bbox[2] + 3, bbox[3] + 2],
                    fill=(255, 255, 255),
                    outline=(120, 120, 120),
                )
                draw.text((cx, cy), label, fill=(60, 60, 60), font=font, anchor="mm")
            except (TypeError, AttributeError):
                # Older PIL doesn't support anchor — fall back to top-left placement.
                draw.text((room.rect.x + 4, room.rect.y + 4), label, fill=(60, 60, 60), font=font)

    return np.array(img)


def planned_to_mask(layout: PlannedLayout) -> np.ndarray:
    """Return a 4-class index mask (uint8 H×W) for the planned layout.

    Class indices match :data:`MASK_BACKGROUND`/``WALL``/``WINDOW``/``DOOR``/``ROOM``.
    """
    mask = Image.new("L", (layout.width, layout.height), color=MASK_BACKGROUND)
    draw = ImageDraw.Draw(mask)

    for room in layout.rooms:
        r = room.rect
        draw.rectangle([r.x, r.y, r.x2, r.y2], fill=MASK_ROOM)

    for poly in _wall_strokes_from_planned(layout):
        draw.polygon(poly.points, fill=MASK_WALL)

    for d in layout.doors:
        if d.orientation == "v":
            draw.rectangle([d.fixed - 6, d.lo, d.fixed + 6, d.hi], fill=MASK_DOOR)
        else:
            draw.rectangle([d.lo, d.fixed - 6, d.hi, d.fixed + 6], fill=MASK_DOOR)

    for w in layout.windows:
        if w.orientation == "v":
            draw.rectangle([w.fixed - 6, w.lo, w.fixed + 6, w.hi], fill=MASK_WINDOW)
        else:
            draw.rectangle([w.lo, w.fixed - 6, w.hi, w.fixed + 6], fill=MASK_WINDOW)

    return np.array(mask, dtype=np.uint8)


def generate_semantic_layout(
    width: int = 640,
    height: int = 480,
    num_rooms: int = 5,
    boundary_shape: str = "auto",
    room_types: list[str] | None = None,
    seed: int | None = None,
    draw_labels: bool = True,
) -> tuple[np.ndarray, list[Polygon], PlannedLayout]:
    """Top-level helper used by API + dataset generation.

    Returns ``(rgb, polygons, plan)``. The plan is exposed so callers can
    inspect / serialise the high-level structure (boundary shape, room types,
    door/window placements).
    """
    plan = plan_layout(
        width=width,
        height=height,
        num_rooms=num_rooms,
        boundary_shape=boundary_shape,
        room_types=room_types,
        seed=seed,
    )
    rgb = render_planned(plan, draw_labels=draw_labels)
    polys = planned_to_polygons(plan)
    return rgb, polys, plan
