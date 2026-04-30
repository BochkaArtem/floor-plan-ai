"""Procedural floor plan generator.

Produces:
    * a rasterised RGB image of a floor plan,
    * a list of polygon annotations (walls, windows, doors, rooms).

Used both as a fast-path baseline generator for the ``/generate`` endpoint
and as a source of synthetic demo data for COCO export and model training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .coco import Polygon

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
