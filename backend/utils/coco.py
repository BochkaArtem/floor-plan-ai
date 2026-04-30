"""COCO format helpers for floor plan annotations.

We use the canonical COCO "object detection" schema (polygons + bbox).
Categories: wall, window, door, room.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .image import CLASS_NAMES

_now = datetime.now(timezone.utc)
COCO_INFO = {
    "description": "Floor Plan AI annotations",
    "version": "0.1",
    "year": _now.year,
    "contributor": "floor-plan-ai",
    "date_created": _now.isoformat(timespec="seconds"),
}


def default_categories() -> list[dict[str, Any]]:
    return [
        {"id": idx + 1, "name": name, "supercategory": "floorplan"}
        for idx, name in enumerate(CLASS_NAMES)
    ]


@dataclass
class Polygon:
    """A polygon annotation in image coordinates (x, y pixels)."""

    points: list[tuple[float, float]]
    category: str

    def flatten(self) -> list[float]:
        flat: list[float] = []
        for x, y in self.points:
            flat.extend([float(x), float(y)])
        return flat

    def bbox(self) -> tuple[float, float, float, float]:
        if not self.points:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)
        return (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))

    def area(self) -> float:
        # Shoelace formula.
        if len(self.points) < 3:
            return 0.0
        s = 0.0
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            s += x1 * y2 - x2 * y1
        return abs(s) / 2.0


@dataclass
class CocoImage:
    file_name: str
    width: int
    height: int
    polygons: list[Polygon] = field(default_factory=list)


def build_coco_dict(images: list[CocoImage]) -> dict[str, Any]:
    categories = default_categories()
    cat_id = {c["name"]: c["id"] for c in categories}

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    next_ann_id = 1

    for image_id, img in enumerate(images, start=1):
        coco_images.append(
            {
                "id": image_id,
                "file_name": img.file_name,
                "width": img.width,
                "height": img.height,
            }
        )
        for poly in img.polygons:
            if poly.category not in cat_id:
                continue
            coco_annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": cat_id[poly.category],
                    "segmentation": [poly.flatten()],
                    "bbox": list(poly.bbox()),
                    "area": poly.area(),
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

    return {
        "info": COCO_INFO,
        "licenses": [],
        "categories": categories,
        "images": coco_images,
        "annotations": coco_annotations,
    }


def save_coco(images: list[CocoImage], path: str | Path) -> None:
    data = build_coco_dict(images)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


def load_coco(path: str | Path) -> list[CocoImage]:
    data = json.loads(Path(path).read_text())
    cat_by_id = {c["id"]: c["name"] for c in data.get("categories", [])}
    images_by_id: dict[int, CocoImage] = {}
    for img in data.get("images", []):
        images_by_id[img["id"]] = CocoImage(
            file_name=img["file_name"], width=img["width"], height=img["height"]
        )
    for ann in data.get("annotations", []):
        cat_name = cat_by_id.get(ann["category_id"])
        if cat_name is None:
            continue
        seg = ann.get("segmentation") or []
        if not seg:
            continue
        flat = seg[0]
        pts = [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]
        target = images_by_id.get(ann["image_id"])
        if target is not None:
            target.polygons.append(Polygon(points=pts, category=cat_name))
    return list(images_by_id.values())
