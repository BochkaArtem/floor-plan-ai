"""Render a small set of generated floor plans as demonstration output.

These illustrate how the ``/api/generate`` endpoint maps user conditions
(boundary shape, room types, area, num_rooms) to a rendered raster floor
plan with semantic labels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ml.generation import GenerationConditions, get_default_generator  # noqa: E402
from backend.utils.image import save_image  # noqa: E402

EXAMPLES = [
    {
        "name": "01_rect_classic_3rooms",
        "width": 640, "height": 480, "num_rooms": 4,
        "boundary_shape": "rect",
        "room_types": ["hall", "living", "kitchen", "bedroom"],
        "seed": 11,
    },
    {
        "name": "02_L_shape_5rooms",
        "width": 720, "height": 540, "num_rooms": 5,
        "boundary_shape": "L",
        "room_types": ["hall", "living", "kitchen", "bedroom", "bathroom"],
        "seed": 23,
    },
    {
        "name": "03_T_shape_family",
        "width": 800, "height": 560, "num_rooms": 6,
        "boundary_shape": "T",
        "room_types": ["hall", "living", "kitchen", "bedroom", "bedroom", "bathroom"],
        "seed": 41,
    },
    {
        "name": "04_U_shape_with_balcony",
        "width": 800, "height": 600, "num_rooms": 7,
        "boundary_shape": "U",
        "room_types": ["hall", "living", "kitchen", "bedroom", "bedroom", "bathroom", "balcony"],
        "seed": 55,
    },
    {
        "name": "05_plus_cross_5rooms",
        "width": 720, "height": 720, "num_rooms": 5,
        "boundary_shape": "plus",
        "room_types": ["hall", "living", "kitchen", "bedroom", "bathroom"],
        "seed": 70,
    },
    {
        "name": "06_random_polyomino",
        "width": 800, "height": 600, "num_rooms": 5,
        "boundary_shape": "random",
        "room_types": ["hall", "living", "kitchen", "bedroom", "bathroom"],
        "seed": 91,
    },
    {
        "name": "07_studio_2rooms",
        "width": 480, "height": 360, "num_rooms": 2,
        "boundary_shape": "rect",
        "room_types": ["hall", "living"],
        "seed": 13,
    },
    {
        "name": "08_three_bed_apartment",
        "width": 800, "height": 560, "num_rooms": 7,
        "boundary_shape": "L",
        "room_types": ["hall", "living", "kitchen", "bedroom", "bedroom", "bedroom", "bathroom"],
        "seed": 84,
    },
]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = get_default_generator()
    print(f"Using generator: {generator.name}")

    for ex in EXAMPLES:
        cond = GenerationConditions(
            width=ex["width"],
            height=ex["height"],
            num_rooms=ex["num_rooms"],
            seed=ex.get("seed"),
            area_m2=ex.get("area_m2"),
            boundary_shape=ex.get("boundary_shape", "rect"),
            room_types=ex.get("room_types", []),
        )
        image, polys = generator.generate(cond)
        out_path = out_dir / f"{ex['name']}.png"
        save_image(image, out_path)
        room_polys = [p for p in polys if p.category == "room"]
        types = [p.subcategory for p in room_polys]
        print(f"  → {out_path}  rooms={len(room_polys)} types={types}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "demo" / "generated")
    args = parser.parse_args()
    main(args.out_dir)
