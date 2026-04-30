"""Render a small set of generated floor plans as demonstration output.

These illustrate how the ``/api/generate`` endpoint maps user conditions
(area, num_rooms, boundary shape) to a rendered raster floor plan.
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
    {"name": "studio_small", "width": 480, "height": 360, "num_rooms": 2, "seed": 11},
    {"name": "two_room_60m2", "width": 640, "height": 480, "num_rooms": 3, "area_m2": 60, "seed": 22},
    {"name": "three_room_compact", "width": 560, "height": 420, "num_rooms": 4, "seed": 33},
    {"name": "four_room_l_shape", "width": 720, "height": 540, "num_rooms": 5, "boundary_shape": "L", "seed": 44},
    {"name": "five_room_large", "width": 800, "height": 540, "num_rooms": 6, "area_m2": 95, "seed": 55},
    {"name": "six_room_large", "width": 800, "height": 600, "num_rooms": 7, "seed": 66},
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
        )
        image, _ = generator.generate(cond)
        out_path = out_dir / f"{ex['name']}.png"
        save_image(image, out_path)
        print(f"  → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "demo" / "generated")
    args = parser.parse_args()
    main(args.out_dir)
