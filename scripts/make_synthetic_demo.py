"""Build the demo synthetic floor plan dataset.

Generates ``N`` procedurally-generated floor plans, writes them as PNGs to
``demo/images/`` and a single COCO JSON to ``demo/annotations/demo.coco.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.utils.coco import CocoImage, save_coco  # noqa: E402
from backend.utils.image import save_image  # noqa: E402
from backend.utils.synth import generate_layout  # noqa: E402


def main(num: int, out_dir: Path, seed: int) -> None:
    images_dir = out_dir / "images"
    annotations_dir = out_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    coco_records: list[CocoImage] = []
    for idx in range(num):
        # Vary canvas / room count for diversity.
        width = 480 + (idx % 4) * 80
        height = 360 + (idx % 3) * 60
        rooms = 3 + (idx % 5)
        image, polygons = generate_layout(
            width=width, height=height, num_rooms=rooms, seed=seed + idx
        )
        file_name = f"plan_{idx:03d}.png"
        save_image(image, images_dir / file_name)
        coco_records.append(
            CocoImage(file_name=file_name, width=width, height=height, polygons=polygons)
        )
        print(f"Saved {file_name} ({width}×{height}, {rooms} rooms, {len(polygons)} polygons)")

    out_path = annotations_dir / "demo.coco.json"
    save_coco(coco_records, out_path)
    print(f"\nWrote COCO annotations: {out_path}")
    print(f"Total: {len(coco_records)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num", type=int, default=12, help="Number of demo plans (default: 12)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "demo",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()
    main(num=args.num, out_dir=args.out_dir, seed=args.seed)
