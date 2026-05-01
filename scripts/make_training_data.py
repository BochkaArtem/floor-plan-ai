"""Synthesise training data for the segmentation U-Net and the conditional
mask generator.

For each sample we produce three artefacts:

* ``images/{idx:04d}.png`` — RGB rendering of the floor plan (used as input
  to the segmentation U-Net),
* ``masks/{idx:04d}.png`` — uint8 indexed mask with class labels matching
  :data:`backend.utils.synth.MASK_*` (used as ground truth for both
  networks),
* ``conditions/{idx:04d}.json`` — boundary shape, requested room types and
  num_rooms (used as conditioning for the generation network).

Boundary masks are recovered at training time from the index masks by
treating any non-background pixel as inside the apartment.

Usage:

    python -m scripts.make_training_data --out data/synthetic --n 1500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from backend.utils.layout_planner import ROOM_TYPES
from backend.utils.synth import (
    generate_semantic_layout,
    planned_to_mask,
)

SHAPES = ["rect", "L", "T", "U", "plus", "random"]


def _sample_room_types(rng: random.Random, n: int) -> list[str]:
    """Pick a plausible set of requested types of length ``n``."""
    base = ["hall", "living"]
    rest = ["kitchen", "bedroom", "bathroom", "bedroom", "balcony", "bedroom"]
    rng.shuffle(rest)
    plan = (base + rest)[:n]
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--n", type=int, default=1500, help="Number of samples")
    parser.add_argument("--width", type=int, default=128, help="Render width")
    parser.add_argument("--height", type=int, default=128, help="Render height")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "conditions").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    written = 0
    idx = 0
    while written < args.n:
        idx += 1
        sample_seed = rng.randint(0, 10_000_000)
        shape = rng.choice(SHAPES)
        num_rooms = rng.randint(3, 7)
        types = _sample_room_types(rng, num_rooms)
        try:
            rgb, _polys, plan = generate_semantic_layout(
                width=args.width,
                height=args.height,
                num_rooms=num_rooms,
                boundary_shape=shape,
                room_types=types,
                seed=sample_seed,
                draw_labels=False,
            )
            mask = planned_to_mask(plan)
        except Exception as exc:  # noqa: BLE001
            # Skip degenerate samples (very rare).
            print(f"  ! skipping idx={idx}: {exc}")
            continue

        Image.fromarray(rgb).save(out / "images" / f"{written:05d}.png")
        Image.fromarray(mask, mode="L").save(out / "masks" / f"{written:05d}.png")
        with (out / "conditions" / f"{written:05d}.json").open("w") as f:
            json.dump(
                {
                    "boundary_shape": plan.boundary_shape,
                    "num_rooms": num_rooms,
                    "room_types": types,
                    "seed": sample_seed,
                    "available_types": list(ROOM_TYPES),
                },
                f,
            )
        written += 1
        if written % 100 == 0:
            print(f"  generated {written}/{args.n}")
    print(f"Done. Wrote {written} samples to {out}")


if __name__ == "__main__":
    main()
