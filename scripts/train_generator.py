"""Train the conditional pix2pix generator on synthetic floor plans.

Pairs a 2-channel conditioning input (boundary mask + room-count plane) with
the rendered floor plan as the target. The model is taken from
``backend.ml.generation.Pix2PixUNet``.

Outputs ``models/generator.pt`` which is automatically picked up by
``backend.ml.generation.Pix2PixGenerator`` when calling ``/api/generate``.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ml.generation import Pix2PixUNet  # noqa: E402
from backend.utils.synth import generate_layout  # noqa: E402


class SyntheticPairs(Dataset):
    """Generates synthetic (condition, target) pairs on the fly."""

    def __init__(self, num_samples: int = 256, image_size: int = 256, seed: int = 0) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        import cv2
        import numpy as np

        rng = random.Random(self.seed + idx)
        rooms = rng.randint(2, 7)
        canvas = self.image_size
        image, _ = generate_layout(
            width=canvas, height=canvas, num_rooms=rooms, seed=self.seed + idx
        )
        target = cv2.resize(image, (self.image_size, self.image_size))
        target_t = torch.from_numpy(target).permute(2, 0, 1).float() / 127.5 - 1.0

        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        m = 20
        mask[m:-m, m:-m] = 1.0
        rooms_plane = np.full(
            (self.image_size, self.image_size), fill_value=min(1.0, rooms / 8.0), dtype=np.float32
        )
        cond = torch.from_numpy(np.stack([mask, rooms_plane], axis=0))
        return cond, target_t


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = SyntheticPairs(num_samples=args.num_samples, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = Pix2PixUNet(in_channels=2, out_channels=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    l1 = nn.L1Loss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for cond, target in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            cond = cond.to(device)
            target = target.to(device)
            pred = model(cond)
            loss = l1(pred, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
        print(f"  L1 = {running / max(1, len(loader)):.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved weights to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out", type=Path, default=Path("models/generator.pt"))
    args = parser.parse_args()
    main(args)
