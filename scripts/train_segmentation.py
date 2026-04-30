"""Train the UNet segmentation model on synthetic floor plans.

Designed to be runnable on a CPU within minutes for sanity-checking the
pipeline. For real-world performance, point ``--data-dir`` at CubiCasa5K (or
any COCO-style dataset) and use a GPU.

Outputs ``models/segmentation.pt`` which is automatically picked up by the
inference path in ``backend.ml.segmentation.UNetSegmenter``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.ml.unet import UNet  # noqa: E402
from backend.utils.coco import load_coco  # noqa: E402
from backend.utils.image import CLASS_NAMES, load_image  # noqa: E402

CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


def rasterise_mask(width: int, height: int, polygons, scale: float = 1.0) -> np.ndarray:
    """Rasterise COCO polygons into a per-pixel class index map (HxW, uint8)."""
    import cv2

    mask = np.full((height, width), fill_value=255, dtype=np.uint8)  # 255 = ignore (background)
    # Render in semantic order — rooms first, walls/doors/windows on top.
    order = ["room", "wall", "window", "door"]
    for cat in order:
        for poly in polygons:
            if poly.category != cat:
                continue
            pts = np.array([(int(x * scale), int(y * scale)) for x, y in poly.points], dtype=np.int32)
            cv2.fillPoly(mask, [pts], CLASS_TO_IDX[cat])
    # Treat unset pixels as room (background) for training stability.
    mask[mask == 255] = CLASS_TO_IDX["room"]
    return mask


class FloorPlanDataset(Dataset):
    def __init__(self, coco_path: Path, images_dir: Path, image_size: int = 256) -> None:
        self.coco = load_coco(coco_path)
        self.images_dir = images_dir
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.coco)

    def __getitem__(self, idx: int):
        import cv2

        record = self.coco[idx]
        img = load_image(str(self.images_dir / record.file_name))
        h, w = img.shape[:2]

        scale_x = self.image_size / w
        scale_y = self.image_size / h
        # Use min scale to preserve aspect, then pad.
        scale = min(scale_x, scale_y)

        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((self.image_size, self.image_size, 3), 245, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        mask = rasterise_mask(new_w, new_h, record.polygons, scale=scale)
        full_mask = np.full((self.image_size, self.image_size), CLASS_TO_IDX["room"], dtype=np.uint8)
        full_mask[:new_h, :new_w] = mask

        x = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(full_mask).long()
        return x, y


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = FloorPlanDataset(
        coco_path=args.coco, images_dir=args.images_dir, image_size=args.image_size
    )
    if len(dataset) == 0:
        raise SystemExit(f"No samples in {args.coco}")
    print(f"Dataset: {len(dataset)} samples")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = UNet(in_channels=3, num_classes=len(CLASS_NAMES)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
        avg = running / max(1, len(loader))
        print(f"  loss = {avg:.4f}")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved weights to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, default=Path("demo/annotations/demo.coco.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("demo/images"))
    parser.add_argument("--out", type=Path, default=Path("models/segmentation.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    main(args)
