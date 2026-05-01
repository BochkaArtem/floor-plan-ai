"""Train both the segmentation U-Net and the conditional mask generator.

Inputs are produced by :mod:`scripts.make_training_data` and live under
``data/synthetic`` with three parallel subdirectories ``images/``, ``masks/``
and ``conditions/`` containing one file per sample.

Both networks output a 5-class softmax over
``{background, wall, window, door, room}``; they only differ in input
channels (RGB image for segmentation, condition tensor for generation).

Usage:

    python -m scripts.train_models \\
        --data data/synthetic --epochs 12 --batch-size 16 \\
        --out-seg models/segmentation.pt --out-gen models/generator.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from backend.ml.generation import (
    NN_INPUT_CHANNELS,
    NN_INPUT_SIZE,
    NN_OUTPUT_CLASSES,
    NN_TYPE_ORDER,
    MaskUNet,
)
from backend.ml.unet import UNet


def _list_indices(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "images").glob("*.png"))


class SegDataset(Dataset):
    """RGB image -> 5-class index mask."""

    def __init__(self, root: Path, indices: list[str], size: int = NN_INPUT_SIZE) -> None:
        self.root = root
        self.indices = indices
        self.size = size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.indices[idx]
        img = Image.open(self.root / "images" / f"{stem}.png").convert("RGB")
        mask = Image.open(self.root / "masks" / f"{stem}.png")
        if img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BILINEAR)
            mask = mask.resize((self.size, self.size), Image.NEAREST)
        x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        y = torch.from_numpy(np.array(mask)).long()
        return x, y


class GenDataset(Dataset):
    """Condition tensor -> 5-class index mask."""

    def __init__(self, root: Path, indices: list[str], size: int = NN_INPUT_SIZE) -> None:
        self.root = root
        self.indices = indices
        self.size = size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.indices[idx]
        mask_img = Image.open(self.root / "masks" / f"{stem}.png")
        if mask_img.size != (self.size, self.size):
            mask_img = mask_img.resize((self.size, self.size), Image.NEAREST)
        mask = np.array(mask_img)
        with (self.root / "conditions" / f"{stem}.json").open() as f:
            cond = json.load(f)

        # Boundary mask = anywhere mask != background.
        boundary = (mask != 0).astype(np.float32)
        rooms_plane = np.full_like(boundary, min(1.0, cond["num_rooms"] / 8.0))
        type_set = set(cond.get("room_types", []))
        type_planes = [
            np.full_like(boundary, 1.0 if t in type_set else 0.0) for t in NN_TYPE_ORDER
        ]
        x = torch.from_numpy(np.stack([boundary, rooms_plane, *type_planes], axis=0)).float()
        y = torch.from_numpy(mask).long()
        return x, y


def _train_one(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    out_path: Path,
    device: torch.device,
) -> dict[str, list[float]]:
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            train_loss += loss.item() * xb.size(0)
            n_train += xb.size(0)
        train_loss /= max(1, n_train)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.numel()
        val_loss /= max(1, len(val_loader.dataset))
        acc = correct / max(1, total)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)

        if val_loss < best_val:
            best_val = val_loss
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, out_path)

        dt = time.time() - t0
        print(
            f"[{name}] epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={acc:.4f}  ({dt:.1f}s)",
            flush=True,
        )

    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--out-seg", type=Path, default=Path("models/segmentation.pt"))
    parser.add_argument("--out-gen", type=Path, default=Path("models/generator.pt"))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--skip", choices=("none", "seg", "gen"), default="none",
        help="Skip training one of the two networks.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    indices = _list_indices(args.data)
    if not indices:
        raise SystemExit(f"No samples found under {args.data}")
    print(f"Loaded {len(indices)} samples from {args.data}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_val = max(1, int(len(indices) * args.val_frac))
    n_train = len(indices) - n_val
    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(indices), generator=rng).tolist()
    train_idx = [indices[i] for i in perm[:n_train]]
    val_idx = [indices[i] for i in perm[n_train:]]

    if args.skip != "seg":
        seg_train = SegDataset(args.data, train_idx)
        seg_val = SegDataset(args.data, val_idx)
        seg_train_loader = DataLoader(
            seg_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
        )
        seg_val_loader = DataLoader(seg_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

        seg_model = UNet(in_channels=3, num_classes=NN_OUTPUT_CLASSES, base=16)
        print(
            f"Segmentation model: {sum(p.numel() for p in seg_model.parameters()):,} params"
        )
        _train_one(
            "seg", seg_model, seg_train_loader, seg_val_loader,
            args.epochs, args.lr, args.out_seg, device,
        )

    if args.skip != "gen":
        gen_train = GenDataset(args.data, train_idx)
        gen_val = GenDataset(args.data, val_idx)
        gen_train_loader = DataLoader(
            gen_train, batch_size=args.batch_size, shuffle=True, num_workers=0,
        )
        gen_val_loader = DataLoader(gen_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

        gen_model = MaskUNet(
            in_channels=NN_INPUT_CHANNELS, num_classes=NN_OUTPUT_CLASSES, base=16
        )
        print(
            f"Generator model: {sum(p.numel() for p in gen_model.parameters()):,} params"
        )
        _train_one(
            "gen", gen_model, gen_train_loader, gen_val_loader,
            args.epochs, args.lr, args.out_gen, device,
        )

    print("Done.")


if __name__ == "__main__":
    main()
