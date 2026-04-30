"""Floor plan generation.

Two backends:

1. ``ProceduralGenerator`` — wraps :func:`backend.utils.synth.generate_layout`.
   Always available, no model weights, deterministic per seed.

2. ``Pix2PixGenerator`` — conditional pix2pix-style generator that accepts a
   building boundary mask (and optionally a room-count condition) and produces
   a colourised plan. Loads pretrained weights from ``models/generator.pt`` if
   present; otherwise the unified ``generate`` API falls back to the procedural
   generator.

The ``Pix2Pix`` building blocks here are sized to be trainable on small data
without GPU. The training script under ``scripts/train_generator.py`` shows
how to adapt them to a larger dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from backend.utils.coco import Polygon
from backend.utils.synth import generate_layout


@dataclass
class GenerationConditions:
    """User-controlled generation conditions."""

    width: int = 640
    height: int = 480
    num_rooms: int = 4
    seed: int | None = None
    area_m2: float | None = None  # interpreted only by Pix2PixGenerator
    boundary_shape: str = "rect"  # "rect" or "L" — only honoured by procedural

    def __post_init__(self) -> None:
        # If an area is given, scale canvas dimensions to roughly preserve it
        # while keeping a reasonable aspect ratio.
        if self.area_m2 and self.area_m2 > 0:
            # 1m == 10px is a comfortable rendering scale for the demo.
            target_px = self.area_m2 * 100  # m^2 → px^2
            ratio = self.width / max(1, self.height)
            new_h = int((target_px / ratio) ** 0.5)
            new_w = int(new_h * ratio)
            self.width = max(320, min(960, new_w))
            self.height = max(240, min(720, new_h))


class ProceduralGenerator:
    name = "procedural"

    def generate(self, cond: GenerationConditions) -> tuple[np.ndarray, list[Polygon]]:
        if cond.boundary_shape == "L":
            return self._generate_l_shape(cond)
        return generate_layout(
            width=cond.width,
            height=cond.height,
            num_rooms=cond.num_rooms,
            seed=cond.seed,
        )

    def _generate_l_shape(self, cond: GenerationConditions) -> tuple[np.ndarray, list[Polygon]]:
        """Compose two rectangular wings into an L-shaped plan."""
        import random

        from backend.utils.synth import Rect, _wall_polygons, render_layout, split_rect

        rng = random.Random(cond.seed)
        margin = 20
        wing_w = (cond.width - 2 * margin) // 2
        wing_h = (cond.height - 2 * margin) // 2
        wing_a = Rect(margin, margin, cond.width - 2 * margin, wing_h)
        wing_b = Rect(margin, margin + wing_h, wing_w, cond.height - 2 * margin - wing_h)

        rooms = split_rect(wing_a, max(1, cond.num_rooms // 2 + 1), rng)
        rooms += split_rect(wing_b, max(1, cond.num_rooms - len(rooms)), rng)

        polys: list[Polygon] = [Polygon(points=r.as_polygon(), category="room") for r in rooms]
        polys.extend(_wall_polygons(rooms))
        image = render_layout(cond.width, cond.height, rooms, polys)
        return image, polys


# ---------------------------------------------------------------------------
# Pix2pix-style generator (compact U-Net) -----------------------------------
# ---------------------------------------------------------------------------


class _DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, normalize: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        return torch.cat([x, skip], dim=1)


class Pix2PixUNet(nn.Module):
    """Compact U-Net generator for 256×256 conditional generation."""

    def __init__(self, in_channels: int = 2, out_channels: int = 3, base: int = 32) -> None:
        super().__init__()
        self.d1 = _DownBlock(in_channels, base, normalize=False)
        self.d2 = _DownBlock(base, base * 2)
        self.d3 = _DownBlock(base * 2, base * 4)
        self.d4 = _DownBlock(base * 4, base * 8)
        self.d5 = _DownBlock(base * 8, base * 8)
        self.d6 = _DownBlock(base * 8, base * 8)

        self.u1 = _UpBlock(base * 8, base * 8, dropout=True)
        self.u2 = _UpBlock(base * 16, base * 8, dropout=True)
        self.u3 = _UpBlock(base * 16, base * 4)
        self.u4 = _UpBlock(base * 8, base * 2)
        self.u5 = _UpBlock(base * 4, base)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)

        u1 = self.u1(d6, d5)
        u2 = self.u2(u1, d4)
        u3 = self.u3(u2, d3)
        u4 = self.u4(u3, d2)
        u5 = self.u5(u4, d1)
        return self.final(u5)


class Pix2PixGenerator:
    """Inference wrapper around :class:`Pix2PixUNet`.

    Falls back to :class:`ProceduralGenerator` when weights are unavailable so
    that the public ``generate`` contract is always satisfied.
    """

    name = "pix2pix"

    def __init__(self, weights_path: str | Path | None = "models/generator.pt") -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self._model: Pix2PixUNet | None = None
        self._fallback = ProceduralGenerator()

    def available(self) -> bool:
        return self.weights_path is not None and self.weights_path.exists()

    def _load(self) -> Pix2PixUNet:
        model = Pix2PixUNet(in_channels=2, out_channels=3)
        state = torch.load(self.weights_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        model.eval()
        return model

    def _condition_tensor(self, cond: GenerationConditions) -> torch.Tensor:
        """Build a 2-channel 256×256 conditioning input.

        Channel 0: building boundary mask (1 inside, 0 outside).
        Channel 1: a constant plane encoding the desired number of rooms
        (normalised to [0, 1]).
        """
        mask = torch.zeros(1, 256, 256)
        if cond.boundary_shape == "L":
            mask[:, 20:140, 20:236] = 1.0
            mask[:, 140:236, 20:128] = 1.0
        else:
            mask[:, 20:236, 20:236] = 1.0
        rooms_plane = torch.full((1, 256, 256), fill_value=min(1.0, cond.num_rooms / 8.0))
        return torch.stack([mask, rooms_plane], dim=1)

    def generate(self, cond: GenerationConditions) -> tuple[np.ndarray, list[Polygon]]:
        if not self.available():
            return self._fallback.generate(cond)

        if self._model is None:
            self._model = self._load()

        with torch.no_grad():
            inp = self._condition_tensor(cond)
            out = self._model(inp).squeeze(0).clamp(-1, 1)
        # Convert from [-1, 1] CHW to uint8 HWC.
        rgb = ((out.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
        # Resize to requested canvas.
        import cv2

        rgb = cv2.resize(rgb, (cond.width, cond.height), interpolation=cv2.INTER_LINEAR)
        # Polygons via classical CV on the generated raster.
        from backend.ml.segmentation import classical_cv_segmentation

        polygons = classical_cv_segmentation(rgb)
        return rgb, polygons


def get_default_generator(prefer_neural: bool = True) -> ProceduralGenerator | Pix2PixGenerator:
    """Return the best available generator."""
    if prefer_neural:
        nn_gen = Pix2PixGenerator()
        if nn_gen.available():
            return nn_gen
    return ProceduralGenerator()
