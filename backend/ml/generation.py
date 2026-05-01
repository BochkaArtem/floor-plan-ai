"""Floor plan generation.

Three backends are available:

1. ``ProceduralGenerator`` — wraps the deterministic
   :func:`backend.utils.synth.generate_semantic_layout` planner. Works with
   varied boundary shapes (rect / L / T / U / plus / random) and assigns
   semantic room types (hall / living / kitchen / bedroom / bathroom /
   balcony) using adjacency rules. Always available.

2. ``MaskUNetGenerator`` — small conditional U-Net trained to predict a
   4-class semantic mask from a 5-channel condition tensor (boundary mask +
   room-count plane + room-type one-hot tiles). The predicted mask is
   converted back into editable polygons via contour extraction. Loads
   weights from ``models/generator.pt`` if present.

3. ``Pix2PixGenerator`` — legacy compact pix2pix-style generator kept for
   backward compatibility. The unified :func:`get_default_generator` prefers
   ``MaskUNetGenerator`` when its weights are available.

The procedural backend is always used as a fallback so the public
``generate`` contract is satisfied even without trained weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from backend.utils.coco import Polygon
from backend.utils.layout_planner import ROOM_TYPES
from backend.utils.synth import (
    MASK_BACKGROUND,
    MASK_DOOR,
    MASK_ROOM,
    MASK_WALL,
    MASK_WINDOW,
    ROOM_TYPE_COLORS,
    generate_semantic_layout,
)


@dataclass
class GenerationConditions:
    """User-controlled generation conditions."""

    width: int = 640
    height: int = 480
    num_rooms: int = 5
    seed: int | None = None
    area_m2: float | None = None
    boundary_shape: str = "auto"  # rect, L, T, U, plus, random, auto
    room_types: list[str] = field(default_factory=list)
    backend: str = "auto"  # auto | nn | procedural

    def __post_init__(self) -> None:
        if self.area_m2 and self.area_m2 > 0:
            target_px = self.area_m2 * 100
            ratio = self.width / max(1, self.height)
            new_h = int((target_px / ratio) ** 0.5)
            new_w = int(new_h * ratio)
            self.width = max(320, min(960, new_w))
            self.height = max(240, min(720, new_h))
        # Filter requested room_types to known set.
        self.room_types = [t for t in self.room_types if t in ROOM_TYPES]


# ---------------------------------------------------------------------------
# Procedural (planner-driven) backend ---------------------------------------
# ---------------------------------------------------------------------------


class ProceduralGenerator:
    name = "procedural"

    def generate(self, cond: GenerationConditions) -> tuple[np.ndarray, list[Polygon]]:
        rgb, polys, _plan = generate_semantic_layout(
            width=cond.width,
            height=cond.height,
            num_rooms=cond.num_rooms,
            boundary_shape=cond.boundary_shape,
            room_types=cond.room_types,
            seed=cond.seed,
            draw_labels=True,
        )
        return rgb, polys


# ---------------------------------------------------------------------------
# Mask U-Net (conditional) backend ------------------------------------------
# ---------------------------------------------------------------------------


# 5-channel condition: [boundary_mask, room_count_plane, hall_present,
# living_present, kitchen_present]. Boundary mask is the only spatial
# signal; other planes are scalars broadcast over all pixels.
NN_INPUT_CHANNELS = 5
NN_OUTPUT_CLASSES = 5  # bg, wall, window, door, room (matches MASK_* constants)
NN_INPUT_SIZE = 128


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MaskUNet(nn.Module):
    """Compact U-Net used by :class:`MaskUNetGenerator`.

    Trained as a multi-class classifier producing a per-pixel softmax over
    {bg, wall, window, door, room}. Architecture is intentionally small
    (~200K params with base=16) so it can be trained on CPU in tens of
    minutes.
    """

    def __init__(self, in_channels: int = NN_INPUT_CHANNELS, num_classes: int = NN_OUTPUT_CLASSES, base: int = 16) -> None:
        super().__init__()
        self.inc = _DoubleConv(in_channels, base)
        self.down1 = _DoubleConv(base, base * 2)
        self.down2 = _DoubleConv(base * 2, base * 4)
        self.down3 = _DoubleConv(base * 4, base * 8)
        self.up1 = _DoubleConv(base * 8 + base * 4, base * 4)
        self.up2 = _DoubleConv(base * 4 + base * 2, base * 2)
        self.up3 = _DoubleConv(base * 2 + base, base)
        self.outc = nn.Conv2d(base, num_classes, 1)
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(self.pool(x1))
        x3 = self.down2(self.pool(x2))
        x4 = self.down3(self.pool(x3))
        u1 = self.up1(torch.cat([self.upsample(x4), x3], dim=1))
        u2 = self.up2(torch.cat([self.upsample(u1), x2], dim=1))
        u3 = self.up3(torch.cat([self.upsample(u2), x1], dim=1))
        return self.outc(u3)


# Order in which one-hot room type signals are stacked into the condition.
NN_TYPE_ORDER: tuple[str, ...] = ("hall", "living", "kitchen")


def build_condition_tensor(
    boundary_mask: np.ndarray,
    num_rooms: int,
    room_types: list[str],
    size: int = NN_INPUT_SIZE,
) -> torch.Tensor:
    """Build a (1, NN_INPUT_CHANNELS, size, size) conditioning tensor.

    Channel 0: resized boundary mask (1 inside the apartment, 0 outside).
    Channel 1: a uniform plane encoding ``num_rooms / 8`` (capped at 1).
    Channels 2-4: a uniform plane per room type from :data:`NN_TYPE_ORDER`.
    """
    bm = cv2.resize(boundary_mask.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)
    rooms_plane = np.full((size, size), min(1.0, num_rooms / 8.0), dtype=np.float32)
    type_set = set(room_types)
    type_planes = [
        np.full((size, size), 1.0 if t in type_set else 0.0, dtype=np.float32)
        for t in NN_TYPE_ORDER
    ]
    stack = np.stack([bm, rooms_plane, *type_planes], axis=0)
    return torch.from_numpy(stack).unsqueeze(0)


def _boundary_mask_for_shape(width: int, height: int, shape: str, seed: int | None) -> np.ndarray:
    """Render a binary boundary mask for the requested shape.

    Reuses the planner so the boundary geometry stays consistent between
    procedural and NN generation (the NN learns to fill the same boundary).
    """
    import random

    from backend.utils.layout_planner import _make_boundary, _pick_shape

    rng = random.Random(seed)
    chosen = _pick_shape(shape, rng)
    cells = _make_boundary(chosen, width, height, rng)
    mask = np.zeros((height, width), dtype=np.uint8)
    for cell in cells:
        x1, y1 = int(cell.x), int(cell.y)
        x2, y2 = int(cell.x2), int(cell.y2)
        mask[max(0, y1):min(height, y2), max(0, x1):min(width, x2)] = 1
    return mask


def _mask_to_polygons(mask: np.ndarray) -> list[Polygon]:
    """Extract editable polygons from a 4-class (or 5-class incl. bg) mask.

    The output uses the canonical category names so it is interchangeable
    with the procedural pipeline output.
    """
    cat_map = {
        MASK_WALL: "wall",
        MASK_WINDOW: "window",
        MASK_DOOR: "door",
        MASK_ROOM: "room",
    }
    polys: list[Polygon] = []
    for class_idx, name in cat_map.items():
        binary = (mask == class_idx).astype(np.uint8) * 255
        if binary.sum() == 0:
            continue
        # Mild closing to fill 1-pixel gaps from softmax noise.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            epsilon = max(1.5, 0.005 * cv2.arcLength(cnt, True))
            simp = cv2.approxPolyDP(cnt, epsilon, True)
            pts = [(float(p[0][0]), float(p[0][1])) for p in simp]
            if len(pts) >= 3:
                polys.append(Polygon(points=pts, category=name))
    return polys


def _render_mask(mask: np.ndarray) -> np.ndarray:
    """Colourise a 5-class mask to RGB using the demo palette."""
    h, w = mask.shape[:2]
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    img[mask == MASK_ROOM] = ROOM_TYPE_COLORS["living"]
    img[mask == MASK_WALL] = (40, 40, 40)
    img[mask == MASK_WINDOW] = (60, 140, 220)
    img[mask == MASK_DOOR] = (220, 100, 60)
    img[mask == MASK_BACKGROUND] = (245, 245, 245)
    return img


class MaskUNetGenerator:
    """NN-driven conditional generator.

    Falls back to :class:`ProceduralGenerator` when weights are missing.
    """

    name = "mask-unet"

    def __init__(self, weights_path: str | Path | None = "models/generator.pt") -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self._model: MaskUNet | None = None
        self._fallback = ProceduralGenerator()

    def available(self) -> bool:
        return self.weights_path is not None and self.weights_path.exists()

    def _load(self) -> MaskUNet:
        model = MaskUNet()
        state = torch.load(self.weights_path, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        model.eval()
        return model

    def generate(self, cond: GenerationConditions) -> tuple[np.ndarray, list[Polygon]]:
        if not self.available():
            return self._fallback.generate(cond)
        if self._model is None:
            self._model = self._load()

        boundary = _boundary_mask_for_shape(
            cond.width, cond.height, cond.boundary_shape, cond.seed
        )
        cond_tensor = build_condition_tensor(
            boundary_mask=boundary,
            num_rooms=cond.num_rooms,
            room_types=cond.room_types,
        )
        with torch.no_grad():
            logits = self._model(cond_tensor)
        pred_low = logits.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)
        # Resize prediction to canvas size and clamp to boundary.
        pred = cv2.resize(pred_low, (cond.width, cond.height), interpolation=cv2.INTER_NEAREST)
        pred[boundary == 0] = MASK_BACKGROUND
        rgb = _render_mask(pred)
        polys = _mask_to_polygons(pred)
        return rgb, polys


# ---------------------------------------------------------------------------
# Legacy pix2pix-style generator (kept for backward compatibility) ----------
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
    """Legacy pix2pix RGB generator. Kept for backward compatibility."""

    name = "pix2pix"

    def __init__(self, weights_path: str | Path | None = "models/generator_pix2pix.pt") -> None:
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

    def generate(self, cond: GenerationConditions) -> tuple[np.ndarray, list[Polygon]]:
        if not self.available():
            return self._fallback.generate(cond)
        if self._model is None:
            self._model = self._load()
        mask = torch.zeros(1, 256, 256)
        if cond.boundary_shape == "L":
            mask[:, 20:140, 20:236] = 1.0
            mask[:, 140:236, 20:128] = 1.0
        else:
            mask[:, 20:236, 20:236] = 1.0
        rooms_plane = torch.full((1, 256, 256), fill_value=min(1.0, cond.num_rooms / 8.0))
        inp = torch.stack([mask, rooms_plane], dim=1)
        with torch.no_grad():
            out = self._model(inp).squeeze(0).clamp(-1, 1)
        rgb = ((out.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
        rgb = cv2.resize(rgb, (cond.width, cond.height), interpolation=cv2.INTER_LINEAR)
        from backend.ml.segmentation import classical_cv_segmentation

        polygons = classical_cv_segmentation(rgb)
        return rgb, polygons


# ---------------------------------------------------------------------------
# Backend selection ---------------------------------------------------------
# ---------------------------------------------------------------------------


def get_default_generator(prefer_neural: bool = True) -> ProceduralGenerator | MaskUNetGenerator:
    """Return the best available generator for runtime use."""
    if prefer_neural:
        nn_gen = MaskUNetGenerator()
        if nn_gen.available():
            return nn_gen
    return ProceduralGenerator()


def select_generator(backend_pref: str) -> ProceduralGenerator | MaskUNetGenerator:
    """Pick a generator according to a user preference (auto/nn/procedural)."""
    pref = (backend_pref or "auto").lower()
    if pref == "procedural":
        return ProceduralGenerator()
    if pref == "nn":
        gen = MaskUNetGenerator()
        if gen.available():
            return gen
        return ProceduralGenerator()
    # auto
    return get_default_generator(prefer_neural=True)
