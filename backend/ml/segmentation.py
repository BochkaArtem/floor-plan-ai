"""Floor plan segmentation.

Two strategies are supported:

1. ``classical_cv_segmentation`` — uses OpenCV (binarisation, morphology,
   contour extraction) to detect walls / windows / doors. Always available
   and works without any pretrained weights.

2. ``UNetSegmenter`` — a thin wrapper around the PyTorch UNet model defined
   in :mod:`backend.ml.unet`. Loads optional pretrained weights from
   ``models/segmentation.pt`` when present; otherwise falls back to
   classical CV automatically.

Both strategies return a list of :class:`backend.utils.coco.Polygon` objects
in image (pixel) coordinates.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from backend.utils.coco import Polygon

# How aggressively to simplify polygon contours (Douglas-Peucker epsilon factor).
POLYGON_EPSILON = 0.005


def _simplify_contour(contour: np.ndarray, epsilon_factor: float = POLYGON_EPSILON) -> np.ndarray:
    perimeter = cv2.arcLength(contour, closed=True)
    epsilon = max(1.5, epsilon_factor * perimeter)
    return cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)


def _contours_to_polygons(
    contours: Iterable[np.ndarray],
    category: str,
    min_area: float = 50.0,
) -> list[Polygon]:
    polygons: list[Polygon] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        simplified = _simplify_contour(cnt)
        pts = [(float(p[0][0]), float(p[0][1])) for p in simplified]
        if len(pts) >= 3:
            polygons.append(Polygon(points=pts, category=category))
    return polygons


def _wall_mask(gray: np.ndarray) -> np.ndarray:
    """Return a binary mask of probable wall pixels (uint8, 0/255)."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Walls in real plans are typically dark on a lighter background.
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Close small gaps and remove specks.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened


def _detect_walls(gray: np.ndarray) -> list[Polygon]:
    mask = _wall_mask(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_polygons(contours, category="wall", min_area=80.0)


def _detect_rooms(gray: np.ndarray) -> list[Polygon]:
    """Rooms = large white connected components inside the wall structure."""
    wall = _wall_mask(gray)
    inverted = cv2.bitwise_not(wall)
    # Erode to detach rooms from the outer background.
    eroded = cv2.erode(inverted, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=4)

    h, w = gray.shape[:2]
    image_area = h * w
    polygons: list[Polygon] = []
    for label in range(1, num_labels):
        x, y, ww, hh, area = stats[label]
        if area < image_area * 0.005:
            continue
        if area > image_area * 0.6:
            # Likely the outer background.
            continue
        component = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons.extend(_contours_to_polygons(contours, category="room", min_area=area * 0.5))
    return polygons


def _detect_windows_doors_from_color(rgb: np.ndarray) -> list[Polygon]:
    """Best-effort heuristic: blue-ish blobs are windows, orange-ish blobs are doors."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    polygons: list[Polygon] = []

    # Windows (blue).
    blue_mask = cv2.inRange(hsv, np.array([90, 80, 80]), np.array([130, 255, 255]))
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons.extend(_contours_to_polygons(contours, category="window", min_area=20.0))

    # Doors (orange).
    orange_mask = cv2.inRange(hsv, np.array([5, 120, 120]), np.array([25, 255, 255]))
    contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons.extend(_contours_to_polygons(contours, category="door", min_area=20.0))

    return polygons


def classical_cv_segmentation(image_rgb: np.ndarray) -> list[Polygon]:
    """Classical OpenCV segmentation pipeline — works on any input image."""
    if image_rgb.ndim == 2:
        rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image_rgb
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    polygons: list[Polygon] = []
    polygons.extend(_detect_rooms(gray))
    polygons.extend(_detect_walls(gray))
    polygons.extend(_detect_windows_doors_from_color(rgb))
    return polygons


class UNetSegmenter:
    """UNet-based semantic segmentation with optional pretrained weights.

    If weights are not available the segmenter transparently falls back to
    classical CV so that the API contract remains the same.
    """

    def __init__(self, weights_path: str | Path | None = "models/segmentation.pt") -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self._model = None  # lazy

    def available(self) -> bool:
        return self.weights_path is not None and self.weights_path.exists()

    def _load_model(self):
        import torch

        from .unet import UNet

        # Try 5-class first (current scheme: bg, wall, window, door, room),
        # fall back to legacy 4-class.
        for num_classes in (5, 4):
            try:
                model = UNet(in_channels=3, num_classes=num_classes, base=16)
                state = torch.load(self.weights_path, map_location="cpu")
                if isinstance(state, dict) and "model" in state:
                    state = state["model"]
                model.load_state_dict(state)
                model.eval()
                self._num_classes = num_classes
                return model
            except RuntimeError:
                # base=16 channel counts didn't match; try base=32 (legacy default).
                try:
                    model = UNet(in_channels=3, num_classes=num_classes, base=32)
                    state = torch.load(self.weights_path, map_location="cpu")
                    if isinstance(state, dict) and "model" in state:
                        state = state["model"]
                    model.load_state_dict(state)
                    model.eval()
                    self._num_classes = num_classes
                    return model
                except RuntimeError:
                    continue
        raise RuntimeError("Could not load segmentation weights")

    def segment(self, image_rgb: np.ndarray) -> list[Polygon]:
        if not self.available():
            return classical_cv_segmentation(image_rgb)

        try:
            import torch
        except ImportError:
            return classical_cv_segmentation(image_rgb)

        if self._model is None:
            self._model = self._load_model()

        h, w = image_rgb.shape[:2]
        # Resize to a network-friendly size matching the training resolution.
        target = 128
        resized = cv2.resize(image_rgb, (target, target), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            logits = self._model(tensor)
        pred = logits.argmax(dim=1).squeeze(0).numpy().astype(np.uint8)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        # Index → category mapping. The 5-class scheme reserves 0 for
        # background so categories start at 1.
        if getattr(self, "_num_classes", 5) == 5:
            mapping = {1: "wall", 2: "window", 3: "door", 4: "room"}
        else:
            mapping = {0: "wall", 1: "window", 2: "door", 3: "room"}

        polygons: list[Polygon] = []
        for class_idx, name in mapping.items():
            mask = (pred == class_idx).astype(np.uint8) * 255
            if mask.sum() == 0:
                continue
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons.extend(_contours_to_polygons(contours, category=name, min_area=40.0))
        return polygons
