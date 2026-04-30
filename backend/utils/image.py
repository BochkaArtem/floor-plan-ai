"""Image I/O helpers used by the Flask backend."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

# Class palette shared with the frontend canvas editor.
CLASS_NAMES: list[str] = ["wall", "window", "door", "room"]
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "wall": (40, 40, 40),
    "window": (60, 140, 220),
    "door": (220, 100, 60),
    "room": (180, 220, 180),
}


def load_image_from_bytes(data: bytes) -> np.ndarray:
    """Load an arbitrary uploaded image into an RGB numpy array."""
    image = Image.open(io.BytesIO(data))
    image = image.convert("RGB")
    return np.array(image)


def load_image(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.array(image)


def save_image(image: np.ndarray, path: str) -> None:
    Image.fromarray(image.astype(np.uint8)).save(path)


def image_to_base64(image: np.ndarray, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    Image.fromarray(image.astype(np.uint8)).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def image_data_url(image: np.ndarray, fmt: str = "PNG") -> str:
    return f"data:image/{fmt.lower()};base64,{image_to_base64(image, fmt=fmt)}"
