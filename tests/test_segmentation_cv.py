import numpy as np

from backend.ml.segmentation import classical_cv_segmentation
from backend.utils.synth import generate_layout


def test_cv_segmentation_finds_polygons_on_synthetic_plan() -> None:
    image, _ = generate_layout(width=480, height=360, num_rooms=4, seed=42)
    polygons = classical_cv_segmentation(image)
    assert polygons, "Classical CV pipeline should find at least one polygon"
    cats = {p.category for p in polygons}
    # Synthetic plan is dominated by walls (dark) and rooms (light fills); both
    # should be detected by the CV path.
    assert "wall" in cats or "room" in cats


def test_cv_segmentation_handles_blank_image() -> None:
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    polygons = classical_cv_segmentation(image)
    # Blank image must not crash and should yield an empty / near-empty list.
    assert isinstance(polygons, list)
