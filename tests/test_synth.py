import numpy as np

from backend.utils.synth import generate_layout


def test_generate_layout_shapes_and_categories() -> None:
    image, polygons = generate_layout(width=480, height=360, num_rooms=4, seed=123)
    assert image.shape == (360, 480, 3)
    assert image.dtype == np.uint8
    cats = {p.category for p in polygons}
    assert "room" in cats
    assert "wall" in cats
    # Room count from splitter should match request when canvas is large enough.
    rooms = [p for p in polygons if p.category == "room"]
    assert len(rooms) == 4


def test_generate_is_deterministic_with_seed() -> None:
    a, _ = generate_layout(seed=7, num_rooms=3)
    b, _ = generate_layout(seed=7, num_rooms=3)
    assert np.array_equal(a, b)


def test_generate_varies_with_seed() -> None:
    a, _ = generate_layout(seed=1, num_rooms=4)
    b, _ = generate_layout(seed=2, num_rooms=4)
    assert not np.array_equal(a, b)
