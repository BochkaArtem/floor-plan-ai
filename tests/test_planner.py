"""Tests for the semantic floor plan planner."""

from __future__ import annotations

import numpy as np
import pytest

from backend.utils.layout_planner import ROOM_TYPES, plan_layout
from backend.utils.synth import (
    MASK_BACKGROUND,
    MASK_DOOR,
    MASK_ROOM,
    MASK_WALL,
    MASK_WINDOW,
    generate_semantic_layout,
    planned_to_mask,
)


@pytest.mark.parametrize("shape", ["rect", "L", "T", "U", "plus", "random"])
def test_all_shapes_produce_valid_layouts(shape: str) -> None:
    rgb, polys, plan = generate_semantic_layout(
        width=320, height=240, num_rooms=5, boundary_shape=shape, seed=7
    )
    assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8
    assert plan.boundary_shape == shape
    assert len(plan.rooms) >= 2
    cats = {p.category for p in polys}
    # Every plan must include rooms and walls.
    assert "room" in cats and "wall" in cats


def test_room_types_include_requested_kitchen_and_living() -> None:
    _rgb, _polys, plan = generate_semantic_layout(
        width=480, height=360, num_rooms=6, boundary_shape="L",
        room_types=["hall", "living", "kitchen", "bedroom", "bathroom", "balcony"],
        seed=11,
    )
    types = {r.type for r in plan.rooms}
    # Kitchen and living must end up assigned somewhere.
    assert "kitchen" in types
    assert "living" in types


def test_kitchen_is_on_outer_wall() -> None:
    """Architectural rule: kitchen must have a window-eligible outer wall."""
    _rgb, _polys, plan = generate_semantic_layout(
        width=480, height=360, num_rooms=5, boundary_shape="rect",
        room_types=["hall", "living", "kitchen", "bedroom", "bathroom"],
        seed=3,
    )
    kitchen = next((r for r in plan.rooms if r.type == "kitchen"), None)
    assert kitchen is not None
    assert kitchen.has_outer_wall is True


def test_planned_mask_has_expected_classes() -> None:
    _rgb, _polys, plan = generate_semantic_layout(
        width=480, height=360, num_rooms=5, boundary_shape="rect", seed=11
    )
    mask = planned_to_mask(plan)
    assert mask.shape == (360, 480) and mask.dtype == np.uint8
    classes = set(np.unique(mask).tolist())
    # Background, walls, doors, windows and rooms must all appear in a
    # reasonably large layout with multiple adjacent rooms.
    assert MASK_BACKGROUND in classes
    assert MASK_WALL in classes
    assert MASK_DOOR in classes
    assert MASK_WINDOW in classes
    assert MASK_ROOM in classes


def test_subcategory_persists_for_rooms() -> None:
    _rgb, polys, _plan = generate_semantic_layout(
        width=320, height=240, num_rooms=4, boundary_shape="rect",
        room_types=["hall", "living", "kitchen", "bedroom"], seed=9,
    )
    room_polys = [p for p in polys if p.category == "room"]
    assert all(rp.subcategory in ROOM_TYPES for rp in room_polys)
    sub_types = {rp.subcategory for rp in room_polys}
    assert "kitchen" in sub_types or "living" in sub_types


def test_seed_is_deterministic() -> None:
    a = plan_layout(width=320, height=240, num_rooms=4, boundary_shape="L", seed=99)
    b = plan_layout(width=320, height=240, num_rooms=4, boundary_shape="L", seed=99)
    assert len(a.rooms) == len(b.rooms)
    for ra, rb in zip(a.rooms, b.rooms, strict=True):
        assert ra.type == rb.type
        assert (ra.rect.x, ra.rect.y, ra.rect.w, ra.rect.h) == (
            rb.rect.x, rb.rect.y, rb.rect.w, rb.rect.h,
        )
