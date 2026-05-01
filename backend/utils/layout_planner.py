"""Semantic floor plan planner.

Produces apartment plans where:

* the building boundary is one of several rectilinear shapes (rect, L, T, U,
  plus, or a random orthogonal polygon),
* rooms are assigned semantic labels (hall, living, kitchen, bedroom,
  bathroom, balcony) based on adjacency / privacy / outer-wall heuristics
  rather than purely geometric splitting,
* doors and windows are placed in places that make architectural sense:
  entrance → hall, kitchen near living + outer wall, bedrooms deep in plan,
  bathroom adjacent to bedroom, balcony on the perimeter.

The output is a :class:`PlannedLayout` describing rooms, walls, doors and
windows. Two consumers use it:

* :mod:`backend.utils.synth` (rendering RGB + COCO polygons for the
  ``/api/generate`` endpoint and demo data),
* :mod:`scripts.make_training_data` (synthesising training pairs for the
  segmentation U-Net and the conditional mask generator).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Recognised semantic room labels.
ROOM_TYPES: tuple[str, ...] = (
    "hall",
    "living",
    "kitchen",
    "bedroom",
    "bathroom",
    "balcony",
)

# Pleasant per-type fill colours so generated plans are easy to read.
ROOM_TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "hall": (224, 224, 224),
    "living": (179, 229, 252),
    "kitchen": (255, 224, 178),
    "bedroom": (197, 225, 165),
    "bathroom": (206, 213, 235),
    "balcony": (220, 237, 200),
    # Fallbacks
    "room": (197, 225, 165),
}


@dataclass
class Cell:
    """An axis-aligned rectangle that participates in the building boundary.

    Cells are first-class so the planner can talk about the boundary as a
    union of cells (a polyomino). We later subdivide cells into rooms, but
    the boundary is fixed by the cell layout.
    """

    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def area(self) -> float:
        return self.w * self.h

    def as_polygon(self) -> list[tuple[float, float]]:
        return [(self.x, self.y), (self.x2, self.y), (self.x2, self.y2), (self.x, self.y2)]

    def centroid(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass
class Room:
    """A room produced by the planner."""

    rect: Cell
    type: str = "living"  # one of ROOM_TYPES
    has_outer_wall: bool = False
    near_entrance: bool = False
    perimeter_outer: float = 0.0  # length of edges that are outside walls

    def as_polygon(self) -> list[tuple[float, float]]:
        return self.rect.as_polygon()


@dataclass
class DoorSpec:
    """A door between two rooms (or between a room and the entrance)."""

    a_idx: int  # index of room A (or -1 for outside / entrance)
    b_idx: int
    orientation: str  # "v" or "h"
    fixed: float
    lo: float
    hi: float


@dataclass
class WindowSpec:
    orientation: str  # "v" or "h"
    fixed: float
    lo: float
    hi: float
    room_idx: int


@dataclass
class PlannedLayout:
    width: int
    height: int
    boundary_cells: list[Cell]
    rooms: list[Room]
    doors: list[DoorSpec]
    windows: list[WindowSpec]
    entrance: tuple[str, float, float, float] | None = None  # ("v"|"h", fixed, lo, hi)
    boundary_shape: str = "rect"
    seed: int | None = None
    room_types_requested: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Boundary shape generators ---------------------------------------------------
# ---------------------------------------------------------------------------

ALL_SHAPES = ("rect", "L", "T", "U", "plus", "random")


def _pick_shape(requested: str, rng: random.Random) -> str:
    if requested == "auto":
        # Bias towards more recognisable shapes.
        return rng.choices(
            ["rect", "L", "T", "U", "plus", "random"],
            weights=[2, 3, 2, 2, 2, 2],
            k=1,
        )[0]
    if requested in ALL_SHAPES:
        return requested
    return "rect"


def _make_boundary(
    shape: str,
    width: int,
    height: int,
    rng: random.Random,
    margin: int = 24,
) -> list[Cell]:
    """Produce the building boundary as a list of axis-aligned cells.

    The returned cells tile the boundary without overlaps.
    """
    inner_w = width - 2 * margin
    inner_h = height - 2 * margin
    cx = margin
    cy = margin

    if shape == "rect":
        return [Cell(cx, cy, inner_w, inner_h)]

    if shape == "L":
        # Two wings: a wide top wing and a narrower bottom-left wing.
        h_top = int(inner_h * rng.uniform(0.45, 0.6))
        h_bottom = inner_h - h_top
        w_bottom = int(inner_w * rng.uniform(0.45, 0.65))
        return [
            Cell(cx, cy, inner_w, h_top),
            Cell(cx, cy + h_top, w_bottom, h_bottom),
        ]

    if shape == "T":
        # Wide bar across the top + a vertical stem in the middle.
        h_bar = int(inner_h * rng.uniform(0.35, 0.5))
        stem_w = int(inner_w * rng.uniform(0.35, 0.55))
        stem_x = cx + (inner_w - stem_w) // 2
        return [
            Cell(cx, cy, inner_w, h_bar),
            Cell(stem_x, cy + h_bar, stem_w, inner_h - h_bar),
        ]

    if shape == "U":
        # Two vertical wings + a connecting bottom bar.
        wing_w = int(inner_w * rng.uniform(0.25, 0.35))
        h_bar = int(inner_h * rng.uniform(0.3, 0.45))
        h_wing = inner_h - h_bar
        return [
            Cell(cx, cy, wing_w, h_wing),
            Cell(cx + inner_w - wing_w, cy, wing_w, h_wing),
            Cell(cx, cy + h_wing, inner_w, h_bar),
        ]

    if shape == "plus":
        # Cross / plus assembled from 5 non-overlapping cells: top arm, left
        # arm, centre, right arm, bottom arm. Using non-overlapping cells
        # keeps room subdivision well-defined inside the cross.
        arm_w = int(inner_w * rng.uniform(0.4, 0.55))
        arm_h = int(inner_h * rng.uniform(0.4, 0.55))
        cx_arm = cx + (inner_w - arm_w) // 2
        cy_arm = cy + (inner_h - arm_h) // 2
        return [
            Cell(cx_arm, cy, arm_w, cy_arm - cy),
            Cell(cx, cy_arm, cx_arm - cx, arm_h),
            Cell(cx_arm, cy_arm, arm_w, arm_h),
            Cell(cx_arm + arm_w, cy_arm, (cx + inner_w) - (cx_arm + arm_w), arm_h),
            Cell(cx_arm, cy_arm + arm_h, arm_w, (cy + inner_h) - (cy_arm + arm_h)),
        ]

    if shape == "random":
        return _random_polyomino(cx, cy, inner_w, inner_h, rng)

    return [Cell(cx, cy, inner_w, inner_h)]


def _random_polyomino(
    x: float, y: float, w: float, h: float, rng: random.Random
) -> list[Cell]:
    """Random rectilinear shape grown by BFS from the centre of a 3×3 grid.

    Always returns a connected polyomino (walls of cells share edges), which
    is important so the produced "apartment" doesn't look like two unrelated
    buildings.
    """
    grid_w = w / 3.0
    grid_h = h / 3.0
    chosen: set[tuple[int, int]] = {(1, 1)}
    target_size = rng.randint(4, 6)
    while len(chosen) < target_size:
        # Find unfilled cells that are 4-connected to the current shape.
        frontier: list[tuple[int, int]] = []
        for col, row in chosen:
            for dc, dr in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nc, nr = col + dc, row + dr
                if 0 <= nc < 3 and 0 <= nr < 3 and (nc, nr) not in chosen:
                    frontier.append((nc, nr))
        if not frontier:
            break
        chosen.add(rng.choice(frontier))
    cells: list[Cell] = []
    for col, row in sorted(chosen):
        cells.append(Cell(x + col * grid_w, y + row * grid_h, grid_w, grid_h))
    # Merge axis-aligned neighbours so we deal with fewer, larger cells.
    return _merge_adjacent_cells(cells)


def _merge_adjacent_cells(cells: list[Cell]) -> list[Cell]:
    """Merge cells that share full edges (best effort, single pass)."""
    eps = 0.5
    cells = list(cells)
    changed = True
    while changed:
        changed = False
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                a, b = cells[i], cells[j]
                # Merge horizontally if same y/h and they touch on x.
                if abs(a.y - b.y) < eps and abs(a.h - b.h) < eps:
                    if abs(a.x2 - b.x) < eps:
                        cells[i] = Cell(a.x, a.y, a.w + b.w, a.h)
                        cells.pop(j)
                        changed = True
                        break
                    if abs(b.x2 - a.x) < eps:
                        cells[i] = Cell(b.x, a.y, a.w + b.w, a.h)
                        cells.pop(j)
                        changed = True
                        break
                # Merge vertically if same x/w and they touch on y.
                if abs(a.x - b.x) < eps and abs(a.w - b.w) < eps:
                    if abs(a.y2 - b.y) < eps:
                        cells[i] = Cell(a.x, a.y, a.w, a.h + b.h)
                        cells.pop(j)
                        changed = True
                        break
                    if abs(b.y2 - a.y) < eps:
                        cells[i] = Cell(a.x, b.y, a.w, a.h + b.h)
                        cells.pop(j)
                        changed = True
                        break
            if changed:
                break
    return cells


# ---------------------------------------------------------------------------
# Room subdivision -----------------------------------------------------------
# ---------------------------------------------------------------------------


def _split_cell(cell: Cell, num_rooms: int, rng: random.Random, min_side: float = 24.0) -> list[Cell]:
    """Recursively BSP-split a cell into ``num_rooms`` rooms."""
    if num_rooms <= 1:
        return [cell]
    horizontal = cell.w >= cell.h
    if cell.w < min_side * 2 and cell.h >= min_side * 2:
        horizontal = False
    elif cell.h < min_side * 2 and cell.w >= min_side * 2:
        horizontal = True
    elif cell.w < min_side * 2 and cell.h < min_side * 2:
        return [cell]

    left_count = max(1, num_rooms // 2)
    right_count = num_rooms - left_count
    ratio = left_count / num_rooms
    jitter = rng.uniform(-0.12, 0.12)
    ratio = max(0.3, min(0.7, ratio + jitter))

    if horizontal:
        cut = cell.w * ratio
        a = Cell(cell.x, cell.y, cut, cell.h)
        b = Cell(cell.x + cut, cell.y, cell.w - cut, cell.h)
    else:
        cut = cell.h * ratio
        a = Cell(cell.x, cell.y, cell.w, cut)
        b = Cell(cell.x, cell.y + cut, cell.w, cell.h - cut)

    return _split_cell(a, left_count, rng, min_side) + _split_cell(b, right_count, rng, min_side)


def _split_boundary(
    boundary: list[Cell],
    num_rooms: int,
    rng: random.Random,
    min_side: float = 24.0,
) -> list[Cell]:
    """Split a boundary (list of cells) into ``num_rooms`` non-overlapping rectangles."""
    total_area = sum(c.area() for c in boundary)
    rooms: list[Cell] = []
    remaining = num_rooms
    for idx, cell in enumerate(boundary):
        share = cell.area() / total_area
        if idx == len(boundary) - 1:
            count = max(1, remaining)
        else:
            count = max(1, round(share * num_rooms))
            count = min(count, remaining - (len(boundary) - idx - 1))
            count = max(1, count)
        rooms.extend(_split_cell(cell, count, rng, min_side=min_side))
        remaining -= count
        if remaining <= 0:
            remaining = 0
    if not rooms:
        rooms = [boundary[0]]
    return rooms


# ---------------------------------------------------------------------------
# Adjacency / connectivity helpers ------------------------------------------
# ---------------------------------------------------------------------------


def _shared_edge(a: Cell, b: Cell) -> tuple[str, float, float, float] | None:
    eps = 0.5
    if abs(a.x2 - b.x) < eps:
        lo = max(a.y, b.y)
        hi = min(a.y2, b.y2)
        if hi - lo > 30:
            return ("v", a.x2, lo, hi)
    if abs(b.x2 - a.x) < eps:
        lo = max(a.y, b.y)
        hi = min(a.y2, b.y2)
        if hi - lo > 30:
            return ("v", a.x, lo, hi)
    if abs(a.y2 - b.y) < eps:
        lo = max(a.x, b.x)
        hi = min(a.x2, b.x2)
        if hi - lo > 30:
            return ("h", a.y2, lo, hi)
    if abs(b.y2 - a.y) < eps:
        lo = max(a.x, b.x)
        hi = min(a.x2, b.x2)
        if hi - lo > 30:
            return ("h", a.y, lo, hi)
    return None


def _outer_perimeter(rect: Cell, boundary: list[Cell]) -> float:
    """Total length of ``rect``'s edges that lie on the outer boundary."""
    eps = 0.5
    outer = 0.0
    edges = [
        ("h", rect.y, rect.x, rect.x2),
        ("h", rect.y2, rect.x, rect.x2),
        ("v", rect.x, rect.y, rect.y2),
        ("v", rect.x2, rect.y, rect.y2),
    ]
    for orient, fixed, lo, hi in edges:
        seg_len = hi - lo
        # Subtract any portion of the edge that is shared with another boundary cell.
        shared = 0.0
        for other in boundary:
            if other is rect:
                continue
            edge = _shared_edge(rect, other)
            if edge is None:
                continue
            o2, f2, lo2, hi2 = edge
            if o2 != orient or abs(f2 - fixed) > eps:
                continue
            shared += max(0.0, min(hi, hi2) - max(lo, lo2))
        outer += max(0.0, seg_len - shared)
    return outer


def _is_in_boundary_perimeter(rect: Cell, boundary: list[Cell]) -> bool:
    return _outer_perimeter(rect, boundary) > 60.0


# ---------------------------------------------------------------------------
# Entrance + room-type assignment -------------------------------------------
# ---------------------------------------------------------------------------


def _pick_entrance(rooms: list[Room], boundary: list[Cell], rng: random.Random) -> tuple[
    int, tuple[str, float, float, float]
] | None:
    """Pick an entrance: an outer edge of one of the rooms, near the lower side."""
    eps = 0.5
    candidates: list[tuple[int, tuple[str, float, float, float], float]] = []
    for idx, room in enumerate(rooms):
        rect = room.rect
        edges = [
            ("h", rect.y, rect.x, rect.x2),
            ("h", rect.y2, rect.x, rect.x2),
            ("v", rect.x, rect.y, rect.y2),
            ("v", rect.x2, rect.y, rect.y2),
        ]
        for orient, fixed, lo, hi in edges:
            # Check the edge lies on the outer boundary.
            shared_inside = False
            for other in boundary:
                if other is rect:
                    continue
                e = _shared_edge(rect, other)
                if e is None:
                    continue
                o2, f2, lo2, hi2 = e
                if o2 == orient and abs(f2 - fixed) < eps:
                    if max(lo, lo2) < min(hi, hi2):
                        # Edge partially shared with another inside cell — skip if fully shared.
                        if min(hi, hi2) - max(lo, lo2) > (hi - lo) - 10:
                            shared_inside = True
                            break
            if shared_inside:
                continue
            if hi - lo < 60:
                continue
            # Prefer entrances on the lower edge (architecturally common).
            score = 1.0 if (orient == "h" and fixed > rect.y + 1) else 0.4
            score += rng.uniform(0.0, 0.3)
            candidates.append((idx, (orient, fixed, lo, hi), score))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[2], reverse=True)
    pick = candidates[0]
    return pick[0], pick[1]


def _assign_room_types(
    rooms: list[Room],
    boundary: list[Cell],
    entrance_room_idx: int,
    requested_types: list[str],
    rng: random.Random,
) -> None:
    """Greedy semantic assignment of room types.

    We honour ``requested_types`` if provided and feasible; otherwise we pick
    a balanced default mix.
    """
    n = len(rooms)
    # Compute outer perimeter for each room (used by kitchen/balcony).
    for r in rooms:
        r.perimeter_outer = _outer_perimeter(r.rect, boundary)
        r.has_outer_wall = r.perimeter_outer > 30.0

    # Figure out which types we want.
    if requested_types:
        plan = list(requested_types)[:n]
    else:
        # A sensible default sequence.
        defaults = ["hall", "living", "kitchen", "bedroom", "bathroom", "bedroom", "balcony"]
        plan = []
        for t in defaults:
            if len(plan) >= n:
                break
            plan.append(t)

    # Pad with bedrooms if needed.
    while len(plan) < n:
        plan.append("bedroom")

    used = [False] * n
    assignments: dict[int, str] = {}

    def assign(idx: int, room_type: str) -> None:
        rooms[idx].type = room_type
        used[idx] = True
        assignments[idx] = room_type

    rooms[entrance_room_idx].near_entrance = True

    # Pass 1: hall — must be the entrance room if requested.
    if "hall" in plan:
        assign(entrance_room_idx, "hall")

    # Pass 2: living — adjacent to hall, large, has outer wall preferred.
    def _adjacent(i: int, j: int) -> bool:
        return _shared_edge(rooms[i].rect, rooms[j].rect) is not None

    def _adjacent_to(i: int, type_set: set[str]) -> bool:
        return any(used[j] and rooms[j].type in type_set and _adjacent(i, j) for j in range(n))

    def _pick(filt) -> int | None:
        cands = [i for i in range(n) if not used[i] and filt(i)]
        if not cands:
            return None
        # Prefer largest by area for "big" types (living), smallest for bathroom.
        rng.shuffle(cands)
        return cands[0]

    if "living" in plan:
        # Prefer adjacent to hall, large area.
        cands = [
            i for i in range(n)
            if not used[i] and (entrance_room_idx not in assignments or _adjacent(i, entrance_room_idx))
        ]
        if not cands:
            cands = [i for i in range(n) if not used[i]]
        cands.sort(key=lambda i: rooms[i].rect.area(), reverse=True)
        if cands:
            assign(cands[0], "living")

    if "kitchen" in plan:
        idx = _pick(lambda i: _adjacent_to(i, {"living", "hall"}) and rooms[i].has_outer_wall)
        if idx is None:
            idx = _pick(lambda i: rooms[i].has_outer_wall)
        if idx is None:
            idx = _pick(lambda i: True)
        if idx is not None:
            assign(idx, "kitchen")

    bathrooms_planned = sum(1 for t in plan if t == "bathroom")
    bedrooms_planned = sum(1 for t in plan if t == "bedroom")
    balconies_planned = sum(1 for t in plan if t == "balcony")

    # Bedrooms: smaller, non-adjacent to entrance, have an outer wall (window).
    placed_bedrooms = 0
    while placed_bedrooms < bedrooms_planned:
        idx = _pick(
            lambda i: rooms[i].has_outer_wall
            and (entrance_room_idx not in assignments or not _adjacent(i, entrance_room_idx))
        )
        if idx is None:
            idx = _pick(lambda i: rooms[i].has_outer_wall)
        if idx is None:
            idx = _pick(lambda i: True)
        if idx is None:
            break
        assign(idx, "bedroom")
        placed_bedrooms += 1

    # Bathrooms: small, adjacent to a bedroom or hall.
    placed_bathrooms = 0
    while placed_bathrooms < bathrooms_planned:
        idx = _pick(lambda i: _adjacent_to(i, {"bedroom", "hall"}))
        if idx is None:
            idx = _pick(lambda i: True)
        if idx is None:
            break
        assign(idx, "bathroom")
        placed_bathrooms += 1

    # Balconies: must be on perimeter (rich outer wall).
    placed_balconies = 0
    while placed_balconies < balconies_planned:
        idx = _pick(lambda i: rooms[i].perimeter_outer > 80.0)
        if idx is None:
            break
        assign(idx, "balcony")
        placed_balconies += 1

    # Anything left becomes an extra bedroom or living.
    for i in range(n):
        if not used[i]:
            rooms[i].type = "bedroom" if rooms[i].has_outer_wall else "living"


# ---------------------------------------------------------------------------
# Top-level planning entrypoint ---------------------------------------------
# ---------------------------------------------------------------------------


def plan_layout(
    width: int = 640,
    height: int = 480,
    num_rooms: int = 5,
    boundary_shape: str = "auto",
    room_types: list[str] | None = None,
    seed: int | None = None,
) -> PlannedLayout:
    """Generate a semantic floor plan.

    Parameters
    ----------
    width, height : int
        Canvas size in pixels.
    num_rooms : int
        Desired total number of rooms.
    boundary_shape : str
        One of ``rect``, ``L``, ``T``, ``U``, ``plus``, ``random``, ``auto``.
    room_types : list[str] | None
        Preferred room types in priority order. Unrecognised types are
        ignored. If ``None`` a sensible default mix is used.
    seed : int | None
        Reproducible randomness.
    """
    rng = random.Random(seed)
    shape = _pick_shape(boundary_shape, rng)
    boundary = _make_boundary(shape, width, height, rng)

    # Subdivide into ``num_rooms`` rooms.
    # Scale min_side and tiny-cell threshold to canvas dimensions so the
    # planner works for both 128×128 (training tiles) and 640×480 (UI demo).
    min_dim = min(width, height)
    min_side = max(18.0, min_dim * 0.12)
    min_area = max(160.0, (min_dim * 0.10) ** 2)
    rooms_cells = _split_boundary(boundary, max(1, num_rooms), rng, min_side=min_side)
    rooms_cells = [c for c in rooms_cells if c.w * c.h > min_area]
    if not rooms_cells:
        # Fallback: keep the boundary cells themselves so we always have at
        # least one room.
        rooms_cells = list(boundary)
    rooms_cells.sort(key=lambda c: (c.y, c.x))

    rooms: list[Room] = [Room(rect=c) for c in rooms_cells]

    # Pick entrance — a perimeter edge of a room near the lower side.
    entrance: tuple[str, float, float, float] | None = None
    entrance_room_idx = 0
    pick = _pick_entrance(rooms, boundary, rng)
    if pick is not None:
        entrance_room_idx, edge = pick
        # Place entrance at the centre of the chosen edge.
        orient, fixed, lo, hi = edge
        centre = (lo + hi) / 2.0
        half = 18.0
        entrance = (orient, fixed, max(lo, centre - half), min(hi, centre + half))

    # Filter requested types to known ones.
    if room_types:
        clean = [t for t in room_types if t in ROOM_TYPES]
    else:
        clean = []
    _assign_room_types(rooms, boundary, entrance_room_idx, clean, rng)

    doors = _plan_doors(rooms, entrance_room_idx, entrance, rng)
    windows = _plan_windows(rooms, boundary, rng)

    return PlannedLayout(
        width=width,
        height=height,
        boundary_cells=boundary,
        rooms=rooms,
        doors=doors,
        windows=windows,
        entrance=entrance,
        boundary_shape=shape,
        seed=seed,
        room_types_requested=clean,
    )


# ---------------------------------------------------------------------------
# Doors + windows ------------------------------------------------------------
# ---------------------------------------------------------------------------


def _plan_doors(
    rooms: list[Room], entrance_idx: int, entrance: tuple[str, float, float, float] | None,
    rng: random.Random,
) -> list[DoorSpec]:
    """Place a door between every adjacent pair of rooms + the entrance door."""
    doors: list[DoorSpec] = []
    n = len(rooms)
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            edge = _shared_edge(rooms[i].rect, rooms[j].rect)
            if edge is None:
                continue
            orient, fixed, lo, hi = edge
            if hi - lo < 40:
                continue
            centre = (lo + hi) / 2.0 + rng.uniform(-8.0, 8.0)
            half = 18.0
            door_lo = max(lo + 4, centre - half)
            door_hi = min(hi - 4, centre + half)
            if door_hi - door_lo < 20:
                continue
            doors.append(DoorSpec(a_idx=i, b_idx=j, orientation=orient, fixed=fixed, lo=door_lo, hi=door_hi))
            seen.add((i, j))

    # Entrance door: between outside (-1) and the entrance room.
    if entrance is not None:
        orient, fixed, lo, hi = entrance
        doors.append(DoorSpec(a_idx=-1, b_idx=entrance_idx, orientation=orient, fixed=fixed, lo=lo, hi=hi))
    return doors


def _plan_windows(rooms: list[Room], boundary: list[Cell], rng: random.Random) -> list[WindowSpec]:
    """Place 1-2 windows on outer walls of rooms that need them."""
    windows: list[WindowSpec] = []
    for idx, room in enumerate(rooms):
        if room.type in {"bathroom", "hall"}:
            # No window for these typically.
            if rng.random() > 0.2:
                continue
        if not room.has_outer_wall:
            continue
        rect = room.rect
        candidates: list[tuple[str, float, float, float]] = []
        for orient, fixed, lo, hi in (
            ("h", rect.y, rect.x + 12, rect.x2 - 12),
            ("h", rect.y2, rect.x + 12, rect.x2 - 12),
            ("v", rect.x, rect.y + 12, rect.y2 - 12),
            ("v", rect.x2, rect.y + 12, rect.y2 - 12),
        ):
            # Edge must be on outer perimeter.
            shared = False
            eps = 0.5
            for other in boundary:
                e = _shared_edge(rect, other)
                if e is None:
                    continue
                o2, f2, lo2, hi2 = e
                if o2 == orient and abs(f2 - fixed) < eps:
                    if min(hi, hi2) - max(lo, lo2) > (hi - lo) - 10:
                        shared = True
                        break
            if shared:
                continue
            if hi - lo < 40:
                continue
            candidates.append((orient, fixed, lo, hi))

        if not candidates:
            continue
        n_windows = 1 if rng.random() < 0.65 else 2
        for orient, fixed, lo, hi in rng.sample(candidates, k=min(n_windows, len(candidates))):
            size = rng.uniform(40.0, min(75.0, hi - lo - 10))
            centre = rng.uniform(lo + size / 2 + 5, hi - size / 2 - 5)
            windows.append(
                WindowSpec(
                    orientation=orient,
                    fixed=fixed,
                    lo=centre - size / 2,
                    hi=centre + size / 2,
                    room_idx=idx,
                )
            )
    return windows
