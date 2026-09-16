"""Predicate definitions for M2-A.1 offline analysis.

P0-P4 predicates operate on CanonicalState (grid + objects + actions).
Each predicate is a function that takes a state and arguments, returns bool.

Predicate levels:
  P0: ActionEnabled(action_id)
  P1: InBounds, Occupied, Empty, ColorAt, OnBorder
  P2: Front, FrontIsEmpty, FrontHasObject, Adjacent
  P3: Connected4, ObjectCells, Area, BBox, TouchesBorder
  P4: AdjacentObjects, RelativePosition, Distance

P5/P6 (counts, temporal, parity, finite modes) are deferred.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.types import CanonicalState, GridObject

# Type aliases
Cell = tuple[int, int]  # (row, col)
Direction = str  # "up", "down", "left", "right"

DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass(frozen=True)
class PredicateFact:
    """A grounded predicate evaluation result."""

    predicate: str  # e.g. "ColorAt"
    args: tuple  # e.g. ((0, 1), 3)
    truth: bool


def _grid_dims(state: CanonicalState) -> tuple[int, int]:
    """Return (height, width) of the grid."""
    if not state.grid:
        return (0, 0)
    return (len(state.grid), len(state.grid[0]) if state.grid[0] else 0)


def _in_bounds(cell: Cell, height: int, width: int) -> bool:
    """Check if a cell coordinate is within grid bounds."""
    return 0 <= cell[0] < height and 0 <= cell[1] < width


def _cell_color(state: CanonicalState, cell: Cell) -> int:
    """Get the palette index at a cell. Assumes cell is in bounds."""
    return state.grid[cell[0]][cell[1]]


def _is_background(color: int) -> bool:
    """Check if a color is background (0)."""
    return color == 0


# --- P0: Action predicates ---


def action_enabled(state: CanonicalState, action_id: int) -> bool:
    """P0: Check if an action is available in the current state.

    Uses CanonicalState.available_actions (from SDK), NOT the selected action.
    """
    return any(a.action_id == action_id for a in state.available_actions)


# --- P1: Local grid predicates ---


def in_bounds(state: CanonicalState, cell: Cell) -> bool:
    """P1: Check if a cell is within the grid bounds."""
    h, w = _grid_dims(state)
    return _in_bounds(cell, h, w)


def occupied(state: CanonicalState, cell: Cell) -> bool:
    """P1: Check if a cell is occupied (non-background color)."""
    if not in_bounds(state, cell):
        return False
    return not _is_background(_cell_color(state, cell))


def empty(state: CanonicalState, cell: Cell) -> bool:
    """P1: Check if a cell is empty (background color 0)."""
    if not in_bounds(state, cell):
        return False
    return _is_background(_cell_color(state, cell))


def color_at(state: CanonicalState, cell: Cell, color: int) -> bool:
    """P1: Check if a cell has a specific palette index."""
    if not in_bounds(state, cell):
        return False
    return _cell_color(state, cell) == color


def on_border(state: CanonicalState, cell: Cell) -> bool:
    """P1: Check if a cell is on the grid border."""
    h, w = _grid_dims(state)
    r, c = cell
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


# --- P2: Directional predicates ---


def front_cell(cell: Cell, direction: Direction) -> Cell:
    """Compute the cell in the given direction from the source cell."""
    dr, dc = DIRECTIONS[direction]
    return (cell[0] + dr, cell[1] + dc)


def front(state: CanonicalState, cell: Cell, direction: Direction) -> bool:
    """P2: Check if the cell in front exists (is in bounds)."""
    fc = front_cell(cell, direction)
    return in_bounds(state, fc)


def front_is_empty(
    state: CanonicalState, cell: Cell, direction: Direction
) -> bool:
    """P2: Check if the cell in front is empty (background)."""
    fc = front_cell(cell, direction)
    if not in_bounds(state, fc):
        return False
    return empty(state, fc)


def front_has_object(
    state: CanonicalState, cell: Cell, direction: Direction
) -> bool:
    """P2: Check if the cell in front is occupied (non-background)."""
    fc = front_cell(cell, direction)
    if not in_bounds(state, fc):
        return False
    return occupied(state, fc)


def adjacent(
    state: CanonicalState, cell1: Cell, cell2: Cell, direction: Direction
) -> bool:
    """P2: Check if cell2 is adjacent to cell1 in the given direction."""
    expected = front_cell(cell1, direction)
    return cell2 == expected and in_bounds(state, cell2)


# --- P3: Connected components / object predicates ---


def _object_by_id(state: CanonicalState, obj_id: str) -> GridObject | None:
    """Find an object by its ID."""
    for obj in state.objects:
        if obj.object_id == obj_id:
            return obj
    return None


def connected4(state: CanonicalState, cell1: Cell, cell2: Cell) -> bool:
    """P3: Check if two cells are 4-connected (adjacent, same color)."""
    if not in_bounds(state, cell1) or not in_bounds(state, cell2):
        return False
    dr = abs(cell1[0] - cell2[0])
    dc = abs(cell1[1] - cell2[1])
    if dr + dc != 1:
        return False  # Not 4-adjacent
    return _cell_color(state, cell1) == _cell_color(state, cell2)


def object_cells(state: CanonicalState, obj_id: str) -> tuple[Cell, ...]:
    """P3: Return all cells belonging to an object.

    Returns empty tuple if object not found.
    """
    obj = _object_by_id(state, obj_id)
    if obj is None:
        return ()
    return obj.cells


def object_area(state: CanonicalState, obj_id: str) -> int:
    """P3: Return the area (cell count) of an object.

    Returns 0 if object not found.
    """
    obj = _object_by_id(state, obj_id)
    if obj is None:
        return 0
    return obj.area


def object_bbox(
    state: CanonicalState, obj_id: str
) -> tuple[int, int, int, int] | None:
    """P3: Return the bounding box of an object.

    Returns None if object not found.
    """
    obj = _object_by_id(state, obj_id)
    if obj is None:
        return None
    return obj.bbox


def touches_border(state: CanonicalState, obj_id: str) -> bool:
    """P3: Check if any cell of an object touches the grid border."""
    obj = _object_by_id(state, obj_id)
    if obj is None:
        return False
    h, w = _grid_dims(state)
    for r, c in obj.cells:
        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            return True
    return False


# --- P4: Object relation predicates ---


def _object_center(obj: GridObject) -> tuple[float, float]:
    """Compute the center of an object's bounding box."""
    min_r, min_c, max_r, max_c = obj.bbox
    return ((min_r + max_r) / 2, (min_c + max_c) / 2)


def adjacent_objects(
    state: CanonicalState,
    obj_id1: str,
    obj_id2: str,
    direction: Direction,
) -> bool:
    """P4: Check if object2 is adjacent to object1 in the given direction.

    Two objects are adjacent in direction d if any cell of obj1 has a
    front cell in direction d that belongs to obj2.
    """
    obj1 = _object_by_id(state, obj_id1)
    obj2 = _object_by_id(state, obj_id2)
    if obj1 is None or obj2 is None:
        return False
    obj2_cells = set(obj2.cells)
    for cell in obj1.cells:
        fc = front_cell(cell, direction)
        if fc in obj2_cells:
            return True
    return False


def relative_position(
    state: CanonicalState, obj_id1: str, obj_id2: str
) -> str | None:
    """P4: Return the relative position of obj2 w.r.t. obj1.

    Returns one of: "north", "south", "east", "west", "northeast",
    "northwest", "southeast", "southwest", "same", or None if either
    object is not found.
    """
    obj1 = _object_by_id(state, obj_id1)
    obj2 = _object_by_id(state, obj_id2)
    if obj1 is None or obj2 is None:
        return None
    c1 = _object_center(obj1)
    c2 = _object_center(obj2)
    dr = c2[0] - c1[0]
    dc = c2[1] - c1[1]
    parts: list[str] = []
    if dr < -0.5:
        parts.append("north")
    elif dr > 0.5:
        parts.append("south")
    if dc < -0.5:
        parts.append("west")
    elif dc > 0.5:
        parts.append("east")
    if not parts:
        return "same"
    return "".join(parts)


def distance(state: CanonicalState, obj_id1: str, obj_id2: str) -> int | None:
    """P4: Manhattan distance between object centers.

    Returns None if either object is not found.
    """
    obj1 = _object_by_id(state, obj_id1)
    obj2 = _object_by_id(state, obj_id2)
    if obj1 is None or obj2 is None:
        return None
    c1 = _object_center(obj1)
    c2 = _object_center(obj2)
    return int(abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]))


# --- Predicate catalog ---


@dataclass(frozen=True)
class PredicateInfo:
    """Metadata about a predicate."""

    name: str
    level: str  # "P0", "P1", "P2", "P3", "P4"
    description: str
    requires_grid: bool  # True if needs full canonical grid
    requires_objects: bool  # True if needs object decomposition


PREDICATE_CATALOG: dict[str, PredicateInfo] = {
    "ActionEnabled": PredicateInfo(
        "ActionEnabled", "P0",
        "Action is available in current state", True, False,
    ),
    "InBounds": PredicateInfo(
        "InBounds", "P1",
        "Cell is within grid bounds", True, False,
    ),
    "Occupied": PredicateInfo(
        "Occupied", "P1",
        "Cell has non-background color", True, False,
    ),
    "Empty": PredicateInfo(
        "Empty", "P1",
        "Cell has background color (0)", True, False,
    ),
    "ColorAt": PredicateInfo(
        "ColorAt", "P1",
        "Cell has specific palette index", True, False,
    ),
    "OnBorder": PredicateInfo(
        "OnBorder", "P1",
        "Cell is on grid border", True, False,
    ),
    "Front": PredicateInfo(
        "Front", "P2",
        "Cell in direction exists", True, False,
    ),
    "FrontIsEmpty": PredicateInfo(
        "FrontIsEmpty", "P2",
        "Cell in direction is background", True, False,
    ),
    "FrontHasObject": PredicateInfo(
        "FrontHasObject", "P2",
        "Cell in direction is occupied", True, False,
    ),
    "Adjacent": PredicateInfo(
        "Adjacent", "P2",
        "Cell2 is adjacent to cell1 in direction", True, False,
    ),
    "Connected4": PredicateInfo(
        "Connected4", "P3",
        "Two cells are 4-connected (adjacent, same color)", True, False,
    ),
    "ObjectCells": PredicateInfo(
        "ObjectCells", "P3",
        "Cells belonging to an object", True, True,
    ),
    "Area": PredicateInfo(
        "Area", "P3",
        "Object area (cell count)", True, True,
    ),
    "BBox": PredicateInfo(
        "BBox", "P3",
        "Object bounding box", True, True,
    ),
    "TouchesBorder": PredicateInfo(
        "TouchesBorder", "P3",
        "Object touches grid border", True, True,
    ),
    "AdjacentObjects": PredicateInfo(
        "AdjacentObjects", "P4",
        "Two objects are adjacent in direction", True, True,
    ),
    "RelativePosition": PredicateInfo(
        "RelativePosition", "P4",
        "Relative position of obj2 w.r.t. obj1", True, True,
    ),
    "Distance": PredicateInfo(
        "Distance", "P4",
        "Manhattan distance between object centers", True, True,
    ),
}


def available_predicates() -> list[str]:
    """Return all predicate names in the catalog."""
    return sorted(PREDICATE_CATALOG.keys())


def predicates_by_level(level: str) -> list[str]:
    """Return predicate names for a specific level (P0-P4)."""
    return sorted(
        name
        for name, info in PREDICATE_CATALOG.items()
        if info.level == level
    )


def grid_required_predicates() -> list[str]:
    """Return predicates that require full canonical grid."""
    return sorted(
        name
        for name, info in PREDICATE_CATALOG.items()
        if info.requires_grid
    )


def object_required_predicates() -> list[str]:
    """Return predicates that require object decomposition."""
    return sorted(
        name
        for name, info in PREDICATE_CATALOG.items()
        if info.requires_objects
    )
