"""Test M2-A.1 predicate evaluation on synthetic canonical states."""

from __future__ import annotations

from core.types import CanonicalAction, CanonicalState, GridObject
from m2.predicates import (
    PREDICATE_CATALOG,
    action_enabled,
    adjacent,
    adjacent_objects,
    color_at,
    connected4,
    distance,
    empty,
    front,
    front_has_object,
    front_is_empty,
    grid_required_predicates,
    in_bounds,
    object_area,
    object_bbox,
    object_cells,
    occupied,
    on_border,
    predicates_by_level,
    relative_position,
    touches_border,
)


def make_state(
    grid: tuple[tuple[int, ...], ...],
    objects: tuple[GridObject, ...] = (),
    actions: tuple[int, ...] = (0, 1),
    status: str | None = "NOT_PLAYED",
) -> CanonicalState:
    """Build a CanonicalState for testing."""
    available = tuple(
        CanonicalAction(
            action_name=f"ACTION{aid}" if aid > 0 else "RESET",
            action_id=aid,
            complex_data=None,
        )
        for aid in actions
    )
    return CanonicalState(
        grid=grid,
        objects=objects,
        available_actions=available,
        status=status,
        state_hash="",  # Not needed for predicate tests
    )


def make_object(
    obj_id: str = "obj1",
    color: int = 1,
    cells: tuple[tuple[int, int], ...] = ((0, 0), (0, 1)),
    bbox: tuple[int, int, int, int] = (0, 0, 0, 1),
) -> GridObject:
    """Build a GridObject for testing."""
    return GridObject(
        object_id=obj_id,
        color=color,
        cells=cells,
        bbox=bbox,
        area=len(cells),
    )


class TestPredicateCatalog:
    """Test predicate catalog metadata."""

    def test_all_levels_present(self) -> None:
        """P0-P4 should all have predicates."""
        for level in ["P0", "P1", "P2", "P3", "P4"]:
            preds = predicates_by_level(level)
            assert len(preds) > 0, f"No predicates for {level}"

    def test_all_predicates_require_grid(self) -> None:
        """All P0-P4 predicates should require grid (they operate on state)."""
        grid_req = grid_required_predicates()
        assert len(grid_req) == len(PREDICATE_CATALOG)

    def test_predicate_count(self) -> None:
        """Should have 18 predicates total."""
        assert len(PREDICATE_CATALOG) == 18


class TestP0Action:
    """P0: ActionEnabled predicate."""

    def test_action_enabled_true(self) -> None:
        state = make_state(grid=((0, 1), (2, 3)), actions=(1, 3))
        assert action_enabled(state, 1) is True
        assert action_enabled(state, 3) is True

    def test_action_enabled_false(self) -> None:
        state = make_state(grid=((0, 1), (2, 3)), actions=(1, 3))
        assert action_enabled(state, 0) is False
        assert action_enabled(state, 5) is False

    def test_uses_available_actions_not_selected(self) -> None:
        """ActionEnabled uses state.available_actions, not the selected action."""
        state = make_state(grid=((0, 1),), actions=(2,))
        assert action_enabled(state, 2) is True
        assert action_enabled(state, 1) is False


class TestP1Grid:
    """P1: Local grid predicates."""

    def test_in_bounds(self) -> None:
        state = make_state(grid=((0, 1), (2, 3)))
        assert in_bounds(state, (0, 0)) is True
        assert in_bounds(state, (1, 1)) is True
        assert in_bounds(state, (2, 0)) is False
        assert in_bounds(state, (0, 2)) is False

    def test_occupied(self) -> None:
        state = make_state(grid=((0, 1), (2, 3)))
        assert occupied(state, (0, 0)) is False  # background
        assert occupied(state, (0, 1)) is True   # color 1
        assert occupied(state, (1, 0)) is True   # color 2

    def test_empty(self) -> None:
        state = make_state(grid=((0, 1), (2, 3)))
        assert empty(state, (0, 0)) is True   # background
        assert empty(state, (0, 1)) is False  # color 1

    def test_color_at(self) -> None:
        state = make_state(grid=((0, 1), (2, 3)))
        assert color_at(state, (0, 0), 0) is True
        assert color_at(state, (0, 1), 1) is True
        assert color_at(state, (1, 0), 2) is True
        assert color_at(state, (0, 1), 2) is False

    def test_on_border(self) -> None:
        state = make_state(grid=((0, 1, 2), (3, 4, 5), (6, 7, 8)))
        assert on_border(state, (0, 0)) is True   # corner
        assert on_border(state, (1, 0)) is True   # left edge
        assert on_border(state, (2, 2)) is True   # corner
        assert on_border(state, (1, 1)) is False  # center


class TestP2Directional:
    """P2: Directional predicates."""

    def test_front_exists(self) -> None:
        state = make_state(grid=((0, 1), (2, 3)))
        assert front(state, (0, 0), "right") is True  # (0,1) exists
        assert front(state, (0, 0), "down") is True   # (1,0) exists
        assert front(state, (0, 0), "up") is False    # (-1,0) out of bounds
        assert front(state, (0, 0), "left") is False  # (0,-1) out of bounds

    def test_front_is_empty(self) -> None:
        state = make_state(grid=((0, 1), (2, 3)))
        assert front_is_empty(state, (0, 1), "down") is False  # (1,1)=3, not empty
        assert front_is_empty(state, (1, 1), "up") is False   # (0,1)=1, not empty
        assert front_is_empty(state, (0, 0), "right") is False  # (0,1)=1, not empty

    def test_front_has_object(self) -> None:
        state = make_state(grid=((0, 1), (2, 3)))
        assert front_has_object(state, (0, 0), "right") is True  # (0,1)=1
        assert front_has_object(state, (0, 0), "down") is True   # (1,0)=2
        assert front_has_object(state, (1, 1), "up") is True     # (0,1)=1

    def test_adjacent(self) -> None:
        state = make_state(grid=((0, 1), (2, 3)))
        assert adjacent(state, (0, 0), (0, 1), "right") is True
        assert adjacent(state, (0, 0), (1, 0), "down") is True
        assert adjacent(state, (0, 0), (1, 1), "right") is False  # not adjacent


class TestP3Objects:
    """P3: Connected components / object predicates."""

    def test_connected4_true(self) -> None:
        state = make_state(grid=((1, 1), (0, 0)))
        assert connected4(state, (0, 0), (0, 1)) is True  # same color, adjacent

    def test_connected4_false_different_color(self) -> None:
        state = make_state(grid=((1, 2), (0, 0)))
        assert connected4(state, (0, 0), (0, 1)) is False

    def test_connected4_false_not_adjacent(self) -> None:
        state = make_state(grid=((1, 1), (1, 0)))
        assert connected4(state, (0, 0), (1, 0)) is True  # diagonal? no, (1,0) is below
        # (0,0) and (1,0) are vertically adjacent
        assert connected4(state, (0, 0), (1, 1)) is False  # diagonal, not 4-adjacent

    def test_object_cells(self) -> None:
        obj = make_object(cells=((0, 0), (0, 1)))
        state = make_state(grid=((1, 1), (0, 0)), objects=(obj,))
        cells = object_cells(state, "obj1")
        assert len(cells) == 2
        assert (0, 0) in cells
        assert (0, 1) in cells

    def test_object_cells_not_found(self) -> None:
        state = make_state(grid=((0, 0),))
        assert object_cells(state, "nonexistent") == ()

    def test_object_area(self) -> None:
        obj = make_object(cells=((0, 0), (0, 1), (1, 0)))
        state = make_state(grid=((1, 1), (1, 0)), objects=(obj,))
        assert object_area(state, "obj1") == 3

    def test_object_bbox(self) -> None:
        obj = make_object(cells=((1, 2), (3, 4)), bbox=(1, 2, 3, 4))
        state = make_state(grid=((0, 0, 0), (0, 1, 0), (0, 0, 0)), objects=(obj,))
        assert object_bbox(state, "obj1") == (1, 2, 3, 4)

    def test_touches_border_true(self) -> None:
        obj = make_object(cells=((0, 0), (0, 1)), bbox=(0, 0, 0, 1))
        state = make_state(grid=((1, 1), (0, 0)), objects=(obj,))
        assert touches_border(state, "obj1") is True

    def test_touches_border_false(self) -> None:
        obj = make_object(cells=((1, 1),), bbox=(1, 1, 1, 1))
        state = make_state(
            grid=((0, 0, 0), (0, 1, 0), (0, 0, 0)), objects=(obj,)
        )
        assert touches_border(state, "obj1") is False


class TestP4Relations:
    """P4: Object relation predicates."""

    def test_adjacent_objects_true(self) -> None:
        obj1 = make_object("o1", cells=((0, 0),), bbox=(0, 0, 0, 0))
        obj2 = make_object("o2", cells=((0, 1),), bbox=(0, 1, 0, 1))
        state = make_state(grid=((1, 2),), objects=(obj1, obj2))
        assert adjacent_objects(state, "o1", "o2", "right") is True
        assert adjacent_objects(state, "o2", "o1", "left") is True

    def test_adjacent_objects_false(self) -> None:
        obj1 = make_object("o1", cells=((0, 0),), bbox=(0, 0, 0, 0))
        obj2 = make_object("o2", cells=((1, 1),), bbox=(1, 1, 1, 1))
        state = make_state(grid=((1, 0), (0, 2)), objects=(obj1, obj2))
        assert adjacent_objects(state, "o1", "o2", "right") is False

    def test_relative_position(self) -> None:
        obj1 = make_object("o1", cells=((1, 1),), bbox=(1, 1, 1, 1))
        obj2 = make_object("o2", cells=((0, 1),), bbox=(0, 1, 0, 1))
        state = make_state(grid=((0, 2), (1, 1)), objects=(obj1, obj2))
        pos = relative_position(state, "o1", "o2")
        assert pos == "north"

    def test_relative_position_same(self) -> None:
        obj1 = make_object("o1", cells=((0, 0),), bbox=(0, 0, 0, 0))
        obj2 = make_object("o2", cells=((0, 0),), bbox=(0, 0, 0, 0))
        state = make_state(grid=((1, 1),), objects=(obj1, obj2))
        pos = relative_position(state, "o1", "o2")
        assert pos == "same"

    def test_distance(self) -> None:
        obj1 = make_object("o1", cells=((0, 0),), bbox=(0, 0, 0, 0))
        obj2 = make_object("o2", cells=((0, 2),), bbox=(0, 2, 0, 2))
        state = make_state(grid=((1, 0, 2),), objects=(obj1, obj2))
        assert distance(state, "o1", "o2") == 2

    def test_distance_none_for_missing(self) -> None:
        obj1 = make_object("o1", cells=((0, 0),), bbox=(0, 0, 0, 0))
        state = make_state(grid=((1, 0),), objects=(obj1,))
        assert distance(state, "o1", "o2") is None
