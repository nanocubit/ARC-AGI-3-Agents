"""Test state canonicalization and delta computation."""

import pytest

from core.state_builder import build_canonical_state, compute_state_delta
from core.types import CanonicalState, GridObject, StateDelta
from arcengine import FrameData, GameState


class TestGridCanonical:
    """Tests for grid canonicalization."""

    def test_simple_grid_canonicalization(self) -> None:
        """Test basic grid extraction."""
        frame = FrameData(
            game_id="test_game",
            frame=[[1, 2], [3, 4]],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            win_levels=0,
            guid="",
            full_reset=False,
            available_actions=[],
        )
        state = build_canonical_state(frame)
        assert state.grid == ((1, 2), (3, 4))
        assert state.status == "NOT_PLAYED"

    def test_empty_grid(self) -> None:
        """Test handling of empty grid."""
        frame = FrameData(
            game_id="test",
            frame=[],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            win_levels=0,
            guid="",
            full_reset=False,
            available_actions=[],
        )
        state = build_canonical_state(frame)
        assert state.grid == ((),)

    def test_missing_grid_attribute(self) -> None:
        """Test handling when frame attribute is None."""
        frame = FrameData(
            game_id="test",
            frame=None,
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            win_levels=0,
            guid="",
            full_reset=False,
            available_actions=[],
        )
        state = build_canonical_state(frame)
        # Should produce some valid grid, even if empty
        assert isinstance(state.grid, tuple)

    def test_integer_normalization(self) -> None:
        """Test that float values are converted to int."""
        frame = FrameData(
            game_id="test",
            frame=[[1.0, 2.5], [3.9, 4.1]],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            win_levels=0,
            guid="",
            full_reset=False,
            available_actions=[],
        )
        state = build_canonical_state(frame)
        assert state.grid == ((1, 2), (3, 4))

    def test_irregular_grid_handling(self) -> None:
        """Test handling of grids with rows of different lengths."""
        frame = FrameData(
            game_id="test",
            frame=[[1, 2, 3], [4, 5]],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            win_levels=0,
            guid="",
            full_reset=False,
            available_actions=[],
        )
        state = build_canonical_state(frame)
        # Should pad shorter rows with 0
        assert len(state.grid[0]) == len(state.grid[1])


class TestConnectedComponents:
    """Tests for object extraction via flood fill."""

    def test_single_connected_component(self) -> None:
        """Test extraction of a single connected object."""
        frame = FrameData(
            game_id="test",
            frame=[[0, 1, 0], [0, 1, 0], [0, 0, 0]],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            win_levels=0,
            guid="",
            full_reset=False,
            available_actions=[],
        )
        state = build_canonical_state(frame)
        assert len(state.objects) == 1
        obj = state.objects[0]
        assert obj.color == 1
        assert obj.area == 3
        assert obj.cells == ((0, 1), (1, 1), (2, 1))

    def test_multiple_components(self) -> None:
        """Test extraction of multiple connected components."""
        frame = FrameData(
            game_id="test",
            frame=[[1, 0, 2], [0, 0, 0], [3, 0, 4]],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            win_levels=0,
            guid="",
            full_reset=False,
            available_actions=[],
        )
        state = build_canonical_state(frame)
        # Should extract 4 objects (colors 1, 2, 3, 4)
        assert len(state.objects) == 4

    def test_same_color_separated_by_another_color(self) -> None:
        """Test that same-color cells separated by another color are separate objects."""
        frame = FrameData(
            game_id="test",
            frame=[[1, 2, 1]],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            win_levels=0,
            guid="",
            full_reset=False,
            available_actions=[],
        )
        state = build_canonical_state(frame)
        # Should have 2 objects of color 1 (not connected) + 1 of color 2
        objects_color_1 = [o for o in state.objects if o.color == 1]
        assert len(objects_color_1) == 2

    def test_deterministic_object_ids(self) -> None:
        """Test that object IDs are deterministic."""
        frame = FrameData(
            game_id="test",
            frame=[[1, 1], [1, 1]],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            win_levels=0,
            guid="",
            full_reset=False,
            available_actions=[],
        )
        state1 = build_canonical_state(frame)
        state2 = build_canonical_state(frame)
        assert state1.objects[0].object_id == state2.objects[0].object_id

    def test_object_ordering_deterministic(self) -> None:
        """Test that objects are ordered deterministically."""
        frame = FrameData(
            game_id="test",
            frame=[[3, 0, 1], [0, 0, 0], [2, 0, 4]],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            win_levels=0,
            guid="",
            full_reset=False,
            available_actions=[],
        )
        state1 = build_canonical_state(frame)
        state2 = build_canonical_state(frame)
        assert [o.object_id for o in state1.objects] == [o.object_id for o in state2.objects]


class TestStateDelta:
    """Tests for state diff computation."""

    def test_changed_cell_delta(self) -> None:
        """Test detection of changed cells."""
        state_before = CanonicalState(
            grid=((1, 2), (3, 4)),
            objects=(),
            available_actions=(),
            status=None,
            state_hash="",
        )
        state_after = CanonicalState(
            grid=((1, 5), (3, 4)),
            objects=(),
            available_actions=(),
            status=None,
            state_hash="",
        )
        delta = compute_state_delta(state_before, state_after)
        assert len(delta.changed_cells) == 1
        assert delta.changed_cells[0] == ((0, 1), 2, 5)

    def test_object_addition(self) -> None:
        """Test detection of added objects."""
        obj1 = GridObject(
            object_id="obj_1", color=1, cells=((0, 0),), bbox=(0, 0, 0, 0), area=1
        )
        obj2 = GridObject(
            object_id="obj_2", color=2, cells=((1, 1),), bbox=(1, 1, 1, 1), area=1
        )

        state_before = CanonicalState(
            grid=((1, 0),),
            objects=(obj1,),
            available_actions=(),
            status=None,
            state_hash="",
        )
        state_after = CanonicalState(
            grid=((1, 2),),
            objects=(obj1, obj2),
            available_actions=(),
            status=None,
            state_hash="",
        )
        delta = compute_state_delta(state_before, state_after)
        assert "obj_2" in delta.added_objects

    def test_object_removal(self) -> None:
        """Test detection of removed objects."""
        obj1 = GridObject(
            object_id="obj_1", color=1, cells=((0, 0),), bbox=(0, 0, 0, 0), area=1
        )

        state_before = CanonicalState(
            grid=((1,),),
            objects=(obj1,),
            available_actions=(),
            status=None,
            state_hash="",
        )
        state_after = CanonicalState(
            grid=((0,),),
            objects=(),
            available_actions=(),
            status=None,
            state_hash="",
        )
        delta = compute_state_delta(state_before, state_after)
        assert "obj_1" in delta.removed_objects

    def test_status_change(self) -> None:
        """Test detection of status changes."""
        state_before = CanonicalState(
            grid=((1,),),
            objects=(),
            available_actions=(),
            status="NOT_PLAYED",
            state_hash="",
        )
        state_after = CanonicalState(
            grid=((1,),),
            objects=(),
            available_actions=(),
            status="WIN",
            state_hash="",
        )
        delta = compute_state_delta(state_before, state_after)
        assert delta.status_changed is True
