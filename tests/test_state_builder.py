"""Test state canonicalization and delta computation."""

import pytest
from arcengine import FrameData, GameState
from pydantic import ValidationError

from core.state_builder import build_canonical_state, compute_state_delta
from core.types import CanonicalState, GridObject


def make_frame(
    grid: list[list[int]],
    state: GameState = GameState.NOT_PLAYED,
    levels_completed: int = 0,
    available_actions: list[int] | None = None,
) -> FrameData:
    """Helper: create a FrameData from a 2D palette-index grid.

    Wraps the 2D grid in an outer list (frame sequence with one frame).
    """
    return FrameData(
        game_id="test",
        frame=[grid],
        state=state,
        levels_completed=levels_completed,
        win_levels=0,
        guid="",
        full_reset=False,
        available_actions=available_actions or [],
    )


class TestGridCanonical:
    """Tests for grid canonicalization."""

    def test_simple_grid_canonicalization(self) -> None:
        """Test basic grid extraction."""
        frame = make_frame([[1, 2], [3, 4]])
        state = build_canonical_state(frame)
        assert state.grid == ((1, 2), (3, 4))
        assert state.status == "NOT_PLAYED"

    def test_empty_grid_raises(self) -> None:
        """Test that empty frame raises ValueError (fail-closed)."""
        frame = make_frame([])
        with pytest.raises(ValueError):
            build_canonical_state(frame)

    def test_none_frame_raises(self) -> None:
        """Test that missing frame data raises ValueError.

        FrameData Pydantic schema requires frame to be a list, so we
        test with an empty frame list which our code should reject.
        """
        frame = FrameData(
            game_id="test",
            frame=[],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
        )
        with pytest.raises(ValueError):
            build_canonical_state(frame)

    def test_large_palette_indices_preserved(self) -> None:
        """Test that large palette indices are preserved (not clamped to 0-9)."""
        frame = make_frame([[10, 15], [20, 255]])
        state = build_canonical_state(frame)
        assert state.grid == ((10, 15), (20, 255))

    def test_irregular_grid_raises(self) -> None:
        """Test that irregular grids raise ValueError (fail-closed)."""
        frame = make_frame([[1, 2, 3], [4, 5]])
        with pytest.raises(ValueError, match="Irregular grid"):
            build_canonical_state(frame)

    def test_negative_cell_raises(self) -> None:
        """Test that negative palette indices raise ValueError."""
        frame = make_frame([[1, -1], [3, 4]])
        with pytest.raises(ValueError, match="negative"):
            build_canonical_state(frame)

    def test_non_int_cell_rejected_by_sdk(self) -> None:
        """Test that non-integer cells are rejected by Pydantic validation.

        The SDK FrameData schema requires list[list[list[int]]], so
        non-integer cells are rejected before our code runs. This test
        documents that contract.
        """
        with pytest.raises(ValidationError):
            FrameData(
                game_id="test",
                frame=[["x", 1], [3, 4]],
                state=GameState.NOT_PLAYED,
                levels_completed=0,
            )

    def test_float_cell_rejected_by_sdk(self) -> None:
        """Test that float cells are rejected by Pydantic validation.

        The SDK FrameData schema requires list[list[list[int]]], so
        fractional floats are rejected before our code runs.
        """
        with pytest.raises(ValidationError):
            FrameData(
                game_id="test",
                frame=[[[1.5, 2.0], [3.0, 4.0]]],
                state=GameState.NOT_PLAYED,
                levels_completed=0,
            )


class TestFrameSequence:
    """Tests for multi-frame sequence handling."""

    def test_last_frame_selected(self) -> None:
        """Test that the last frame in a sequence is selected."""
        frame = FrameData(
            game_id="test",
            frame=[
                [[1, 2], [3, 4]],  # first frame
                [[5, 6], [7, 8]],  # last frame
            ],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            available_actions=[],
        )
        state = build_canonical_state(frame)
        assert state.grid == ((5, 6), (7, 8))

    def test_same_frame_sequence_same_hash(self) -> None:
        """Test that the same multi-frame sequence produces the same hash."""
        frame1 = FrameData(
            game_id="test",
            frame=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            available_actions=[],
        )
        frame2 = FrameData(
            game_id="test",
            frame=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            state=GameState.NOT_PLAYED,
            levels_completed=0,
            available_actions=[],
        )
        state1 = build_canonical_state(frame1)
        state2 = build_canonical_state(frame2)
        assert state1.state_hash == state2.state_hash


class TestConnectedComponents:
    """Tests for object extraction via flood fill."""

    def test_single_connected_component(self) -> None:
        """Test extraction of a single connected object."""
        frame = make_frame(
            [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        )
        state = build_canonical_state(frame)
        assert len(state.objects) == 1
        obj = state.objects[0]
        assert obj.color == 1
        assert obj.area == 3
        assert obj.cells == ((0, 1), (1, 1), (2, 1))

    def test_multiple_components(self) -> None:
        """Test extraction of multiple connected components."""
        frame = make_frame(
            [[1, 0, 2], [0, 0, 0], [3, 0, 4]]
        )
        state = build_canonical_state(frame)
        # Should extract 4 objects (colors 1, 2, 3, 4)
        assert len(state.objects) == 4

    def test_same_color_separated_by_another_color(self) -> None:
        """Test that same-color cells separated by another color are separate objects."""
        frame = make_frame([[1, 2, 1]])
        state = build_canonical_state(frame)
        # Should have 2 objects of color 1 (not connected) + 1 of color 2
        objects_color_1 = [o for o in state.objects if o.color == 1]
        assert len(objects_color_1) == 2

    def test_deterministic_object_ids(self) -> None:
        """Test that object IDs are deterministic."""
        frame = make_frame([[1, 1], [1, 1]])
        state1 = build_canonical_state(frame)
        state2 = build_canonical_state(frame)
        assert state1.objects[0].object_id == state2.objects[0].object_id

    def test_object_ordering_deterministic(self) -> None:
        """Test that objects are ordered deterministically."""
        frame = make_frame(
            [[3, 0, 1], [0, 0, 0], [2, 0, 4]]
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


class TestCanonicalActions:
    """Tests for canonical action extraction from available_actions."""

    def test_available_actions_extraction(self) -> None:
        """Test that integer action IDs are converted to CanonicalAction."""
        frame = make_frame(
            [[0, 0], [0, 0]],
            available_actions=[1, 6, 7],
        )
        state = build_canonical_state(frame)
        assert len(state.available_actions) == 3
        ids = [a.action_id for a in state.available_actions]
        assert ids == [1, 6, 7]
        names = [a.action_name for a in state.available_actions]
        assert names == ["ACTION1", "ACTION6", "ACTION7"]

    def test_action6_no_fabricated_coordinates(self) -> None:
        """Test that ACTION6 has complex_data=None when from available_actions."""
        frame = make_frame(
            [[0, 0], [0, 0]],
            available_actions=[6],
        )
        state = build_canonical_state(frame)
        assert len(state.available_actions) == 1
        action = state.available_actions[0]
        assert action.action_name == "ACTION6"
        assert action.complex_data is None

    def test_actions_sorted_deterministically(self) -> None:
        """Test that actions are sorted by action_id."""
        frame = make_frame(
            [[0, 0], [0, 0]],
            available_actions=[7, 1, 3],
        )
        state = build_canonical_state(frame)
        ids = [a.action_id for a in state.available_actions]
        assert ids == [1, 3, 7]

    def test_empty_available_actions(self) -> None:
        """Test that no available actions produces empty tuple."""
        frame = make_frame([[0, 0], [0, 0]])
        state = build_canonical_state(frame)
        assert state.available_actions == ()

    def test_reset_action(self) -> None:
        """Test that RESET (id=0) is properly converted."""
        frame = make_frame(
            [[0, 0], [0, 0]],
            available_actions=[0],
        )
        state = build_canonical_state(frame)
        assert len(state.available_actions) == 1
        assert state.available_actions[0].action_name == "RESET"
        assert state.available_actions[0].action_id == 0


class TestStateHashContract:
    """Tests for state hash determinism through the full pipeline."""

    def test_same_frame_same_hash(self) -> None:
        """Test that the same FrameData produces the same state_hash."""
        frame1 = make_frame([[1, 2], [3, 4]])
        frame2 = make_frame([[1, 2], [3, 4]])
        state1 = build_canonical_state(frame1)
        state2 = build_canonical_state(frame2)
        assert state1.state_hash == state2.state_hash

    def test_one_cell_changed_different_hash(self) -> None:
        """Test that changing one cell changes the state_hash."""
        frame1 = make_frame([[1, 2], [3, 4]])
        frame2 = make_frame([[1, 2], [3, 5]])
        state1 = build_canonical_state(frame1)
        state2 = build_canonical_state(frame2)
        assert state1.state_hash != state2.state_hash

    def test_positional_change_different_hash(self) -> None:
        """Test that swapping cell positions changes the state_hash."""
        frame1 = make_frame([[1, 2], [3, 4]])
        frame2 = make_frame([[2, 1], [3, 4]])
        state1 = build_canonical_state(frame1)
        state2 = build_canonical_state(frame2)
        assert state1.state_hash != state2.state_hash

    def test_status_change_different_hash(self) -> None:
        """Test that changing game status changes the state_hash."""
        frame1 = make_frame([[1, 2], [3, 4]], state=GameState.NOT_FINISHED)
        frame2 = make_frame([[1, 2], [3, 4]], state=GameState.WIN)
        state1 = build_canonical_state(frame1)
        state2 = build_canonical_state(frame2)
        assert state1.state_hash != state2.state_hash
