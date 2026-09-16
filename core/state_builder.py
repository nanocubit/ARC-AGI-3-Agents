"""Convert SDK FrameData to canonical state representation."""

import logging

from arcengine import FrameData, GameAction

from .hashing import state_hash
from .types import CanonicalAction, CanonicalState, GridObject, StateDelta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame extraction and grid normalization
# ---------------------------------------------------------------------------


def _extract_rendered_frame(frame_data: FrameData) -> list[list[int]]:
    """Select the last rendered frame from the frame sequence.

    SDK FrameData.frame is a sequence of 2D palette-index grids rendered
    during a single action. We canonicalize only the final observed state.

    Args:
        frame_data: SDK frame data object.

    Returns:
        2D list of palette indices (last rendered frame).

    Raises:
        ValueError: If frame is missing, empty, or has no valid frames.
    """
    if not hasattr(frame_data, "frame") or frame_data.frame is None:
        raise ValueError("FrameData has no frame attribute or frame is None")

    frames = frame_data.frame
    if not frames:
        raise ValueError("FrameData.frame is empty")

    last_frame = frames[-1]
    if last_frame is None:
        raise ValueError("Last rendered frame is None")
    if not last_frame:
        raise ValueError("Last rendered frame is empty")

    return last_frame


def _normalize_palette_index_grid(
    frame: list[list[int]],
) -> tuple[tuple[int, ...], ...]:
    """Convert a 2D list of palette indices to immutable tuple form.

    Validates that every cell is a non-negative integer. Does not
    silently coerce invalid values to 0.

    Args:
        frame: 2D list of palette index values.

    Returns:
        Tuple of tuples representing the normalized grid.

    Raises:
        ValueError: If any cell is not a non-negative integer, or if
            rows have irregular lengths.
    """
    if not frame:
        return ((),)

    normalized: list[list[int]] = []
    expected_width: int | None = None

    for row_idx, row in enumerate(frame):
        if row is None:
            raise ValueError(f"Row {row_idx} is None")

        normalized_row: list[int] = []
        if expected_width is None:
            expected_width = len(row)

        if len(row) != expected_width:
            raise ValueError(
                f"Irregular grid: row {row_idx} has width {len(row)}, "
                f"expected {expected_width}"
            )

        for col_idx, cell in enumerate(row):
            # Reject bools (isinstance(True, int) is True in Python)
            if isinstance(cell, bool):
                raise TypeError(
                    f"Cell ({row_idx}, {col_idx}) is bool, not int: {cell}"
                )
            if not isinstance(cell, (int, float)):
                raise TypeError(
                    f"Cell ({row_idx}, {col_idx}) is not numeric: {cell!r}"
                )
            cell_int = int(cell)
            if cell_int < 0:
                raise ValueError(
                    f"Cell ({row_idx}, {col_idx}) is negative: {cell_int}"
                )
            normalized_row.append(cell_int)

        if normalized_row:
            normalized.append(normalized_row)

    if not normalized:
        return ((),)

    return tuple(tuple(row) for row in normalized)


def _validate_canonical_grid(grid: tuple[tuple[int, ...], ...]) -> None:
    """Validate that a canonical grid is well-formed.

    Args:
        grid: Tuple of tuples of non-negative integers.

    Raises:
        ValueError: If the grid is malformed.
    """
    if not grid:
        raise ValueError("Canonical grid is empty")

    expected_width = len(grid[0])
    if expected_width == 0:
        raise ValueError("Canonical grid has zero-width rows")

    for row_idx, row in enumerate(grid):
        if len(row) != expected_width:
            raise ValueError(
                f"Canonical grid row {row_idx} has width {len(row)}, "
                f"expected {expected_width}"
            )


def _extract_grid(frame_data: FrameData) -> tuple[tuple[int, ...], ...]:
    """Extract and normalize the canonical grid from FrameData.

    Pipeline:
        1. Select last rendered frame from the frame sequence.
        2. Normalize 2D palette-index list to immutable tuple form.
        3. Validate the resulting canonical grid.

    Args:
        frame_data: SDK frame data object.

    Returns:
        Tuple of tuples representing the canonical grid.
    """
    last_frame = _extract_rendered_frame(frame_data)
    grid = _normalize_palette_index_grid(last_frame)
    _validate_canonical_grid(grid)
    return grid


# ---------------------------------------------------------------------------
# Object extraction
# ---------------------------------------------------------------------------


def _extract_objects(grid: tuple[tuple[int, ...], ...]) -> tuple[GridObject, ...]:
    """Extract connected components from grid using 4-neighbor connectivity.

    Returns:
        Sorted tuple of GridObject instances (deterministic order).
    """
    if not grid or not grid[0]:
        return ()

    height = len(grid)
    width = len(grid[0]) if grid[0] else 0
    visited = [[False] * width for _ in range(height)]
    objects: list[GridObject] = []

    def flood_fill(
        start_r: int, start_c: int, color: int
    ) -> tuple[list[tuple[int, int]], int, int, int, int]:
        """Flood fill to find connected component.

        Returns: (cells, min_r, min_c, max_r, max_c)
        """
        stack = [(start_r, start_c)]
        cells = []
        min_r, min_c = start_r, start_c
        max_r, max_c = start_r, start_c

        while stack:
            r, c = stack.pop()
            if r < 0 or r >= height or c < 0 or c >= width:
                continue
            if visited[r][c]:
                continue
            if grid[r][c] != color:
                continue

            visited[r][c] = True
            cells.append((r, c))
            min_r = min(min_r, r)
            min_c = min(min_c, c)
            max_r = max(max_r, r)
            max_c = max(max_c, c)

            # 4-neighbor connectivity
            stack.append((r + 1, c))
            stack.append((r - 1, c))
            stack.append((r, c + 1))
            stack.append((r, c - 1))

        return cells, min_r, min_c, max_r, max_c

    # Extract components in deterministic row-major order
    for r in range(height):
        for c in range(width):
            if not visited[r][c]:
                color = grid[r][c]
                if color == 0:
                    # Skip background (color 0)
                    visited[r][c] = True
                    continue

                cells, min_r, min_c, max_r, max_c = flood_fill(r, c, color)
                if cells:
                    # Sort cells for determinism
                    sorted_cells = tuple(sorted(set(cells)))
                    area = len(sorted_cells)

                    # Deterministic object ID: hash of (color, sorted_cells)
                    obj_id = f"obj_{color}_{min_r}_{min_c}_{area}"

                    obj = GridObject(
                        object_id=obj_id,
                        color=color,
                        cells=sorted_cells,
                        bbox=(min_r, min_c, max_r, max_c),
                        area=area,
                    )
                    objects.append(obj)

    # Sort objects deterministically by color, then by min coordinates
    sorted_objects = tuple(
        sorted(
            objects,
            key=lambda o: (o.color, o.bbox[0], o.bbox[1]),
        )
    )
    return sorted_objects


# ---------------------------------------------------------------------------
# Status and action extraction
# ---------------------------------------------------------------------------


def _state_status(frame_data: FrameData) -> str | None:
    """Extract and normalize game status from FrameData."""
    if hasattr(frame_data, "state") and frame_data.state is not None:
        # GameState is an enum; convert to string name
        if hasattr(frame_data.state, "name"):
            return frame_data.state.name
        return str(frame_data.state)
    return None


def _extract_canonical_actions(
    frame_data: FrameData,
) -> tuple[CanonicalAction, ...]:
    """Convert SDK available_actions (list[int]) to CanonicalAction tuple.

    SDK provides available_actions as a list of integer action IDs.
    We convert each to a GameAction enum via from_id(), then extract
    the name and ID. No raw SDK objects are stored.

    For ACTION6, complex_data is None because coordinates are not
    known at the available_actions level — they are only set when
    the agent selects a specific action.

    Returns:
        Sorted tuple of canonical actions (deterministic order).
    """
    if not hasattr(frame_data, "available_actions") or not frame_data.available_actions:
        return ()

    canonical_actions: list[CanonicalAction] = []
    for action_id in frame_data.available_actions:
        try:
            if not isinstance(action_id, int) or isinstance(action_id, bool):
                logger.warning(f"Skipping non-integer action: {action_id!r}")
                continue

            sdk_action = GameAction.from_id(action_id)

            canonical = CanonicalAction(
                action_name=sdk_action.name,
                action_id=sdk_action.value,
                complex_data=None,
            )
            canonical_actions.append(canonical)
        except Exception as e:  # noqa: BLE001 - action extraction must not crash
            logger.warning(f"Failed to canonicalize action {action_id}: {e}")

    # Sort deterministically by action_id, then name
    sorted_actions = tuple(
        sorted(
            canonical_actions,
            key=lambda a: (a.action_id, a.action_name),
        )
    )
    return sorted_actions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_canonical_state(frame_data: FrameData) -> CanonicalState:
    """Convert SDK FrameData to CanonicalState.

    Args:
        frame_data: SDK frame data object.

    Returns:
        Immutable, hashable canonical state.
    """
    grid = _extract_grid(frame_data)
    objects = _extract_objects(grid)
    available_actions = _extract_canonical_actions(frame_data)
    status = _state_status(frame_data)

    # Create state without hash first
    state_without_hash = CanonicalState(
        grid=grid,
        objects=objects,
        available_actions=available_actions,
        status=status,
        state_hash="",  # Placeholder
    )

    # Compute hash
    computed_hash = state_hash(state_without_hash)

    # Create final state with hash
    return CanonicalState(
        grid=grid,
        objects=objects,
        available_actions=available_actions,
        status=status,
        state_hash=computed_hash,
    )


def compute_state_delta(
    state_before: CanonicalState, state_after: CanonicalState
) -> StateDelta:
    """Compute deterministic delta between two consecutive states.

    Conservative approach: only identify exact matches.

    Args:
        state_before: Previous canonical state.
        state_after: Current canonical state.

    Returns:
        StateDelta describing changes.
    """
    changed_cells: list[tuple[tuple[int, int], int, int]] = []
    moved_objects: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    added_objects: list[str] = []
    removed_objects: list[str] = []

    # Changed cells
    height_before = len(state_before.grid)
    height_after = len(state_after.grid)
    width_before = len(state_before.grid[0]) if height_before > 0 else 0
    width_after = len(state_after.grid[0]) if height_after > 0 else 0

    # Compare grids cell by cell
    min_height = min(height_before, height_after)
    min_width = min(width_before, width_after)

    for r in range(min_height):
        for c in range(min_width):
            color_before = state_before.grid[r][c]
            color_after = state_after.grid[r][c]
            if color_before != color_after:
                changed_cells.append(((r, c), color_before, color_after))

    # Handle grid size changes (conservative: report as changes at boundary)
    if height_before != height_after or width_before != width_after:
        # For now, just note that size changed; don't try to infer moves
        pass

    # Objects: exact matching by object_id
    before_obj_map = {obj.object_id: obj for obj in state_before.objects}
    after_obj_map = {obj.object_id: obj for obj in state_after.objects}

    for obj_id in after_obj_map:
        if obj_id not in before_obj_map:
            added_objects.append(obj_id)

    for obj_id in before_obj_map:
        if obj_id not in after_obj_map:
            removed_objects.append(obj_id)

    # Moved objects: if same object exists, check if bbox changed
    for obj_id, before_obj in before_obj_map.items():
        if obj_id in after_obj_map:
            after_obj = after_obj_map[obj_id]
            if before_obj.bbox != after_obj.bbox:
                # Object moved; use bbox center as position proxy
                before_pos = (
                    (before_obj.bbox[0] + before_obj.bbox[2]) // 2,
                    (before_obj.bbox[1] + before_obj.bbox[3]) // 2,
                )
                after_pos = (
                    (after_obj.bbox[0] + after_obj.bbox[2]) // 2,
                    (after_obj.bbox[1] + after_obj.bbox[3]) // 2,
                )
                moved_objects.append((obj_id, before_pos, after_pos))

    status_changed = state_before.status != state_after.status

    return StateDelta(
        changed_cells=tuple(changed_cells),
        moved_objects=tuple(moved_objects),
        added_objects=tuple(added_objects),
        removed_objects=tuple(removed_objects),
        status_changed=status_changed,
    )
