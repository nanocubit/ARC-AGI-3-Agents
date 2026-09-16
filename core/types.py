"""Immutable typed dataclasses for canonical state and action representation."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GridObject:
    """Immutable representation of a connected object in the grid.
    
    Coordinates use (row, col) convention with (0, 0) at top-left.
    """

    object_id: str  # Deterministic ID based on object properties
    color: int  # 0-9 ARC color value
    cells: tuple[tuple[int, int], ...]  # Sorted immutable set of (row, col)
    bbox: tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    area: int  # Number of cells


@dataclass(frozen=True)
class CanonicalState:
    """Canonical, immutable representation of observable game state.
    
    All fields are deterministic primitives suitable for hashing and
    JSON serialization. Contains no raw SDK objects.
    """

    grid: tuple[tuple[int, ...], ...]  # 2D tuple of integers (colors)
    objects: tuple[GridObject, ...]  # Sorted, deterministic order
    available_actions: tuple["CanonicalAction", ...]  # SDK actions normalized
    status: str | None  # Game state (e.g., "WIN", "GAME_OVER", "NOT_PLAYED")
    state_hash: str  # SHA-256 hex digest


@dataclass(frozen=True)
class CanonicalAction:
    """Immutable representation of an action suitable for hashing and equality.
    
    Preserves enough information to reconstruct or identify the original
    SDK GameAction without storing the raw object.
    """

    action_name: str  # E.g., "RESET", "ACTION1", etc.
    action_id: int  # Numeric ID from GameAction enum
    complex_data: tuple[tuple[str, int], ...] | None  # For ACTION6: (("x", 5), ("y", 7))


@dataclass(frozen=True)
class StateDelta:
    """Immutable record of changes between two consecutive states."""

    changed_cells: tuple[tuple[tuple[int, int], int, int], ...]  # ((row, col), color_before, color_after)
    moved_objects: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...]  # (object_id, old_pos, new_pos)
    added_objects: tuple[str, ...]  # object_ids
    removed_objects: tuple[str, ...]  # object_ids
    status_changed: bool


@dataclass(frozen=True)
class ActionReceipt:
    """Immutable receipt of a single attempted action and its outcome.
    
    No raw SDK objects. All fields are deterministic primitives suitable
    for JSON serialization.
    """

    receipt_id: str  # UUID
    game_id: str
    level_id: str
    step_index: int
    state_hash_before: str  # SHA-256 hex
    action_hash: str  # SHA-256 hex
    selected_action: CanonicalAction
    state_hash_after: str  # SHA-256 hex
    actual_delta: StateDelta
    verification_outcome: Literal["confirmed", "contradicted", "unknown"]
