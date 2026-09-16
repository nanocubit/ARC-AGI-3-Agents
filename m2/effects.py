"""Effect classification from StateDelta.

Classifies transitions based on what changed between before and after states.
Works from StateDelta alone — does not require full canonical grids.

Effect taxonomy:
  - no_change: delta is empty
  - cell_change: one or more cells changed color
  - object_move: one or more objects moved
  - object_add: one or more objects appeared
  - object_remove: one or more objects disappeared
  - status_change: game status changed (win/lose)
  - composite: multiple effect types combined
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.types import StateDelta


class EffectType(str, Enum):
    """Classification of transition effects."""

    NO_CHANGE = "no_change"
    CELL_CHANGE = "cell_change"
    OBJECT_MOVE = "object_move"
    OBJECT_ADD = "object_add"
    OBJECT_REMOVE = "object_remove"
    STATUS_CHANGE = "status_change"
    COMPOSITE = "composite"


@dataclass(frozen=True)
class TransitionEffect:
    """Classified effect of a state transition.

    Derived from StateDelta. Does not require full canonical grids.
    """

    effect_type: EffectType
    cell_change_count: int
    object_move_count: int
    object_add_count: int
    object_remove_count: int
    status_changed: bool

    @property
    def is_empty(self) -> bool:
        """True if no changes occurred."""
        return (
            self.cell_change_count == 0
            and self.object_move_count == 0
            and self.object_add_count == 0
            and self.object_remove_count == 0
            and not self.status_changed
        )

    @property
    def component_types(self) -> list[EffectType]:
        """List all component effect types present in this transition."""
        types: list[EffectType] = []
        if self.cell_change_count > 0:
            types.append(EffectType.CELL_CHANGE)
        if self.object_move_count > 0:
            types.append(EffectType.OBJECT_MOVE)
        if self.object_add_count > 0:
            types.append(EffectType.OBJECT_ADD)
        if self.object_remove_count > 0:
            types.append(EffectType.OBJECT_REMOVE)
        if self.status_changed:
            types.append(EffectType.STATUS_CHANGE)
        return types

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for report output."""
        return {
            "effect_type": self.effect_type.value,
            "cell_change_count": self.cell_change_count,
            "object_move_count": self.object_move_count,
            "object_add_count": self.object_add_count,
            "object_remove_count": self.object_remove_count,
            "status_changed": self.status_changed,
            "components": [t.value for t in self.component_types],
        }


class EffectClassifier:
    """Classifies transitions based on StateDelta.

    Does not require full canonical grids — works from delta alone.
    """

    @staticmethod
    def classify(delta: StateDelta) -> TransitionEffect:
        """Classify a state delta into a transition effect.

        Rules:
        1. If delta is completely empty → NO_CHANGE
        2. If exactly one component type → that type
        3. If multiple component types → COMPOSITE
        """
        cell_count = len(delta.changed_cells)
        move_count = len(delta.moved_objects)
        add_count = len(delta.added_objects)
        remove_count = len(delta.removed_objects)
        status = delta.status_changed

        component_types: list[EffectType] = []
        if cell_count > 0:
            component_types.append(EffectType.CELL_CHANGE)
        if move_count > 0:
            component_types.append(EffectType.OBJECT_MOVE)
        if add_count > 0:
            component_types.append(EffectType.OBJECT_ADD)
        if remove_count > 0:
            component_types.append(EffectType.OBJECT_REMOVE)
        if status:
            component_types.append(EffectType.STATUS_CHANGE)

        if not component_types:
            effect_type = EffectType.NO_CHANGE
        elif len(component_types) == 1:
            effect_type = component_types[0]
        else:
            effect_type = EffectType.COMPOSITE

        return TransitionEffect(
            effect_type=effect_type,
            cell_change_count=cell_count,
            object_move_count=move_count,
            object_add_count=add_count,
            object_remove_count=remove_count,
            status_changed=status,
        )

    @staticmethod
    def classify_batch(
        deltas: list[StateDelta],
    ) -> list[TransitionEffect]:
        """Classify a batch of deltas."""
        return [EffectClassifier.classify(d) for d in deltas]
