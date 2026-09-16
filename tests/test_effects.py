"""Test M2-A.1 effect classification from StateDelta."""

from __future__ import annotations

from core.types import StateDelta
from m2.effects import EffectClassifier, EffectType


def make_delta(
    changed_cells: tuple = (),
    moved_objects: tuple = (),
    added_objects: tuple = (),
    removed_objects: tuple = (),
    status_changed: bool = False,
) -> StateDelta:
    """Build a StateDelta for testing."""
    return StateDelta(
        changed_cells=changed_cells,
        moved_objects=moved_objects,
        added_objects=added_objects,
        removed_objects=removed_objects,
        status_changed=status_changed,
    )


class TestEffectClassification:
    """Test effect type classification."""

    def test_no_change(self) -> None:
        delta = make_delta()
        effect = EffectClassifier.classify(delta)
        assert effect.effect_type == EffectType.NO_CHANGE
        assert effect.is_empty is True
        assert effect.component_types == []

    def test_cell_change(self) -> None:
        delta = make_delta(
            changed_cells=(((0, 0), 0, 1),)
        )
        effect = EffectClassifier.classify(delta)
        assert effect.effect_type == EffectType.CELL_CHANGE
        assert effect.cell_change_count == 1
        assert effect.is_empty is False

    def test_object_move(self) -> None:
        delta = make_delta(
            moved_objects=(("obj1", (0, 0), (1, 1)),)
        )
        effect = EffectClassifier.classify(delta)
        assert effect.effect_type == EffectType.OBJECT_MOVE
        assert effect.object_move_count == 1

    def test_object_add(self) -> None:
        delta = make_delta(added_objects=("obj2",))
        effect = EffectClassifier.classify(delta)
        assert effect.effect_type == EffectType.OBJECT_ADD
        assert effect.object_add_count == 1

    def test_object_remove(self) -> None:
        delta = make_delta(removed_objects=("obj1",))
        effect = EffectClassifier.classify(delta)
        assert effect.effect_type == EffectType.OBJECT_REMOVE
        assert effect.object_remove_count == 1

    def test_status_change(self) -> None:
        delta = make_delta(status_changed=True)
        effect = EffectClassifier.classify(delta)
        assert effect.effect_type == EffectType.STATUS_CHANGE
        assert effect.status_changed is True

    def test_composite(self) -> None:
        delta = make_delta(
            changed_cells=(((0, 0), 0, 1),),
            moved_objects=(("obj1", (0, 0), (1, 1)),),
        )
        effect = EffectClassifier.classify(delta)
        assert effect.effect_type == EffectType.COMPOSITE
        assert len(effect.component_types) == 2

    def test_composite_with_status(self) -> None:
        delta = make_delta(
            changed_cells=(((0, 0), 0, 1),),
            status_changed=True,
        )
        effect = EffectClassifier.classify(delta)
        assert effect.effect_type == EffectType.COMPOSITE
        assert EffectType.CELL_CHANGE in effect.component_types
        assert EffectType.STATUS_CHANGE in effect.component_types


class TestTransitionEffectSerialization:
    """Test TransitionEffect serialization."""

    def test_to_dict(self) -> None:
        delta = make_delta(
            changed_cells=(((0, 0), 0, 1),),
            status_changed=True,
        )
        effect = EffectClassifier.classify(delta)
        d = effect.to_dict()
        assert d["effect_type"] == "composite"
        assert d["cell_change_count"] == 1
        assert d["status_changed"] is True
        assert "cell_change" in d["components"]
        assert "status_change" in d["components"]

    def test_batch_classify(self) -> None:
        deltas = [
            make_delta(),
            make_delta(changed_cells=(((0, 0), 0, 1),)),
            make_delta(status_changed=True),
        ]
        effects = EffectClassifier.classify_batch(deltas)
        assert len(effects) == 3
        assert effects[0].effect_type == EffectType.NO_CHANGE
        assert effects[1].effect_type == EffectType.CELL_CHANGE
        assert effects[2].effect_type == EffectType.STATUS_CHANGE


class TestEffectEdgeCases:
    """Test edge cases in effect classification."""

    def test_empty_delta_is_no_change(self) -> None:
        delta = StateDelta(
            changed_cells=(),
            moved_objects=(),
            added_objects=(),
            removed_objects=(),
            status_changed=False,
        )
        effect = EffectClassifier.classify(delta)
        assert effect.effect_type == EffectType.NO_CHANGE

    def test_all_types_composite(self) -> None:
        delta = make_delta(
            changed_cells=(((0, 0), 0, 1),),
            moved_objects=(("obj1", (0, 0), (1, 1)),),
            added_objects=("obj2",),
            removed_objects=("obj3",),
            status_changed=True,
        )
        effect = EffectClassifier.classify(delta)
        assert effect.effect_type == EffectType.COMPOSITE
        assert len(effect.component_types) == 5
