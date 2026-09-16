"""Test M2-A.1 inventory aggregation and trajectory handling."""

from __future__ import annotations

import json
from pathlib import Path

from core.hashing import state_hash
from core.types import CanonicalAction, CanonicalState, GridObject, StateDelta
from m2.inventory import InventoryBuilder
from m2.report import InventoryReport
from m2.trajectory import (
    CanonicalSnapshot,
    ObservedTransition,
    ReceiptRecord,
    TrajectoryReader,
    join_receipts_with_snapshots,
)


def make_grid_state(
    grid: tuple[tuple[int, ...], ...],
    objects: tuple[GridObject, ...] = (),
    actions: tuple[int, ...] = (0, 1),
    status: str | None = "NOT_PLAYED",
) -> CanonicalState:
    """Build a CanonicalState with computed hash."""
    available = tuple(
        CanonicalAction(
            action_name=f"ACTION{a}" if a > 0 else "RESET",
            action_id=a,
            complex_data=None,
        )
        for a in actions
    )
    state = CanonicalState(
        grid=grid,
        objects=objects,
        available_actions=available,
        status=status,
        state_hash="",  # Will be computed
    )
    return CanonicalState(
        grid=grid,
        objects=objects,
        available_actions=available,
        status=status,
        state_hash=state_hash(state),
    )


def make_delta(
    changed_cells: tuple = (),
    moved_objects: tuple = (),
    added_objects: tuple = (),
    removed_objects: tuple = (),
    status_changed: bool = False,
) -> StateDelta:
    return StateDelta(
        changed_cells=changed_cells,
        moved_objects=moved_objects,
        added_objects=added_objects,
        removed_objects=removed_objects,
        status_changed=status_changed,
    )


def make_receipt(
    game_id: str = "game_1",
    level_key: str = "game_1:level:0",
    step: int = 0,
    action_id: int = 1,
    delta: StateDelta | None = None,
    state_hash_before: str = "hash_before_0",
    state_hash_after: str = "hash_after_0",
) -> ReceiptRecord:
    """Build a ReceiptRecord for testing."""
    delta = delta or make_delta()
    action = CanonicalAction(
        action_name=f"ACTION{action_id}" if action_id > 0 else "RESET",
        action_id=action_id,
        complex_data=None,
    )
    return ReceiptRecord(
        receipt_id=f"test_{game_id}_{step}",
        game_id=game_id,
        derived_level_key=level_key,
        step_index=step,
        state_hash_before=state_hash_before,
        action_hash="action_hash_0",
        selected_action=action,
        state_hash_after=state_hash_after,
        actual_delta=delta,
        verification_outcome="confirmed",
        schema_version="1",
    )


def make_full_transition(
    state_before: CanonicalState,
    state_after: CanonicalState,
    receipt: ReceiptRecord,
) -> ObservedTransition:
    """Build a transition with full snapshots."""
    sb = CanonicalSnapshot.from_state(state_before)
    sa = CanonicalSnapshot.from_state(state_after)
    # Update receipt hashes to match
    return ObservedTransition(
        receipt=ReceiptRecord(
            receipt_id=receipt.receipt_id,
            game_id=receipt.game_id,
            derived_level_key=receipt.derived_level_key,
            step_index=receipt.step_index,
            state_hash_before=sb.state_hash,
            action_hash=receipt.action_hash,
            selected_action=receipt.selected_action,
            state_hash_after=sa.state_hash,
            actual_delta=receipt.actual_delta,
            verification_outcome=receipt.verification_outcome,
            schema_version="1",
        ),
        state_before=sb,
        state_after=sa,
    )


class TestReceiptParsing:
    """Test receipt JSONL parsing."""

    def test_receipt_from_dict(self) -> None:
        data = {
            "receipt_id": "test_1",
            "game_id": "game_1",
            "derived_level_key": "game_1:level:0",
            "step_index": 0,
            "state_hash_before": "abc",
            "action_hash": "def",
            "selected_action": {
                "action_name": "ACTION1",
                "action_id": 1,
                "complex_data": None,
            },
            "state_hash_after": "ghi",
            "actual_delta": {
                "changed_cells": [],
                "moved_objects": [],
                "added_objects": [],
                "removed_objects": [],
                "status_changed": False,
            },
            "verification_outcome": "confirmed",
            "schema_version": "1",
        }
        r = ReceiptRecord.from_dict(data)
        assert r.game_id == "game_1"
        assert r.action_id == 1
        assert r.schema_version == "1"

    def test_receipt_wrong_schema_version_raises(self) -> None:
        data = {
            "receipt_id": "test_1",
            "game_id": "game_1",
            "derived_level_key": "game_1:level:0",
            "step_index": 0,
            "state_hash_before": "abc",
            "action_hash": "def",
            "selected_action": {
                "action_name": "ACTION1",
                "action_id": 1,
                "complex_data": None,
            },
            "state_hash_after": "ghi",
            "actual_delta": {
                "changed_cells": [],
                "moved_objects": [],
                "added_objects": [],
                "removed_objects": [],
                "status_changed": False,
            },
            "verification_outcome": "confirmed",
            "schema_version": "2",
        }
        try:
            ReceiptRecord.from_dict(data)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_read_receipts_from_jsonl(self, tmp_path: Path) -> None:
        receipts = [
            make_receipt(step=0),
            make_receipt(step=1, action_id=2),
            make_receipt(game_id="game_2", level_key="game_2:level:0", step=0),
        ]
        path = tmp_path / "receipts.jsonl"
        with open(path, "w") as f:
            for r in receipts:
                # Serialize as JSONL
                d = {
                    "receipt_id": r.receipt_id,
                    "game_id": r.game_id,
                    "derived_level_key": r.derived_level_key,
                    "step_index": r.step_index,
                    "state_hash_before": r.state_hash_before,
                    "action_hash": r.action_hash,
                    "selected_action": {
                        "action_name": r.selected_action.action_name,
                        "action_id": r.selected_action.action_id,
                        "complex_data": r.selected_action.complex_data,
                    },
                    "state_hash_after": r.state_hash_after,
                    "actual_delta": {
                        "changed_cells": list(r.actual_delta.changed_cells),
                        "moved_objects": list(r.actual_delta.moved_objects),
                        "added_objects": list(r.actual_delta.added_objects),
                        "removed_objects": list(r.actual_delta.removed_objects),
                        "status_changed": r.actual_delta.status_changed,
                    },
                    "verification_outcome": r.verification_outcome,
                    "schema_version": r.schema_version,
                }
                f.write(json.dumps(d) + "\n")

        read = TrajectoryReader.read_receipts(path)
        assert len(read) == 3
        assert read[0].game_id == "game_1"
        assert read[2].game_id == "game_2"


class TestTrajectoryGrouping:
    """Test trajectory grouping by game/level."""

    def test_group_by_game(self) -> None:
        receipts = [
            make_receipt(game_id="game_1", step=0),
            make_receipt(game_id="game_1", step=1),
            make_receipt(game_id="game_2", step=0),
        ]
        trajs = TrajectoryReader.group_trajectories(receipts)
        assert len(trajs) == 2
        game1 = [t for t in trajs if t.game_id == "game_1"]
        assert len(game1) == 1
        assert game1[0].length == 2

    def test_group_by_level_within_game(self) -> None:
        receipts = [
            make_receipt(game_id="game_1", level_key="game_1:level:0", step=0),
            make_receipt(game_id="game_1", level_key="game_1:level:1", step=0),
        ]
        trajs = TrajectoryReader.group_trajectories(receipts)
        assert len(trajs) == 2  # Two different levels = two trajectories


class TestSnapshotJoin:
    """Test receipt-snapshot joining with fail-closed hash matching."""

    def test_join_with_matching_hashes(self) -> None:
        state = make_grid_state(grid=((0, 1), (2, 3)))
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt(
            state_hash_before=snap.state_hash,
            state_hash_after=snap.state_hash,
        )
        snapshots = {snap.state_hash: snap}
        trajs = join_receipts_with_snapshots([receipt], snapshots)
        assert len(trajs) == 1
        assert trajs[0].transitions[0].data_completeness == "full"

    def test_join_hash_mismatch_raises(self) -> None:
        snap = CanonicalSnapshot(
            canonical_state=make_grid_state(grid=((0, 1),)),
            state_hash="wrong_hash",
        )
        receipt = make_receipt(state_hash_before="wrong_hash")
        snapshots = {"wrong_hash": snap}
        try:
            join_receipts_with_snapshots([receipt], snapshots)
            assert False, "Should have raised"
        except ValueError:
            pass

    def test_receipt_only_without_snapshots(self) -> None:
        receipt = make_receipt()
        trajs = TrajectoryReader.group_trajectories([receipt])
        assert trajs[0].completeness == "receipt_only"
        assert trajs[0].transitions[0].data_completeness == "receipt_only"


class TestInventoryBuilding:
    """Test inventory aggregation from trajectories."""

    def test_receipt_only_inventory(self) -> None:
        receipts = [
            make_receipt(game_id="game_1", step=0, action_id=1),
            make_receipt(game_id="game_1", step=1, action_id=2),
            make_receipt(game_id="game_2", step=0, action_id=1),
        ]
        trajs = TrajectoryReader.group_trajectories(receipts)
        builder = InventoryBuilder()
        inv = builder.build(trajs)

        assert inv.data_completeness == "receipt_only"
        assert inv.total_trajectories == 2
        assert inv.total_transitions == 3
        assert inv.action_stats is not None
        assert inv.action_stats.counts[1] == 2
        assert inv.action_stats.counts[2] == 1
        assert inv.can_evaluate_predicates is False

    def test_full_inventory_with_snapshots(self) -> None:
        state = make_grid_state(grid=((0, 1), (2, 3)))
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt(
            state_hash_before=snap.state_hash,
            state_hash_after=snap.state_hash,
        )
        snapshots = {snap.state_hash: snap}
        trajs = join_receipts_with_snapshots([receipt], snapshots)
        builder = InventoryBuilder()
        inv = builder.build(trajs)

        assert inv.data_completeness == "full"
        assert inv.can_evaluate_predicates is True
        assert len(inv.predicate_stats) > 0
        assert inv.representation_stats is not None

    def test_effect_taxonomy_built(self) -> None:
        delta = make_delta(changed_cells=(((0, 0), 0, 1),))
        receipt = make_receipt(delta=delta)
        trajs = TrajectoryReader.group_trajectories([receipt])
        inv = InventoryBuilder().build(trajs)

        assert inv.effect_stats is not None
        assert "cell_change" in inv.effect_stats.counts

    def test_trajectory_split(self) -> None:
        receipts = []
        for game in ["game_1", "game_2", "game_3"]:
            for level in range(2):
                for step in range(2):
                    receipts.append(
                        make_receipt(
                            game_id=game,
                            level_key=f"{game}:level:{level}",
                            step=step,
                        )
                    )
        trajs = TrajectoryReader.group_trajectories(receipts)
        inv = InventoryBuilder().build(trajs, train_split=0.6)

        assert inv.trajectory_split is not None
        assert inv.trajectory_split.train_count > 0
        assert inv.trajectory_split.test_count > 0


class TestFullGridEvaluation:
    """R1.2 regression tests: full-grid predicate evaluation.

    Ensures objects outside the first 8 rows/cols are detected.
    """

    def test_occupied_detects_sprite_outside_sample_window(self) -> None:
        """Object at (16,16) must be detected with full-grid evaluation."""
        # Build a 20x20 grid with a sprite at row 16, col 16
        row = tuple([0] * 20)
        grid = [list(row) for _ in range(20)]
        grid[16][16] = 1  # Non-background cell at (16, 16)
        grid = tuple(tuple(r) for r in grid)

        obj = GridObject(
            object_id="o1", color=1,
            cells=((16, 16),), bbox=(16, 16, 16, 16), area=1,
        )
        state = make_grid_state(grid=grid, objects=(obj,))
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt(
            state_hash_before=snap.state_hash,
            state_hash_after=snap.state_hash,
        )
        snapshots = {snap.state_hash: snap}
        trajs = join_receipts_with_snapshots([receipt], snapshots)
        inv = InventoryBuilder().build(trajs)

        assert inv.data_completeness == "full"
        assert inv.predicate_stats is not None
        # Occupied must be > 0 — the bug was that 8x8 sampling missed row 16
        occupied_stats = inv.predicate_stats.get("Occupied")
        assert occupied_stats is not None
        assert occupied_stats.true_count > 0, (
            f"Occupied should be > 0 with full-grid eval, got {occupied_stats.true_count}"
        )

    def test_full_grid_evaluates_all_cells(self) -> None:
        """Full grid evaluation should count all cells, not just 64."""
        # 12x12 grid with one non-background cell at (11, 11)
        row = tuple([0] * 12)
        grid = [list(row) for _ in range(12)]
        grid[11][11] = 1
        grid = tuple(tuple(r) for r in grid)

        state = make_grid_state(grid=grid)
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt(
            state_hash_before=snap.state_hash,
            state_hash_after=snap.state_hash,
        )
        snapshots = {snap.state_hash: snap}
        trajs = join_receipts_with_snapshots([receipt], snapshots)
        inv = InventoryBuilder().build(trajs)

        # Full grid: 144 cells evaluated
        empty_stats = inv.predicate_stats.get("Empty")
        assert empty_stats is not None
        assert empty_stats.eval_count == 144  # 12*12
        assert empty_stats.true_count == 143  # 144 - 1

    def test_color_at_all_colors(self) -> None:
        """ColorAt should evaluate for all colors present, not just top 4."""
        # Grid with 5 colors
        grid = (
            (0, 1, 2, 3, 4),
            (0, 0, 0, 0, 0),
        )
        state = make_grid_state(grid=grid)
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt(
            state_hash_before=snap.state_hash,
            state_hash_after=snap.state_hash,
        )
        snapshots = {snap.state_hash: snap}
        trajs = join_receipts_with_snapshots([receipt], snapshots)
        inv = InventoryBuilder().build(trajs)

        # All 5 colors should be in predicate stats
        for color in range(5):
            key = f"ColorAt({color})"
            assert key in inv.predicate_stats, f"Missing ColorAt({color})"


class TestReportGeneration:
    """Test markdown report generation."""

    def test_receipt_only_report(self) -> None:
        receipts = [make_receipt()]
        trajs = TrajectoryReader.group_trajectories(receipts)
        inv = InventoryBuilder().build(trajs)
        report = InventoryReport.generate(inv)

        assert "Data Completeness" in report
        assert "receipt_only" in report
        assert "Action Distribution" in report
        assert "Effect Taxonomy" in report
        assert "requires full canonical snapshots" in report

    def test_full_report(self) -> None:
        state = make_grid_state(grid=((0, 1), (2, 3)))
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt(
            state_hash_before=snap.state_hash,
            state_hash_after=snap.state_hash,
        )
        snapshots = {snap.state_hash: snap}
        trajs = join_receipts_with_snapshots([receipt], snapshots)
        inv = InventoryBuilder().build(trajs)
        report = InventoryReport.generate(inv)

        assert "Predicate Frequencies" in report
        assert "Truth Tables" in report
        assert "Representation Comparison" in report
        assert "offline only" in report
