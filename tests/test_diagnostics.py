"""Test M2-A.1-R1 snapshot diagnostics."""

from __future__ import annotations

from core.hashing import state_hash
from core.types import (
    CanonicalAction,
    CanonicalState,
    GridObject,
    StateDelta,
)
from m2.diagnostics import SnapshotDiagnostics
from m2.trajectory import (
    CanonicalSnapshot,
    ObservedTransition,
    ReceiptRecord,
    Trajectory,
)


def make_grid_state(
    grid: tuple[tuple[int, ...], ...],
    objects: tuple[GridObject, ...] = (),
    actions: tuple[int, ...] = (1,),
    status: str | None = "NOT_PLAYED",
) -> CanonicalState:
    available = tuple(
        CanonicalAction(
            action_name=f"ACTION{a}" if a > 0 else "RESET",
            action_id=a,
            complex_data=None,
        )
        for a in actions
    )
    state = CanonicalState(
        grid=grid, objects=objects,
        available_actions=available, status=status, state_hash="",
    )
    return CanonicalState(
        grid=grid, objects=objects,
        available_actions=available, status=status,
        state_hash=state_hash(state),
    )


def make_obj(oid="o1", color=1, cells=((0, 0),), bbox=(0, 0, 0, 0)):
    return GridObject(
        object_id=oid, color=color, cells=cells, bbox=bbox, area=len(cells),
    )


def make_receipt(game_id="g1", level="g1:l:0", step=0):
    return ReceiptRecord(
        receipt_id="test", game_id=game_id, derived_level_key=level,
        step_index=step, state_hash_before="h1", action_hash="a1",
        selected_action=CanonicalAction("A1", 1, None),
        state_hash_after="h2", actual_delta=StateDelta((), (), (), (), False),
        verification_outcome="confirmed", schema_version="1",
    )


class TestGridDiagnostic:
    """Test single-snapshot diagnostics."""

    def test_empty_grid(self):
        state = make_grid_state(grid=((0, 0), (0, 0)))
        snap = CanonicalSnapshot.from_state(state)
        diag = SnapshotDiagnostics.diagnose_snapshot(snap)
        assert diag.height == 2
        assert diag.width == 2
        assert diag.total_cells == 4
        assert diag.non_background_count == 0
        assert diag.background_count == 4
        assert diag.palette_sum_ok is True
        assert diag.non_bg_ok is True

    def test_with_non_background(self):
        state = make_grid_state(grid=((0, 1), (2, 0)))
        snap = CanonicalSnapshot.from_state(state)
        diag = SnapshotDiagnostics.diagnose_snapshot(snap)
        assert diag.non_background_count == 2
        assert diag.background_count == 2
        assert diag.palette == {0: 2, 1: 1, 2: 1}
        assert diag.palette_sum_ok is True

    def test_with_objects(self):
        obj = make_obj("o1", 1, ((0, 0),), (0, 0, 0, 0))
        state = make_grid_state(grid=((1, 0), (0, 0)), objects=(obj,))
        snap = CanonicalSnapshot.from_state(state)
        diag = SnapshotDiagnostics.diagnose_snapshot(snap)
        assert diag.object_count == 1
        assert diag.object_details[0]["object_id"] == "o1"
        assert diag.object_details[0]["color"] == 1
        assert diag.object_details[0]["area"] == 1

    def test_all_background_no_objects(self):
        state = make_grid_state(grid=((0, 0, 0), (0, 0, 0)))
        snap = CanonicalSnapshot.from_state(state)
        diag = SnapshotDiagnostics.diagnose_snapshot(snap)
        assert diag.non_background_count == 0
        assert diag.object_count == 0
        assert diag.palette_sum_ok is True


class TestDiagnosticSummary:
    """Test aggregate diagnostics across trajectories."""

    def test_empty_trajectories(self):
        summary = SnapshotDiagnostics.diagnose_trajectories([])
        assert summary.total_snapshots == 0
        assert summary.has_non_background is False
        assert summary.has_objects is False

    def test_all_background_snapshots(self):
        state = make_grid_state(grid=((0, 0), (0, 0)))
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt()
        trans = ObservedTransition(
            receipt=receipt, state_before=snap, state_after=snap,
        )
        traj = Trajectory(
            game_id="g1", derived_level_key="g1:l:0",
            transitions=(trans,),
        )
        summary = SnapshotDiagnostics.diagnose_trajectories([traj])
        assert summary.total_snapshots == 1
        assert summary.has_non_background is False
        assert summary.has_objects is False
        assert summary.occupied_should_be_nonzero is False

    def test_with_non_background(self):
        obj = make_obj("o1", 1, ((0, 0),), (0, 0, 0, 0))
        state = make_grid_state(grid=((1, 0), (0, 0)), objects=(obj,))
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt()
        trans = ObservedTransition(
            receipt=receipt, state_before=snap, state_after=snap,
        )
        traj = Trajectory(
            game_id="g1", derived_level_key="g1:l:0",
            transitions=(trans,),
        )
        summary = SnapshotDiagnostics.diagnose_trajectories([traj])
        assert summary.total_snapshots == 1
        assert summary.has_non_background is True
        assert summary.has_objects is True
        assert summary.occupied_should_be_nonzero is True

    def test_deduplicates_same_state(self):
        state = make_grid_state(grid=((1, 0), (0, 0)))
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt()
        trans = ObservedTransition(
            receipt=receipt, state_before=snap, state_after=snap,
        )
        traj = Trajectory(
            game_id="g1", derived_level_key="g1:l:0",
            transitions=(trans, trans),
        )
        summary = SnapshotDiagnostics.diagnose_trajectories([traj])
        assert summary.total_snapshots == 1  # Deduplicated


class TestReportGeneration:
    """Test markdown report generation."""

    def test_report_with_non_background(self):
        obj = make_obj("o1", 1, ((0, 0),), (0, 0, 0, 0))
        state = make_grid_state(grid=((1, 0), (0, 0)), objects=(obj,))
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt()
        trans = ObservedTransition(
            receipt=receipt, state_before=snap, state_after=snap,
        )
        traj = Trajectory(
            game_id="g1", derived_level_key="g1:l:0",
            transitions=(trans,),
        )
        summary = SnapshotDiagnostics.diagnose_trajectories([traj])
        report = SnapshotDiagnostics.generate_report(summary)

        assert "OK" in report
        assert "non-background" in report.lower()
        assert "Proceed to R1.2" in report

    def test_report_all_background(self):
        state = make_grid_state(grid=((0, 0), (0, 0)))
        snap = CanonicalSnapshot.from_state(state)
        receipt = make_receipt()
        trans = ObservedTransition(
            receipt=receipt, state_before=snap, state_after=snap,
        )
        traj = Trajectory(
            game_id="g1", derived_level_key="g1:l:0",
            transitions=(trans,),
        )
        summary = SnapshotDiagnostics.diagnose_trajectories([traj])
        report = SnapshotDiagnostics.generate_report(summary)

        assert "CRITICAL" in report
        assert "UPSTREAM" in report
