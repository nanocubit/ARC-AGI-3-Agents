"""Trajectory reconstruction from M1 receipts and optional canonical snapshots.

Receipts contain state hashes and deltas but NOT full canonical grids.
For P1-P4 predicate evaluation, canonical snapshots must be provided
alongside receipts. The join is fail-closed: hash mismatch raises.

This module does NOT fabricate states. If only receipts are available,
data_completeness is "receipt_only" and grid predicates cannot be evaluated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.hashing import state_hash
from core.types import CanonicalAction, CanonicalState, StateDelta

DataCompleteness = Literal["receipt_only", "full"]


@dataclass(frozen=True)
class ReceiptRecord:
    """Parsed receipt from M1 JSONL output.

    Contains hashes and deltas but NOT full canonical grid states.
    """

    receipt_id: str
    game_id: str
    derived_level_key: str
    step_index: int
    state_hash_before: str
    action_hash: str
    selected_action: CanonicalAction
    state_hash_after: str
    actual_delta: StateDelta
    verification_outcome: str
    schema_version: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReceiptRecord:
        """Parse a receipt dictionary from JSONL.

        Raises ValueError if schema_version is not "1".
        """
        sv = data.get("schema_version", "unknown")
        if sv != "1":
            raise ValueError(
                f"Unsupported receipt schema_version: {sv!r}. Expected '1'."
            )

        action_data = data["selected_action"]
        action = CanonicalAction(
            action_name=action_data["action_name"],
            action_id=action_data["action_id"],
            complex_data=tuple(
                tuple(p) for p in action_data["complex_data"]
            )
            if action_data.get("complex_data")
            else None,
        )

        delta_data = data["actual_delta"]
        delta = StateDelta(
            changed_cells=tuple(
                (
                    tuple(cc[0]),
                    cc[1],
                    cc[2],
                )
                for cc in delta_data["changed_cells"]
            ),
            moved_objects=tuple(
                (mo[0], tuple(mo[1]), tuple(mo[2]))
                for mo in delta_data["moved_objects"]
            ),
            added_objects=tuple(delta_data["added_objects"]),
            removed_objects=tuple(delta_data["removed_objects"]),
            status_changed=delta_data["status_changed"],
        )

        return cls(
            receipt_id=data["receipt_id"],
            game_id=data["game_id"],
            derived_level_key=data["derived_level_key"],
            step_index=data["step_index"],
            state_hash_before=data["state_hash_before"],
            action_hash=data["action_hash"],
            selected_action=action,
            actual_delta=delta,
            state_hash_after=data["state_hash_after"],
            verification_outcome=data["verification_outcome"],
            schema_version=sv,
        )

    @property
    def action_id(self) -> int:
        """Numeric action ID from selected action."""
        return self.selected_action.action_id


@dataclass(frozen=True)
class CanonicalSnapshot:
    """Full canonical state with verified hash.

    The hash must match state_hash(canonical_state). If it does not,
    this snapshot is invalid and must not be used for predicate evaluation.
    """

    canonical_state: CanonicalState
    state_hash: str

    @classmethod
    def from_state(cls, state: CanonicalState) -> CanonicalSnapshot:
        """Create a snapshot from a canonical state, computing its hash."""
        return cls(
            canonical_state=state,
            state_hash=state_hash(state),
        )

    def verify(self, expected_hash: str) -> None:
        """Verify that this snapshot's hash matches the expected hash.

        Raises ValueError on mismatch (fail-closed).
        """
        actual = state_hash(self.canonical_state)
        if actual != expected_hash:
            raise ValueError(
                f"Hash mismatch: expected {expected_hash}, "
                f"got {actual}. Snapshot rejected."
            )
        if self.state_hash != expected_hash:
            raise ValueError(
                f"Stored hash {self.state_hash} != expected {expected_hash}"
            )


@dataclass(frozen=True)
class ObservedTransition:
    """A single observed transition with optional full state data.

    If state_before/state_after are None, data_completeness is "receipt_only".
    If both are present, data_completeness is "full".
    """

    receipt: ReceiptRecord
    state_before: CanonicalSnapshot | None
    state_after: CanonicalSnapshot | None

    @property
    def data_completeness(self) -> DataCompleteness:
        """Level of data available for this transition."""
        if self.state_before is not None and self.state_after is not None:
            return "full"
        return "receipt_only"

    @property
    def game_id(self) -> str:
        return self.receipt.game_id

    @property
    def derived_level_key(self) -> str:
        return self.receipt.derived_level_key

    @property
    def step_index(self) -> int:
        return self.receipt.step_index


@dataclass(frozen=True)
class Trajectory:
    """A sequence of observed transitions for one game/level.

    Grouped by (game_id, derived_level_key).
    """

    game_id: str
    derived_level_key: str
    transitions: tuple[ObservedTransition, ...]

    @property
    def length(self) -> int:
        return len(self.transitions)

    @property
    def is_full(self) -> bool:
        """True if all transitions have full canonical snapshots."""
        return all(
            t.data_completeness == "full" for t in self.transitions
        )

    @property
    def completeness(self) -> DataCompleteness:
        """Overall completeness of this trajectory."""
        return "full" if self.is_full else "receipt_only"


class TrajectoryReader:
    """Reads M1 receipt JSONL files and constructs trajectories.

    Optionally joins with canonical snapshots if provided.
    """

    @staticmethod
    def read_receipts(
        path: str | Path,
    ) -> list[ReceiptRecord]:
        """Read receipts from a JSONL file.

        Each line is a JSON object matching the ReceiptWriter serialization.
        Lines that fail to parse are skipped with a warning to stderr.
        """
        import sys

        receipts: list[ReceiptRecord] = []
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    receipts.append(ReceiptRecord.from_dict(data))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(
                        f"Warning: skipping line {line_no}: {e}",
                        file=sys.stderr,
                    )
        return receipts

    @staticmethod
    def group_trajectories(
        receipts: list[ReceiptRecord],
        snapshots: dict[str, CanonicalSnapshot] | None = None,
    ) -> list[Trajectory]:
        """Group receipts into trajectories by (game_id, derived_level_key).

        If snapshots dict is provided, it maps state_hash → CanonicalSnapshot.
        Snapshots are joined by matching state_hash_before/state_hash_after.
        Hash mismatches raise ValueError (fail-closed).
        """
        snapshots = snapshots or {}

        # Group receipts by (game_id, derived_level_key)
        groups: dict[tuple[str, str], list[ReceiptRecord]] = {}
        for r in receipts:
            key = (r.game_id, r.derived_level_key)
            groups.setdefault(key, []).append(r)

        trajectories: list[Trajectory] = []
        for (game_id, level_key), receipt_list in groups.items():
            receipt_list.sort(key=lambda r: r.step_index)
            transitions: list[ObservedTransition] = []
            for r in receipt_list:
                sb = snapshots.get(r.state_hash_before)
                if sb is not None:
                    sb.verify(r.state_hash_before)
                sa = snapshots.get(r.state_hash_after)
                if sa is not None:
                    sa.verify(r.state_hash_after)
                transitions.append(
                    ObservedTransition(
                        receipt=r,
                        state_before=sb,
                        state_after=sa,
                    )
                )
            trajectories.append(
                Trajectory(
                    game_id=game_id,
                    derived_level_key=level_key,
                    transitions=tuple(transitions),
                )
            )
        return trajectories


def join_receipts_with_snapshots(
    receipts: list[ReceiptRecord],
    snapshots: dict[str, CanonicalSnapshot],
) -> list[Trajectory]:
    """Join receipts with canonical snapshots.

    Each snapshot's hash must match the corresponding receipt hash.
    Raises ValueError on any mismatch (fail-closed).
    """
    return TrajectoryReader.group_trajectories(receipts, snapshots)
