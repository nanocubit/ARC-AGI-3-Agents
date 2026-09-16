"""Test append-only JSONL receipt writer."""

import json
import tempfile
from pathlib import Path

import pytest

from core.receipts import ReceiptWriter
from core.types import ActionReceipt, CanonicalAction, StateDelta


class TestReceiptWriter:
    """Tests for ReceiptWriter JSONL logging."""

    def test_one_receipt_produces_one_json_line(self) -> None:
        """Test that a single receipt produces exactly one JSON line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_receipts.jsonl"
            writer = ReceiptWriter(output_path)

            receipt = ActionReceipt(
                receipt_id="test-id-1",
                game_id="game_1",
                level_id="level_1",
                step_index=0,
                state_hash_before="hash_before",
                action_hash="action_hash",
                selected_action=CanonicalAction(
                    action_name="RESET", action_id=0, complex_data=None
                ),
                state_hash_after="hash_after",
                actual_delta=StateDelta(
                    changed_cells=(),
                    moved_objects=(),
                    added_objects=(),
                    removed_objects=(),
                    status_changed=False,
                ),
                verification_outcome="unknown",
            )

            success = writer.write(receipt)
            assert success is True

            with open(output_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 1

    def test_json_can_be_parsed(self) -> None:
        """Test that serialized receipts are valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_receipts.jsonl"
            writer = ReceiptWriter(output_path)

            receipt = ActionReceipt(
                receipt_id="test-id-2",
                game_id="game_2",
                level_id="level_2",
                step_index=1,
                state_hash_before="before",
                action_hash="action",
                selected_action=CanonicalAction(
                    action_name="ACTION1", action_id=1, complex_data=None
                ),
                state_hash_after="after",
                actual_delta=StateDelta(
                    changed_cells=(),
                    moved_objects=(),
                    added_objects=(),
                    removed_objects=(),
                    status_changed=False,
                ),
                verification_outcome="unknown",
            )

            writer.write(receipt)

            with open(output_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    assert data is not None
                    assert isinstance(data, dict)

    def test_all_fields_are_serializable_primitives(self) -> None:
        """Test that all receipt fields serialize to JSON primitives."""
        receipt = ActionReceipt(
            receipt_id="test-id-3",
            game_id="game_3",
            level_id="level_3",
            step_index=5,
            state_hash_before="hash_b",
            action_hash="hash_a",
            selected_action=CanonicalAction(
                action_name="ACTION6",
                action_id=6,
                complex_data=(("x", 10), ("y", 20)),
            ),
            state_hash_after="hash_a",
            actual_delta=StateDelta(
                changed_cells=(((0, 1), 1, 2),),
                moved_objects=(("obj_1", (0, 0), (1, 1)),),
                added_objects=("obj_2",),
                removed_objects=(),
                status_changed=True,
            ),
            verification_outcome="confirmed",
        )

        serialized = ReceiptWriter._serialize_receipt(receipt)
        data = json.loads(serialized)

        # Verify all fields exist and are primitive types
        assert isinstance(data["receipt_id"], str)
        assert isinstance(data["game_id"], str)
        assert isinstance(data["level_id"], str)
        assert isinstance(data["step_index"], int)
        assert isinstance(data["state_hash_before"], str)
        assert isinstance(data["action_hash"], str)
        assert isinstance(data["state_hash_after"], str)
        assert isinstance(data["verification_outcome"], str)
        assert isinstance(data["selected_action"], dict)
        assert isinstance(data["actual_delta"], dict)

    def test_append_only_behavior(self) -> None:
        """Test that multiple receipts are appended to the same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_receipts.jsonl"
            writer = ReceiptWriter(output_path)

            for i in range(3):
                receipt = ActionReceipt(
                    receipt_id=f"test-id-{i}",
                    game_id="game",
                    level_id="level",
                    step_index=i,
                    state_hash_before="before",
                    action_hash="action",
                    selected_action=CanonicalAction(
                        action_name="RESET", action_id=0, complex_data=None
                    ),
                    state_hash_after="after",
                    actual_delta=StateDelta(
                        changed_cells=(),
                        moved_objects=(),
                        added_objects=(),
                        removed_objects=(),
                        status_changed=False,
                    ),
                    verification_outcome="unknown",
                )
                writer.write(receipt)

            with open(output_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 3

    def test_logging_failure_does_not_raise(self) -> None:
        """Test that write failures are logged but do not raise exceptions."""
        # Use an invalid path that can't be written to (directory instead of file)
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "subdir" / "receipts.jsonl"
            invalid_path.parent.mkdir(exist_ok=True)
            # Create a directory where the file should be
            invalid_path.mkdir(exist_ok=True)

            writer = ReceiptWriter(invalid_path)
            receipt = ActionReceipt(
                receipt_id="test-id",
                game_id="game",
                level_id="level",
                step_index=0,
                state_hash_before="before",
                action_hash="action",
                selected_action=CanonicalAction(
                    action_name="RESET", action_id=0, complex_data=None
                ),
                state_hash_after="after",
                actual_delta=StateDelta(
                    changed_cells=(),
                    moved_objects=(),
                    added_objects=(),
                    removed_objects=(),
                    status_changed=False,
                ),
                verification_outcome="unknown",
            )

            # Should not raise, but return False
            success = writer.write(receipt)
            assert success is False

    def test_no_raw_sdk_objects_in_serialized_output(self) -> None:
        """Test that serialized receipts contain no raw SDK objects."""
        receipt = ActionReceipt(
            receipt_id="test-id",
            game_id="game",
            level_id="level",
            step_index=0,
            state_hash_before="before",
            action_hash="action",
            selected_action=CanonicalAction(
                action_name="ACTION1", action_id=1, complex_data=None
            ),
            state_hash_after="after",
            actual_delta=StateDelta(
                changed_cells=(),
                moved_objects=(),
                added_objects=(),
                removed_objects=(),
                status_changed=False,
            ),
            verification_outcome="unknown",
        )

        serialized = ReceiptWriter._serialize_receipt(receipt)
        data = json.loads(serialized)

        # Verify no unexpected object types
        serialized_str = json.dumps(data)
        # Should only contain basic types and no class references
        assert "<" not in serialized_str  # No object representations
        assert "object at" not in serialized_str  # No Python object refs

    def test_parent_directory_creation(self) -> None:
        """Test that parent directories are created automatically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "receipts" / "deep" / "nested" / "receipts.jsonl"
            writer = ReceiptWriter(nested_path)

            assert nested_path.parent.exists()

    def test_generate_receipt_id_is_unique(self) -> None:
        """Test that generate_receipt_id produces unique IDs."""
        ids = {ReceiptWriter.generate_receipt_id() for _ in range(100)}
        assert len(ids) == 100  # All unique
