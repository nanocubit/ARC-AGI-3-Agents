"""Test append-only JSONL receipt writer."""

import json
import tempfile
from pathlib import Path

from core.receipts import ReceiptWriter
from core.types import ActionReceipt, CanonicalAction, StateDelta


def make_receipt(
    receipt_id: str = "test-id",
    game_id: str = "game_1",
    derived_level_key: str = "game_1:level:0",
    step_index: int = 0,
    state_hash_before: str = "hash_before",
    action_hash: str = "action_hash",
    state_hash_after: str = "hash_after",
    selected_action: CanonicalAction | None = None,
    actual_delta: StateDelta | None = None,
    verification_outcome: str = "unknown",
) -> ActionReceipt:
    """Helper: create an ActionReceipt with defaults."""
    return ActionReceipt(
        receipt_id=receipt_id,
        game_id=game_id,
        derived_level_key=derived_level_key,
        step_index=step_index,
        state_hash_before=state_hash_before,
        action_hash=action_hash,
        selected_action=selected_action or CanonicalAction(
            action_name="RESET", action_id=0, complex_data=None
        ),
        state_hash_after=state_hash_after,
        actual_delta=actual_delta or StateDelta(
            changed_cells=(),
            moved_objects=(),
            added_objects=(),
            removed_objects=(),
            status_changed=False,
        ),
        verification_outcome=verification_outcome,
    )


class TestReceiptWriter:
    """Tests for ReceiptWriter JSONL logging."""

    def test_one_receipt_produces_one_json_line(self) -> None:
        """Test that a single receipt produces exactly one JSON line."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_receipts.jsonl"
            writer = ReceiptWriter(output_path)

            receipt = make_receipt(receipt_id="test-id-1")

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

            receipt = make_receipt(
                receipt_id="test-id-2",
                game_id="game_2",
                derived_level_key="game_2:level:1",
                step_index=1,
                selected_action=CanonicalAction(
                    action_name="ACTION1", action_id=1, complex_data=None
                ),
            )

            writer.write(receipt)

            with open(output_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    assert data is not None
                    assert isinstance(data, dict)

    def test_all_fields_are_serializable_primitives(self) -> None:
        """Test that all receipt fields serialize to JSON primitives."""
        receipt = make_receipt(
            receipt_id="test-id-3",
            game_id="game_3",
            derived_level_key="game_3:level:5",
            step_index=5,
            selected_action=CanonicalAction(
                action_name="ACTION6",
                action_id=6,
                complex_data=(("x", 10), ("y", 20)),
            ),
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
        assert isinstance(data["derived_level_key"], str)
        assert isinstance(data["step_index"], int)
        assert isinstance(data["state_hash_before"], str)
        assert isinstance(data["action_hash"], str)
        assert isinstance(data["state_hash_after"], str)
        assert isinstance(data["verification_outcome"], str)
        assert isinstance(data["selected_action"], dict)
        assert isinstance(data["actual_delta"], dict)
        assert isinstance(data["schema_version"], str)
        assert data["schema_version"] == "1"

    def test_append_only_behavior(self) -> None:
        """Test that multiple receipts are appended to the same file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_receipts.jsonl"
            writer = ReceiptWriter(output_path)

            for i in range(3):
                receipt = make_receipt(
                    receipt_id=f"test-id-{i}",
                    step_index=i,
                )
                writer.write(receipt)

            with open(output_path, "r") as f:
                lines = f.readlines()
            assert len(lines) == 3

    def test_logging_failure_does_not_raise(self) -> None:
        """Test that write failures are logged but do not raise exceptions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_path = Path(tmpdir) / "subdir" / "receipts.jsonl"
            invalid_path.parent.mkdir(exist_ok=True)
            # Create a directory where the file should be
            invalid_path.mkdir(exist_ok=True)

            writer = ReceiptWriter(invalid_path)
            receipt = make_receipt()

            # Should not raise, but return False
            success = writer.write(receipt)
            assert success is False

    def test_no_raw_sdk_objects_in_serialized_output(self) -> None:
        """Test that serialized receipts contain no raw SDK objects."""
        receipt = make_receipt(
            selected_action=CanonicalAction(
                action_name="ACTION1", action_id=1, complex_data=None
            ),
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
            ReceiptWriter(nested_path)

            assert nested_path.parent.exists()

    def test_schema_version_in_serialized_output(self) -> None:
        """Test that schema_version is present in serialized output."""
        receipt = make_receipt()
        serialized = ReceiptWriter._serialize_receipt(receipt)
        data = json.loads(serialized)
        assert "schema_version" in data
        assert data["schema_version"] == "1"


class TestDeterministicReceiptId:
    """Tests for deterministic receipt ID generation."""

    def test_same_transition_same_receipt_id(self) -> None:
        """Test that the same transition produces the same receipt_id."""
        rid1 = ReceiptWriter.generate_receipt_id(
            game_id="game_1",
            derived_level_key="game_1:level:0",
            step_index=0,
            state_hash_before="abc",
            action_hash="def",
            state_hash_after="ghi",
        )
        rid2 = ReceiptWriter.generate_receipt_id(
            game_id="game_1",
            derived_level_key="game_1:level:0",
            step_index=0,
            state_hash_before="abc",
            action_hash="def",
            state_hash_after="ghi",
        )
        assert rid1 == rid2

    def test_different_transition_different_receipt_id(self) -> None:
        """Test that different transitions produce different receipt_ids."""
        rid1 = ReceiptWriter.generate_receipt_id(
            game_id="game_1",
            derived_level_key="game_1:level:0",
            step_index=0,
            state_hash_before="abc",
            action_hash="def",
            state_hash_after="ghi",
        )
        rid2 = ReceiptWriter.generate_receipt_id(
            game_id="game_1",
            derived_level_key="game_1:level:0",
            step_index=1,
            state_hash_before="abc",
            action_hash="def",
            state_hash_after="ghi",
        )
        assert rid1 != rid2

    def test_receipt_id_is_hex_string(self) -> None:
        """Test that receipt_id is a hex string."""
        rid = ReceiptWriter.generate_receipt_id(
            game_id="game_1",
            derived_level_key="game_1:level:0",
            step_index=0,
            state_hash_before="abc",
            action_hash="def",
            state_hash_after="ghi",
        )
        assert isinstance(rid, str)
        assert len(rid) == 32
        int(rid, 16)  # Should parse as hex
