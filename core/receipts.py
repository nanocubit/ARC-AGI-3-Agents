"""Append-only JSONL logging of action receipts."""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from .types import ActionReceipt

logger = logging.getLogger(__name__)


class ReceiptWriter:
    """Append-only writer for ActionReceipt objects to JSONL format.
    
    Properties:
    - One receipt per line (valid JSON)
    - Deterministic serialization (no SDK objects)
    - Configurable local path
    - Logging failures do NOT crash the agent
    - Parent directory creation handled automatically
    """

    def __init__(self, output_path: str | Path | None = None) -> None:
        """Initialize the receipt writer.
        
        Args:
            output_path: Path to JSONL file. Defaults to ./receipts/receipts.jsonl
        """
        if output_path is None:
            output_path = Path("receipts") / "receipts.jsonl"
        else:
            output_path = Path(output_path)

        self.output_path = output_path
        self._ensure_parent_directory()

    def _ensure_parent_directory(self) -> None:
        """Create parent directories if they do not exist."""
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create parent directory for {self.output_path}: {e}")

    def write(self, receipt: ActionReceipt) -> bool:
        """Append a receipt to the JSONL file.
        
        Args:
            receipt: The ActionReceipt to log.
            
        Returns:
            True if successful, False otherwise. Never raises.
        """
        try:
            serialized = self._serialize_receipt(receipt)
            with open(self.output_path, "a", encoding="utf-8") as f:
                f.write(serialized + "\n")
            logger.debug(f"Wrote receipt {receipt.receipt_id} to {self.output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to write receipt {receipt.receipt_id}: {e}")
            return False

    @staticmethod
    def _serialize_receipt(receipt: ActionReceipt) -> str:
        """Convert ActionReceipt to JSON string.
        
        Args:
            receipt: The receipt to serialize.
            
        Returns:
            JSON string representation.
        """
        receipt_dict: dict[str, Any] = {
            "receipt_id": receipt.receipt_id,
            "game_id": receipt.game_id,
            "level_id": receipt.level_id,
            "step_index": receipt.step_index,
            "state_hash_before": receipt.state_hash_before,
            "action_hash": receipt.action_hash,
            "selected_action": {
                "action_name": receipt.selected_action.action_name,
                "action_id": receipt.selected_action.action_id,
                "complex_data": receipt.selected_action.complex_data,
            },
            "state_hash_after": receipt.state_hash_after,
            "actual_delta": {
                "changed_cells": receipt.actual_delta.changed_cells,
                "moved_objects": receipt.actual_delta.moved_objects,
                "added_objects": receipt.actual_delta.added_objects,
                "removed_objects": receipt.actual_delta.removed_objects,
                "status_changed": receipt.actual_delta.status_changed,
            },
            "verification_outcome": receipt.verification_outcome,
        }
        return json.dumps(receipt_dict, separators=(",", ":"))

    @staticmethod
    def generate_receipt_id() -> str:
        """Generate a unique receipt ID (UUID)."""
        return str(uuid.uuid4())
