"""Deterministic canonical serialization and hashing.

All hashes use SHA-256 and are stable across processes.
Dictionary/object ordering is deterministic.
No Python hash(), timestamps, or random values are used for identity.
"""

import hashlib
import json
from typing import Any

from .types import CanonicalAction, CanonicalState, StateDelta


def _canonical_json(obj: Any) -> str:
    """Serialize an object to canonical (sorted) JSON.
    
    Ensures that identical logical values produce identical byte sequences
    regardless of construction order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def canonicalize(state: CanonicalState) -> bytes:
    """Convert a CanonicalState to deterministic bytes for hashing.
    
    Args:
        state: The canonical state to serialize.
        
    Returns:
        Canonical byte representation.
    """
    state_dict = {
        "grid": state.grid,
        "objects": [
            {
                "object_id": obj.object_id,
                "color": obj.color,
                "cells": obj.cells,
                "bbox": obj.bbox,
                "area": obj.area,
            }
            for obj in state.objects
        ],
        "available_actions": [
            {
                "action_name": action.action_name,
                "action_id": action.action_id,
                "complex_data": action.complex_data,
            }
            for action in state.available_actions
        ],
        "status": state.status,
    }
    return _canonical_json(state_dict).encode("utf-8")


def state_hash(state: CanonicalState) -> str:
    """Compute SHA-256 hash of a canonical state.
    
    Args:
        state: The canonical state.
        
    Returns:
        Hex-encoded SHA-256 digest.
    """
    canonical_bytes = canonicalize(state)
    return hashlib.sha256(canonical_bytes).hexdigest()


def action_hash(action: CanonicalAction) -> str:
    """Compute SHA-256 hash of a canonical action.
    
    Args:
        action: The canonical action.
        
    Returns:
        Hex-encoded SHA-256 digest.
    """
    action_dict = {
        "action_name": action.action_name,
        "action_id": action.action_id,
        "complex_data": action.complex_data,
    }
    canonical_bytes = _canonical_json(action_dict).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def delta_signature(delta: StateDelta) -> str:
    """Compute a signature for a state delta (not a cryptographic hash).
    
    Useful for logging and debugging, but not used for verification.
    
    Args:
        delta: The state delta.
        
    Returns:
        Hex-encoded SHA-256 digest.
    """
    delta_dict = {
        "changed_cells": delta.changed_cells,
        "moved_objects": delta.moved_objects,
        "added_objects": delta.added_objects,
        "removed_objects": delta.removed_objects,
        "status_changed": delta.status_changed,
    }
    canonical_bytes = _canonical_json(delta_dict).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()
