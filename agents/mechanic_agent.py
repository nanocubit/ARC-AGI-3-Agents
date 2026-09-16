"""MechanicAgent: Milestone 1 skeleton for ARC-AGI-3.

This agent:
1. Converts FrameData to CanonicalState
2. Verifies the previous action
3. Computes StateDelta
4. Writes ActionReceipt
5. Selects deterministic fallback action
6. Returns SDK GameAction to the harness

No controller, solver, planner, or LLM at this stage.
"""

import logging
from dataclasses import dataclass
from typing import Any

from arcengine import FrameData, GameAction

from agents.agent import Agent
from core.hashing import action_hash
from core.receipts import ReceiptWriter
from core.state_builder import build_canonical_state, compute_state_delta
from core.types import ActionReceipt, CanonicalAction, CanonicalState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingActionContext:
    """Minimal context retained for verification on next invocation."""

    state_hash_before: str
    canonical_state_before: CanonicalState
    selected_action: CanonicalAction
    step_index: int
    game_id: str
    derived_level_key: str


class MechanicAgent(Agent):
    """Milestone 1 agent: canonical state + deterministic receipts.

    Does NOT call arc_env.step() directly.
    Returns GameAction from choose_action() only.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.receipt_writer = ReceiptWriter()
        self.pending_action: PendingActionContext | None = None
        logger.info(f"MechanicAgent initialized for game {self.game_id}")

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing.

        Args:
            frames: All frames in episode.
            latest_frame: Most recent frame.

        Returns:
            True if game is complete.
        """
        from arcengine import GameState

        if latest_frame.state is GameState.WIN:
            logger.info(f"Game {self.game_id}: WIN state reached")
            return True

        return False

    def _derive_level_key(self, latest_frame: FrameData) -> str:
        """Derive a level identity from available SDK fields.

        levels_completed is a count of completed levels, not a true
        level ID. We use it as a derived level context key, clearly
        labeled as derived rather than SDK-provided.

        Args:
            latest_frame: Current frame.

        Returns:
            Derived level key string.
        """
        levels_completed = getattr(latest_frame, "levels_completed", 0) or 0
        return f"{self.game_id}:level:{levels_completed}"

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose next action.

        Lifecycle:
        1. Convert latest_frame to CanonicalState
        2. Verify pending previous action (if any)
        3. Compute StateDelta
        4. Write ActionReceipt
        5. Select deterministic fallback action
        6. Remember pending action context
        7. Return SDK GameAction

        Args:
            frames: All frames in episode.
            latest_frame: Current frame.

        Returns:
            SDK GameAction for the harness to execute.
        """
        # 1. Convert to canonical state
        try:
            canonical_state = build_canonical_state(latest_frame)
        except Exception as e:  # noqa: BLE001 - receipt I/O must not crash agent
            logger.error(f"Failed to build canonical state: {e}")
            return GameAction.RESET

        derived_level_key = self._derive_level_key(latest_frame)

        # 2 & 3. Verify previous action and compute delta if applicable
        if self.pending_action is not None:
            try:
                delta = compute_state_delta(
                    self.pending_action.canonical_state_before, canonical_state
                )

                # 4. Determine verification outcome (conservative: default to "unknown")
                verification_outcome = self._verify_action(delta)

                act_hash = action_hash(self.pending_action.selected_action)

                # 5. Write receipt for previous action
                receipt_id = ReceiptWriter.generate_receipt_id(
                    game_id=self.pending_action.game_id,
                    derived_level_key=self.pending_action.derived_level_key,
                    step_index=self.pending_action.step_index,
                    state_hash_before=self.pending_action.state_hash_before,
                    action_hash=act_hash,
                    state_hash_after=canonical_state.state_hash,
                )

                receipt = ActionReceipt(
                    receipt_id=receipt_id,
                    game_id=self.pending_action.game_id,
                    derived_level_key=self.pending_action.derived_level_key,
                    step_index=self.pending_action.step_index,
                    state_hash_before=self.pending_action.state_hash_before,
                    action_hash=act_hash,
                    selected_action=self.pending_action.selected_action,
                    state_hash_after=canonical_state.state_hash,
                    actual_delta=delta,
                    verification_outcome=verification_outcome,
                )
                self.receipt_writer.write(receipt)
            except Exception as e:  # noqa: BLE001 - receipt I/O must not crash agent
                logger.error(f"Failed to verify previous action: {e}")

        # 6. Select deterministic fallback action
        selected_action = self._select_fallback_action(canonical_state)

        # 7. Remember pending action context for next invocation
        self.pending_action = PendingActionContext(
            state_hash_before=canonical_state.state_hash,
            canonical_state_before=canonical_state,
            selected_action=selected_action,
            step_index=self.action_counter,
            game_id=self.game_id,
            derived_level_key=derived_level_key,
        )

        # 8. Convert canonical action back to SDK GameAction
        sdk_action = self._canonical_to_sdk_action(selected_action, latest_frame)
        return sdk_action

    def _verify_action(self, delta: Any) -> str:
        """Determine verification outcome for the previous action.

        Milestone 1: No prediction engine, so outcome is "unknown" by default.
        Future milestones will implement hypothesis-based verification.

        Args:
            delta: The computed StateDelta.

        Returns:
            One of: "confirmed", "contradicted", "unknown"
        """
        # For now, always unknown (no prediction model)
        return "unknown"

    def _select_fallback_action(self, state: CanonicalState) -> CanonicalAction:
        """Select a deterministic fallback action.

        Policy:
        - Only select from available actions
        - Use deterministic tie-breaking (sorted by action_id, then name)
        - Never fabricate unavailable actions

        Args:
            state: Current canonical state.

        Returns:
            Deterministically selected CanonicalAction.
        """
        if not state.available_actions:
            logger.warning("No available actions; defaulting to RESET")
            return CanonicalAction(
                action_name="RESET",
                action_id=0,
                complex_data=None,
            )

        # Deterministic selection: first available action (already sorted)
        selected = state.available_actions[0]
        logger.debug(f"Selected fallback action: {selected.action_name}")
        return selected

    def _canonical_to_sdk_action(
        self, canonical: CanonicalAction, latest_frame: FrameData
    ) -> GameAction:
        """Convert CanonicalAction back to SDK GameAction.

        Args:
            canonical: The canonical action to convert.
            latest_frame: Current frame (for context).

        Returns:
            SDK GameAction ready for the harness.
        """
        try:
            # Find matching SDK action by name
            sdk_action = GameAction[canonical.action_name]

            # If complex data, set it
            if canonical.complex_data is not None:
                data_dict = dict(canonical.complex_data)
                sdk_action.set_data(data_dict)

            return sdk_action
        except (KeyError, AttributeError) as e:
            logger.error(
                f"Failed to convert canonical action {canonical.action_name}: {e}"
            )
            return GameAction.RESET
