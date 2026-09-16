"""Inventory aggregation for M2-A.1.

Aggregates predicate statistics, effect taxonomies, and truth tables
from trajectories. Supports train/held-out splits and cross-trajectory
stability checks.

In receipt_only mode:
  - Action distribution (from selected_action)
  - Effect taxonomy (from StateDelta)
  - Raw transition count
  Cannot produce P1-P4 truth tables.

In full mode (with canonical snapshots):
  - All of the above, plus:
  - P0-P4 predicate frequencies
  - Predicate/effect co-occurrence
  - Object-vs-grid representation comparison
  - Compression proxy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from m2.effects import EffectClassifier, EffectType
from m2.predicates import (
    PREDICATE_CATALOG,
    action_enabled,
    color_at,
    empty,
    in_bounds,
    object_cells,
    occupied,
    on_border,
    touches_border,
)
from m2.trajectory import DataCompleteness, Trajectory


@dataclass(frozen=True)
class ActionStats:
    """Distribution of selected actions across trajectories."""

    counts: dict[int, int]  # action_id → count
    total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": {str(k): v for k, v in sorted(self.counts.items())},
            "total": self.total,
        }


@dataclass(frozen=True)
class EffectStats:
    """Distribution of effect types across transitions."""

    counts: dict[str, int]  # effect_type → count
    total: int
    composite_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "total": self.total,
            "composite_count": self.composite_count,
        }


@dataclass(frozen=True)
class PredicateStats:
    """Frequency statistics for a single predicate type.

    Counts how many times the predicate was evaluated and how many
    times it was true across all states.
    """

    predicate: str
    level: str
    true_count: int
    eval_count: int

    @property
    def frequency(self) -> float:
        """Fraction of evaluations where predicate was true."""
        if self.eval_count == 0:
            return 0.0
        return self.true_count / self.eval_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "level": self.level,
            "true_count": self.true_count,
            "eval_count": self.eval_count,
            "frequency": round(self.frequency, 4),
        }


@dataclass(frozen=True)
class TruthTableEntry:
    """A row in the predicate/effect co-occurrence truth table.

    Records how many transitions had a specific effect type and a
    specific predicate true in the before-state.
    """

    predicate: str
    effect_type: str
    count: int


@dataclass(frozen=True)
class RepresentationStats:
    """Size comparison between three representations.

    A. Raw trajectory: number of distinct (state_hash, action, delta) tuples
    B. Object-based: number of distinct (object_set, action, object_delta) tuples
    C. Grid-first: predicate library + rule count + residuals
    """

    raw_count: int
    object_count: int
    grid_first_count: int

    @property
    def compression_raw_vs_grid(self) -> float:
        """Compression ratio: raw / grid_first."""
        if self.grid_first_count == 0:
            return 0.0
        return self.raw_count / self.grid_first_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_count": self.raw_count,
            "object_count": self.object_count,
            "grid_first_count": self.grid_first_count,
            "compression_ratio": round(self.compression_raw_vs_grid, 4),
        }


@dataclass(frozen=True)
class TrajectorySplit:
    """Train/held-out split of trajectories.

    Splits by game_id: some games → train, others → test.
    Within same game_id: first levels → train, later levels → test.
    """

    train: tuple[Trajectory, ...]
    test: tuple[Trajectory, ...]
    split_method: str  # "by_game" or "by_level"

    @property
    def train_count(self) -> int:
        return len(self.train)

    @property
    def test_count(self) -> int:
        return len(self.test)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_method": self.split_method,
            "train_trajectories": self.train_count,
            "test_trajectories": self.test_count,
            "train_transitions": sum(t.length for t in self.train),
            "test_transitions": sum(t.length for t in self.test),
        }


@dataclass
class Inventory:
    """Aggregated statistics from trajectory analysis.

    Mutable during building, frozen conceptually after build.
    """

    # Data completeness
    data_completeness: DataCompleteness = "receipt_only"
    total_trajectories: int = 0
    total_transitions: int = 0
    full_transitions: int = 0
    receipt_only_transitions: int = 0

    # Action stats
    action_stats: ActionStats | None = None

    # Effect stats
    effect_stats: EffectStats | None = None

    # Predicate stats (only in full mode)
    predicate_stats: dict[str, PredicateStats] = field(default_factory=dict)

    # Truth table entries (only in full mode)
    truth_table: list[TruthTableEntry] = field(default_factory=list)

    # Representation comparison (only in full mode)
    representation_stats: RepresentationStats | None = None

    # Trajectory split
    trajectory_split: TrajectorySplit | None = None

    # Cross-trajectory stability
    cross_trajectory_stability: dict[str, float] = field(default_factory=dict)

    @property
    def can_evaluate_predicates(self) -> bool:
        """True if we have enough data for P1-P4 evaluation."""
        return self.data_completeness == "full"

    def to_dict(self) -> dict[str, Any]:
        """Serialize inventory to dictionary for report generation."""
        result: dict[str, Any] = {
            "data_completeness": self.data_completeness,
            "total_trajectories": self.total_trajectories,
            "total_transitions": self.total_transitions,
            "full_transitions": self.full_transitions,
            "receipt_only_transitions": self.receipt_only_transitions,
            "action_stats": (
                self.action_stats.to_dict() if self.action_stats else None
            ),
            "effect_stats": (
                self.effect_stats.to_dict() if self.effect_stats else None
            ),
            "predicate_stats": (
                {k: v.to_dict() for k, v in self.predicate_stats.items()}
                if self.predicate_stats
                else {}
            ),
            "truth_table": [
                {"predicate": e.predicate, "effect_type": e.effect_type, "count": e.count}
                for e in self.truth_table
            ],
            "representation_stats": (
                self.representation_stats.to_dict()
                if self.representation_stats
                else None
            ),
            "trajectory_split": (
                self.trajectory_split.to_dict()
                if self.trajectory_split
                else None
            ),
            "cross_trajectory_stability": dict(
                sorted(self.cross_trajectory_stability.items())
            ),
        }
        return result


class InventoryBuilder:
    """Builds an Inventory from trajectories.

    In receipt_only mode: action stats + effect taxonomy only.
    In full mode: also P0-P4 predicate frequencies + truth tables.
    """

    def __init__(self) -> None:
        self._inventory = Inventory()

    def build(
        self,
        trajectories: list[Trajectory],
        train_split: float = 0.6,
    ) -> Inventory:
        """Build a complete inventory from trajectories.

        Args:
            trajectories: List of trajectories to analyze.
            train_split: Fraction of games for training (rest = held-out).
        """
        inv = self._inventory

        # Determine overall data completeness
        any_full = any(t.is_full for t in trajectories)
        all_full = all(t.is_full for t in trajectories) if trajectories else False
        if all_full:
            inv.data_completeness = "full"
        elif any_full:
            inv.data_completeness = "receipt_only"
        else:
            inv.data_completeness = "receipt_only"

        # Count trajectories and transitions
        inv.total_trajectories = len(trajectories)
        inv.total_transitions = sum(t.length for t in trajectories)
        inv.full_transitions = sum(
            1
            for t in trajectories
            for tr in t.transitions
            if tr.data_completeness == "full"
        )
        inv.receipt_only_transitions = inv.total_transitions - inv.full_transitions

        # Build action stats (always available)
        inv.action_stats = self._build_action_stats(trajectories)

        # Build effect stats (always available from delta)
        inv.effect_stats = self._build_effect_stats(trajectories)

        # Build trajectory split
        inv.trajectory_split = self._build_split(trajectories, train_split)

        # Build predicate stats + truth tables (only if full data available)
        if any_full:
            inv.predicate_stats = self._build_predicate_stats(trajectories)
            inv.truth_table = self._build_truth_table(trajectories)
            inv.representation_stats = self._build_representation_stats(
                trajectories
            )
            inv.cross_trajectory_stability = (
                self._build_cross_trajectory_stability(trajectories)
            )

        return inv

    @staticmethod
    def _build_action_stats(
        trajectories: list[Trajectory],
    ) -> ActionStats:
        """Build action distribution from receipts."""
        counts: dict[int, int] = {}
        total = 0
        for traj in trajectories:
            for trans in traj.transitions:
                aid = trans.receipt.action_id
                counts[aid] = counts.get(aid, 0) + 1
                total += 1
        return ActionStats(counts=counts, total=total)

    @staticmethod
    def _build_effect_stats(
        trajectories: list[Trajectory],
    ) -> EffectStats:
        """Build effect taxonomy from deltas."""
        effect_counts: dict[str, int] = {}
        composite_count = 0
        total = 0
        for traj in trajectories:
            for trans in traj.transitions:
                effect = EffectClassifier.classify(
                    trans.receipt.actual_delta
                )
                key = effect.effect_type.value
                effect_counts[key] = effect_counts.get(key, 0) + 1
                if effect.effect_type == EffectType.COMPOSITE:
                    composite_count += 1
                total += 1
        return EffectStats(
            counts=effect_counts, total=total, composite_count=composite_count
        )

    @staticmethod
    def _build_split(
        trajectories: list[Trajectory],
        train_split: float,
    ) -> TrajectorySplit:
        """Split trajectories into train/held-out by game_id.

        Games are sorted by game_id. First fraction → train, rest → test.
        Within same game_id: first levels → train, later → test.
        """
        # Group by game_id
        by_game: dict[str, list[Trajectory]] = {}
        for t in trajectories:
            by_game.setdefault(t.game_id, []).append(t)

        sorted_games = sorted(by_game.keys())
        n_games = len(sorted_games)
        n_train = max(1, int(n_games * train_split)) if n_games > 1 else n_games

        train: list[Trajectory] = []
        test: list[Trajectory] = []

        for i, game_id in enumerate(sorted_games):
            game_trajs = sorted(
                by_game[game_id], key=lambda t: t.derived_level_key
            )
            if i < n_train:
                # Within game: first levels → train, later → test
                n_levels = len(game_trajs)
                if n_levels <= 1:
                    train.extend(game_trajs)
                else:
                    n_train_levels = max(1, int(n_levels * train_split))
                    train.extend(game_trajs[:n_train_levels])
                    test.extend(game_trajs[n_train_levels:])
            else:
                test.extend(game_trajs)

        return TrajectorySplit(
            train=tuple(train),
            test=tuple(test),
            split_method="by_game" if n_games > 1 else "by_level",
        )

    @staticmethod
    def _build_predicate_stats(
        trajectories: list[Trajectory],
    ) -> dict[str, PredicateStats]:
        """Build predicate frequency statistics from full trajectories.

        Only evaluates predicates on transitions with full snapshots.
        """
        stats: dict[str, int] = {}  # predicate → true_count
        evals: dict[str, int] = {}  # predicate → eval_count

        for traj in trajectories:
            for trans in traj.transitions:
                if trans.data_completeness != "full":
                    continue
                state = trans.state_before
                if state is None:
                    continue
                cs = state.canonical_state

                # P0: ActionEnabled — evaluate for all 8 action IDs
                for aid in range(8):
                    key = f"ActionEnabled({aid})"
                    val = action_enabled(cs, aid)
                    stats[key] = stats.get(key, 0) + (1 if val else 0)
                    evals[key] = evals.get(key, 0) + 1

                # P1: Grid predicates — sample cells
                h, w = len(cs.grid), len(cs.grid[0]) if cs.grid else 0
                for r in range(min(h, 8)):  # Cap at 8x8 sample for performance
                    for c in range(min(w, 8)):
                        cell = (r, c)
                        for pred_name, pred_fn in [
                            ("InBounds", lambda s, c=cell: in_bounds(s, c)),
                            ("Occupied", lambda s, c=cell: occupied(s, c)),
                            ("Empty", lambda s, c=cell: empty(s, c)),
                            ("OnBorder", lambda s, c=cell: on_border(s, c)),
                        ]:
                            val = pred_fn(cs)
                            stats[pred_name] = stats.get(pred_name, 0) + (1 if val else 0)
                            evals[pred_name] = evals.get(pred_name, 0) + 1

                        # ColorAt — evaluate for top 4 colors
                        colors_present = set()
                        for r2 in range(min(h, 8)):
                            for c2 in range(min(w, 8)):
                                colors_present.add(cs.grid[r2][c2])
                        for color in sorted(colors_present)[:4]:
                            key = f"ColorAt({color})"
                            val = color_at(cs, cell, color)
                            stats[key] = stats.get(key, 0) + (1 if val else 0)
                            evals[key] = evals.get(key, 0) + 1

                # P3/P4: Object predicates
                for obj in cs.objects:
                    obj_id = obj.object_id
                    for pred_name, pred_fn in [
                        ("TouchesBorder", lambda s, oid=obj_id: touches_border(s, oid)),
                        ("ObjectCells", lambda s, oid=obj_id: len(object_cells(s, oid)) > 0),
                    ]:
                        val = pred_fn(cs)
                        stats[pred_name] = stats.get(pred_name, 0) + (1 if val else 0)
                        evals[pred_name] = evals.get(pred_name, 0) + 1

        # Build PredicateStats for each predicate
        result: dict[str, PredicateStats] = {}
        all_keys = set(list(stats.keys()) + list(evals.keys()))
        for key in sorted(all_keys):
            base_name = key.split("(")[0]
            info = PREDICATE_CATALOG.get(base_name)
            level = info.level if info else "?"
            result[key] = PredicateStats(
                predicate=key,
                level=level,
                true_count=stats.get(key, 0),
                eval_count=evals.get(key, 0),
            )
        return result

    @staticmethod
    def _build_truth_table(
        trajectories: list[Trajectory],
    ) -> list[TruthTableEntry]:
        """Build predicate/effect co-occurrence truth table.

        For each transition with full data, records which predicates were
        true in the before-state and what effect occurred.
        """
        co_occurrence: dict[tuple[str, str], int] = {}

        for traj in trajectories:
            for trans in traj.transitions:
                if trans.data_completeness != "full":
                    continue
                state = trans.state_before
                if state is None:
                    continue
                cs = state.canonical_state

                effect = EffectClassifier.classify(
                    trans.receipt.actual_delta
                )

                # Evaluate key predicates on before-state
                h, w = len(cs.grid), len(cs.grid[0]) if cs.grid else 0
                predicates_true: list[str] = []

                # P0
                for aid in range(8):
                    if action_enabled(cs, aid):
                        predicates_true.append(f"ActionEnabled({aid})")

                # P1 (sample)
                for r in range(min(h, 4)):
                    for c in range(min(w, 4)):
                        cell = (r, c)
                        if occupied(cs, cell):
                            predicates_true.append("Occupied")
                        if on_border(cs, cell):
                            predicates_true.append("OnBorder")

                # P3
                for obj in cs.objects:
                    if touches_border(cs, obj.object_id):
                        predicates_true.append("TouchesBorder")

                # Record co-occurrences
                for pred in predicates_true:
                    key = (pred, effect.effect_type.value)
                    co_occurrence[key] = co_occurrence.get(key, 0) + 1

        return [
            TruthTableEntry(
                predicate=pred,
                effect_type=eff,
                count=count,
            )
            for (pred, eff), count in sorted(co_occurrence.items())
        ]

    @staticmethod
    def _build_representation_stats(
        trajectories: list[Trajectory],
    ) -> RepresentationStats:
        """Build representation comparison statistics.

        A. Raw: distinct (state_hash, action_hash, delta) tuples
        B. Object: distinct (object_set, action, object_delta) tuples
        C. Grid-first: predicate library + rule count + residuals
        """
        raw_set: set[tuple[str, str]] = set()
        object_set: set[tuple[str, str]] = set()

        for traj in trajectories:
            for trans in traj.transitions:
                if trans.data_completeness != "full":
                    continue
                # Raw representation
                raw_key = (
                    trans.receipt.state_hash_before,
                    trans.receipt.action_hash,
                )
                raw_set.add(raw_key)

                # Object representation
                state = trans.state_before
                if state:
                    obj_ids = tuple(
                        sorted(o.object_id for o in state.canonical_state.objects)
                    )
                    obj_key = (str(obj_ids), trans.receipt.action_hash)
                    object_set.add(obj_key)

        # Grid-first: predicate count + unique effects + residuals
        predicate_count = len(PREDICATE_CATALOG)
        effect_types = set()
        for traj in trajectories:
            for trans in traj.transitions:
                effect = EffectClassifier.classify(
                    trans.receipt.actual_delta
                )
                effect_types.add(effect.effect_type.value)
        grid_first_count = predicate_count + len(effect_types) + len(raw_set)

        return RepresentationStats(
            raw_count=len(raw_set),
            object_count=len(object_set),
            grid_first_count=grid_first_count,
        )

    @staticmethod
    def _build_cross_trajectory_stability(
        trajectories: list[Trajectory],
    ) -> dict[str, float]:
        """Check if different trajectories produce the same predicate basis.

        For each pair of trajectories with full data, compute Jaccard
        similarity of their predicate true-sets.
        """
        traj_predicates: list[set[str]] = []
        for traj in trajectories:
            preds: set[str] = set()
            for trans in traj.transitions:
                if trans.data_completeness != "full":
                    continue
                state = trans.state_before
                if state is None:
                    continue
                cs = state.canonical_state

                h, w = len(cs.grid), len(cs.grid[0]) if cs.grid else 0
                for r in range(min(h, 4)):
                    for c in range(min(w, 4)):
                        cell = (r, c)
                        if occupied(cs, cell):
                            preds.add("Occupied")
                        if empty(cs, cell):
                            preds.add("Empty")
                        if on_border(cs, cell):
                            preds.add("OnBorder")

                for obj in cs.objects:
                    if touches_border(cs, obj.object_id):
                        preds.add("TouchesBorder")
            if preds:
                traj_predicates.append(preds)

        if len(traj_predicates) < 2:
            return {"stability": 1.0, "pairs_compared": 0}

        # Compute pairwise Jaccard similarity
        similarities: list[float] = []
        for i in range(len(traj_predicates)):
            for j in range(i + 1, len(traj_predicates)):
                s1, s2 = traj_predicates[i], traj_predicates[j]
                union = s1 | s2
                if union:
                    sim = len(s1 & s2) / len(union)
                else:
                    sim = 1.0
                similarities.append(sim)

        avg_sim = sum(similarities) / len(similarities) if similarities else 1.0
        min_sim = min(similarities) if similarities else 1.0
        max_sim = max(similarities) if similarities else 1.0

        return {
            "stability": round(avg_sim, 4),
            "min": round(min_sim, 4),
            "max": round(max_sim, 4),
            "pairs_compared": len(similarities),
        }
