"""Report generation for M2-A.1 Predicate Inventory.

Generates a markdown report from an Inventory object.
The report honestly reflects data completeness: if only receipts are
available, it reports that P1-P4 predicates are unavailable.
"""

from __future__ import annotations

from typing import Any

from m2.inventory import Inventory


class InventoryReport:
    """Generate a markdown report from an Inventory.

    The report structure:
    1. Data completeness summary
    2. Action distribution
    3. Effect taxonomy
    4. Predicate frequencies (only if full data)
    5. Truth tables (only if full data)
    6. Representation comparison (only if full data)
    7. Trajectory split
    8. Cross-trajectory stability (only if full data)
    9. Compression proxy
    10. Conclusions and next steps
    """

    @staticmethod
    def generate(inventory: Inventory) -> str:
        """Generate the full markdown report."""
        sections: list[str] = []
        sections.append("# M2-A.1 Predicate Inventory Report\n")
        sections.append(InventoryReport._data_completeness(inventory))
        sections.append(InventoryReport._action_distribution(inventory))
        sections.append(InventoryReport._effect_taxonomy(inventory))
        sections.append(InventoryReport._predicate_frequencies(inventory))
        sections.append(InventoryReport._truth_tables(inventory))
        sections.append(InventoryReport._representation_comparison(inventory))
        sections.append(InventoryReport._trajectory_split(inventory))
        sections.append(InventoryReport._cross_trajectory_stability(inventory))
        sections.append(InventoryReport._compression_proxy(inventory))
        sections.append(InventoryReport._conclusions(inventory))
        return "\n".join(sections)

    @staticmethod
    def _data_completeness(inv: Inventory) -> str:
        lines = [
            "## Data Completeness\n",
            f"- **Mode**: `{inv.data_completeness}`",
            f"- **Total trajectories**: {inv.total_trajectories}",
            f"- **Total transitions**: {inv.total_transitions}",
            f"- **Full transitions** (with snapshots): {inv.full_transitions}",
            f"- **Receipt-only transitions**: {inv.receipt_only_transitions}",
            "",
        ]
        if inv.data_completeness == "receipt_only":
            lines.append(
                "> **Note**: P1-P4 predicate truth tables are unavailable "
                "without canonical snapshots. Only action distribution and "
                "effect taxonomy are reported.\n"
            )
        return "\n".join(lines)

    @staticmethod
    def _action_distribution(inv: Inventory) -> str:
        lines = ["## Action Distribution\n"]
        if inv.action_stats is None:
            lines.append("No action data available.\n")
            return "\n".join(lines)

        stats = inv.action_stats
        lines.append(f"- **Total actions**: {stats.total}")
        lines.append("")
        lines.append("| Action ID | Count | Percentage |")
        lines.append("|-----------|-------|------------|")
        for aid in sorted(stats.counts.keys()):
            count = stats.counts[aid]
            pct = (count / stats.total * 100) if stats.total > 0 else 0.0
            lines.append(f"| {aid} | {count} | {pct:.1f}% |")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _effect_taxonomy(inv: Inventory) -> str:
        lines = ["## Effect Taxonomy\n"]
        if inv.effect_stats is None:
            lines.append("No effect data available.\n")
            return "\n".join(lines)

        stats = inv.effect_stats
        lines.append(f"- **Total transitions**: {stats.total}")
        lines.append(f"- **Composite transitions**: {stats.composite_count}")
        lines.append("")
        lines.append("| Effect Type | Count | Percentage |")
        lines.append("|-------------|-------|------------|")
        for eff_type, count in sorted(stats.counts.items()):
            pct = (count / stats.total * 100) if stats.total > 0 else 0.0
            lines.append(f"| {eff_type} | {count} | {pct:.1f}% |")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _predicate_frequencies(inv: Inventory) -> str:
        lines = ["## Predicate Frequencies\n"]
        if not inv.predicate_stats:
            lines.append(
                "No predicate data available (requires full canonical snapshots).\n"
            )
            return "\n".join(lines)

        # Group by level
        by_level: dict[str, list[Any]] = {}
        for stats in inv.predicate_stats.values():
            level = stats.level
            by_level.setdefault(level, []).append(stats)

        for level in sorted(by_level.keys()):
            lines.append(f"### {level}\n")
            lines.append("| Predicate | True | Evaluated | Frequency |")
            lines.append("|-----------|-------|-----------|-----------|")
            for stats in sorted(by_level[level], key=lambda s: s.predicate):
                lines.append(
                    f"| {stats.predicate} | {stats.true_count} | "
                    f"{stats.eval_count} | {stats.frequency:.4f} |"
                )
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _truth_tables(inv: Inventory) -> str:
        lines = ["## Truth Tables (Predicate / Effect Co-occurrence)\n"]
        if not inv.truth_table:
            lines.append(
                "No truth table data available (requires full canonical snapshots).\n"
            )
            return "\n".join(lines)

        lines.append("| Predicate | Effect Type | Count |")
        lines.append("|-----------|-------------|-------|")
        for entry in inv.truth_table[:50]:  # Cap at 50 rows
            lines.append(
                f"| {entry.predicate} | {entry.effect_type} | {entry.count} |"
            )
        if len(inv.truth_table) > 50:
            lines.append(f"| ... | ... | ({len(inv.truth_table) - 50} more) |")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _representation_comparison(inv: Inventory) -> str:
        lines = ["## Representation Comparison\n"]
        if inv.representation_stats is None:
            lines.append(
                "No representation comparison available "
                "(requires full canonical snapshots).\n"
            )
            return "\n".join(lines)

        rep = inv.representation_stats
        lines.append("| Representation | Size |")
        lines.append("|----------------|------|")
        lines.append(f"| A. Raw trajectory | {rep.raw_count} |")
        lines.append(f"| B. Object-based | {rep.object_count} |")
        lines.append(f"| C. Grid-first Fabric | {rep.grid_first_count} |")
        lines.append("")
        lines.append(
            f"- **Compression ratio** (raw / grid_first): "
            f"{rep.compression_raw_vs_grid:.4f}\n"
        )
        return "\n".join(lines)

    @staticmethod
    def _trajectory_split(inv: Inventory) -> str:
        lines = ["## Trajectory Split (Train / Held-out)\n"]
        if inv.trajectory_split is None:
            lines.append("No trajectory split available.\n")
            return "\n".join(lines)

        split = inv.trajectory_split
        lines.append(f"- **Split method**: {split.split_method}")
        lines.append(f"- **Train trajectories**: {split.train_count}")
        lines.append(f"- **Test trajectories**: {split.test_count}")
        lines.append(
            f"- **Train transitions**: "
            f"{sum(t.length for t in split.train)}"
        )
        lines.append(
            f"- **Test transitions**: "
            f"{sum(t.length for t in split.test)}"
        )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _cross_trajectory_stability(inv: Inventory) -> str:
        lines = ["## Cross-Trajectory Stability\n"]
        if not inv.cross_trajectory_stability:
            lines.append(
                "No stability data available "
                "(requires full canonical snapshots).\n"
            )
            return "\n".join(lines)

        stability = inv.cross_trajectory_stability
        lines.append(
            f"- **Average Jaccard similarity**: {stability.get('stability', 0.0)}"
        )
        lines.append(f"- **Min similarity**: {stability.get('min', 0.0)}")
        lines.append(f"- **Max similarity**: {stability.get('max', 0.0)}")
        lines.append(
            f"- **Pairs compared**: {stability.get('pairs_compared', 0)}"
        )
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _compression_proxy(inv: Inventory) -> str:
        lines = ["## Compression Proxy\n"]
        if inv.representation_stats is None:
            lines.append(
                "Compression proxy requires full canonical snapshots.\n"
            )
            return "\n".join(lines)

        rep = inv.representation_stats
        if rep.raw_count == 0:
            lines.append("No data for compression calculation.\n")
            return "\n".join(lines)

        compression = rep.compression_raw_vs_grid
        lines.append(f"- **Raw transition count**: {rep.raw_count}")
        lines.append(f"- **Grid-first size**: {rep.grid_first_count}")
        lines.append(f"- **Compression ratio**: {compression:.4f}")
        if compression > 1.0:
            lines.append("- **Verdict**: Grid-first representation compresses data.")
        else:
            lines.append("- **Verdict**: No compression benefit from grid-first.")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _conclusions(inv: Inventory) -> str:
        lines = ["## Conclusions and Next Steps\n"]

        if inv.data_completeness == "receipt_only":
            lines.append(
                "1. **Data sufficiency**: Only receipt data available. "
                "P1-P4 predicate truth tables require canonical snapshots "
                "or game replay."
            )
            lines.append(
                "2. **Effect taxonomy**: Available from StateDelta. "
                "Review distribution of effect types."
            )
            lines.append(
                "3. **Next step**: Either (a) run agent with snapshot capture, "
                "or (b) build replay engine to reconstruct canonical states "
                "from game logs."
            )
        else:
            lines.append(
                "1. **Predicate coverage**: Review P0-P4 frequency tables."
            )
            if inv.representation_stats:
                rep = inv.representation_stats
                if rep.compression_raw_vs_grid > 1.0:
                    lines.append(
                        "2. **Compression**: Grid-first representation "
                        "shows compression benefit."
                    )
                else:
                    lines.append(
                        "2. **Compression**: No compression benefit detected."
                    )
            lines.append(
                "3. **Next step**: If predicate basis is stable and "
                "compression is beneficial, proceed to M2-A.2 "
                "(bounded guard synthesis)."
            )

        lines.append("")
        lines.append(
            "> **Important**: M2-A.1 is offline only. "
            "No rules, hypotheses, or online learning have been introduced."
        )
        lines.append("")
        return "\n".join(lines)
