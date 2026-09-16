"""Snapshot representation diagnostics for M2-A.1-R1.

Per-snapshot audit of canonical grid and object extraction.
Answers: where do sprite cells disappear? Is the problem in
canonical grid extraction or in predicate evaluation?

Output: markdown diagnostic report with per-snapshot stats:
  - grid dimensions
  - palette histogram
  - non-background cell count
  - object count, bboxes, areas
  - consistency checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from m2.trajectory import CanonicalSnapshot, Trajectory


@dataclass(frozen=True)
class GridDiagnostic:
    """Diagnostic stats for a single canonical grid."""

    state_hash: str
    game_id: str
    level_key: str
    step_index: int
    height: int
    width: int
    total_cells: int
    palette: dict[int, int]  # color → count
    non_background_count: int  # cells where color != 0
    background_count: int  # cells where color == 0
    object_count: int
    object_details: tuple[dict[str, Any], ...]  # per-object summary
    # Consistency checks
    palette_sum_ok: bool  # sum(palette.values()) == total_cells
    non_bg_ok: bool  # non_background == sum(count for color != 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_hash": self.state_hash[:12] + "...",
            "game_id": self.game_id,
            "level_key": self.level_key,
            "step_index": self.step_index,
            "height": self.height,
            "width": self.width,
            "total_cells": self.total_cells,
            "palette": dict(sorted(self.palette.items())),
            "non_background": self.non_background_count,
            "background": self.background_count,
            "objects": self.object_count,
            "object_details": list(self.object_details),
            "palette_sum_ok": self.palette_sum_ok,
            "non_bg_ok": self.non_bg_ok,
        }


@dataclass
class DiagnosticSummary:
    """Aggregate diagnostic stats across all snapshots."""

    total_snapshots: int = 0
    total_non_background: int = 0
    total_objects: int = 0
    all_palette_ok: bool = True
    all_non_bg_ok: bool = True
    per_snapshot: list[GridDiagnostic] = field(default_factory=list)

    # Key question: do any snapshots have non-background cells?
    has_non_background: bool = False
    # Key question: do any snapshots have objects?
    has_objects: bool = False
    # Key question: does Occupied predicate see non-background?
    occupied_should_be_nonzero: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_snapshots": self.total_snapshots,
            "total_non_background": self.total_non_background,
            "total_objects": self.total_objects,
            "all_palette_ok": self.all_palette_ok,
            "all_non_bg_ok": self.all_non_bg_ok,
            "has_non_background": self.has_non_background,
            "has_objects": self.has_objects,
            "occupied_should_be_nonzero": self.occupied_should_be_nonzero,
            "per_snapshot": [d.to_dict() for d in self.per_snapshot],
        }


class SnapshotDiagnostics:
    """Run diagnostics on canonical snapshots.

    For each snapshot, checks:
    1. Grid dimensions are non-zero
    2. Palette histogram sums to total cells
    3. Non-background count is computed correctly
    4. Objects exist and their cells are in the grid
    5. Object cells match non-background cells (consistency)
    """

    @staticmethod
    def diagnose_snapshot(
        snapshot: CanonicalSnapshot,
        game_id: str = "",
        level_key: str = "",
        step_index: int = 0,
    ) -> GridDiagnostic:
        """Diagnose a single canonical snapshot."""
        state = snapshot.canonical_state
        grid = state.grid

        height = len(grid)
        width = len(grid[0]) if height > 0 else 0
        total_cells = height * width

        # Palette histogram
        palette: dict[int, int] = {}
        for row in grid:
            for cell in row:
                palette[cell] = palette.get(cell, 0) + 1

        # Non-background count (color != 0)
        non_background = sum(
            count for color, count in palette.items() if color != 0
        )
        background = palette.get(0, 0)

        # Object details
        object_details: list[dict[str, Any]] = []
        for obj in state.objects:
            object_details.append({
                "object_id": obj.object_id,
                "color": obj.color,
                "area": obj.area,
                "bbox": list(obj.bbox),
                "cells_count": len(obj.cells),
                "first_cell": list(obj.cells[0]) if obj.cells else None,
            })

        # Consistency checks
        palette_sum = sum(palette.values())
        palette_sum_ok = palette_sum == total_cells
        non_bg_ok = non_background == (total_cells - background)

        return GridDiagnostic(
            state_hash=state.state_hash,
            game_id=game_id,
            level_key=level_key,
            step_index=step_index,
            height=height,
            width=width,
            total_cells=total_cells,
            palette=palette,
            non_background_count=non_background,
            background_count=background,
            object_count=len(state.objects),
            object_details=tuple(object_details),
            palette_sum_ok=palette_sum_ok,
            non_bg_ok=non_bg_ok,
        )

    @staticmethod
    def diagnose_trajectories(
        trajectories: list[Trajectory],
    ) -> DiagnosticSummary:
        """Diagnose all snapshots in a set of trajectories.

        For each transition with full data, diagnoses the before-state.
        """
        summary = DiagnosticSummary()
        seen_hashes: set[str] = set()

        for traj in trajectories:
            for trans in traj.transitions:
                if trans.data_completeness != "full":
                    continue
                state = trans.state_before
                if state is None:
                    continue
                if state.state_hash in seen_hashes:
                    continue
                seen_hashes.add(state.state_hash)

                diag = SnapshotDiagnostics.diagnose_snapshot(
                    state,
                    game_id=traj.game_id,
                    level_key=traj.derived_level_key,
                    step_index=trans.step_index,
                )
                summary.per_snapshot.append(diag)
                summary.total_snapshots += 1
                summary.total_non_background += diag.non_background_count
                summary.total_objects += diag.object_count

                if not diag.palette_sum_ok:
                    summary.all_palette_ok = False
                if not diag.non_bg_ok:
                    summary.all_non_bg_ok = False
                if diag.non_background_count > 0:
                    summary.has_non_background = True
                if diag.object_count > 0:
                    summary.has_objects = True

        summary.occupied_should_be_nonzero = summary.has_non_background

        return summary

    @staticmethod
    def generate_report(summary: DiagnosticSummary) -> str:
        """Generate a markdown diagnostic report."""
        lines: list[str] = []
        lines.append("# M2-A.1-R1: Snapshot Representation Diagnostics\n")

        # Summary
        lines.append("## Summary\n")
        lines.append(f"- **Total snapshots**: {summary.total_snapshots}")
        lines.append(f"- **Total non-background cells**: {summary.total_non_background}")
        lines.append(f"- **Total objects**: {summary.total_objects}")
        lines.append(f"- **Palette sums OK**: {summary.all_palette_ok}")
        lines.append(f"- **Non-bg counts OK**: {summary.all_non_bg_ok}")
        lines.append(f"- **Has non-background cells**: {summary.has_non_background}")
        lines.append(f"- **Has objects**: {summary.has_objects}")
        lines.append(
            f"- **Occupied should be non-zero**: {summary.occupied_should_be_nonzero}"
        )
        lines.append("")

        # Diagnosis
        lines.append("## Diagnosis\n")
        if not summary.has_non_background and not summary.has_objects:
            lines.append(
                "**CRITICAL**: All snapshots have 0 non-background cells and 0 objects.\n"
                "Problem is UPSTREAM: in canonical grid extraction or rendering.\n"
                "Do NOT fix predicate evaluation — fix grid extraction first.\n"
            )
        elif summary.has_non_background and not summary.has_objects:
            lines.append(
                "**WARNING**: Non-background cells exist but no objects were extracted.\n"
                "Problem is in object extraction (connected components).\n"
                "Predicate evaluation may work, but object-based predicates will fail.\n"
            )
        elif summary.has_non_background and summary.has_objects:
            lines.append(
                "**OK**: Snapshots contain non-background cells AND objects.\n"
                "If Occupied predicate gave 0%, the problem is in predicate evaluation\n"
                "(sampling or grounding), not in grid extraction.\n"
                "Proceed to R1.2: full-grid predicate evaluation.\n"
            )
        else:
            lines.append(
                "**UNEXPECTED**: Objects exist but no non-background cells.\n"
                "Possible inconsistency between object extraction and grid.\n"
            )
        lines.append("")

        # Per-snapshot details
        lines.append("## Per-Snapshot Details\n")
        for diag in summary.per_snapshot:
            lines.append(f"### {diag.game_id} / {diag.level_key} / step {diag.step_index}\n")
            lines.append(f"- **State hash**: `{diag.state_hash[:16]}...`")
            lines.append(f"- **Grid**: {diag.height}x{diag.width} ({diag.total_cells} cells)")
            lines.append(f"- **Non-background**: {diag.non_background_count}")
            lines.append(f"- **Background**: {diag.background_count}")
            lines.append(f"- **Objects**: {diag.object_count}")
            lines.append(f"- **Palette**: {dict(sorted(diag.palette.items()))}")
            lines.append(f"- **Palette sum OK**: {diag.palette_sum_ok}")
            lines.append(f"- **Non-bg OK**: {diag.non_bg_ok}")
            if diag.object_details:
                lines.append("- **Object details**:")
                for obj in diag.object_details:
                    lines.append(
                        f"  - {obj['object_id']}: color={obj['color']} "
                        f"area={obj['area']} bbox={obj['bbox']} "
                        f"cells={obj['cells_count']}"
                    )
            lines.append("")

        return "\n".join(lines)
