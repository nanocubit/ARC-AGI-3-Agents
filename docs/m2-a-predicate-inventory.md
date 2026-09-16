# M2-A.1: Predicate Inventory & Offline Analysis

## Purpose

Verify the hypothesis that a significant portion of ARC-AGI-3 dynamics can be
expressed as a small set of deterministic grid/object/relation predicates and
short Boolean guards. The experiment is designed to potentially falsify this
hypothesis.

## Scope

M2-A.1 is **offline only**. It does not:

- Modify the online `MechanicAgent`
- Introduce runtime rules, hypotheses, or online learning
- Build a planner, explorer, or controller
- Use an LLM or transition model
- Change the M1 receipt schema

## Architecture

### Data Flow

```
M1 receipts (JSONL)
      │
      ▼
TrajectoryReader ──→ Trajectory (grouped by game_id + level_key)
      │                   │
      │                   ├── ReceiptRecord (always available)
      │                   └── CanonicalSnapshot (optional, for full mode)
      │
      ▼
InventoryBuilder
      │
      ├── ActionStats (from receipts)
      ├── EffectStats (from StateDelta)
      ├── PredicateStats (only in full mode)
      ├── TruthTable (only in full mode)
      ├── RepresentationStats (only in full mode)
      └── TrajectorySplit (train / held-out)
      │
      ▼
InventoryReport (markdown)
```

### Data Completeness

| Mode | Available | Unavailable |
|------|-----------|-------------|
| `receipt_only` | Action distribution, effect taxonomy, trajectory split | P1-P4 truth tables, predicate frequencies, representation comparison |
| `full` | All of the above + P0-P4 predicates, truth tables, compression proxy | — |

In `receipt_only` mode, the report honestly states that P1-P4 predicates are
unavailable without canonical snapshots. This is not an error — it is a data
sufficiency result.

### Snapshot Join (Fail-Closed)

When canonical snapshots are provided alongside receipts, the join is
fail-closed: if `state_hash(canonical_state) != receipt.state_hash_before`,
a `ValueError` is raised. No heuristic matching is performed.

## Predicate Catalog

### P0: Action
- `ActionEnabled(action_id)` — Action is available in current state (from `available_actions`, NOT selected action)

### P1: Local Grid
- `InBounds(cell)` — Cell is within grid bounds
- `Occupied(cell)` — Cell has non-background color
- `Empty(cell)` — Cell has background color (0)
- `ColorAt(cell, color)` — Cell has specific palette index
- `OnBorder(cell)` — Cell is on grid border

### P2: Directional
- `Front(cell, direction)` — Cell in direction exists (in bounds)
- `FrontIsEmpty(cell, direction)` — Cell in direction is background
- `FrontHasObject(cell, direction)` — Cell in direction is occupied
- `Adjacent(cell1, cell2, direction)` — Cell2 is adjacent to cell1 in direction

### P3: Connected Components
- `Connected4(cell1, cell2)` — Two cells are 4-connected (adjacent, same color)
- `ObjectCells(object)` — Cells belonging to an object
- `Area(object)` — Object area (cell count)
- `BBox(object)` — Object bounding box
- `TouchesBorder(object)` — Object touches grid border

### P4: Object Relations
- `AdjacentObjects(obj1, obj2, direction)` — Two objects are adjacent in direction
- `RelativePosition(obj1, obj2)` — Relative position of obj2 w.r.t. obj1
- `Distance(obj1, obj2)` — Manhattan distance between object centers

### P5/P6 (Deferred)
- Counts, distance thresholds, parity
- Temporal predicates, finite modes, history
- These will be added only if residual evidence shows P0-P4 is insufficient.

## Effect Taxonomy

| Effect Type | Description |
|-------------|-------------|
| `no_change` | Delta is empty |
| `cell_change` | One or more cells changed color |
| `object_move` | One or more objects moved |
| `object_add` | One or more objects appeared |
| `object_remove` | One or more objects disappeared |
| `status_change` | Game status changed (win/lose) |
| `composite` | Multiple effect types combined |

## Metrics

| Metric | What it checks |
|--------|---------------|
| Transition coverage | How many transitions are explained |
| Rule precision | How often a rule is correct (deferred to M2-A.2) |
| Effect type coverage | How many mechanics types are found |
| Contradiction rate | Whether rules conflict with observations (M2-A.2) |
| Predicate count | Size of computational vocabulary |
| Compression proxy | Whether grid-first compresses raw trajectories |
| Held-out accuracy | Generalization across games/levels (M2-A.2) |
| Cross-trajectory stability | Whether different trajectories share predicate basis |

## Trajectory Split

### By Game
Games are sorted by `game_id`. First 60% → train, rest → test.

### By Level (within same game)
First levels → train, later levels → test.

## Representation Comparison

Three representations are compared:
- **A. Raw trajectory**: distinct (state_hash, action_hash) tuples
- **B. Object-based**: distinct (object_set, action) tuples
- **C. Grid-first Fabric**: predicate library + effect types + residuals

Compression = Raw / Grid-first

## Next Steps

### M2-A.2 (only if A.1 provides material)
- Bounded guard synthesis
- Train/held-out validation
- Compression / MDL
- Contradiction detection

### Prerequisites for Full Truth Tables
To evaluate P1-P4 predicates, canonical snapshots are needed. Options:
1. Run agent with snapshot capture (sidecar to receipts)
2. Build a replay engine to reconstruct states from game logs
3. Extend receipt format to include grid (requires M1 schema change — not recommended)

Option 1 is preferred — it does not change M1 and is purely additive.
