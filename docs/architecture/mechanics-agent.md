# Mechanics Agent Architecture

## Scope

The mechanics agent is an evidence-backed learner for ARC-AGI-3 environments. It does not assume a universal end-to-end policy. It observes state transitions, induces game-scoped rules, proposes actions, and uses a deterministic controller as the only action-execution authority.

The initial formal model is a Bayes-adaptive MDP with unknown game mechanics:

```text
s_{t+1} = T_theta(s_t, a_t)
```

`theta` denotes unknown action semantics, controllable objects, collision behavior, push behavior, toggles, and terminal conditions. A POMDP extension is introduced only when observations demonstrably omit state relevant to transitions.

## Architecture

```text
ARC environment
  -> SDK/state adapter
  -> canonical GameState
  -> mechanics router
  -> deterministic or optional model-backed proposal sources
  -> deterministic central controller
  -> existing harness action path
  -> verifier
  -> evidence store, state graph, and JSONL receipt
```

Only the central controller may select an action for execution. Solvers and optional models may emit proposals, but they never call the environment directly. An observed environment transition is the only source of rule confirmation.

## Identity Layers

### Exact state hash

`state_hash` is a stable hash of the full canonical state. It is used for exact replay, receipt identity, state-graph nodes, cache keys, and literal loop detection. A positional change must change `state_hash`.

### Mechanics signature

`mechanics_signature` is a stable canonical representation of mechanics-relevant structure rather than exact placement. It may include object shape, color, area, border contact, relative spatial relations, normalized action availability, and status class. It is used to recognize materially equivalent experiment contexts within one game.

A mechanics signature must not be used to transfer rules between unrelated `game_id` values. On a new game, transferred knowledge starts as speculative.

### Precondition signature

`precondition_signature` is local to a rule/action pair. It records conditions relevant to a predicted effect, such as action, actor type, inferred direction, front-cell occupancy, boundary relation, and adjacent target type. It prevents duplicate evidence from the same equivalent experiment from being counted as independent confirmation.

## Evidence and Rule Status

Facts are observed states and state transitions. Hypotheses are predictions about transitions under explicit preconditions. They remain separate in storage and APIs.

```text
speculative -- first compatible confirmed transition --> observed
observed -- second compatible confirmation under a distinct
            precondition signature and no contradiction --> verified
verified -- compatible confirmation on a subsequent level of
            the same game --> transferable

any active status -- contradiction under equivalent preconditions --> rejected
```

The minimum verification policy is:

```text
verified =
  at least two compatible observed transitions
  in at least two distinct precondition signatures
  with no contradiction under equivalent preconditions
```

A contradiction must be retained. The explorer must not repeat a contradicted `(mechanics_signature, action, precondition_signature)` experiment unless preconditions materially differ.

## Exploration and Control

Exploration and plan execution are separate modes. In early milestones, expected information gain uses a deterministic proxy:

```text
EIG(action) = number of distinct normalized predicted outcomes
              among active hypotheses
```

The controller applies a transparent, mode-dependent robust utility rather than majority voting:

```text
utility =
  w_progress * estimated_progress
+ w_information * information_gain
+ w_confidence * evidence_confidence
- w_risk * risk_game_over
- w_repeat * repeated_experiment_penalty
- w_loop * loop_penalty
- w_disagreement * disagreement_penalty
- w_cost
```

It rejects unavailable actions, malformed action representations, invalid coordinates, known contradicted experiments in equivalent contexts, and high-risk proposals lacking strong evidence. Ties are broken by canonical action serialization. If every proposal is rejected, it selects a conservative available fallback and records why.

## Planning

The planner operates only over verified deterministic transitions. Its initial implementation uses bounded breadth-first search over exact `state_hash` nodes and avoids loops with a visited set. Speculative or observed hypotheses may guide exploration but are never treated as deterministic edges in a plan.

A terminal `WIN` observation is definitive evidence. Before that, goal candidates remain hypotheses.

## Receipts and Reproducibility

Every attempted environment action produces a JSON-serializable `ActionReceipt`. A receipt records the exact prior state hash, canonical available actions, proposals, selected action, selection mode and rationale, predicted delta, observed post-action state hash, actual delta, and verification outcome.

Receipts must not contain raw SDK objects, secrets, API keys, or unbounded histories. JSONL writing is best-effort: logging failure must not terminate the agent run.

## Model Boundary

Deterministic mechanics solvers are the MVP. A future LLM, VLM, or learned transition model is optional and must implement a proposal contract such as:

```text
GameState + canonical action -> predicted StateDelta | proposal
```

Model output is speculative evidence only. It cannot execute actions, mark rules verified, or bypass controller validation and post-transition verification.

## Milestone Mapping

1. Observability foundation: canonical state, exact hashes, mechanics signatures, deltas, receipts, and deterministic tests.
2. Mechanics and verification: scoped memory, hypotheses, router, solvers, verifier, and EIG proxy.
3. Control and exploration: deterministic controller, bounded explorer, proposal validation, and agent integration.
4. Planning: verified transition graph, bounded BFS, candidate goals, and regression tests.
