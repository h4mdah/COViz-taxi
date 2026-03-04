# Critical States Module — `critical_states.py`

## Overview

This module defines what counts as a "critical state" in the Taxi-v3 environment. The idea is that not every state is equally important — some states represent key decision points where the agent's choice really matters (like being at a pickup or dropoff location). 

The main class is `TaxiCriticalStates`. Given a state index, it can tell you whether it's a pickup opportunity, a dropoff opportunity, whether an action would cause a penalty, and whether the taxi is next to a wall. It also computes an importance score that combines environment-level criticality with the agent's own uncertainty (regret).


## Class: `TaxiCriticalStates`

### Constructor

| `env` | `gymnasium.Env` or `None` | An existing environment instance can be passed in. If not provided, a fallback fast math decoding function gets used internally instead of creating temporary environments. |

### Landmark Mapping

The four landmark positions are hardcoded since they're fixed in Taxi-v3:

```python
LOCATIONS = {
    0: (0, 0),  # R (Red)
    1: (0, 4),  # G (Green)
    2: (4, 0),  # Y (Yellow)
    3: (4, 3)   # B (Blue)
}
```

### Wall Mapping

The internal walls from the Taxi-v3 grid are also encoded. The map looks like this:

```
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

The `|` characters are walls that block East/West movement. They're stored as pairs of adjacent cells that are separated by a wall:

```python
WALLS = {
    ((0, 1), (0, 2)),  # wall between col 1 and col 2 at row 0
    ((1, 1), (1, 2)),  # wall between col 1 and col 2 at row 1
    ((3, 0), (3, 1)),  # wall between col 0 and col 1 at row 3
    ((4, 0), (4, 1)),  # wall between col 0 and col 1 at row 4
    ((3, 2), (3, 3)),  # wall between col 2 and col 3 at row 3
    ((4, 2), (4, 3)),  # wall between col 2 and col 3 at row 4
}
```

On top of these internal walls, the grid boundaries also block movement (can't go past row 0/4 or col 0/4).

---

### Methods

#### `decode(state) → list`
Takes a discrete state integer (0–499) and breaks it down into its four components:

| Component | Meaning |
|---|---|
| `taxi_row` | Row position of the taxi (0–4) |
| `taxi_col` | Column position of the taxi (0–4) |
| `pass_idx` | Where the passenger is (0–3 = at a landmark, 4 = in the taxi) |
| `dest_idx` | Destination landmark index (0–3) |

```python
cs = TaxiCriticalStates()
cs.decode(123)  # → [2, 2, 3, 0]  (taxi at row 2, col 2, passenger at B, destination R)
```

---

#### `is_at_any_landmark(row, col) → bool`
Simple check — returns `True` if the taxi is sitting on one of the four landmarks (R, G, Y, B).

#### `is_pickup_possible(state) → bool`
Returns `True` when the taxi is at the **same spot as the passenger** and the passenger hasn't been picked up yet (`pass_idx ≠ 4`). Basically, this is when doing `action=4 (Pickup)` would actually work.

---

#### `is_dropoff_possible(state) → bool`
Returns `True` when the passenger **is in the taxi** (`pass_idx == 4`) and the taxi is at the **destination**. This is the moment where `action=5 (Dropoff)` would succeed and give the `+20` reward.

---

#### `is_penalty_pickup(state) → bool`
Returns `True` if doing a Pickup here would cause a **−10 penalty**. That happens when:
- The passenger is already in the taxi, or
- The taxi isn't at the passenger's location.

---

#### `is_penalty_dropoff(state) → bool`
Returns `True` if doing a Dropoff here would cause a **−10 penalty**. That happens when:
- The passenger isn't in the taxi yet, or
- The taxi isn't at the destination.

---

#### `is_bottleneck(state) → bool`
Returns `True` if the taxi is standing on one of the key transit tiles in the row-2 corridor — the only open horizontal passage through the internal wall segments.

Row 2 has no internal walls, making it the only route between the areas separated by the three wall segments:
- **Top wall** (rows 0–1, between col 1 and col 2): must pass through `(2,1)` or `(2,2)`
- **Bottom-left wall** (rows 3–4, between col 0 and col 1): must pass through `(2,0)` or `(2,1)`
- **Bottom-right wall** (rows 3–4, between col 2 and col 3): must pass through `(2,2)` or `(2,3)`

The union of these transit tiles is the entire row-2 corridor: `{(2,0), (2,1), (2,2), (2,3)}`.

---

#### `is_one_step_away(state) → bool`
Returns `True` if the taxi is exactly one Manhattan step from the active target, and the correct structural move lands the taxi on the target.

---

#### `is_high_uncertainty(state, agent) → bool`
Returns `True` when a given PPO agent policy has a difference of `< 0.15` between its top-1 and top-2 predicted action probabilities, indicating hesitation.

---

#### `is_alignment_turning(state) → bool`
Returns `True` if the taxi shares a row/col with the target, but a wall interrupts the direct line of sight, forcing a 90-degree detour.

---

#### `is_divergence_point(state) → bool`
Detects states where choosing the wrong movement direction creates a large
detour to the active target. The implementation simulates each legal movement
from the current taxi cell, computes the shortest-path length from the
resulting cell to the target (via `_shortest_path_length` which respects walls
and boundaries), and compares distances. If any legal action increases the
distance by at least `DIVERGENCE_DETOUR_THRESHOLD` (default `2`) relative to
the best available move, the state is flagged as a divergence point.

If the BFS cannot reach the target from a simulated successor cell the
shortest-path helper returns the sentinel `10**6`, which will count as a very
large detour when evaluating divergence.

---

#### `get_criticality_category(state, agent=None) → str`
The main categorization method. Returns a label depending on what's going on in the state:

| Category | When it applies |
|---|---|
| `"PICKUP_ZONE"` | Taxi is at the passenger's location and the passenger is waiting |
| `"DROPOFF_ZONE"` | Taxi is at the destination with the passenger on board |
| `"HIGH_UNCERTAINTY"`| PPO agent action probabilities are very close (requires passing `agent`) |
| `"ONE_STEP_AWAY"` | Taxi is exactly one step from target and next valid move lands on target |
| `"ALIGNMENT_TURNING"`| Taxi shares row/col with target but wall forces a 90-degree detour |
| `"DIVERGENCE_POINT"` | State where a wrong movement choice causes a large simulated detour to the target |
| `"CHOKEPOINT"` | Taxi is on one of the key transit tiles in the row-2 corridor: `(2,0)`, `(2,1)`, `(2,2)`, or `(2,3)` |
| `"LANDMARK"` | Taxi is at one of the landmarks but none of the above apply |
| `"NORMAL"` | Everything else |

```python
cs = TaxiCriticalStates()
cs.get_criticality_category(123)  # → "NORMAL"
```

---

#### `get_importance_score(state, agent=None) → float`
Computes a numeric score for how important a state is. Higher means more critical — this is what gets used for ranking which states to highlight in the visualization.

**How the score works:**
1. **Base structural contribution (prioritised):** exactly one structural bucket contributes using strict precedence — the first matching category yields the base score and lower-priority structural bonuses are ignored.
    - `PICKUP_ZONE` / `DROPOFF_ZONE`: **+5.0**
    - `ONE_STEP_AWAY`: **+3.0**
    - `BOTTLENECK`: **+2.0**
    - `ALIGNMENT_TURNING`: **+1.5**
    - `LANDMARK` (when none of the above): **+0.5**
2. **Agent-derived signal (added on top):** an uncertainty/regret measure in `[0,1]` is computed from the agent's `get_state_action_values(state)` output and scaled by `2.0`, so agent contributions are in `[0,2.0]`.

    - For PPO-like probability outputs (non-negative and summing ≈1): `normalised = 1 - (top1 - top2)` (small gap → high uncertainty → larger contribution).
    - For Q-value outputs (DQN): compute a stable scaled regret `regret = (top1 - top2) / (|top1| + |top2| + eps)` and use `normalised = 1 - clip(regret, 0, 1)`.

Total score = (single highest-priority structural base) + (agent normalised × 2.0).

| Parameter | Type | Description |
|---|---|---|
| `state` | `int` | Discrete state index (0–499) |
| `agent` | object or `None` | Any agent that has a `get_state_action_values(state)` method |

```python
cs = TaxiCriticalStates()
score = cs.get_importance_score(state=328, agent=my_agent)
# score = 5.0 + 1.2  (pickup possible + high uncertainty)
```

---
