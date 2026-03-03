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
Returns `True` if the taxi is at one of the doorway tiles in the middle horizontal corridor (row 2). These tiles are forced passage points — the taxi must cross through them to move between segregated areas of the grid. A wrong move here forces a long detour (multiple `-1` step penalties), making these states highly critical.

Bottleneck positions:
- `(2, 0)` — left end of middle corridor
- `(2, 1)` — passage between top wall and bottom-left wall
- `(2, 2)` — passage between top wall and bottom-right wall
- `(2, 3)` — right end of middle corridor

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

#### `get_criticality_category(state, agent=None) → str`
The main categorization method. Returns a label depending on what's going on in the state:

| Category | When it applies |
|---|---|
| `"PICKUP_ZONE"` | Taxi is at the passenger's location and the passenger is waiting |
| `"DROPOFF_ZONE"` | Taxi is at the destination with the passenger on board |
| `"ONE_STEP_AWAY"` | Taxi is exactly one step from target and next valid move lands on target |
| `"BOTTLENECK"` | Taxi is at one of the doorway tiles in row 2: (2,0), (2,1), (2,2), (2,3) |
| `"HIGH_UNCERTAINTY"`| PPO agent action probabilities are very close (requires passing `agent`) |
| `"ALIGNMENT_TURNING"`| Taxi shares row/col with target but wall forces a 90-degree detour |
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
1. **+5.0** if the agent can do a successful pickup or dropoff here.
2. **+3.0** if the taxi is one step away from the target location.
3. **+2.0** if the taxi is at a bottleneck chokepoint.
4. **+1.5** if the taxi is aligned with the target but a wall forces a turning detour.
5. **+0.5** if the taxi is at a landmark (and none of the above are true).
6. **+uncertainty** (up to +2.0) if an agent is provided — this rewards states where the agent is highly uncertain (a small gap between best and second-best action value for PPO, or inverted scaled regret for DQN). A higher uncertainty score means the state is a critical decision boundary.

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
