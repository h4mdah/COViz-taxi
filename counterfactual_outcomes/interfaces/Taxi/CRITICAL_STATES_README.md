# Critical States Module — `critical_states.py`

## Overview

This module defines what counts as a "critical state" in the Taxi-v3 environment. The idea is that not every state is equally important — some states represent key decision points where the agent's choice really matters (like being at a pickup or dropoff location). 

The main class is `TaxiCriticalStates`. Given a state index, it can tell you whether it's a pickup opportunity, a dropoff opportunity, whether an action would cause a penalty, and whether the taxi is next to a wall. It also computes an importance score that combines environment-level criticality with the agent's own uncertainty (regret).


## Class: `TaxiCriticalStates`

### Constructor

```python
TaxiCriticalStates(env=None)
```

| Parameter | Type | Description |
|---|---|---|
| `env` | `gymnasium.Env` or `None` | An existing environment instance can be passed in. If not provided, a temporary Taxi-v3 env gets created internally to access the `decode` function. |

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

---

#### `is_wall_hit(row, col, action) → bool`
Checks if a specific movement action from `(row, col)` would result in hitting a wall or the grid boundary. When this returns `True`, the taxi wouldn't actually move but would still get the `-1` step penalty — basically a wasted move.

Actions are: `0=South, 1=North, 2=East, 3=West`.

```python
cs = TaxiCriticalStates()
cs.is_wall_hit(0, 0, 1)  # True — can't go North from row 0
cs.is_wall_hit(0, 1, 2)  # True — wall between col 1 and col 2 at row 0
cs.is_wall_hit(2, 2, 2)  # False — row 2 has no internal walls
```

---

#### `get_wall_actions(state) → list`
Returns which movement actions (0–3) would hit a wall from this state. Useful for quickly seeing how constrained the taxi's movement is.

```python
cs = TaxiCriticalStates()
cs.get_wall_actions(0)  # → [1, 3]  (can't go North or West from corner R)
```

---

#### `is_near_wall(state) → bool`
Returns `True` if at least one movement direction is blocked. Helpful for flagging states where the agent has to be more careful about which direction it picks.

---

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

#### `get_criticality_category(state) → str`
The main categorization method. Returns a label depending on what's going on in the state:

| Category | When it applies |
|---|---|
| `"PICKUP_ZONE"` | Taxi is at the passenger's location and the passenger is waiting |
| `"DROPOFF_ZONE"` | Taxi is at the destination with the passenger on board |
| `"LANDMARK"` | Taxi is at one of the landmarks but can't do a pickup or dropoff |
| `"WALL_CORNER"` | Taxi has 2+ directions blocked by walls/boundaries (corner or dead-end) |
| `"WALL_ADJACENT"` | Taxi has exactly 1 direction blocked by a wall/boundary |
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
2. **+1.5** if the taxi is in a corner or dead-end (2+ blocked directions) — these are spots where a wrong move wastes a step.
3. **+0.5** if the taxi has one direction blocked by a wall.
4. **+regret** if an agent is provided — this is the gap between the best and second-best action value. A big gap means the agent is very "sure" about what to do, which usually means the state matters a lot.

| Parameter | Type | Description |
|---|---|---|
| `state` | `int` | Discrete state index (0–499) |
| `agent` | object or `None` | Any agent that has a `get_state_action_values(state)` method — works with both DQN (Q-values) and PPO (action probabilities) |

```python
cs = TaxiCriticalStates(env)
score = cs.get_importance_score(state=328, agent=my_agent)
# score = 5.0 + 3.2  (pickup possible + high regret)
```

---
