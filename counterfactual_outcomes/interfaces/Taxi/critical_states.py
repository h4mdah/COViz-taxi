import numpy as np
import gymnasium as gym

class TaxiCriticalStates:
    """
    Identifies and categorizes critical states in the Taxi-v3 environment.

    Categories (in decreasing priority):
      PICKUP_ZONE       – Taxi is at the passenger location (pick-up possible).
      DROPOFF_ZONE      – Passenger is in taxi and taxi is at destination.
      ONE_STEP_AWAY     – Taxi is exactly one Manhattan step from the target and
                          the correct next action is a movement onto the target.
      BOTTLENECK        – Taxi is at one of the two map chokepoints that bridge
                          the left/right segregated areas (cols 1↔2 passages).
      HIGH_UNCERTAINTY  – Top-2 PPO action probabilities are very close (agent
                          is uncertain). Only meaningful for PPO; needs agent.
      ALIGNMENT_TURNING – Taxi shares row or column with the target but a wall
                          or aisle forces a 90-degree turn before it can reach
                          the target directly.
      LANDMARK          – Taxi is at any of the four pickup/drop-off spots.
      NORMAL            – None of the above.
    """

    LOCATIONS = {
        0: (0, 0),  # R
        1: (0, 4),  # G
        2: (4, 0),  # Y
        3: (4, 3),  # B
    }

    # Action indices
    SOUTH, NORTH, EAST, WEST = 0, 1, 2, 3

    # ─────────────────────────────────────────────────────────────────────────
    # Bottleneck chokepoints
    # These chokepoints are single-cell passages that connect segregated areas
    # of the map. Making a wrong move at these cells forces the agent to take
    # a long detour, incurring multiple -1 step penalties — they are therefore
    # the most critical purely navigational states.
    #
    # Bottleneck pairs are defined in `BOTTLENECK_PAIRS` below. Each entry is
    # a (start, end) coordinate pair that connects segregated pockets of the
    # grid; a wrong move at the start cell forces a long detour to the end
    # cell (multiple -1 step penalties).
    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Bottleneck chokepoints (pairs)
    # These are pairs of coordinates (start, end) that represent connected
    # segregated areas of the map. A wrong move at the start cell while
    # navigating toward the end cell forces a long detour (multiple -1 step
    # penalties), so we treat these as critical navigational pairs.
    BOTTLENECK_PAIRS = {
        ((0, 2), (4, 1)),
        ((0, 2), (3, 2)),
        ((1, 2), (3, 1)),
        ((2, 1), (2, 2)),
        ((1, 2), (2, 2)),
    }

    # NOTE: derive positions lazily to keep the pair list authoritative and
    # avoid stale derived sets if callers modify `BOTTLENECK_PAIRS` at runtime.
    @property
    def bottleneck_positions(self):
        """Return the set of all endpoint coordinates referenced by pairs.

        Access via `self.bottleneck_positions` (preferred) or use
        `self.is_bottleneck_start(...)` / `self.is_bottleneck(...)` helpers.
        """
        return {p for pair in self.BOTTLENECK_PAIRS for p in pair}

    def is_bottleneck_start(self, state=None, coords=None):
        """Return True if the given state or (row,col) tuple is an endpoint of
        any chokepoint pair. Pairs are treated as undirected connections: an
        endpoint behaves the same regardless of which side is considered
        'start' or 'end'.

        Provide either `state` (discrete int) or `coords` as a (row,col) pair.
        """
        if state is not None:
            row, col, _, _ = self.decode(state)
        elif coords is not None:
            row, col = coords
        else:
            raise ValueError("Provide either state or coords")

        # Treat pairs as undirected — check membership in the endpoint set.
        return (row, col) in self.bottleneck_positions

    # Backwards-compatible alias: older callers expecting a "start" check can
    # continue to call `is_bottleneck_start`, which now treats endpoints
    # bidirectionally. For clarity, provide an explicit alias name too.
    is_bottleneck_endpoint = is_bottleneck_start

    # Internal walls: pairs ((row, col_left), (row, col_right)) separated by
    # a vertical wall segment.
    WALLS = {
        ((0, 1), (0, 2)),  # row 0 between col 1 and col 2
        ((1, 1), (1, 2)),  # row 1 between col 1 and col 2
        ((3, 0), (3, 1)),  # row 3 between col 0 and col 1
        ((4, 0), (4, 1)),  # row 4 between col 0 and col 1
        ((3, 2), (3, 3)),  # row 3 between col 2 and col 3
        ((4, 2), (4, 3)),  # row 4 between col 2 and col 3
    }

    # ─── Alignment / turning thresholds ──────────────────────────────────────
    # A state is considered an alignment/turning state when the taxi is in the
    # same row OR column as the target but a wall forces a detour.
    HIGH_UNCERTAINTY_THRESHOLD = 0.15  # max prob difference for top-2 actions

    def __init__(self, env=None):
        self._env = env
        self.coords_to_loc = {v: k for k, v in self.LOCATIONS.items()}

    # ─── Environment helpers ──────────────────────────────────────────────────

    def _get_decoder(self):
        """Returns the decode function from the environment."""
        if hasattr(self, '_cached_decoder'):
            return self._cached_decoder

        if self._env is not None:
            candidate = self._env
            for _ in range(10):
                if hasattr(candidate, "decode"):
                    self._cached_decoder = candidate.decode
                    return candidate.decode
                if hasattr(candidate, "unwrapped"):
                    candidate = candidate.unwrapped
                elif hasattr(candidate, "env"):
                    candidate = candidate.env
                else:
                    break

        def decode(state):
            dest_idx = state % 4
            state = state // 4
            pass_idx = state % 5
            state = state // 5
            taxi_col = state % 5
            taxi_row = state // 5
            return (taxi_row, taxi_col, pass_idx, dest_idx)
            
        self._cached_decoder = decode
        return decode

    def decode(self, state):
        """Decodes the discrete state index → (taxi_row, taxi_col, pass_idx, dest_idx)."""
        decoder = self._get_decoder()
        return list(decoder(state))

    # ─── Geometry helpers ─────────────────────────────────────────────────────

    def is_at_any_landmark(self, row, col):
        """True if taxi is at one of the four R/G/Y/B landmarks."""
        return (row, col) in self.LOCATIONS.values()

    def _wall_blocks(self, row, col, action):
        """True if action from (row, col) hits a wall or grid boundary."""
        if action == self.SOUTH:
            return row >= 4
        elif action == self.NORTH:
            return row <= 0
        elif action == self.EAST:
            if col >= 4:
                return True
            return ((row, col), (row, col + 1)) in self.WALLS
        elif action == self.WEST:
            if col <= 0:
                return True
            return ((row, col - 1), (row, col)) in self.WALLS
        return False

    def _movement_result(self, row, col, action):
        """Returns (new_row, new_col) after action, clamped by walls/boundaries."""
        if self._wall_blocks(row, col, action):
            return row, col
        if action == self.SOUTH:
            return row + 1, col
        elif action == self.NORTH:
            return row - 1, col
        elif action == self.EAST:
            return row, col + 1
        elif action == self.WEST:
            return row, col - 1
        return row, col

    # ─── Immediate action checks (state only) ─────────────────────────────────

    def is_pickup_possible(self, state):
        """True if taxi is at passenger's location (pick-up is valid)."""
        row, col, pass_idx, _ = self.decode(state)
        if pass_idx == 4:
            return False
        return (row, col) == self.LOCATIONS[pass_idx]

    def is_dropoff_possible(self, state):
        """True if passenger is in taxi and taxi is at destination."""
        row, col, pass_idx, dest_idx = self.decode(state)
        if pass_idx != 4:
            return False
        return (row, col) == self.LOCATIONS[dest_idx]

    def is_penalty_pickup(self, state):
        """True if PICKUP action would yield −10 penalty."""
        row, col, pass_idx, _ = self.decode(state)
        if pass_idx == 4:
            return True
        return (row, col) != self.LOCATIONS[pass_idx]

    def is_penalty_dropoff(self, state):
        """True if DROPOFF action would yield −10 penalty."""
        row, col, pass_idx, dest_idx = self.decode(state)
        if pass_idx != 4:
            return True
        return (row, col) != self.LOCATIONS[dest_idx]

    # ─── New critical-state predicates ───────────────────────────────────────

    def is_bottleneck(self, state):
        """
        True if taxi is at one of the map's major chokepoints. These single-cell
        passages connect segregated pockets of the grid; a wrong move here
        forces a long detour (multiple -1 step penalties), making the state
        highly critical from a navigational perspective.

        Coordinates used: endpoints of `BOTTLENECK_PAIRS` —
        ((0,2) -> (4,1)), ((0,2) -> (3,2)), ((1,2) -> (3,1)),
        ((2,1) -> (2,2)), ((1,2) -> (2,2)).
        """
        row, col, _, _ = self.decode(state)
        return (row, col) in self.bottleneck_positions

    def is_one_step_away(self, state):
        """
        True if taxi is exactly one Manhattan step from the current target
        (passenger location when not picked up, or destination when carrying)
        AND the next correct movement action would land the taxi directly on
        the target.
        """
        row, col, pass_idx, dest_idx = self.decode(state)

        # Determine the active target
        if pass_idx != 4:
            target = self.LOCATIONS[pass_idx]
        else:
            target = self.LOCATIONS[dest_idx]

        tr, tc = target
        manhattan = abs(row - tr) + abs(col - tc)
        if manhattan != 1:
            return False

        # Verify there is at least one movement that lands on the target
        for action in (self.SOUTH, self.NORTH, self.EAST, self.WEST):
            nr, nc = self._movement_result(row, col, action)
            if (nr, nc) == target:
                return True
        return False

    def is_high_uncertainty(self, state, agent):
        """
        True when the PPO policy's top-2 action probabilities are closer than
        HIGH_UNCERTAINTY_THRESHOLD apart, indicating the agent is uncertain.

        This predicate is only meaningful for PPO agents (values sum ≈ 1).
        For DQN it will still compute, but the threshold has no semantic meaning.
        Caller is responsible for passing the right kind of agent.
        """
        if agent is None:
            return False
        try:
            vals = np.asarray(agent.get_state_action_values(state), dtype=float)
            if vals.size < 2:
                return False
            sorted_desc = np.sort(vals)[::-1]
            return (sorted_desc[0] - sorted_desc[1]) < self.HIGH_UNCERTAINTY_THRESHOLD
        except Exception:
            return False

    def is_alignment_turning(self, state):
        """
        True if taxi shares a row or column with the active target but a wall
        prevents direct straight-line movement, forcing a 90-degree turn.

        This tests whether the agent has learned that line-of-sight does not
        equal reachability in Taxi-v3.
        """
        row, col, pass_idx, dest_idx = self.decode(state)

        if pass_idx != 4:
            target = self.LOCATIONS[pass_idx]
        else:
            target = self.LOCATIONS[dest_idx]

        tr, tc = target

        # Must be aligned on at least one axis
        aligned_row = (row == tr)
        aligned_col = (col == tc)
        if not aligned_row and not aligned_col:
            return False

        # Check whether a direct path along the alignment axis is blocked
        if aligned_row:
            # Same row: verify wall between taxi and target
            if tc > col:
                # Target is to the East – any wall between col..tc?
                for c in range(col, tc):
                    if ((row, c), (row, c + 1)) in self.WALLS:
                        return True
            else:
                # Target is to the West
                for c in range(tc, col):
                    if ((row, c), (row, c + 1)) in self.WALLS:
                        return True

        if aligned_col:
            # Same column: North/South movement is never blocked by internal walls
            # (Taxi-v3 only has vertical wall segments on columns).
            # So if taxi and target share a column, they can always move directly.
            # Still, an adjacent-row alignment is interesting if the taxi is one
            # off in column (i.e., almost aligned).  We keep the col check for
            # completeness but it will generally return False here.
            pass

        return False

    # ─── Aggregate categorisation ─────────────────────────────────────────────

    def get_criticality_category(self, state, agent=None):
        """
        Returns the highest-priority criticality label for the state.

        Priority order (highest first):
          PICKUP_ZONE → DROPOFF_ZONE → ONE_STEP_AWAY → BOTTLENECK →
          HIGH_UNCERTAINTY → ALIGNMENT_TURNING → LANDMARK → NORMAL
        """
        if self.is_pickup_possible(state):
            return "PICKUP_ZONE"
        if self.is_dropoff_possible(state):
            return "DROPOFF_ZONE"
        if self.is_one_step_away(state):
            return "ONE_STEP_AWAY"
        if self.is_bottleneck(state):
            return "BOTTLENECK"
        if agent is not None and self.is_high_uncertainty(state, agent):
            return "HIGH_UNCERTAINTY"
        if self.is_alignment_turning(state):
            return "ALIGNMENT_TURNING"

        row, col, _, _ = self.decode(state)
        if self.is_at_any_landmark(row, col):
            return "LANDMARK"

        return "NORMAL"

    # ─── Importance score (scale-normalised) ─────────────────────────────────

    def get_importance_score(self, state, agent=None):
        """
        Returns a numeric importance score for the state ∈ [0, ∞).
        Scores are designed to be comparable across DQN and PPO agents by
        normalising the agent-derived signal to [0, 1] before adding it.

        Base scores (structural, agent-independent):
          PICKUP_ZONE / DROPOFF_ZONE   →  +5.0
          ONE_STEP_AWAY                →  +3.0
          BOTTLENECK                   →  +2.0
          ALIGNMENT_TURNING            →  +1.5
          LANDMARK (no other category) →  +0.5

        Agent-derived signal (normalised, added on top):
          For DQN  – scaled regret = 1 - ((max_Q − 2nd_max_Q) / (|max_Q| + ε))
          For PPO  – inverted uncertainty gap = 1 - (top1_prob − top2_prob)
        Either way the contribution is a value in [0, 1] scaled by 2.0, so that
        highly uncertain states get up to +2.0 score.
        """
        score = 0.0

        if self.is_pickup_possible(state) or self.is_dropoff_possible(state):
            score += 5.0
        elif self.is_one_step_away(state):
            score += 3.0
        elif self.is_bottleneck(state):
            score += 2.0
        elif self.is_alignment_turning(state):
            score += 1.5
        else:
            row, col, _, _ = self.decode(state)
            if self.is_at_any_landmark(row, col):
                score += 0.5

        if agent is not None:
            try:
                vals = np.asarray(agent.get_state_action_values(state),
                                  dtype=float)
                if vals.size >= 2:
                    sorted_desc = np.sort(vals)[::-1]
                    top1, top2 = sorted_desc[0], sorted_desc[1]
                    gap = top1 - top2

                    # Detect whether these are probabilities (PPO) or Q-values (DQN)
                    is_probs = (
                        np.all(vals >= 0)
                        and np.isclose(np.sum(vals), 1.0, atol=0.05)
                    )

                    if is_probs:
                        # PPO: uncertainty gap = 1 - (top1 - top2). High uncertainty -> small gap -> higher score.
                        normalised = 1.0 - float(np.clip(gap, 0.0, 1.0))
                    else:
                        # DQN: compute a stable scaled regret using both top Q-values
                        # to avoid inflating the measure when top1 is near zero.
                        # denom = |top1| + |top2| + eps ensures bounded behaviour.
                        denom = abs(top1) + abs(top2) + 1e-6
                        regret = gap / denom
                        normalised = 1.0 - float(np.clip(regret, 0.0, 1.0))

                    score += normalised * 2.0  # max contribution: 2.0 for both agents

            except Exception:
                pass

        return score
