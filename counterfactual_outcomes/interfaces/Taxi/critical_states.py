import numpy as np
import gymnasium as gym

class TaxiCriticalStates:
    """
    Identifies and categorizes critical states in the Taxi-v3 environment.
    
    A state is considered critical if it represents a major decision point,
    such as being at a pickup or drop-off location, or if the agent's 
    potential actions lead to significantly different outcomes (high regret).
    """
    
    LOCATIONS = {
        0: (0, 0),  # R
        1: (0, 4),  # G
        2: (4, 0),  # Y
        3: (4, 3)   # B
    }

    # Internal walls in Taxi-v3, encoded as pairs of adjacent cells
    # that are separated by a wall: ((row, col_left), (row, col_right))
    # Derived from the Taxi-v3 map:
    #   +---------+
    #   |R: | : :G|
    #   | : | : : |
    #   | : : : : |
    #   | | : | : |
    #   |Y| : |B: |
    #   +---------+
    WALLS = {
        ((0, 1), (0, 2)),  # wall between col 1 and col 2 at row 0
        ((1, 1), (1, 2)),  # wall between col 1 and col 2 at row 1
        ((3, 0), (3, 1)),  # wall between col 0 and col 1 at row 3
        ((4, 0), (4, 1)),  # wall between col 0 and col 1 at row 4
        ((3, 2), (3, 3)),  # wall between col 2 and col 3 at row 3
        ((4, 2), (4, 3)),  # wall between col 2 and col 3 at row 4
    }

    def __init__(self, env=None):
        self._env = env
        # Cache mapping of location index to coordinates for quick lookup
        self.coords_to_loc = {v: k for k, v in self.LOCATIONS.items()}

    def _get_decoder(self):
        """Returns the decoder function from the environment if available."""
        if self._env is None:
            # Fallback to a fresh environment if none provided
            temp_env = gym.make("Taxi-v3")
            return temp_env.unwrapped.decode
        
        # Try to find decode method in wrappers or unwrapped env
        candidate = self._env
        for _ in range(10):
            if hasattr(candidate, 'decode'):
                return candidate.decode
            if hasattr(candidate, 'unwrapped'):
                candidate = candidate.unwrapped
            elif hasattr(candidate, 'env'):
                candidate = candidate.env
            else:
                break
        
        # Ultimate fallback
        return gym.make("Taxi-v3").unwrapped.decode

    def decode(self, state):
        """Decodes the discrete state into its components."""
        # taxi_row, taxi_col, pass_idx, dest_idx
        decoder = self._get_decoder()
        return list(decoder(state))

    def is_at_any_landmark(self, row, col):
        """Checks if the taxi is at one of the four landmarks R, G, Y, B."""
        return (row, col) in self.LOCATIONS.values()

    def is_wall_hit(self, row, col, action):
        """
        Returns True if the given movement action from (row, col) would
        result in hitting a wall or the grid boundary (wasted move).
        Actions: 0=South, 1=North, 2=East, 3=West
        """
        if action == 0:  # South
            return row >= 4
        elif action == 1:  # North
            return row <= 0
        elif action == 2:  # East
            if col >= 4:
                return True
            return ((row, col), (row, col + 1)) in self.WALLS
        elif action == 3:  # West
            if col <= 0:
                return True
            return ((row, col - 1), (row, col)) in self.WALLS
        return False

    def get_wall_actions(self, state):
        """
        Returns a list of movement actions (0-3) that would hit a wall
        or boundary from this state. These are wasted moves.
        """
        row, col, _, _ = self.decode(state)
        blocked = []
        for action in range(4):  # only movement actions
            if self.is_wall_hit(row, col, action):
                blocked.append(action)
        return blocked

    def is_near_wall(self, state):
        """
        Returns True if at least one movement direction is blocked by a wall
        or boundary. Useful for identifying constrained navigation states.
        """
        return len(self.get_wall_actions(state)) > 0

    def is_pickup_possible(self, state):
        """Returns True if the taxi is at the same location as the passenger."""
        row, col, pass_idx, dest_idx = self.decode(state)
        if pass_idx == 4:  # Passenger already in taxi
            return False
        
        pass_loc = self.LOCATIONS[pass_idx]
        return (row, col) == pass_loc

    def is_dropoff_possible(self, state):
        """Returns True if the passenger is in the taxi and taxi is at the destination."""
        row, col, pass_idx, dest_idx = self.decode(state)
        if pass_idx != 4:  # Passenger not in taxi
            return False
        
        dest_loc = self.LOCATIONS[dest_idx]
        return (row, col) == dest_loc

    def is_penalty_pickup(self, state):
        """True if 'Pick up' action would lead to -10 penalty."""
        row, col, pass_idx, dest_idx = self.decode(state)
        # Penalty if taxi is NOT at passenger location
        if pass_idx == 4: return True # Already in taxi
        pass_loc = self.LOCATIONS[pass_idx]
        return (row, col) != pass_loc

    def is_penalty_dropoff(self, state):
        """True if 'Drop off' action would lead to -10 penalty."""
        row, col, pass_idx, dest_idx = self.decode(state)
        # Penalty if:
        # 1. Passenger is not in taxi
        # 2. Taxi is not at destination
        if pass_idx != 4: return True
        dest_loc = self.LOCATIONS[dest_idx]
        return (row, col) != dest_loc

    def get_criticality_category(self, state):
        """Categorizes the state's criticality for better visualization/logic."""
        if self.is_pickup_possible(state):
            return "PICKUP_ZONE"
        if self.is_dropoff_possible(state):
            return "DROPOFF_ZONE"
        
        row, col, pass_idx, dest_idx = self.decode(state)
        if self.is_at_any_landmark(row, col):
            return "LANDMARK"

        wall_actions = self.get_wall_actions(state)
        if len(wall_actions) >= 2:
            return "WALL_CORNER"
        elif len(wall_actions) == 1:
            return "WALL_ADJACENT"

        return "NORMAL"

    def get_importance_score(self, state, agent=None):
        """
        Calculates a numeric importance score for the state.
        Higher score means more critical.
        """
        score = 0.0
        if self.is_pickup_possible(state) or self.is_dropoff_possible(state):
            score += 5.0
        
        # Wall-adjacent states are trickier to navigate, bump importance slightly
        wall_actions = self.get_wall_actions(state)
        if len(wall_actions) >= 2:
            score += 1.5  # corner or dead-end
        elif len(wall_actions) == 1:
            score += 0.5  # wall on one side

        if agent is not None:
            try:
                # Regret-based importance
                state_vals = agent.get_state_action_values(state)
                if state_vals is not None and len(state_vals) > 0:
                    sorted_vals = np.sort(state_vals)
                    regret = sorted_vals[-1] - sorted_vals[-2]
                    score += regret
            except Exception:
                pass
        return score
