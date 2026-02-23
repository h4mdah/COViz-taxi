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
            
        return "NORMAL"

    def get_importance_score(self, state, agent=None):
        """
        Calculates a numeric importance score for the state.
        Higher score means more critical.
        """
        score = 0.0
        if self.is_pickup_possible(state) or self.is_dropoff_possible(state):
            score += 5.0
        
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
