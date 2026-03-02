import time
import numpy as np
import gymnasium as gym

from counterfactual_outcomes.interfaces.Taxi.critical_states import TaxiCriticalStates

class DummyAgent:
    def __init__(self, is_ppo=True):
        self.is_ppo = is_ppo

    def get_state_action_values(self, state):
        if self.is_ppo:
            # high uncertainty state
            return [0.4, 0.4, 0.1, 0.1]
        else:
            return [10.0, 10.0, 0.0, 0.0]

def main():
    start_time = time.time()
    
    env = gym.make("Taxi-v3")
    cs = TaxiCriticalStates() # no env passed!
    
    # Test performance of calling decode multiple times
    try:
        for i in range(500):
            cs.get_criticality_category(i)
    except Exception as e:
        print(f"Error evaluating 500 states: {e}")
        
    print(f"Time taken for 500 evaluates without env: {time.time() - start_time:.4f}s")
    
    # Check important state logic
    agent_ppo = DummyAgent(is_ppo=True)
    state = 0
    score = cs.get_importance_score(state, agent_ppo)
    cat = cs.get_criticality_category(state, agent_ppo)
    print(f"PPO Agent (high uncertainty) - Category: {cat}, Score: {score}")

    agent_ppo_certain = DummyAgent(is_ppo=True)
    agent_ppo_certain.get_state_action_values = lambda s: [0.97, 0.01, 0.01, 0.01]
    score_certain = cs.get_importance_score(state, agent_ppo_certain)
    cat_certain = cs.get_criticality_category(state, agent_ppo_certain)
    print(f"PPO Agent (high certainty) - Category: {cat_certain}, Score: {score_certain}")

if __name__ == '__main__':
    main()
