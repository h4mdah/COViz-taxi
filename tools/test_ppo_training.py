
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import gymnasium as gym
import numpy as np
import torch
from tools.custom_ppo import PPO, Memory

def test_ppo_training():
    env_id = "Taxi-v3"
    env = gym.make(env_id)
    
    state_dim = 500
    action_dim = env.action_space.n
    
    ppo_conf = {
        'lr': 0.002,
        'betas': (0.9, 0.999),
        'gamma': 0.99,
        'eps_clip': 0.2,
        'K_epochs': 4,
        'nn_type': 'tanh',
        'action_std': 0.6,
        'lam_a': 0.0,
        'normalize_rewards': False
    }
    
    print("Initializing PPO...")
    try:
        ppo_agent = PPO(state_dim, action_dim, ppo_conf, use_gpu=False, is_continuous=False)
        print("PPO Initialized.")
    except Exception as e:
        print(f"Failed to initialize PPO: {e}")
        return

    memory = Memory()
    max_ep_len = 20
    update_timestep = 50
    time_step = 0
    total_timesteps = 100
    
    print("Starting Training Loop...")
    try:
        for i_episode in range(1, 5):
            state, _ = env.reset()
            for t in range(max_ep_len):
                time_step += 1
                
                state_vec = np.zeros(500)
                state_vec[state] = 1.0
                
                action = ppo_agent.select_action(state_vec, memory)
                state, reward, done, truncated, _ = env.step(action)
                
                memory.rewards.append(reward)
                memory.is_terminals.append(done or truncated)
                
                if time_step % update_timestep == 0:
                    print(f"Updating PPO at timestep {time_step}")
                    ppo_agent.update(memory, use_gpu=False)
                    memory.clear_memory()
                    time_step = 0
                
                if done or truncated:
                    break
        print("Training Loop Completed Successfully.")
    except Exception as e:
        print(f"Training Loop Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ppo_training()
