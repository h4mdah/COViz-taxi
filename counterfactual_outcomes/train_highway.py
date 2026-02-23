import sys
from pathlib import Path
import json
import time

# Make repo root importable
REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT), str(Path.cwd())):
    if p not in sys.path:
        sys.path.insert(0, p)

# Register Highway environments
import counterfactual_outcomes.interfaces.Highway.environments  # registers 'Highway-v0-COViz'

try:
    import gymnasium as gym
except ImportError:
    import gym

from stable_baselines3 import DQN, PPO
import numpy as np
from counterfactual_outcomes.common import Trace, State, save_traces, load_traces
import imageio

def safe_reset(env):
    res = env.reset()
    return res[0] if isinstance(res, tuple) else res

def safe_step(env, action):
    res = env.step(int(action))
    if len(res) == 5:
        obs, reward, terminated, truncated, info = res
        return obs, float(reward), (terminated or truncated), info
    obs, reward, done, info = res
    return obs, float(reward), done, info

def render_frame(env):
    try:
        return env.render()
    except Exception:
        return None

def one_hot_action(action, n_actions):
    vec = [0] * n_actions
    try:
        vec[int(action)] = 1
    except Exception:
        pass
    return vec

def eval_and_collect_traces(model, env_id, n_episodes, k_steps=15):
    try:
        env = gym.make(env_id, render_mode='rgb_array')
    except Exception:
        env = gym.make(env_id)
    
    if hasattr(env.unwrapped, 'configure'):
        env.unwrapped.configure({
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 40,
        })
    if hasattr(env.unwrapped, 'define_spaces'):
        env.unwrapped.define_spaces()

    traces = []
    traces_root = REPO_ROOT / 'traces' / env_id.replace('/', '_')
    traces_root.mkdir(parents=True, exist_ok=True)

    for ti in range(n_episodes):
        obs = safe_reset(env)
        done = False
        trace = Trace(idx=ti, k_steps=k_steps)
        trace_dir = traces_root / f"trace_{ti}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        step_idx = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            img = render_frame(env)
            
            img_path = None
            if img is not None:
                img_name = f"frame_{step_idx:04d}.png"
                img_path = trace_dir / img_name
                imageio.imwrite(str(img_path), img.astype('uint8'))
            
            n_actions = getattr(env.action_space, "n", 0)
            action_vector = one_hot_action(int(action), n_actions)
            
            state_obj = State(id=(ti, step_idx), obs=obs, state=obs,
                              action_vector=action_vector, img=None, features=None)
            state_obj.image_path = str(img_path) if img_path else None
            
            next_obs, reward, done, info = safe_step(env, action)
            trace.update(obs=obs, r=reward, done=done, infos=info, a=int(action), state_id=state_obj)
            
            obs = next_obs
            step_idx += 1
            if step_idx > 100: # safety break for highway
                 break
        
        traces.append(trace)
    
    env.close()
    return traces

def main(
    env_id="Highway-v0-COViz",
    total_timesteps=2000,
    algo="DQN",
    model_dir="agents/Highway/trained_model",
    traces_dir="traces/highway",
    eval_episodes=5,
    eval_interval=10_000
):
    repo = Path(REPO_ROOT)
    model_path_dir = repo / model_dir
    model_path_dir.mkdir(parents=True, exist_ok=True)
    traces_path = repo / traces_dir
    traces_path.mkdir(parents=True, exist_ok=True)

    print(f"Starting training on {env_id} via {algo}...")
    
    train_env = gym.make(env_id)
    # Highway environments often need explicit configuration
    if hasattr(train_env.unwrapped, 'configure'):
        train_env.unwrapped.configure({
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 40,
            "lanes_count": 4,
            "vehicles_count": 50,
        })
    if hasattr(train_env.unwrapped, 'define_spaces'):
        train_env.unwrapped.define_spaces()
    
    if algo == "PPO":
        model = PPO("MlpPolicy", train_env, verbose=1)
    else:
        model = DQN("MlpPolicy", train_env, verbose=1)

    accumulated_timesteps = 0
    while accumulated_timesteps < total_timesteps:
        to_learn = min(eval_interval, total_timesteps - accumulated_timesteps)
        model.learn(total_timesteps=to_learn, reset_num_timesteps=False)
        accumulated_timesteps += to_learn
        
        print(f"Progress: {accumulated_timesteps}/{total_timesteps}")
        
        # Collection
        new_traces = eval_and_collect_traces(model, env_id, eval_episodes)
        save_traces(new_traces, str(traces_path), name=f"traces_{accumulated_timesteps}.pkl")
        
    final_model = model_path_dir / "model_final.zip"
    model.save(str(final_model))
    print(f"Training complete. Model saved to {final_model}")
    train_env.close()

if __name__ == "__main__":
    main()
