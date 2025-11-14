import sys
from pathlib import Path
import json
import time

# Make repo root importable so package imports (and env registration) work
REPO_ROOT = Path(__file__).resolve().parents[1]  # COViz-1
# ensure repo root and current working dir are on sys.path (avoid ModuleNotFoundError)
for p in (str(REPO_ROOT), str(Path.cwd())):
    if p not in sys.path:
        sys.path.insert(0, p)

# ensure the custom Taxi env is registered
import counterfactual_outcomes.interfaces.Taxi.environments  # registers 'Taxi-v3-COViz'

# gym fallback between gymnasium and gym
try:
    import gymnasium as gym
except Exception:
    import gym

from stable_baselines3 import DQN
import numpy as np

def safe_reset(env):
    r = env.reset()
    return r[0] if isinstance(r, tuple) else r

def safe_step(env, action):
    r = env.step(int(action))
    if len(r) == 5:
        obs, reward, terminated, truncated, info = r
        done = terminated or truncated
        return obs, float(reward), done, info
    obs, reward, done, info = r
    return obs, float(reward), done, info

def eval_and_collect_traces(model, env_id, n_episodes):
    env = gym.make(env_id)
    traces = []
    for _ in range(n_episodes):
        obs = safe_reset(env)
        done = False
        trace = {"observations": [], "actions": [], "rewards": [], "dones": []}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, info = safe_step(env, action)
            # Taxi observations are typically integers; ensure JSON-serializable
            trace["observations"].append(int(obs) if np.isscalar(obs) else (np.array(obs).tolist()))
            trace["actions"].append(int(action))
            trace["rewards"].append(float(reward))
            trace["dones"].append(bool(done))
            obs = next_obs
        traces.append(trace)
    env.close()
    return traces

def main(
    env_id="Taxi-v3-COViz",
    total_timesteps=100_000,
    learn_chunk=10_000,
    eval_episodes=5,
    eval_interval=10_000,
    save_model_interval=20_000,
    out_traces_file="traces/taxi_traces.json",
    model_dir="agents/taxi_sb3"
):
    repo = Path(REPO_ROOT)
    out_traces_path = repo / out_traces_file
    (repo / out_traces_path.parent).mkdir(parents=True, exist_ok=True)
    model_path_dir = repo / model_dir
    model_path_dir.mkdir(parents=True, exist_ok=True)

    # create training env
    train_env = gym.make(env_id)

    model = DQN("MlpPolicy", train_env, verbose=1)
    accumulated_timesteps = 0
    all_traces = []
    # if existing traces file present, load to append
    if out_traces_path.exists():
        try:
            with open(out_traces_path, "r") as f:
                all_traces = json.load(f)
        except Exception:
            all_traces = []

    while accumulated_timesteps < total_timesteps:
        to_learn = min(learn_chunk, total_timesteps - accumulated_timesteps)
        model.learn(total_timesteps=to_learn, reset_num_timesteps=False)
        accumulated_timesteps += to_learn
        print(f"[{time.strftime('%H:%M:%S')}] Learned {accumulated_timesteps}/{total_timesteps} timesteps")

        # periodic evaluation & trace collection
        if accumulated_timesteps % eval_interval == 0 or accumulated_timesteps == total_timesteps:
            traces = eval_and_collect_traces(model, env_id, eval_episodes)
            all_traces.extend(traces)
            with open(out_traces_path, "w") as f:
                json.dump(all_traces, f)
            print(f"[{time.strftime('%H:%M:%S')}] Collected {len(traces)} eval traces -> {out_traces_path}")

        # periodic model save
        if accumulated_timesteps % save_model_interval == 0 or accumulated_timesteps == total_timesteps:
            model_file = model_path_dir / f"model_{accumulated_timesteps}.zip"
            model.save(str(model_file))
            print(f"[{time.strftime('%H:%M:%S')}] Saved model -> {model_file}")

    train_env.close()
    # final save
    final_model = model_path_dir / "model_final.zip"
    model.save(str(final_model))
    print(f"Training finished. Final model saved to {final_model}")
    print(f"Total traces saved: {len(all_traces)} at {out_traces_path}")

if __name__ == "__main__":
    main()