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
#import counterfactual_outcomes.interfaces.Highway.environments
# gym fallback between gymnasium and gym
try:
    import gymnasium as gym
except Exception:
    import gym

from stable_baselines3 import DQN
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

def create_obs_image(obs, size=(84, 84)):
    """Fallback: render a small image with the obs text if env.render returns None."""
    try:
        if Image is None:
            return None
        img = Image.new('RGB', size, color='white')
        draw = ImageDraw.Draw(img)
        txt = str(obs)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((4, 4), txt, fill='black', font=font)
        return np.array(img)
    except Exception:
        return None

# import Trace/State helpers to produce Traces.pkl compatible with the rest of the library
from counterfactual_outcomes.common import Trace, State, save_traces, load_traces

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

def render_frame(env):
    # try common render signatures and return an RGB array (or None)
    try:
        # gymnasium/gym may accept no args if env created with render_mode='rgb_array'
        frm = env.render()
        if frm is None:
            # fallback to explicit mode argument
            frm = env.render(mode='rgb_array')
        return frm
    except Exception:
        try:
            return env.render(mode='rgb_array')
        except Exception:
            return None

def one_hot_action(action, n_actions):
    vec = [0] * n_actions
    try:
        vec[int(action)] = 1
    except Exception:
        pass
    return vec

def eval_and_collect_traces(model, env_id, n_episodes, k_steps=10):
    # try to request RGB rendering at creation if supported
    try:
        env = gym.make(env_id, render_mode='rgb_array')
    except Exception:
        env = gym.make(env_id)
    traces = []
    for ti in range(n_episodes):
        obs = safe_reset(env)
        done = False
        trace = Trace(idx=ti, k_steps=k_steps)
        step_idx = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # capture frame before taking the action (state at current timestep)
            img = render_frame(env)
            if img is None:
                # fallback image built from the observation so downstream code always has an image
                img = create_obs_image(obs)
            action_int = int(action)
            # create State for current timestep
            n_actions = getattr(env.action_space, "n", None) or 0
            action_vector = one_hot_action(action_int, n_actions) if n_actions else []
            state_obj = State(id=(ti, step_idx), obs=obs, state=obs,
                              action_vector=action_vector, img=img, features=None)
            # perform step
            next_obs, reward, done, info = safe_step(env, action)
            # update Trace with same signature as common.Trace.update(obs, r, done, infos, a, state_id)
            trace.update(obs=obs, r=reward, done=done, infos=info, a=action_int, state_id=state_obj)
            obs = next_obs
            step_idx += 1
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
    out_traces_file="traces/taxi/taxi_traces.json",
    model_dir="agents/taxi_sb3",
    k_steps=10
):
    repo = Path(REPO_ROOT)
    out_traces_path = repo / out_traces_file
    traces_dir = repo / out_traces_path.parent
    traces_dir.mkdir(parents=True, exist_ok=True)
    model_path_dir = repo / model_dir
    model_path_dir.mkdir(parents=True, exist_ok=True)

    # if an existing Traces.pkl is present, load to append
    existing_traces = []
    try:
        existing_traces = load_traces(str(traces_dir))
    except Exception:
        existing_traces = []

    # create training env
    train_env = gym.make(env_id)

    model = DQN("MlpPolicy", train_env, verbose=1)
    accumulated_timesteps = 0
    # if existing model artifacts in model_dir you'd like to load, add logic here

    while accumulated_timesteps < total_timesteps:
        to_learn = min(learn_chunk, total_timesteps - accumulated_timesteps)
        model.learn(total_timesteps=to_learn, reset_num_timesteps=False)
        accumulated_timesteps += to_learn
        print(f"[{time.strftime('%H:%M:%S')}] Learned {accumulated_timesteps}/{total_timesteps} timesteps")

        # periodic evaluation & trace collection (produces Trace objects compatible with common.Trace)
        if accumulated_timesteps % eval_interval == 0 or accumulated_timesteps == total_timesteps:
            new_traces = eval_and_collect_traces(model, env_id, eval_episodes, k_steps=k_steps)
            # if loaded existing traces were plain lists/other, attempt to concatenate
            combined_traces = []
            if existing_traces:
                combined_traces.extend(existing_traces)
            combined_traces.extend(new_traces)
            # save as Traces.pkl so the contrastive pipeline can load them
            save_traces(combined_traces, str(traces_dir))
            # also write a small JSON summary for human inspection (optional)
            try:
                summary = []
                for t in new_traces:
                    summary.append({"trace_idx": t.trace_idx, "length": t.length, "reward_sum": t.reward_sum})
                with open(traces_dir / "taxi_traces_summary.json", "w") as f:
                    json.dump(summary, f)
            except Exception:
                pass

            existing_traces = combined_traces
            print(f"[{time.strftime('%H:%M:%S')}] Collected {len(new_traces)} eval traces -> {traces_dir / 'Traces.pkl'}")

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
    print(f"Total traces saved: {len(existing_traces)} at {traces_dir / 'Traces.pkl'}")

if __name__ == "__main__":
    main()
    #main(env_id='Plain-v0', total_timesteps=15000, out_traces_file='traces/highway/highway_traces.pkl', model_dir='agents/highway_sb3')