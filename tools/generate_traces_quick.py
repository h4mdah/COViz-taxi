import sys, os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from counterfactual_outcomes.train_model import eval_and_collect_traces
from counterfactual_outcomes.common import save_traces
import gym

# try to load a saved SB3 model
model = None
model_dir = REPO_ROOT / "agents" / "taxi_sb3"
if model_dir.exists():
    for z in model_dir.glob("*.zip"):
        try:
            from stable_baselines3 import DQN
            model = DQN.load(str(z))
            break
        except Exception:
            model = None

# fallback random policy wrapper
if model is None:
    class RandModel:
        def __init__(self, action_space):
            self.action_space = action_space
        def predict(self, obs, deterministic=True):
            return (self.action_space.sample(), None)
    try:
        env = gym.make("Taxi-v3", render_mode="rgb_array")
    except Exception:
        env = gym.make("Taxi-v3")
    model = RandModel(env.action_space)
    env.close()

# collect traces (adjust n_episodes if you want fewer)
traces = eval_and_collect_traces(model, env_id="Taxi-v3-COViz", n_episodes=20, k_steps=5)
save_traces(traces, str(REPO_ROOT / "traces"))
print("Wrote traces/Traces.pkl with", len(traces), "traces")