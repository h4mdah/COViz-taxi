"""Regenerate traces using an existing SB3 model and save as Traces.pkl

This script loads the model at `agents/taxi_sb3/model_final.zip`, runs
`eval_and_collect_traces` from `counterfactual_outcomes.train_model` and
writes the Traces.pkl to `traces/taxi/`.

Adjust `MODEL_PATH`, `ENV_ID`, and `N_EPISODES` as needed.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from counterfactual_outcomes.train_model import eval_and_collect_traces
from counterfactual_outcomes.common import save_traces
from stable_baselines3 import DQN

MODEL_PATH = Path(REPO_ROOT) / "agents" / "taxi_sb3" / "model_final.zip"
ENV_ID = "Taxi-v3-COViz"
N_EPISODES = 10
K_STEPS = 15
OUT_DIR = Path(REPO_ROOT) / "traces" / "taxi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading model from: {MODEL_PATH}")
model = DQN.load(str(MODEL_PATH))

print(f"Collecting {N_EPISODES} evaluation traces from {ENV_ID}...")
traces = eval_and_collect_traces(model, ENV_ID, N_EPISODES, k_steps=K_STEPS)

print(f"Saving {len(traces)} traces to {OUT_DIR}")
save_traces(traces, str(OUT_DIR))
print("Done.")
