import os
import sys
import pickle
from pathlib import Path
from os.path import join, abspath

# make repo root importable so unpickling can find 'counterfactual_outcomes'
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

traces_dir = "traces"
p = join(traces_dir, "Traces.pkl")
if not os.path.exists(p):
    print("No Traces.pkl found at", p)
    raise SystemExit(1)

with open(p, "rb") as f:
    try:
        traces = pickle.load(f)
    except ModuleNotFoundError as e:
        print("Unpickle failed because a module referenced by the pickle wasn't importable:")
        print(e)
        print("Ensure the repository root is on sys.path or run this script from the repo root.")
        raise

changed = False
for t in traces:
    if not hasattr(t, "contrastive"):
        t.contrastive = []
        changed = True

if changed:
    backup = join(traces_dir, "Traces.pkl.bak")
    os.rename(p, backup)
    with open(p, "wb") as f:
        pickle.dump(traces, f)
    print("Updated Traces.pkl (backup at)", backup)
else:
    print("Nothing to change.")