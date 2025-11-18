import sys, pickle
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

p = "C:\\Users\\hmari\\Desktop\\THI_KI\\S5\\COViz-taxi\\results\\run_2025-11-18_10-34-09_13008\\traces.pkl"
with open(p, "rb") as f:
    traces = pickle.load(f)

for ti, t in enumerate(traces[:5]):
    print(f"Trace[{ti}] trace_idx={getattr(t,'trace_idx',None)} contrastive_count={len(getattr(t,'contrastive',[]))}")
    for ci, c in enumerate(getattr(t,'contrastive',[])[:5]):
        print(f"  Contrast[{ci}] importance={getattr(c,'importance',None)} traj_end={getattr(c,'traj_end_state',None)} n_states={len(getattr(c,'states',[]))}")