import pickle, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

p = REPO / "traces" / "Traces.pkl"
with open(p, "rb") as f:
    traces = pickle.load(f)
t = traces[0]
print("Trace states:", len(getattr(t, "states", [])))
s0 = t.states[0]
print("attrs on state[0]:", [a for a in dir(s0) if not a.startswith("_")])
print("img:", getattr(s0, "img", None) is not None, "image:", getattr(s0, "image", None) is not None)
print("type(img):", type(getattr(s0, "img", None)), "type(image):", type(getattr(s0, "image", None)))