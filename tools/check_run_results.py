import os
import glob
import pickle
from pathlib import Path

repo = Path('.').resolve()
results_dir = repo / 'results'
if not results_dir.exists():
    print("No results/ folder found")
    raise SystemExit(0)

runs = sorted([d for d in results_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
if not runs:
    print("No run_* folders inside results/")
    raise SystemExit(0)

latest = runs[0]
print("Latest run:", latest.name)
videos_dir = latest / 'Highlight_Videos'
nomark_videos_dir = latest / 'NoMark_Videos'
frames_dir = latest / 'Highlight_Frames'
nomark_frames_dir = latest / 'NoMark_Frames'

def list_dir(p):
    if not p.exists():
        print(f"  {p.name}  -- MISSING")
        return []
    files = sorted(p.glob('*'))
    print(f"  {p.name}  -- {len(files)} files")
    for f in files[:20]:
        print("    ", f.name)
    if len(files) > 20:
        print("    ...")
    return files

print("\nContents of latest run:")
list_dir(videos_dir)
list_dir(nomark_videos_dir)
list_dir(frames_dir)
list_dir(nomark_frames_dir)

# check traces pickles: first check Traces.pkl in traces/ and in run folder
for candidate in [repo / 'traces' / 'Traces.pkl', latest / 'Traces.pkl', latest / 'Selected_Highlights.pkl']:
    if candidate.exists():
        print("\nFound pickle:", candidate)
        try:
            with open(candidate, 'rb') as f:
                obj = pickle.load(f)
            # if it's a list of traces, inspect first trace
            if isinstance(obj, list) and obj:
                t = obj[0]
                states = getattr(t, 'states', None)
                print("  object is a list of length", len(obj), "first trace has", len(states or []), "states")
                if states and hasattr(states[0], 'image'):
                    img = states[0].image
                    print("  states[0].image is", "present" if img is not None else "None")
                else:
                    print("  no image attribute on states (or states empty)")
        except Exception as e:
            print("  failed to unpickle or inspect:", e)

print("\nIf frames exist but videos are missing, install imageio-ffmpeg and rerun save_highlights or re-run run.py.")
print("To re-run contrastive generation: python run.py --traces-path traces")