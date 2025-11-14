import os, pickle, glob
runs = sorted([d for d in os.listdir('results') if d.startswith('run_')], key=lambda x: os.path.getmtime(os.path.join('results',x)), reverse=True)
if not runs:
    print("no runs")
else:
    latest = runs[0]
    p = os.path.join('results', latest, 'Selected_Highlights.pkl')
    print("selected highlights:", p, os.path.exists(p))
    if os.path.exists(p):
        print(pickle.load(open(p,'rb')))