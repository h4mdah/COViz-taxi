import pickle, sys, os
paths=[
 'results\\Traces.pkl',
 'results\\run_2025-11-18_12-18-34_23912\\Traces.pkl',
 'results\\run_2025-11-18_12-20-50_2132\\Traces.pkl',
 'results\\run_2025-11-18_12-10-17_21040\\Traces.pkl',
 'traces\\taxi\\Traces.pkl'
]
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
for p in paths:
    if not os.path.exists(p):
        print('MISSING',p)
        continue
    try:
        t=pickle.load(open(p,'rb'))
        print(p,'-> type',type(t),'len',len(t))
    except Exception as e:
        print('ERROR',p,e)
