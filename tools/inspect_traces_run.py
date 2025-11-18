import pickle, pprint, sys, os

# ensure repo root importable for unpickling custom classes
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

p = sys.argv[1] if len(sys.argv) > 1 else 'results\\run_2025-11-18_12-20-50_2132\\Traces.pkl'
try:
    traces = pickle.load(open(p, 'rb'))
except Exception as e:
    print('ERROR loading', p, e)
    raise
print('type', type(traces))
try:
    print('len', len(traces))
except Exception:
    print('len unknown')
# Inspect trace 9 and 198 if present
for idx in [9, 170, 177, 184, 191, 198]:
    try:
        t = traces[idx]
        print('\nTRACE IDX', idx, 'type', type(t))
        # try attributes
        for attr in ['states', 'trajectory', 'steps', 'frames', 'images']:
            if hasattr(t, attr):
                v = getattr(t, attr)
                try:
                    print(' -', attr, 'len', len(v))
                except Exception:
                    print(' -', attr, 'type', type(v))
        # fallback: if trace is list-like
        try:
            L = len(t)
            print(' - trace is list-like, len', L)
            sample = t[0]
            print(' - sample type', type(sample))
            for a in ['state', 'image', 'img']:
                if hasattr(sample, a):
                    print('   sample has', a, '->', type(getattr(sample,a)))
        except Exception:
            pass
    except Exception as e:
        print('Could not access trace', idx, e)
