import pickle, itertools, pprint, sys
p = sys.argv[1] if len(sys.argv) > 1 else 'results\\run_2025-11-18_12-20-50_2132\\Selected_Highlights.pkl'
try:
    d = pickle.load(open(p, 'rb'))
except Exception as e:
    print('ERROR loading', p, e)
    raise
print('type', type(d))
try:
    print('len', len(d))
except Exception:
    print('len unknown')
print('first 10 items:')
for i, x in enumerate(itertools.islice(d, 10)):
    print('\nITEM', i, '->', type(x))
    pprint.pprint(x)
