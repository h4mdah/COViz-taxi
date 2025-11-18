"""Inspect a Selected_Highlights.pkl file and print sample entries."""
import sys, os, pickle
p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if p not in sys.path:
    sys.path.insert(0, p)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()

fn = args.file
with open(fn, 'rb') as f:
    sel = pickle.load(f)

print('type(sel)=', type(sel))
try:
    print('len(sel)=', len(sel))
except Exception:
    pass

# Print first 10 entries with repr truncated
from pprint import pformat

for i, e in enumerate(sel[:10]):
    print('--- entry', i, 'type=', type(e))
    try:
        print(pformat(e))
    except Exception as ex:
        print('repr failed:', ex)

print('\nSample access attempts:')
if len(sel) > 0:
    e = sel[0]
    print('as dict?', isinstance(e, dict))
    try:
        print('e[0]=', e[0])
    except Exception as ex:
        print('e[0] failed:', ex)
    try:
        print("e.trace_idx:", getattr(e, 'trace_idx', None))
        print("e.contrastive_idx:", getattr(e, 'contrastive_idx', None))
    except Exception as ex:
        print('attr access failed:', ex)
