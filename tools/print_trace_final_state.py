"""
Safely inspect a Traces.pkl and print the final state's information without importing project modules.

Usage:
    python tools\print_trace_final_state.py "C:\path\to\Traces.pkl"

This script uses a custom Unpickler that returns placeholder objects for unknown classes so we can safely introspect attributes that were stored in instance dicts.
"""
import sys
import pickle
from pathlib import Path

class Placeholder:
    def __init__(self, *args, **kwargs):
        self.__dict__['_placeholder_args'] = args
        self.__dict__['_placeholder_kwargs'] = kwargs
    def __setstate__(self, state):
        # state is often a dict with attributes
        try:
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__['_state'] = state
        except Exception:
            self.__dict__['_state_repr'] = repr(state)
    def __repr__(self):
        cls = self.__dict__.get('__class__', 'Placeholder')
        return f"<Placeholder {cls} {list(self.__dict__.keys())}>"

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # allow safe builtins
        if module == 'builtins':
            return super().find_class(module, name)
        # return a placeholder class for all other globals to avoid importing project modules
        return Placeholder


def load_all_safe(path):
    objs = []
    with open(path, 'rb') as f:
        unpickler = SafeUnpickler(f)
        while True:
            try:
                objs.append(unpickler.load())
            except EOFError:
                break
            except Exception as e:
                print('Warning: failed to unpickle one object:', e)
                break
    return objs


def inspect_trace_obj(obj):
    # Try common attribute patterns for trace objects
    info = {}
    # try .states
    states = None
    if hasattr(obj, 'states'):
        states = getattr(obj, 'states')
    elif isinstance(obj, dict) and 'states' in obj:
        states = obj['states']
    elif hasattr(obj, 'trace'):
        states = getattr(obj, 'trace')
    info['n_states'] = len(states) if states is not None else 0
    last_state = None
    if states:
        last_state = states[-1]
    # last_state may be a Placeholder, dict, or primitive
    if last_state is None:
        info['last_state_repr'] = None
        return info
    # try common fields: .state, .s, .obs, .frame
    def get_attr(x, names):
        for n in names:
            try:
                if hasattr(x, n):
                    return getattr(x, n)
                if isinstance(x, dict) and n in x:
                    return x[n]
            except Exception:
                continue
        return None
    state_id = get_attr(last_state, ['state_id', 'state_idx', 'id', 'state'])
    internal_s = get_attr(last_state, ['s', 'state', 'state_s'])
    obs = get_attr(last_state, ['obs', 'observation', 'ob'])
    reward = get_attr(last_state, ['reward', 'r'])
    info.update({'state_id': state_id, 'internal_s': internal_s, 'obs_repr': repr(obs)[:200], 'reward': reward})
    return info


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to Traces.pkl')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if not args.path:
        print('Usage: python tools\\print_trace_final_state.py path/to/Traces.pkl')
        sys.exit(1)
    p = Path(args.path)
    if not p.exists():
        print('File not found:', p)
        sys.exit(1)
    objs = load_all_safe(str(p))
    if args.verbose:
        print(f'Loaded {len(objs)} top-level object(s) from pickle')
        for i, o in enumerate(objs[:5]):
            try:
                print(f'  obj[{i}]: type={type(o)}, repr={repr(o)[:200]}')
            except Exception:
                print(f'  obj[{i}]: type={type(o)} (repr failed)')
    if not objs:
        print('No objects found in pickle')
        sys.exit(0)
    # Flatten top-level lists (some pickles store a single list of traces)
    candidates = []
    for o in objs:
        if isinstance(o, list):
            candidates.extend(o)
        else:
            candidates.append(o)

    # Try to find the last trace-like object among candidates
    trace_obj = None
    for o in reversed(candidates):
        if hasattr(o, 'states') or (isinstance(o, dict) and 'states' in o):
            trace_obj = o
            break
    if trace_obj is None:
        # fallback to last candidate
        trace_obj = candidates[-1] if candidates else objs[-1]
        print('No object with "states" found; using last object in file')
    info = inspect_trace_obj(trace_obj)
    print('Trace summary:')
    for k, v in info.items():
        print(f'  {k}: {v}')

if __name__ == '__main__':
    main()
