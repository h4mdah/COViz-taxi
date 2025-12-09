"""
Generate a Trace starting from a specific Taxi state.

Usage examples:

# Use an existing Traces.pkl and start from trace index/state index
python tools\generate_trace_from_state.py --tracefile results\run_xxx\Traces.pkl --trace_idx 0 --state_idx 10 --n_steps 50 --out generated

# Start from a raw Taxi internal state integer
python tools\generate_trace_from_state.py --state_s 123 --n_steps 50 --out generated

# Use an SB3 model for actions
python tools\generate_trace_from_state.py --state_s 123 --model_zip agents\taxi_sb3\best.zip --n_steps 50 --out generated

Flags:
  --compute_rd    Attempt to compute RD values using TaxiInterface.get_state_RD_action_values (best-effort).
"""
import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gym
import numpy as np
from copy import deepcopy

from counterfactual_outcomes.common import save_traces
# TaxiTrace and State live in the Taxi interface module
try:
    from counterfactual_outcomes.interfaces.Taxi.taxi_interface import TaxiTrace, State
except Exception:
    # Fallback to common Trace/State if TaxiTrace not importable
    from counterfactual_outcomes.common import Trace as TaxiTrace, State


def load_trace_from_pkl(path, trace_idx=None):
    objs = []
    try:
        with open(path, 'rb') as f:
            while True:
                try:
                    objs.append(pickle.load(f))
                except EOFError:
                    break
    except Exception:
        return None
    # convert dicts to traces if needed elsewhere; here we just return the list
    if not objs:
        return None
    if trace_idx is None:
        return objs[0]
    # find trace with matching trace_idx attribute if available
    for o in objs:
        try:
            if getattr(o, 'trace_idx', None) == trace_idx:
                return o
        except Exception:
            continue
    # fallback: index by position
    if 0 <= trace_idx < len(objs):
        return objs[trace_idx]
    return None


def make_env():
    try:
        env = gym.make('Taxi-v3-COViz', render_mode='rgb_array')
    except Exception:
        try:
            env = gym.make('Taxi-v3', render_mode='rgb_array')
        except Exception:
            env = gym.make('Taxi-v3')
    return env


def set_inner_state(env, s_val):
    inner = getattr(env, 'unwrapped', None) or getattr(env, 'env', None) or env
    try:
        inner.s = int(s_val)
        return True
    except Exception:
        try:
            # Some wrappers store state differently
            setattr(inner, 's', int(s_val))
            return True
        except Exception:
            return False


def build_agent_from_sb3(model_zip, env):
    try:
        from stable_baselines3 import DQN
    except Exception:
        return None
    try:
        model = DQN.load(str(model_zip))
        return model
    except Exception:
        return None


def run_trace_from_state(start_state_s=None, start_state_obj=None, model=None, n_steps=50, compute_rd=False):
    env = make_env()
    inner = getattr(env, 'unwrapped', None) or getattr(env, 'env', None) or env

    if start_state_s is not None:
        ok = set_inner_state(env, start_state_s)
        if not ok:
            print('Warning: could not set internal state directly')
    elif start_state_obj is not None:
        # if start_state_obj contains a .state attribute, try to set it
        s_val = getattr(start_state_obj, 'state', None)
        if s_val is not None:
            set_inner_state(env, s_val)

    # Reset to make sure env internal structures are initialized
    try:
        res = env.reset()
        obs = res[0] if isinstance(res, tuple) else res
    except Exception:
        obs = env.reset()

    # Prepare output trace
    trace = TaxiTrace(0, n_steps)
    traceRd = []

    done = False
    step = 0
    while not done and step < n_steps:
        # select action
        a = None
        try:
            if model is None:
                a = env.action_space.sample()
            else:
                # SB3 model
                if hasattr(model, 'predict'):
                    a, _ = model.predict(obs, deterministic=True)
                else:
                    # fallback to sample
                    a = env.action_space.sample()
        except Exception:
            a = env.action_space.sample()

        out = env.step(int(a))
        if isinstance(out, tuple) and len(out) >= 4:
            # gymnasium returns (obs, reward, terminated, truncated, info) or gym (obs, reward, done, info)
            if len(out) == 5:
                obs, r, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, r, done, info = out
        else:
            # unexpected format
            obs = out
            r = 0
            done = False
            info = {}

        # Build State: use inner.s for state if available
        s_val = getattr(inner, 's', None)
        features = None
        frame = None
        try:
            frame = env.render()
        except Exception:
            frame = None

        state_obj = State((0, step), obs, s_val, None, frame, features)
        # TaxiTrace.update expects (state_object, obs, r, done, infos, a, state_id)
        try:
            trace.update(state_obj, obs, r, done, info, a, (0, step))
        except TypeError:
            # Fallback for base Trace signature
            try:
                trace.update(obs, r, done, info, a, (0, step))
            except Exception:
                pass

        # Optionally compute RD for this state using TaxiInterface if available
        rd_val = None
        if compute_rd:
            try:
                from counterfactual_outcomes.interfaces.Taxi.taxi_interface import TaxiInterface
                # create a minimal interface instance (best-effort)
                iface = TaxiInterface({'env': {'id': 'Taxi-v3-COViz'}}, '.', None)
                rd_val = iface.get_state_RD_action_values(None, getattr(state_obj, 'state', None))
            except Exception:
                rd_val = None
        traceRd.append(rd_val)

        step += 1

    # store rd values on trace if any
    if any(x is not None for x in traceRd):
        try:
            trace.RD_vals = traceRd
        except Exception:
            pass

    try:
        env.close()
    except Exception:
        pass

    return trace


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tracefile', type=str, default=None, help='Path to Traces.pkl to read a starting state from')
    p.add_argument('--trace_idx', type=int, default=None)
    p.add_argument('--state_idx', type=int, default=None)
    p.add_argument('--state_s', type=int, default=None, help='Raw Taxi internal state integer')
    p.add_argument('--model_zip', type=str, default=None)
    p.add_argument('--n_steps', type=int, default=50)
    p.add_argument('--out', type=str, default='generated')
    p.add_argument('--compute_rd', action='store_true')
    args = p.parse_args()

    start_state_obj = None
    start_state_s = None
    if args.tracefile and args.state_idx is not None:
        tr = load_trace_from_pkl(args.tracefile, args.trace_idx)
        if tr is None:
            print('Could not load trace from', args.tracefile)
            sys.exit(1)
        try:
            start_state_obj = tr.states[args.state_idx]
        except Exception:
            print('Could not get state from trace at index', args.state_idx)
            sys.exit(1)
    elif args.state_s is not None:
        start_state_s = args.state_s
    else:
        print('You must provide --state_s or --tracefile with --state_idx')
        sys.exit(1)

    model = None
    if args.model_zip:
        model = build_agent_from_sb3(args.model_zip, None)
        if model is None:
            print('Warning: could not load SB3 model; using random policy')

    trace = run_trace_from_state(start_state_s=start_state_s, start_state_obj=start_state_obj, model=model, n_steps=args.n_steps, compute_rd=args.compute_rd)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_traces([trace], str(out_dir))
    print('Saved generated trace to', out_dir)
