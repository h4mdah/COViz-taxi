"""Run a single online trace and profile memory.

Usage:
  python tools/mem_profile_trace.py --config config.json --output_dir results/mem_profile --n_traces 1

This will import the contrastive_online runner and execute one trace while sampling memory.
"""
import argparse
import time
import os
from pathlib import Path
import psutil
import threading
import json

# ensure repo root importable
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from counterfactual_outcomes import contrastive_online
from counterfactual_outcomes.get_agent import get_agent


def sample_mem(pid, interval, stop_event, results):
    p = psutil.Process(pid)
    peak = 0
    samples = []
    while not stop_event.is_set():
        try:
            mem = p.memory_info().rss
        except Exception:
            mem = 0
        samples.append(mem)
        if mem > peak:
            peak = mem
        time.sleep(interval)
    results['peak'] = peak
    results['samples'] = samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--output_dir', default='results/mem_profile')
    parser.add_argument('--n_traces', type=int, default=1)
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--sample_interval', type=float, default=0.5)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # build minimal args object for contrastive_online.runner
    class A:
        pass
    myargs = A()
    myargs.output_dir = str(out)
    myargs.n_traces = args.n_traces
    myargs.k_steps = 10
    myargs.verbose = False
    myargs.contra_action_counter = 1
    # trigger mem snapshot earlier for profiling (percent of system memory)
    myargs.mem_snapshot_threshold = 60

    # load config dict and build a minimal args object expected by get_agent
    from counterfactual_outcomes.get_agent import get_config
    cfg = None
    try:
        cfg = get_config(args.config, 'metadata') if args.config else None
    except Exception:
        cfg = None
    # Ensure minimal env config exists for Taxi interface
    if not cfg or not isinstance(cfg, dict) or 'env' not in cfg:
        cfg = cfg or {}
        cfg.setdefault('env', {})
        cfg['env'].setdefault('id', 'Taxi-v3')
        cfg.setdefault('interface', cfg.get('interface', 'Taxi'))
        cfg.setdefault('model_dir', cfg.get('model_dir', 'agents/taxi_sb3'))

    class GArgs:
        pass

    gargs = GArgs()
    # interface name comes from config or default to Taxi
    gargs.interface = cfg.get('interface') if cfg and isinstance(cfg, dict) else 'Taxi'
    gargs.config = cfg
    gargs.output_dir = str(out)
    # load_path hint (model dir) from config
    gargs.load_path = cfg.get('model_dir') if cfg and isinstance(cfg, dict) else None

    # get a pair of agents/envs via get_agent (uses config metadata)
    env1, agent1 = get_agent(gargs)
    env2, agent2 = get_agent(gargs)

    stop_event = threading.Event()
    results = {}
    sampler = threading.Thread(target=sample_mem, args=(os.getpid(), args.sample_interval, stop_event, results))
    sampler.start()

    t0 = time.time()
    try:
        traces = contrastive_online.online_comparison(env1, agent1, env2, agent2, myargs)
    except Exception as e:
        import traceback
        print('Error running online comparison:')
        traceback.print_exc()
        traces = None
    t1 = time.time()

    stop_event.set()
    sampler.join()

    peak = results.get('peak', None)
    samples = results.get('samples', [])

    report = {
        'duration_sec': t1 - t0,
        'peak_rss_bytes': peak,
        'samples': samples,
        'n_traces_returned': len(traces) if traces is not None else 0
    }

    with open(out / 'mem_profile_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print('Memory profile saved to', out / 'mem_profile_report.json')
    print('Peak RSS (MB):', (peak or 0) / (1024*1024))


if __name__ == '__main__':
    main()
