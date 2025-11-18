import json
import os
import glob
import json
from os import listdir
from os.path import join, isfile, isdir


def get_agent(args):
    """Implement here for specific agent and environment loading scheme"""
    if args.interface == "Highway":
        from counterfactual_outcomes.interfaces.Highway.highway_interface import HighwayInterface
        interface = HighwayInterface(args.config, args.output_dir, args.load_path)
    elif args.interface == "Taxi":
        from counterfactual_outcomes.interfaces.Taxi.taxi_interface import TaxiInterface
        interface = TaxiInterface(args.config, args.output_dir, args.load_path)
    else:
        raise NotImplementedError(f"Interface '{args.interface}' not implemented by get_agent")

    env, agent = interface.initiate()
    agent.interface = interface
    # Robust seeding: some env wrappers (gym/gymnasium or custom wrappers) do not
    # implement `seed`. Attempt multiple safe approaches and ignore failures.
    try:
        if hasattr(env, 'seed') and callable(getattr(env, 'seed')):
            env.seed(0)
    except Exception:
        pass
    try:
        if hasattr(env, 'reset'):
            try:
                env.reset(seed=0)
            except TypeError:
                pass
    except Exception:
        pass
    return env, agent

def get_config(load_path, filename, changes=None):
    """Load a JSON config file.

    Behavior:
    - If ``load_path`` is a path to a file, try to open it as JSON.
    - If ``load_path`` is a directory, find the first file in it whose name contains ``filename``.
    - If not found, fall back to the latest `results/run_*/metadata.json`, then any
      `agents/*/metadata.json` file.
    Raises FileNotFoundError when no suitable config is found.
    Applies ``changes`` (dict of section->key/values) in-place if provided.
    """
    config = None

    # Direct file path
    if load_path and isfile(load_path):
        with open(load_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # If load_path is a directory, search for a matching filename
        if load_path and isdir(load_path):
            try:
                candidates = [x for x in listdir(load_path) if filename in x]
            except Exception:
                candidates = []
            if candidates:
                cfg_file = join(load_path, candidates[0])
                with open(cfg_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

    # Fallback: latest results run metadata
    if config is None:
        res_candidates = sorted(glob.glob(join('results', 'run_*', 'metadata.json')),
                                key=os.path.getmtime, reverse=True)
        if res_candidates:
            with open(res_candidates[0], 'r', encoding='utf-8') as f:
                config = json.load(f)

    # Fallback: any agents/*/metadata.json
    if config is None:
        agent_meta = glob.glob(join('agents', '*', 'metadata.json'))
        if agent_meta:
            with open(agent_meta[0], 'r', encoding='utf-8') as f:
                config = json.load(f)
    print("path", load_path)
    if config is None:
        raise FileNotFoundError(f"Could not find a config containing '{filename}' at '{load_path}' and no fallback metadata found.")

    # Apply changes if provided
    if changes:
        for section in changes:
            for k, v in changes[section].items():
                config.setdefault(section, {})[k] = v

    return config