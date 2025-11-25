import logging
import imageio
import os
from pathlib import Path
from counterfactual_outcomes.common import log_msg
from counterfactual_outcomes.common import State, append_trace_singlefile
import psutil
import io
import sys
from datetime import datetime

# Module-level frames root (set at start of online run)
FRAMES_ROOT = None


def print_env_interface_info(agent, env):
    try:
        print("=== Agent Interface / Environment ===")
        print(f"agent.interface type: {type(getattr(agent, 'interface', agent))}")
        try:
            cfg = getattr(agent.interface, 'config', None)
            print(f"interface.config: {cfg}")
        except Exception:
            pass
        try:
            print(f"env repr: {repr(env)}")
            inner = getattr(env, 'unwrapped', None) or getattr(env, 'env', None) or env
            print(f"unwrapped type: {type(inner)}")
            if hasattr(inner, 'decode'):
                print("env has 'decode' (Taxi-like)")
            if hasattr(inner, 's'):
                print(f"inner.s example: {getattr(inner, 's', None)}")
            print(f"observation_space: {getattr(env, 'observation_space', None)}")
            print(f"action_space: {getattr(env, 'action_space', None)}")
        except Exception:
            pass
        print("=== End Agent Interface / Environment ===")
    except Exception:
        print("Failed to print agent/interface info")


class ContrastiveTrajectory(object):
    def __init__(self, state_id, k_steps, trace):
        self.importance = 0
        self.k_steps = k_steps
        self.id = state_id
        self.start_idx = (state_id[1] - k_steps) if (state_id[1] - k_steps) >= 0 else 0
        self.rewards = []
        self.states = trace.states[self.start_idx:]
        self.actions = []

    def update(self, state_obj, r, action):
        self.states.append(state_obj)
        self.rewards.append(r)
        self.actions.append(action)

    def get_contrastive_trajectory(self, env, agent, state_id, contra_action, contra_counter):
        action = contra_action
        for step in range(state_id[1] + 1, state_id[1] + self.k_steps + 1):
            out = env.step(action)
            if isinstance(out, tuple) and len(out) == 5:
                obs, r, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, r, done, info = out
            contra_counter -= 1  # reduce contra counter
            s = agent.interface.get_state_from_obs(agent, obs)
            s_a_values = agent.interface.get_state_action_values(agent, s)
            frame = None
            try:
                frame = env.render(mode='rgb_array')
            except Exception:
                try:
                    frame = env.render()
                except Exception:
                    frame = None
            # stream frame to disk if available
            image_path = None
            try:
                if frame is not None and FRAMES_ROOT is not None:
                    trace_idx = state_id[0]
                    trace_dir = Path(FRAMES_ROOT) / f"trace_{trace_idx}"
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    img_name = f"trace_{trace_idx}_contr_{step:04d}.png"
                    image_path = trace_dir / img_name
                    imageio.imwrite(str(image_path), frame.astype('uint8') if hasattr(frame, 'astype') else frame)
            except Exception:
                image_path = None
            features = agent.interface.get_features(env)
            contra_state_id = (state_id[0], step)
            state_obj = State(contra_state_id, obs, s, s_a_values, None, features)
            state_obj.image_path = str(image_path) if image_path is not None else None
            self.update(state_obj, r, action)
            if done: break
            if contra_counter > 0: continue
            action = agent.interface.get_next_action(agent, obs, s)


def get_contrastive_trajectory(state_id, trace, env, agent, contra_action, k_steps,
                               contra_counter):
    traj = ContrastiveTrajectory(state_id, k_steps, trace)
    traj.get_contrastive_trajectory(env, agent, state_id, contra_action, contra_counter)
    return traj


def online_comparison(env1, agent1, env2, agent2, args, evaluation1=None, evaluation2=None):
    """
    get all contrastive trajectories a given agent
    """
    """Run"""
    traces = []
    # prepare frames root on disk
    global FRAMES_ROOT
    try:
        FRAMES_ROOT = Path(args.output_dir) / 'trace_images'
        FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception:
        FRAMES_ROOT = None
    # Print interface/env details once before generating traces (before per-trace logs)
    if getattr(args, 'verbose', True):
        print_env_interface_info(agent1, env1)
    for n in range(args.n_traces):
        log_msg(f'Executing Trace number: {n}', args.verbose)
        memory = psutil.virtual_memory()
        print(f"Memory usage before trace {n}: {memory.percent}%")
        # prepare mem_profile output folder for this run
        mem_out = None
        try:
            mem_out = Path(getattr(args, 'output_dir', '.')) / 'mem_profile'
            mem_out.mkdir(parents=True, exist_ok=True)
        except Exception:
            mem_out = None
        # allow one snapshot per trace when threshold crossed
        _snapshot_taken = False
        trace = agent1.interface.contrastive_trace(n, args.k_steps)
        """initial state"""
        res1 = env1.reset()
        res2 = env2.reset()
        obs = res1[0] if isinstance(res1, tuple) else res1
        _obs = res2[0] if isinstance(res2, tuple) else res2
        if not (getattr(obs, 'tolist', None) and getattr(_obs, 'tolist', None) and obs.tolist() == _obs.tolist()):
            log_msg('Warning: initial observations differ between env1 and env2; continuing', args.verbose)
            _obs = obs
        step, r, done, infos, agent1_a = 0, 0, False, {}, None
        agent1.previous_state = agent2.previous_state = obs  # required

        # for _ in range(30):  # TODO remove
        while not done:
            logging.debug(f'time-step number: {step}')
            # check memory and take snapshot if needed
            try:
                vm = psutil.virtual_memory()
                proc = psutil.Process()
                rss = proc.memory_info().rss
                threshold = getattr(args, 'mem_snapshot_threshold', None) or 80
                if (not _snapshot_taken) and (vm.percent >= float(threshold)):
                    # attempt to import and run mem_inspect.snapshot, capture stdout
                    try:
                        from tools.mem_inspect import snapshot
                        buf = io.StringIO()
                        old_stdout = sys.stdout
                        try:
                            sys.stdout = buf
                            snapshot(top=50, min_size=1024)
                        finally:
                            sys.stdout = old_stdout
                        txt = buf.getvalue()
                        if mem_out is not None:
                            fname = f'snapshot_trace_{n}_step_{step}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
                            with open(mem_out / fname, 'w', encoding='utf-8') as f:
                                f.write(f'vm_percent={vm.percent}, rss={rss}\n')
                                f.write(txt)
                        _snapshot_taken = True
                    except Exception:
                        # best-effort: ignore failures to snapshot
                        _snapshot_taken = True
            except Exception:
                pass
            state = agent1.interface.get_state_from_obs(agent1, obs, [r, done])
            s_a_values = agent1.interface.get_state_action_values(agent1, state)
            state_id = (n, step)
            # attempt to render frame
            frame = None
            try:
                frame = env1.render(mode='rgb_array')
            except Exception:
                try:
                    frame = env1.render()
                except Exception:
                    frame = None
            # fallback: if no frame returned by the env, ask the interface to render
            if frame is None:
                try:
                    if hasattr(agent1.interface, 'render_observation'):
                        # prefer passing env and observation when available
                        fb = None
                        try:
                            fb = agent1.interface.render_observation(obs=obs, env=env1)
                        except Exception:
                            try:
                                fb = agent1.interface.render_observation(obs=obs)
                            except Exception:
                                fb = None
                        if fb is not None:
                            frame = fb
                except Exception:
                    pass
            image_path = None
            try:
                if frame is not None and FRAMES_ROOT is not None:
                    trace_dir = Path(FRAMES_ROOT) / f"trace_{n}"
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    img_name = f"trace_{n}_state_{step:04d}.png"
                    image_path = trace_dir / img_name
                    imageio.imwrite(str(image_path), frame.astype('uint8') if hasattr(frame, 'astype') else frame)
            except Exception:
                image_path = None
            features = agent1.interface.get_features(env1)
            state_obj = State(state_id, obs, state, s_a_values, None, features)
            state_obj.image_path = str(image_path) if image_path is not None else None
            trace.update(state_obj, obs, r, done, infos, agent1_a, state_id)
            """actions"""
            agent1_a = agent1.interface.get_next_action(agent1, obs, state) if not done else None
            agent2_a = sorted(list(enumerate(s_a_values)), key=lambda x: x[1])[-2][0]
            """contrastive trajectory"""
            pre_vars = agent2.interface.pre_contrastive(env1)
            trace.contrastive.append(
                get_contrastive_trajectory(state_id, trace, env2, agent2, agent2_a, args.k_steps,
                                           args.contra_action_counter))
            """return agent 2 environment to the current state"""
            env2 = agent2.interface.post_contrastive(agent1, agent2, pre_vars)
            """Transition both agent's based on agent 1 action"""
            step += 1
            out = env1.step(agent1_a)
            if isinstance(out, tuple) and len(out) == 5:
                obs, r, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, r, done, info = out
            if done: break
            out2 = env2.step(agent1_a)
            obs2 = out2[0] if isinstance(out2, tuple) else out2
            if getattr(obs, 'tolist', None) and getattr(obs2, 'tolist', None):
                if obs.tolist() != obs2.tolist():
                    log_msg('Warning: environment transition produced different observations; continuing', args.verbose)
                    obs2 = obs

        """end of episode"""
        # Before persisting, remove any in-memory image objects (e.g., pygame Surfaces)
        # and replace full State objects with lightweight dicts to avoid pickling GUI objects.
        def compact_state_obj(s):
            if s is None:
                return None
            # If already a dict-like compact state, return minimal fields
            if isinstance(s, dict):
                return {
                    'id': s.get('id'),
                    'action_vector': s.get('action_vector'),
                    'image_path': s.get('image_path'),
                    'features': s.get('features')
                }
            try:
                # drop any image-like attributes (Surfaces) and keep a path if present
                img_path = getattr(s, 'image_path', None)
                return {
                    'id': getattr(s, 'id', None),
                    'action_vector': getattr(s, 'action_vector', None),
                    'image_path': img_path,
                    'features': getattr(s, 'features', None)
                }
            except Exception:
                return {'id': getattr(s, 'id', None)}

        # compact original states
        try:
            compact_states = [compact_state_obj(s) for s in getattr(trace, 'states', [])]
            trace.states = compact_states
        except Exception:
            pass

        # compact contrastive trajectories (their .states may contain objects)
        try:
            for c in getattr(trace, 'contrastive', []):
                try:
                    c.states = [compact_state_obj(s) for s in getattr(c, 'states', [])]
                except Exception:
                    # If c.states isn't iterable or already compacted, skip
                    pass
        except Exception:
            pass

        # free other large auxiliary lists that are not needed downstream
        try:
            trace.obs = None
            trace.previous_actions = None
            trace.infos = None
        except Exception:
            pass

        # Persist this compacted trace to disk (append to single Traces.pkl)
        try:
            traces_path = os.path.join(getattr(args, 'output_dir', '.'), 'Traces.pkl')
            # Build a safe serializable representation (dict) to avoid pickling objects
            safe_trace = {
                'trace_idx': getattr(trace, 'trace_idx', None),
                'k_steps': getattr(trace, 'k_steps', None),
                'length': getattr(trace, 'length', None),
                'states': getattr(trace, 'states', []),
                'contrastive': [
                    {
                        'id': getattr(c, 'id', None),
                        'start_idx': getattr(c, 'start_idx', None),
                        'states': getattr(c, 'states', []),
                        'rewards': getattr(c, 'rewards', [])
                    }
                    for c in getattr(trace, 'contrastive', [])
                ]
            }
            append_trace_singlefile(traces_path, safe_trace)
            # keep a lightweight in-memory summary
            traces.append({'trace_idx': safe_trace.get('trace_idx'),
                           'length': safe_trace.get('length'),
                           'file': traces_path})
        except Exception as e:
            # If writing the trace fails, fall back to storing the compacted trace
            logging.warning(f'Failed to append trace to disk: {e}')
            traces.append(trace)

        # hint Python to free unused memory now
        try:
            import gc
            gc.collect()
            del trace
        except Exception:
            pass
    return traces
