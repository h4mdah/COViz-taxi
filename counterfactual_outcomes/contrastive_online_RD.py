from os.path import abspath
import logging
import imageio
from pathlib import Path

from counterfactual_outcomes.common import log_msg, save_traces
from counterfactual_outcomes.common import State
from counterfactual_outcomes.contrastive_online import get_contrastive_trajectory
import counterfactual_outcomes.contrastive_online as co_module


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


def online_comparison_RD(env1, agent1, env2, agent2, args, evaluation1=None, evaluation2=None):
    """
    get all contrastive trajectories a given agent
    """
    """Run"""
    traces, reward_decomps = [], []
    # Print interface/env details once before generating traces (before per-trace logs)
    if getattr(args, 'verbose', True):
        print_env_interface_info(agent1, env1)
    # prepare frames root and set it in the contrastive_online module so contrastive calls
    # also write into the same folder
    try:
        frames_root = Path(args.output_dir) / 'trace_images'
        frames_root.mkdir(parents=True, exist_ok=True)
        co_module.FRAMES_ROOT = frames_root
    except Exception:
        co_module.FRAMES_ROOT = None
    for n in range(args.n_traces):
        log_msg(f'Executing Trace number: {n}', args.verbose)
        trace = agent1.interface.contrastive_trace(n, args.k_steps)
        rd_vals = []
        """initial state"""
        res1 = env1.reset()
        res2 = env2.reset()
        # gymnasium may return (obs, info); extract observation for compatibility
        obs = res1[0] if isinstance(res1, tuple) else res1
        _obs = res2[0] if isinstance(res2, tuple) else res2
        # If environments don't start identically, warn and continue using env1's obs
        if not (getattr(obs, 'tolist', None) and getattr(_obs, 'tolist', None) and obs.tolist() == _obs.tolist()):
            log_msg('Warning: initial observations differ between env1 and env2; continuing', args.verbose)
            _obs = obs
        step, r, done, infos, agent1_a = 0, 0, False, {}, None
        agent1.previous_state = agent2.previous_state = obs  # required

        # for _ in range(30):  # TODO remove
        while not done:
            logging.debug(f'time-step number: {step}')
            state = agent1.interface.get_state_from_obs(agent1, obs, [r, done])
            s_a_values = agent1.interface.get_state_action_values(agent1, state)
            rd_vals.append(agent1.interface.get_state_RD_action_values(agent1, state))
            state_id = (n, step)
            frame = None
            try:
                frame = env1.render(mode='rgb_array')
            except Exception:
                try:
                    frame = env1.render()
                except Exception:
                    frame = None
            image_path = None
            try:
                if frame is not None and co_module.FRAMES_ROOT is not None:
                    trace_dir = Path(co_module.FRAMES_ROOT) / f"trace_{n}"
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
            # gymnasium step may return (obs, reward, terminated, truncated, info)
            if isinstance(out, tuple) and len(out) == 5:
                obs, r, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, r, done, info = out
            if done: break
            out2 = env2.step(agent1_a)  # dont need returned values beyond obs for comparison
            obs2 = out2[0] if isinstance(out2, tuple) else out2
            # ensure arrays are comparable; if they differ, warn and continue using env1 obs
            if getattr(obs, 'tolist', None) and getattr(obs2, 'tolist', None):
                if obs.tolist() != obs2.tolist():
                    log_msg('Warning: environment transition produced different observations; continuing', args.verbose)
                    obs2 = obs

        """end of episode"""
        traces.append(trace)
        trace.RD_vals = rd_vals
        reward_decomps.append(rd_vals)
    """save RD traces"""
    save_traces(reward_decomps, args.output_dir, name='RD_Values.pkl')
    save_traces(traces, abspath('results'), name='RD_Values.pkl')
    return traces
