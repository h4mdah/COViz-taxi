import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
import numpy as np
from tqdm import trange
from memory_profiler import profile

from counterfactual_outcomes.common import log_msg, save_traces
from counterfactual_outcomes.common import State
from counterfactual_outcomes.contrastive_online import get_contrastive_trajectory


def online_comparison_RD(env1, agent1, env2, agent2, args, evaluation1=None, evaluation2=None):
    """
    Get all contrastive trajectories for a given agent, collecting reward decomposition data.
    Uses the same structure as contrastive_online but additionally stores RD action values.
    """
    traces = []
    rd_values = []  # Store RD action values for all states across all traces
    
    # Use a single trange progress bar and update its description per-iteration.
    pbar = trange(args.n_traces, desc="Traces", unit="trace")
    for t in pbar:
        # Update the bar description instead of printing a new line each iteration
        pbar.set_description(f"Trace {t+1}/{args.n_traces}")
        trace = agent1.interface.contrastive_trace(t, args.k_steps)
        trace_rd_vals = []  # Collect RD values for this trace
        
        """initial state"""
        res1 = agent1.interface.reset_env(env1, state=getattr(args, 'start_state', None))
        res2 = agent2.interface.reset_env(env2, state=getattr(args, 'start_state', None))

        if getattr(args, 'start_state', None) is not None:
             log_msg(f"Using start state: {args.start_state}", args.verbose)

        obs = res1[0] if isinstance(res1, tuple) else res1
        _obs = res2[0] if isinstance(res2, tuple) else res2
        if not (getattr(obs, 'tolist', None) and getattr(_obs, 'tolist', None) and obs.tolist() == _obs.tolist()):
            log_msg('Warning: initial observations differ between env1 and env2; continuing', args.verbose)
            _obs = obs
        step, r, done, infos, agent1_a = 0, 0, False, {}, None
        agent1.previous_state = agent2.previous_state = obs  # required

        # Implement optional lockstep prefix before forking contrastive
        sync_prefix = int(getattr(args, 'sync_prefix', 0) or 0)

        # 1) Run prefix steps in lockstep using agent1's action for both envs
        while not done and step < sync_prefix:
            logging.debug(f'prefix time-step number: {step}')
            state = agent1.interface.get_state_from_obs(agent1, obs, [r, done])
            s_a_values = agent1.interface.get_state_action_values(agent1, state)
            rd_action_values = agent1.interface.get_state_RD_action_values(agent1, state)
            trace_rd_vals.append(rd_action_values)
            
            state_id, frame = (t, step), env1.render()
            features = agent1.interface.get_features(env1, obs)
            state_obj = State(state_id, obs, state, s_a_values, frame, features)
            # previous action (agent1_a) is the action that led to this state
            trace.update(state_obj, obs, r, done, infos, agent1_a, state_id)
            # keep contrastive list aligned with states
            trace.contrastive.append(None)

            # both agents take agent1's action during prefix
            agent1_a = agent1.interface.get_next_action(agent1, obs, state) if not done else None
            agent1.previous_state = agent2.previous_state = obs

            step += 1
            out = env1.step(agent1_a)
            if isinstance(out, tuple) and len(out) == 5:
                obs, r, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, r, done, info = out
            if done:
                break
            out2 = env2.step(agent1_a)
            obs2 = out2[0] if isinstance(out2, tuple) else out2
            if getattr(obs, 'tolist', None) and getattr(obs2, 'tolist', None):
                if obs.tolist() != obs2.tolist():
                    log_msg('Warning: environment transition produced different observations; continuing', args.verbose)
                    obs2 = obs

        # if episode ended during prefix, finish this trace
        if done:
            trace.RD_vals = trace_rd_vals
            traces.append(trace)
            rd_values.append(trace_rd_vals)
            continue

        # 2) At fork time-step: create fork state and contrastive trajectory
        logging.debug(f'fork at time-step: {step}')
        state = agent1.interface.get_state_from_obs(agent1, obs, [r, done])
        s_a_values = agent1.interface.get_state_action_values(agent1, state)
        rd_action_values = agent1.interface.get_state_RD_action_values(agent1, state)
        trace_rd_vals.append(rd_action_values)
        
        state_id, frame = (t, step), env1.render()
        features = agent1.interface.get_features(env1, obs)
        state_obj = State(state_id, obs, state, s_a_values, frame, features)
        # update trace with fork state (previous action is agent1_a)
        trace.update(state_obj, obs, r, done, infos, agent1_a, state_id)
        
        # derive contrastive action for agent2
        try:
             # Delegate to the interface to decide the counterfactual action
             # This handles DQN (2nd highest Q) and PPO (2nd highest Prob)
             if hasattr(agent2.interface, 'get_counterfactual_action'):
                 agent2_a = agent2.interface.get_counterfactual_action(agent2, obs, s_a_values)
             else:
                 # Fallback to legacy logic if interface doesn't implement it
                vals = np.asarray(s_a_values)
                if vals.size == 0 or np.allclose(vals, vals.flat[0]):
                    n_actions = vals.size if vals.size > 0 else getattr(getattr(agent1, 'action_space', None), 'n', None)
                    if n_actions is None or n_actions == 0:
                        agent2_pref = agent2.interface.get_next_action(agent2, obs, state)
                        agent2_a = agent2_pref if agent2_pref is not None else 0
                    else:
                        base = agent1_a if agent1_a is not None else 0
                        agent2_a = (base + 1) % int(n_actions)
                else:
                    agent2_a = sorted(list(enumerate(vals)), key=lambda x: x[1])[-2][0]
        except Exception:
            try:
                agent2_a = agent2.interface.get_next_action(agent2, obs, state)
            except Exception:
                agent2_a = 0

        # create contrastive trajectory from env2 starting at this fork state
        pre_vars = agent2.interface.pre_contrastive(env1)
        contra_traj = get_contrastive_trajectory(state_id, trace, pre_vars, agent2, agent2_a, args.k_steps,
                                                 args.contra_action_counter)
        trace.contrastive.append(contra_traj)
        # Compute and store RD values for the contrastive trajectory.
        # For prefix states, copy from the original trace.RD_vals where available;
        # for appended contrastive states, compute using agent2's interface.
        try:
            contra_rd_list = []
            trace_rd = getattr(trace, 'RD_vals', None)
            for j, st in enumerate(contra_traj.states):
                global_idx = contra_traj.start_idx + j
                # Use original trace RD if available for this prefix state
                if trace_rd is not None and global_idx < len(trace_rd):
                    contra_rd_list.append(trace_rd[global_idx])
                else:
                    # Contrastive state -> compute RD from agent2's interface
                    try:
                        # st is a State object; use its .state attribute
                        rd_val = agent2.interface.get_state_RD_action_values(agent2, getattr(st, 'state', None))
                    except Exception:
                        rd_val = None
                    contra_rd_list.append(rd_val)
            contra_traj.RD_vals = contra_rd_list
        except Exception:
            # best-effort: if anything fails, leave RD_vals unset
            contra_traj.RD_vals = None
        # we do not call post_contrastive here; env2 has been consumed by contra_traj

        # 3) Continue original (env1) until episode end — record true future
        # ensure agent1 has a valid action for the fork step
        if agent1_a is None:
            try:
                agent1_a = agent1.interface.get_next_action(agent1, obs, state)
            except Exception:
                # fallback to 0 if action cannot be computed
                agent1_a = 0
        step += 1
        out = env1.step(agent1_a)
        if isinstance(out, tuple) and len(out) == 5:
            obs, r, terminated, truncated, info = out
            done = bool(terminated or truncated)
        else:
            obs, r, done, info = out
        
        # continue remaining episode steps for env1
        if not done:
            while not done:
                logging.debug(f'post-fork env1 time-step: {step}')
                state = agent1.interface.get_state_from_obs(agent1, obs, [r, done])
                s_a_values = agent1.interface.get_state_action_values(agent1, state)
                rd_action_values = agent1.interface.get_state_RD_action_values(agent1, state)
                trace_rd_vals.append(rd_action_values)
                
                state_id, frame = (t, step), env1.render()
                features = agent1.interface.get_features(env1, obs)
                state_obj = State(state_id, obs, state, s_a_values, frame, features)
                trace.update(state_obj, obs, r, done, infos, agent1_a, state_id)
                trace.contrastive.append(None)

                agent1_a = agent1.interface.get_next_action(agent1, obs, state) if not done else None
                step += 1
                out = env1.step(agent1_a)
                if isinstance(out, tuple) and len(out) == 5:
                    obs, r, terminated, truncated, info = out
                    done = bool(terminated or truncated)
                else:
                    obs, r, done, info = out
        else:
            # if done immediately after fork, just finish
            pass

        # Record the final terminal state
        if done:
            logging.debug(f'terminal env1 time-step: {step}')
            state = agent1.interface.get_state_from_obs(agent1, obs, [r, done])
            s_a_values = agent1.interface.get_state_action_values(agent1, state)
            rd_action_values = agent1.interface.get_state_RD_action_values(agent1, state)
            trace_rd_vals.append(rd_action_values)

            state_id, frame = (t, step), env1.render()
            features = agent1.interface.get_features(env1, obs)
            state_obj = State(state_id, obs, state, s_a_values, frame, features)
            trace.update(state_obj, obs, r, done, infos, None, state_id)
            trace.contrastive.append(None)

        """end of episode"""
        trace.RD_vals = trace_rd_vals
        traces.append(trace)
        rd_values.append(trace_rd_vals)
    
    """save traces and RD values"""
    save_traces(traces, REPO_ROOT / 'results', name='Traces_RD.pkl')
    save_traces(rd_values, REPO_ROOT / 'results', name='RD_Values.pkl')
    return traces
