import sys, pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from counterfactual_outcomes.common import log_msg
from counterfactual_outcomes.common import State
from tqdm import trange
from memory_profiler import profile
import numpy as np

class ContrastiveTrajectory(object):
    def __init__(self, state_id, k_steps, trace):
        self.importance = 0
        self.k_steps = k_steps
        self.id = state_id
        self.start_idx = (state_id[1] - k_steps) if (state_id[1] - k_steps) >= 0 else 0
        # Pre-fill rewards with the original trajectory rewards from the start_idx
        # so that rewards and states align for prefix frames.
        try:
            self.rewards = list(trace.rewards[self.start_idx:]) if getattr(trace, 'rewards', None) is not None else []
        except Exception:
            self.rewards = []
        self.states = list(trace.states[self.start_idx:])
        self.actions = []

    def update(self, state_obj, r, action):
        self.states.append(state_obj)
        self.rewards.append(r)
        self.actions.append(action)

    def get_contrastive_trajectory(self, env, agent, state_id, contra_action, contra_counter):
        # Apply the counterfactual action for exactly one step, then follow the
        # agent's policy for up to `self.k_steps` steps or until the episode ends.
        action = contra_action
        current_step_idx = state_id[1] + 1
        done = False
        steps_taken = 0
        max_steps = max(1, int(self.k_steps))  # ensure integer >=1

        while not done and steps_taken < max_steps:
            out = env.step(action)
            if isinstance(out, tuple) and len(out) == 5:
                obs, r, terminated, truncated, info = out
                done = bool(terminated or truncated)
            else:
                obs, r, done, info = out

            # Build state object for this contrastive step and store it
            s = agent.interface.get_state_from_obs(agent, obs)
            s_a_values = agent.interface.get_state_action_values(agent, s)
            frame = env.render()
            features = agent.interface.get_features(env, obs)
            contra_state_id = (state_id[0], current_step_idx)
            state_obj = State(contra_state_id, obs, s, s_a_values, frame, features)
            self.update(state_obj, r, action)

            steps_taken += 1
            if done:
                break

            # After the first (counterfactual) step, switch to the agent's policy
            # for all subsequent steps in this contrastive rollout.
            action = agent.interface.get_next_action(agent, obs, s)
            current_step_idx += 1


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
    # Use a single trange progress bar and update its description per-iteration.
    pbar = trange(args.n_traces, desc="Traces", unit="trace")
    for t in pbar:
        # Update the bar description instead of printing a new line each iteration
        pbar.set_description(f"Trace {t+1}/{args.n_traces}")
        trace = agent1.interface.contrastive_trace(t, args.k_steps)
        """initial state"""
        res1 = env1.reset()
        res2 = env2.reset()
        
        # If a specific start state is requested (e.g. for Taxi), force it.
        # This assumes the environment supports setting state via .s or similar
        # and that the observation is consistent w/ that state or simply the state itself.
        if getattr(args, 'start_state', None) is not None:
             try:
                 env1.unwrapped.s = args.start_state
                 env2.unwrapped.s = args.start_state
                 # For Taxi, obs is just the state index
                 res1 = args.start_state
                 res2 = args.start_state
                 env1.unwrapped.s = args.start_state
                 env2.unwrapped.s = args.start_state
                 # For Taxi, obs is just the state index
                 res1 = args.start_state
                 res2 = args.start_state
                 pbar.write(f"Using start state: {args.start_state}")
                 log_msg(f"Using start state: {args.start_state}", args.verbose)
             except Exception:
                 log_msg("Warning: could not set start_state on environment", args.verbose)

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
            traces.append(trace)
            continue

        # 2) At fork time-step: create fork state and contrastive trajectory
        logging.debug(f'fork at time-step: {step}')
        state = agent1.interface.get_state_from_obs(agent1, obs, [r, done])
        s_a_values = agent1.interface.get_state_action_values(agent1, state)
        state_id, frame = (t, step), env1.render()
        features = agent1.interface.get_features(env1, obs)
        state_obj = State(state_id, obs, state, s_a_values, frame, features)
        # update trace with fork state (previous action is agent1_a)
        trace.update(state_obj, obs, r, done, infos, agent1_a, state_id)
        # derive contrastive action for agent2 (reuse robust logic)
        try:
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
        
        
        contra_traj = get_contrastive_trajectory(state_id, trace, pre_vars, agent2, agent2_a, args.k_steps, #changed from env2 to pre_vars
                                                 args.contra_action_counter)
        trace.contrastive.append(contra_traj)
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
        # append placeholder for contrastive alignment for this new state (we already added contra for fork)
        if not done:
            # loop through remaining episode steps for env1
            while not done:
                logging.debug(f'post-fork env1 time-step: {step}')
                state = agent1.interface.get_state_from_obs(agent1, obs, [r, done])
                s_a_values = agent1.interface.get_state_action_values(agent1, state)
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
            # if done immediately after fork, just append and finish
            traces.append(trace)
            continue

        """end of episode"""
        traces.append(trace)
    return traces
