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
        self.rewards = []
        self.states = trace.states[self.start_idx:]
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

@profile
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
        contra_traj = get_contrastive_trajectory(state_id, trace, env2, agent2, agent2_a, args.k_steps,
                                                 args.contra_action_counter)
        trace.contrastive.append(contra_traj)

        # Extra logging for debugging/validation: print fork features and
        # the first few contrastive actions/states so the user can verify
        # the contrastive rollout actually diverged from the main trajectory.
        try:
            logging.info(f"Fork: trace={t} step={step} state_id={state_id} features={features}")
            logging.info(f"Fork actions: agent1_a={agent1_a} agent2_a={agent2_a}")
            # Diagnostic: inspect inner env and obs to understand missing features
            try:
                inner1 = getattr(env1, 'unwrapped', None) or getattr(env1, 'env', None) or env1
                s1 = getattr(inner1, 's', None)
                has_decode = hasattr(inner1, 'decode')
                logging.info(f"Fork diag (env1): inner={type(inner1)} has_decode={has_decode} s={s1}")
            except Exception as _:
                logging.info(f"Fork diag (env1): could not inspect inner env: {_}")
            try:
                logging.info(f"Fork diag obs: obs={obs} type={type(obs)}")
            except Exception:
                logging.info("Fork diag obs: (unprintable)")
            # print first few contrastive actions
            first_actions = getattr(contra_traj, 'actions', [])[:10]
            logging.info(f"Contrastive first actions: {first_actions}")
            # print first few contrastive states' features if available
            contra_state_feats = [getattr(s, 'features', None) for s in getattr(contra_traj, 'states', [])[:10]]
            logging.info(f"Contrastive first state features: {contra_state_feats}")
        except Exception:
            pass

        # Human-readable paired report for the fork: include a few steps before
        # the fork and N steps after so you can compare trajectories easily.
        try:
            N = int(getattr(args, 'fork_report_steps', 10) or 10)
            pre = int(getattr(args, 'fork_report_pre', 2) or 2)
            # build action name mapping for readability
            if getattr(args, 'interface', '').lower() == 'taxi':
                ACTION_DICT = {0: 'SOUTH', 1: 'NORTH', 2: 'EAST', 3: 'WEST', 4: 'PICKUP', 5: 'DROPOFF'}
            elif getattr(args, 'interface', '').lower() == 'highway':
                ACTION_DICT = {0: 'LANE_LEFT', 1: 'IDLE', 2: 'LANE_RIGHT', 3: 'FASTER', 4: 'SLOWER'}
            else:
                ACTION_DICT = {}

            # Window: start a few steps before fork, end N steps after
            start_idx = max(0, state_id[1] - pre)
            end_idx = state_id[1] + N

            # precompute contrastive states by their trace-step index
            contra_states = getattr(contra_traj, 'states', [])
            contra_actions = getattr(contra_traj, 'actions', [])
            contra_rewards = getattr(contra_traj, 'rewards', [])
            contra_map = {getattr(s, 'id', (None, None))[1]: s for s in contra_states if getattr(s, 'id', None)}

            # find fork position index in contra_states to align actions
            fork_pos_in_contra = next((i for i, s in enumerate(contra_states) if getattr(s, 'id', None) == state_id), None)

            logging.info(f"Fork report (from step {start_idx} to {end_idx}) — fork_state={state_id} features={features}")
            for idx in range(start_idx, end_idx + 1):
                # True entry
                if idx < len(trace.states):
                    st = trace.states[idx]
                    act = trace.previous_actions[idx] if idx < len(trace.previous_actions) else None
                    rew = trace.rewards[idx] if idx < len(trace.rewards) else None
                    try:
                        pos = st.features.get('position') if getattr(st, 'features', None) else None
                    except Exception:
                        pos = None
                    true_entry = (ACTION_DICT.get(act, act), pos, rew)
                else:
                    true_entry = (None, None, None)

                # Contrastive entry (may be missing before fork)
                if idx in contra_map:
                    stc = contra_map[idx]
                    # compute action index offset relative to fork
                    if fork_pos_in_contra is not None:
                        offset = idx - state_id[1]
                        action_idx = offset - 1
                        if action_idx >= 0 and action_idx < len(contra_actions):
                            actc = contra_actions[action_idx]
                            rewc = contra_rewards[action_idx] if action_idx < len(contra_rewards) else None
                        else:
                            actc = None; rewc = None
                    else:
                        actc = None; rewc = None
                    try:
                        posc = stc.features.get('position') if getattr(stc, 'features', None) else None
                    except Exception:
                        posc = None
                    contra_entry = (ACTION_DICT.get(actc, actc), posc, rewc)
                else:
                    contra_entry = (None, None, None)

                rel = idx - state_id[1]
                label = f"Step{rel:+d}" if rel != 0 else "Fork"
                logging.info(f"{label}: TRUE: {true_entry} | CONTRA: {contra_entry}")
        except Exception:
            pass
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

