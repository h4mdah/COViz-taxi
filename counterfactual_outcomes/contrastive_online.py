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
            features = agent.interface.get_features(env)
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

        # for _ in range(30):  # TODO remove
        while not done:
            logging.debug(f'time-step number: {step}')
            state = agent1.interface.get_state_from_obs(agent1, obs, [r, done])
            s_a_values = agent1.interface.get_state_action_values(agent1, state)
            state_id, frame = (t, step), env1.render()
            features = agent1.interface.get_features(env1)
            state_obj = State(state_id, obs, state, s_a_values, frame, features)
            trace.update(state_obj, obs, r, done, infos, agent1_a, state_id)
            """actions"""
            agent1_a = agent1.interface.get_next_action(agent1, obs, state) if not done else None
            # derive a contrastive action robustly. If state-action values are
            # flat or unavailable (e.g., SB3 fallback returning zeros), pick a
            # deterministic alternative to the agent1 action instead of a
            # seemingly-random choice.
            try:
                vals = np.asarray(s_a_values)
                # if values are non-informative (all equal), fall back
                if vals.size == 0 or np.allclose(vals, vals.flat[0]):
                    # prefer an action different from agent1's action
                    n_actions = vals.size if vals.size > 0 else getattr(getattr(agent1, 'action_space', None), 'n', None)
                    if n_actions is None or n_actions == 0:
                        # last resort: ask agent2 for its preferred action
                        agent2_pref = agent2.interface.get_next_action(agent2, obs, state)
                        agent2_a = agent2_pref if agent2_pref is not None else 0
                    else:
                        base = agent1_a if agent1_a is not None else 0
                        agent2_a = (base + 1) % int(n_actions)
                else:
                    agent2_a = sorted(list(enumerate(vals)), key=lambda x: x[1])[-2][0]
            except Exception:
                # conservative fallback
                try:
                    agent2_a = agent2.interface.get_next_action(agent2, obs, state)
                except Exception:
                    agent2_a = 0
            """contrastive trajectory"""
            pre_vars = agent2.interface.pre_contrastive(env1)
            trace.contrastive.append(
                get_contrastive_trajectory(state_id, trace, env2, agent2, agent2_a, args.k_steps,
                                           args.contra_action_counter))
            """return agent 2 environment to the current state"""
            env2 = agent2.interface.post_contrastive(agent1, agent2, pre_vars)
            # Prefer restoring the exact Taxi state without calling `reset()`
            # because `reset()` may change RNG or other hidden state. If the
            # underlying env exposes `s` (Taxi's integer state), use it as the
            # observation and avoid reset. Fall back to `reset()` only when
            # `s` is not available.
            obs2_reset = None
            try:
                inner = getattr(env2, 'unwrapped', None) or getattr(env2, 'env', None) or env2
                s_val = getattr(inner, 's', None)
                if s_val is not None:
                    obs2_reset = int(s_val)
                    # if rng was saved in pre_vars, try to restore it here
                    if isinstance(pre_vars, dict) and pre_vars.get('type') == 'state_s':
                        rng = pre_vars.get('rng')
                        if rng:
                            try:
                                import random as _rnd, numpy as _npy
                                _rnd.setstate(rng['py']); _npy.random.set_state(rng['np'])
                            except Exception:
                                pass
                else:
                    # last-resort: call reset() so wrappers that enforce reset
                    # before stepping are satisfied
                    res_reset = env2.reset()
                    obs2_reset = res_reset[0] if isinstance(res_reset, tuple) else res_reset
            except Exception:
                try:
                    res_reset = env2.reset()
                    obs2_reset = res_reset[0] if isinstance(res_reset, tuple) else res_reset
                except Exception:
                    obs2_reset = None

            # sync agent previous_state if we obtained an observation
            if obs2_reset is not None:
                agent1.previous_state = agent2.previous_state = obs2_reset
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
        traces.append(trace)
    return traces

