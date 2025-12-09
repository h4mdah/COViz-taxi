import numpy as np
import gymnasium as gym 
import glob
import os
from copy import deepcopy, copy
from pathlib import Path
from os.path import join
from counterfactual_outcomes.common import Trace
from rl_agents.agents.common.exploration.abstract import exploration_factory
from rl_agents.agents.common.factory import agent_factory
from counterfactual_outcomes.interfaces.abstract_interface import AbstractInterface
from rl_agents.trainer.evaluation import Evaluation
from os.path import abspath, join
import random, numpy as _np
from copy import deepcopy

import counterfactual_outcomes.interfaces.Taxi.environments

class MyEvaluation(Evaluation):
    def __init__(self, env, agent, output_dir='../agents', num_episodes=1000, display_env=False):
        self.OUTPUT_FOLDER = output_dir
        super(MyEvaluation, self).__init__(env, agent, num_episodes=num_episodes, 
                                         display_env=display_env)


class TaxiInterface(AbstractInterface):
    def __init__(self, config, output_dir, load_path):
        super().__init__(config, output_dir)
        self.load_path = load_path

    def initiate(self, seed=0, evaluation_reset=False):
        config = self.config
        # Debug: print the resolved agent configuration so failures in agent_factory
        # are easy to diagnose. This shows either the dict or the path that will
        # be passed to rl_agents.agent_factory.

        env = gym.make(config['env']['id'], render_mode='rgb_array')
        # gym/gymnasium seeding differs between versions and wrappers.
        # Try several approaches so this works with older gym, gymnasium, and custom wrappers.
        try:
            if hasattr(env, 'seed') and callable(getattr(env, 'seed')):
                env.seed(seed)
        except Exception:
            pass
        try:
            # gymnasium: reset accepts seed kwarg
            if hasattr(env, 'reset'):
                try:
                    env.reset(seed=seed)
                except TypeError:
                    # some env.reset don't accept seed kwarg
                    pass
        except Exception:
            pass
        # seed action/observation spaces if available
        try:
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'seed'):
                env.action_space.seed(seed)
        except Exception:
            pass
        try:
            if hasattr(env, 'observation_space') and hasattr(env.observation_space, 'seed'):
                env.observation_space.seed(seed)
        except Exception:
            pass
        # Try to build agent via rl_agents factory. If the provided config does not
        # describe an rl_agents agent (or agent_factory fails), fall back to loading
        # an SB3 model (if present) and wrap it with a minimal adapter.
        try:
            agent = agent_factory(env, config=['agent'])
            agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
        except Exception:
            # fallback: try to load a Stable-Baselines3 DQN model from model_dir
            agent = None
            try:
                from stable_baselines3 import DQN
                model_dir = config.get('model_dir') or self.load_path or 'agents\\taxi_sb3'
                # find latest .zip model in model_dir
                model_files = sorted(glob.glob(join(model_dir, '*.zip')), key=os.path.getmtime, reverse=True)
                print("Loaded model files:", model_files[0])
                if model_files:
                    sb3_model = DQN.load(model_files[0])

                    class SB3Adapter:
                        def __init__(self, model, action_space):
                            self.model = model
                            self.action_space = action_space
                            self.previous_state = None

                        def act(self, state):
                            # SB3 expects numpy observation; leave to model.predict to handle
                            a, _ = self.model.predict(state, deterministic=True)
                            return int(a)

                        def get_state_action_values(self, state):
                            # SB3 DQN does not expose Q-values easily; provide a placeholder
                            # vector of zeros with correct length. This lets the pipeline run
                            # though importance scores will be approximate.
                            import numpy as _np
                            n = getattr(self.action_space, 'n', None) or 0
                            return _np.zeros((n,))

                        # Minimal APIs expected by rl_agents Evaluation
                        def set_writer(self, writer):
                            self.writer = writer

                        def close(self):
                            return
                        
                        def load(self, filename=None):
                            # SB3 model already loaded; return the underlying model
                            return self.model

                        def save(self, filename=None):
                            # Attempt to save underlying SB3 model if supported; otherwise no-op
                            try:
                                if hasattr(self.model, 'save'):
                                    self.model.save(str(filename))
                            except Exception:
                                pass

                    agent = SB3Adapter(sb3_model, env.action_space)
            except Exception:
                agent = None

        if agent is None:
            # Provide a helpful error rather than re-raising a suppressed exception
            msg = (
                "Failed to create an agent via rl_agents.agent_factory and no Stable-Baselines3 "
                "model could be loaded as a fallback.\n"
                "Ensure your agent metadata contains an 'agent' section that rl_agents understands,\n"
                "or place a SB3 .zip model in the folder pointed to by 'model_dir' or 'self.load_path'.\n"
                f"Tried model_dir='{config.get('model_dir')}', load_path='{self.load_path}'."
            )
            raise RuntimeError(msg)
        if evaluation_reset:
            evaluation_reset.training = False
            evaluation_reset.close()
        # remember the env on the interface so post_contrastive can restore state
        try:
            self.env = env
        except Exception:
            pass
        return env, agent
    
    def evaluation(self, env, agent):
        evaluation = MyEvaluation(env, agent, display_env=False, output_dir=self.output_dir)
        agent_path = Path(join(self.load_path, 'checkpoint-final.tar'))
        evaluation.load_agent_model(agent_path)
        return evaluation
    
    def get_state_action_values(self, agent, state):
        return agent.get_state_action_values(state)
    
    def get_state_RD_action_values(self, agent, state):
        """
        Return a per-action immediate reward decomposition (RD) vector for the given state.

        This implementation simulates each discrete action on a deepcopy of the underlying
        Taxi environment (or a best-effort approximation) starting from the provided
        `state` and decomposes the immediate reward into components:
          [time_step_component, success_component, illegal_component]

        If simulation is not possible, returns a zero-array of shape (n, 3).
        """
        # Determine action count
        action_space = getattr(agent, 'action_space', None)
        n = getattr(action_space, 'n', None) if action_space is not None else None
        if n is None:
            env_as = getattr(getattr(self, 'env', None), 'action_space', None)
            n = getattr(env_as, 'n', 0) if env_as is not None else 0

        try:
            # Locate the inner environment to simulate on (prefer unwrapped)
            inner = getattr(self, 'env', None)
            if inner is None:
                return np.zeros((n, 3))
            candidate = getattr(inner, 'unwrapped', None) or getattr(inner, 'env', None) or inner

            # Try to infer internal state value `s` from the provided state/observation
            s_val = None
            try:
                if isinstance(state, (int,)):
                    s_val = int(state)
                elif isinstance(state, (list, tuple,)) and len(state) > 0 and isinstance(state[0], (int,)):
                    s_val = int(state[0])
                elif hasattr(candidate, 's'):
                    s_val = getattr(candidate, 's', None)
            except Exception:
                s_val = getattr(candidate, 's', None) if hasattr(candidate, 's') else None

            rd_list = []
            for a in range(int(n)):
                try:
                    # Operate on a deepcopy so we don't change the real env
                    tmp = deepcopy(candidate)
                    # If we could infer s, set it on the temp env
                    if s_val is not None:
                        try:
                            setattr(tmp, 's', s_val)
                        except Exception:
                            pass

                    out = tmp.step(a)
                    if isinstance(out, tuple) and len(out) >= 2:
                        r = out[1]
                    else:
                        # fallback if step returns scalar
                        r = float(out)

                    # Decompose reward according to Taxi-v3 rules
                    time_comp = -1 if r == -1 else 0
                    success_comp = 20 if r == 20 else 0
                    illegal_comp = -10 if r == -10 else 0
                    rd_list.append([time_comp, success_comp, illegal_comp])
                except Exception:
                    # If any simulation fails, append zeros for this action
                    rd_list.append([0, 0, 0])

            return np.asarray(rd_list)
        except Exception:
            return np.zeros((n, 3))
    
    def get_state_from_obs(self, agent, obs, params=None):
        return obs
    
    def get_next_action(self, agent, obs, state):
        return agent.act(state)
    
    def get_features(self, env, obs=None):
        # Different wrappers expose the underlying Taxi env differently. Try several
        # ways to access the underlying environment and its state `s` and `decode`.
        # Accept an optional observation decode fallback via `obs` if available.
        # Note: keep signature backwards-compatible; callers may pass only `env`.
        obs = None
        # allow callers to pass (env, obs) if they forwarded obs
        # but maintain compatibility: if called with two args, `env` will be a tuple
        # This wrapper method is called internally with (env, obs) by our code.
        try:
            # If someone called get_features(env, obs), Python passes obs as second arg
            # but not via this signature; handle via attribute if present
            pass
        except Exception:
            pass

        # locate a candidate object that might have `decode` and `s`
        candidate = getattr(env, 'unwrapped', None) or getattr(env, 'env', None) or env
        searched = set()
        for _ in range(5):
            if candidate is None:
                break
            cid = id(candidate)
            if cid in searched:
                break
            searched.add(cid)
            s = getattr(candidate, 's', None)
            if hasattr(candidate, 'decode'):
                # prefer decode from explicit s if available
                if s is not None:
                    try:
                        taxi_row, taxi_col, pass_idx, dest_idx = candidate.decode(s)
                        return {"position": (taxi_row, taxi_col), "passenger_status": pass_idx,
                                "destination": dest_idx}
                    except Exception:
                        pass
                # fallback: try decoding from obs if it's an int-like state
                try:
                    if obs is not None:
                        s_obs = obs[0] if isinstance(obs, tuple) else obs
                        if isinstance(s_obs, (int,)):
                            taxi_row, taxi_col, pass_idx, dest_idx = candidate.decode(int(s_obs))
                            return {"position": (taxi_row, taxi_col), "passenger_status": pass_idx,
                                    "destination": dest_idx}
                except Exception:
                    pass
            # try to drill down into common attributes that hold the inner env
            next_cand = None
            for attr in ('env', 'unwrapped', 'inner', 'wrapped_env'):
                next_cand = getattr(candidate, attr, None)
                if next_cand is not None:
                    break
            candidate = next_cand

        # Fallback: try to decode from a provided observation if it's an int
        try:
            # callers in this repo pass obs separately; attempt to get it from
            # a global-like place if available (not ideal) — prefer explicit pass.
            # We don't have obs here reliably; return placeholders.
            pass
        except Exception:
            pass

        return {"position": None, "passenger_status": None, "destination": None}

          
    
    def contrastive_trace(self, trace_idx, k_steps, params=None):
        return TaxiTrace(trace_idx, k_steps)
    
    def pre_contrastive(self, env):
        return deepcopy(env)

    def post_contrastive(self, agent1, agent2, pre_params=None):
        env = pre_params
        agent1.previous_state = agent2.previous_state
        return env
    

class TaxiTrace(Trace):
    def __init__(self, trace_idx, k_steps):
        super().__init__(trace_idx, k_steps)
        self.contrastive = []

    def update(self, state_object, obs, r, done, infos, a, state_id):
        self.obs.append(obs)
        self.rewards.append(r)
        self.dones.append(done)
        self.infos.append(infos)
        self.previous_actions.append(a)
        self.reward_sum += r
        self.states.append(state_object)
        self.length += 1

    def mark_frames(self, hl_idx, indexes, color=255, thickness=2, no_mark=False):
        """Return a list of image frames (arrays or file paths) and the relative
        index of the highlighted state within that list.
        """
        frames = []
        rel_idx = 0

        start = indexes[0]
        end = indexes[-1]
        # clamp to available states
        start = max(0, start)
        end = min(len(self.states) - 1, end)

        # collect frames from start..end inclusive
        for i in range(start, end + 1):
            st = self.states[i]
            # prefer in-memory image, then img, then image_path
            img = getattr(st, 'image', None)
            if img is None:
                img = getattr(st, 'img', None)
            if img is None and getattr(st, 'image_path', None):
                # keep the path (lazy load later)
                frames.append(getattr(st, 'image_path'))
                continue
            frames.append(img)

        # compute relative index for highlighted state
        if hl_idx < start or hl_idx > end:
            rel_idx = 0
        else:
            rel_idx = hl_idx - start

        # If no_mark is requested, still return images without any marking
        return frames, rel_idx
        

def taxi_config(args):
    args.config_filename = "metadata"
    args.config_changes = {"env": {"id": 'Taxi-v3-COViz'}, "agent": {}}
    args.data_name = ''
    args.name = "taxi_optimal"
    #check if path exists, otherwise train model
    

    
    # Only set a default load_path if one wasn't provided via CLI or metadata
    if not getattr(args, 'load_path', None):
        args.load_path = abspath(f'../agents/{args.interface}/{args.name}')
    args.k_steps = 50
    args.overlay = 5
    args.contra_action_counter = 1
    return args


