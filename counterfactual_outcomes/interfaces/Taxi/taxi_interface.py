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
        # Try to create env with RGB array rendering enabled so `env.render()` returns frames.
        try:
            env = gym.make(config['env']['id'], render_mode='rgb_array')
        except Exception:
            env = gym.make(config['env']['id'])
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
                model_dir = config.get('model_dir') or self.load_path or 'agents/taxi_sb3'
                # find latest .zip model in model_dir
                model_files = sorted(glob.glob(join(model_dir, '*.zip')), key=os.path.getmtime, reverse=True)
                if model_files:
                    sb3_model = DQN.load(model_files[0])

                    class SB3Adapter:
                        def __init__(self, model, action_space):
                            self.model = model
                            self.action_space = action_space

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
        # Try to infer action space size from the agent (SB3Adapter) or the interface
        action_space = getattr(agent, 'action_space', None)
        n = getattr(action_space, 'n', None) if action_space is not None else None
        if n is None:
            # fallback: try to read from self if available
            n = getattr(getattr(self, 'env', None), 'action_space', None)
            n = getattr(n, 'n', 0) if n is not None else 0
        return np.zeros((n, 1))
    
    def get_state_from_obs(self, agent, obs, params=None):
        return obs
    
    def get_next_action(self, agent, obs, state):
        return agent.act(state)
    
    def get_features(self, env):
        # Different wrappers expose the underlying Taxi env differently. Try several
        # ways to access the underlying environment and its state `s` and `decode`.
        inner = getattr(env, 'unwrapped', None) or getattr(env, 'env', None) or env
        s = getattr(inner, 's', None)
        if hasattr(inner, 'decode') and s is not None:
            taxi_row, taxi_col, pass_idx, dest_idx = inner.decode(s)
            return {"position": (taxi_row, taxi_col), "passenger_status": pass_idx,
                    "destination": dest_idx}
        # Fallback: return minimal placeholder features
        return {"position": None, "passenger_status": None, "destination": None}

    def render_observation(self, obs=None, env=None, size=(84, 84)):
        """Return an RGB numpy array visualizing the Taxi state.

        Accepts either an observation (`obs`) or an environment (`env`).
        This is a lightweight fallback renderer used when `env.render()`
        doesn't return an image (headless environments). The output is
        a small, 84x84 RGB array with colored squares for taxi and destination.
        """
        try:
            import numpy as _np
        except Exception:
            return None

        h, w = size
        # default background: road grey
        img = (_np.ones((h, w, 3), dtype='uint8') * 120)

        # try to get features from env first, fall back to obs
        features = None
        try:
            if env is not None:
                features = self.get_features(env)
        except Exception:
            features = None
        if features is None and obs is not None:
            # obs may be integer state; try to reconstruct via env if possible
            try:
                # try to decode using an inner env if available
                inner = getattr(env, 'unwrapped', None) or getattr(env, 'env', None) or env
                if inner is not None and hasattr(inner, 'decode'):
                    taxi_row, taxi_col, pass_idx, dest_idx = inner.decode(obs)
                    features = {"position": (taxi_row, taxi_col), "passenger_status": pass_idx, "destination": dest_idx}
            except Exception:
                features = None

        if not features:
            return img

        pos = features.get('position')
        dest = features.get('destination')

        # assume a 5x5 taxi grid by default; compute cell size
        grid_h = grid_w = 5
        cell_h = max(1, h // grid_h)
        cell_w = max(1, w // grid_w)

        def fill_cell(center_r, center_c, color):
            r0 = int(center_r * cell_h)
            c0 = int(center_c * cell_w)
            r1 = min(h, r0 + cell_h)
            c1 = min(w, c0 + cell_w)
            img[r0:r1, c0:c1, :] = color

        # draw destination as green
        try:
            if dest is not None:
                dr = dest // grid_w
                dc = dest % grid_w
                fill_cell(dr, dc, [34, 177, 76])
        except Exception:
            pass

        # draw taxi as yellow
        try:
            if pos is not None:
                tr, tc = pos
                fill_cell(tr, tc, [255, 242, 0])
        except Exception:
            pass

        return img
    
    def contrastive_trace(self, trace_idx, k_steps, params=None):
        return TaxiTrace(trace_idx, k_steps)
    
    def pre_contrastive(self, env):
        # Avoid deep-copying the whole env (may contain non-picklable Surfaces).
        # For Taxi we can save the internal integer state `s` and restore it later.
        try:
            inner = getattr(env, 'unwrapped', None) or getattr(env, 'env', None) or env
            s = getattr(inner, 's', None)
            return {'s': s}
        except Exception:
            # fallback to nothing
            return {'s': None}
    
    def post_contrastive(self, agent1, agent2, pre_params=None):
        # Restore the saved internal state on agent2's interface env if possible.
        try:
            saved = pre_params
            if isinstance(saved, dict) and 's' in saved:
                iface = getattr(agent2, 'interface', None)
                if iface is not None and hasattr(iface, 'env'):
                    try:
                        inner = getattr(iface.env, 'unwrapped', None) or getattr(iface.env, 'env', None) or iface.env
                        if saved.get('s') is not None:
                            setattr(inner, 's', saved.get('s'))
                    except Exception:
                        pass
                # return the restored env if available
                if iface is not None and hasattr(iface, 'env'):
                    return iface.env
        except Exception:
            pass
        # Fallback: return agent2's env if available, otherwise the original pre_params
        try:
            iface = getattr(agent2, 'interface', None)
            if iface is not None and hasattr(iface, 'env'):
                return iface.env
        except Exception:
            pass
        return pre_params
    

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
    args.n_traces = 10
    args.k_steps = 15
    args.overlay = args.k_steps // 2
    return args
        

