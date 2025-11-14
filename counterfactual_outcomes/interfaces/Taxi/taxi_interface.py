import numpy as np
import gymnasium as gym 
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
        env = gym.make(config['env']['id'])
        env.seed(seed)
        agent = agent_factory(env, config=['agent'])
        agent.exploration_policy = exploration_factory({'method': 'Greedy'}, env.action_space)
        if evaluation_reset:
            evaluation_reset.training = False
            evaluation_reset.close()
        return env, agent
    
    def evaluation(self, env, agent):
        evaluation = MyEvaluation(env, agent, display_env=False, output_dir=self.output_dir)
        agent_path = Path(join(self.load_path, 'checkpoint-final.tar'))
        evaluation.load_agent_model(agent_path)
        return evaluation
    
    def get_state_action_values(self, agent, state):
        return agent.get_state_action_values(state)
    
    def get_state_RD_action_values(self, agent, state):
        return np.zeros((env.action_space.n, 1))
    
    def get_state_from_obs(self, agent, obs, params=None):
        return obs
    
    def get_next_action(self, agent, obs, state):
        return agent.act(state)
    
    def get_features(self, env):
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(env.s)
        return {"position": (taxi_row, taxi_col), "passenger_status": pass_idx, 
                "destination": dest_idx}
    
    def contrastive_trace(self, trace_idx, k_steps, params=None):
        return TaxiTrace(trace_idx, k_steps)
    
    def pre_contrastive(self, env):
        return deepcopy(env)
    
    def post_contrastive(self, agent1, agent2, pre_params=None):
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
        frames, rel_idx = [], 0

        if no_mark:
            for i in range(indexes[0], indexes[-1]+1):
                frames.append(self.states[i])
            return frames, rel_idx
        
        for i in range(indexes[0], hl_idx):
            frames.append(self.states[i])
            rel_idx = len(frames)
            marked_frame = copy(self.states[hl_idx])
            frames.append(marked_frame)

        for i in range(indexes[-1]-hl_idx):
            frames.append(self.states[hl_idx+1+1])
            return frames, rel_idx
        

def taxi_config(args):
    args.config_filename = "metadata"
    args.config_changes = {"env": {"id": 'Taxi-v3-COViz'}, "agent": {}}
    args.data_name = ''
    args.name = "taxi_optimal"
    #check if path exists, otherwise train model
    

    
    args.load_path = abspath(f'../agents/{args.interface}/{args.name}')
    args.n_traces = 10
    args.k_steps = 15
    args.overlay = args.k_steps // 2
    return args
        

