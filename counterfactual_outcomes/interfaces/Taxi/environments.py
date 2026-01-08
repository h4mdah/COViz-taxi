import gymnasium as gym
from gymnasium.envs.registration import register
from typing import Dict, Any, Union
from gymnasium import Wrapper

class TaxiEnvWrapper(gym.Wrapper):
    """Wrapper to expose the Gymnasium new API for Taxi-v3 and ensure state access."""
    metadata = {"render_modes": ["rgb_array"]}
    
    def __init__(self, **kwargs):
        env = gym.make('Taxi-v3', render_mode = 'rgb_array')
        super().__init__(env)
        self.env_id = 'Taxi-v3'
        self.metadata = getattr(env, 'metadata', {"render_modes": ["rgb_array"]})

    @property
    def s(self):
        return self.env.unwrapped.s

    @s.setter
    def s(self, value):
        self.env.unwrapped.s = value

    def reset(self, **kwargs):
        """Return either (obs, info) if inner env supports new API, else (obs, {})"""
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            return res
        return res, {}

    def step(self, action):
        """Return (obs, reward, terminated, truncated, info)."""
        res = self.env.step(action)
        if isinstance(res, tuple) and len(res) == 5:
            return res
        obs, reward, done, info = res
        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info

    def __getstate__(self):
        state = self.__dict__.copy()
        state['unwrapped_state_s'] = self.env.unwrapped.s
        del state['env']
        return state
    
    def __setstate__(self, state):
        unwrapped_state_s = state.pop('unwrapped_state_s')
        self.__dict__.update(state)
        self.env = gym.make(self.env_id, render_mode='rgb_array')
        self.env.reset()
        self.env.unwrapped.s = unwrapped_state_s


register(id='Taxi-v3-COViz', entry_point='counterfactual_outcomes.interfaces.Taxi.environments:TaxiEnvWrapper')
