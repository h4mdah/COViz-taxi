import gymnasium as gym
from gymnasium.envs.registration import register
from typing import Dict, Any, Union
from gymnasium import Env

class TaxiEnvWrapper(Env):
    """Wrapper to expose the Gymnasium new API for Taxi-v3."""
    def __init__(self, **kwargs):
        self.env = gym.make('Taxi-v3')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        # keep metadata/reward_range if present
        self.metadata = getattr(self.env, "metadata", {})
        self.reward_range = getattr(self.env, "reward_range", None)

    def reset(self, **kwargs):
        """Return either (obs, info) if inner env supports new API, else (obs, {})"""
        res = self.env.reset(**kwargs)
        # gymnasium new API returns (obs, info)
        if isinstance(res, tuple) and len(res) == 2:
            return res
        # old Gym returns obs only
        return res, {}

    def step(self, action):
        """Return (obs, reward, terminated, truncated, info)."""
        res = self.env.step(action)
        # gymnasium new API returns 5-tuple
        if isinstance(res, tuple) and len(res) == 5:
            return res
        # old Gym returns 4-tuple (obs, reward, done, info) -> map to new API
        obs, reward, done, info = res
        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode='rgb_array'):
        return self.env.render()

    def close(self):
        return self.env.close()

register(id='Taxi-v3-COViz', entry_point='counterfactual_outcomes.interfaces.Taxi.environments:TaxiEnvWrapper')
