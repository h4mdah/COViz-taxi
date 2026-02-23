import numpy as np
import gymnasium as gym
from highway_env.envs import HighwayEnv
from gymnasium.envs.registration import register
from highway_env.utils import lmap
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.envs.common.action import Action


class Plain(HighwayEnv):
    """rewarded for driving in parallel to a car"""
    metadata = {"render_modes": ["rgb_array"]}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "keep_distance_reward": 1.0,
            "high_speed_reward": 0.4,
            "collision_reward": -1.0,
            "reward_speed_range": [20, 30],
        })
        return config

    def _reward(self, action: Action) -> float:
        obs = self.observation_type.observe()
        other_cars = obs[1:]
        dist_closest_car_in_lane = [x[1] for x in other_cars if x[1] > 0 and abs(x[2]) <= 0.05]
        scaled_speed = lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])

        # safety distance from car in same lane
        if not dist_closest_car_in_lane or dist_closest_car_in_lane[0] > 0.02:
            keeping_distance = 1
        else:
            keeping_distance = -1

        reward = \
            + self.config["keep_distance_reward"] * keeping_distance \
            + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
            + self.config["collision_reward"] * self.vehicle.crashed

        reward = -10 if not self.vehicle.on_road else reward
        return reward


register(
    id='Plain-v0',
    entry_point='counterfactual_outcomes.interfaces.Highway.environments:Plain',
)

class HighwayEnvWrapper(gym.Wrapper):
    """Wrapper to ensure consistent COViz API for Highway environments."""
    metadata = {"render_modes": ["rgb_array"]}
    def __init__(self, env_id='Plain-v0', **kwargs):
        # ensure render_mode is set for COViz visualization
        if 'render_mode' not in kwargs:
            kwargs['render_mode'] = 'rgb_array'
        env = gym.make(env_id, **kwargs)
        super().__init__(env)
        self.env_id = env_id

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            return res
        return res, {}

    def step(self, action):
        res = self.env.step(action)
        if isinstance(res, tuple) and len(res) == 5:
            return res
        obs, reward, done, info = res
        return obs, reward, bool(done), False, info

    def __getstate__(self):
        state = self.__dict__.copy()
        # Highway environments often have complex internal state; 
        # for COViz we mostly care about the wrapper state and underlying env recreation
        del state['env']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.env = gym.make(self.env_id, render_mode='rgb_array')
        self.env.reset()

register(
    id='Highway-v0-COViz',
    entry_point='counterfactual_outcomes.interfaces.Highway.environments:HighwayEnvWrapper',
    kwargs={'env_id': 'Plain-v0'}
)