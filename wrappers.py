import numpy as np
import torch
import gym

from gym.spaces import Box
from torchvision import transforms as T

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunc, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation

class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_strategy):
        super(CustomRewardWrapper, self).__init__(env)
        self.reward_strategy = reward_strategy
        self.last_x_pos = 0

    def reset(self, **kwargs):
        self.last_x_pos = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        current_x_pos = info.get('x_pos', 0)
        points = info.get('score', 0)

        if self.reward_strategy == 'points':
            # Reward based on points
            reward = points
        elif self.reward_strategy == 'x_pos':
            # Reward based on x position
            distance_reward = current_x_pos - self.last_x_pos
            reward = distance_reward
        elif self.reward_strategy == 'hybrid':
            # Hybrid approach
            distance_reward = current_x_pos - self.last_x_pos
            hybrid_reward = distance_reward + points
            reward = hybrid_reward
        
        self.last_x_pos = current_x_pos
        return obs, reward, done, trunc, info
    