import numpy as np
import gym
from gym.spaces import Box
from pyrsistent import s
from torchvision import transforms
import torch

class Counter(dict):

    def __init__(self, size=1):
        super().__init__()
        self.size = size

    def __getitem__(self, idx):
        idx = str(idx)
        self.setdefault(idx, np.zeros(self.size))
        return dict.__getitem__(self, idx)



class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(
            low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)

    def observation(self, observation):
        transform = transforms.Grayscale()
        result = transform(torch.tensor(np.transpose(
            observation, (2, 0, 1)).copy(), dtype=torch.float))
        return result


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transformations = transforms.Compose(
            [transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        return transformations(observation).squeeze(0)

class Experience:
    
    def __init__(self, state, next_state, action, reward, done, td_error) -> None:
        self.state = state
        self.next_state = next_state
        self.action = action
        self.reward = reward
        self.done = done
        self.td_error = td_error
        
    def __lt__(self, other):
        return self.td_error < other.td_error