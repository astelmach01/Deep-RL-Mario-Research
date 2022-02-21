import numpy as np
import gym
from gym.wrappers import *
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from torchvision import transforms
import torch
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_AND_JUMP, TEST

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
    
def setup_environment(actions=SIMPLE_MOVEMENT, skip=4, world='1', level='1'):
    env = gym_super_mario_bros.make('SuperMarioBros-' + str(world) + '-' + str(level) + '-v0')
    env = JoypadSpace(env, actions)
    env = FrameStack(ResizeObservation(GrayScaleObservation(
    SkipFrame(env, skip)), shape=84), num_stack=4)
    env.seed(42)
    env.action_space.seed(42)
    
    return env