import copy
import os
from queue import PriorityQueue
import random
from collections import deque
from os.path import exists


import matplotlib.pyplot as plt
import numpy as np
import torch

from gym.wrappers import *
from nes_py.wrappers import JoypadSpace
from torch import nn
from torch.distributions import *

from util import *
import gym_super_mario_bros


torch.manual_seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

class DDQNSolver(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        return self.online(input) if model == "online" else self.target(input)


class DDQNAgent:
    def __init__(self, action_dim, save_directory):
        self.action_dim = action_dim
        self.save_directory = save_directory
        self.net = DDQNSolver(self.action_dim).cuda()
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.999
        self.exploration_rate_min = 0.01
        self.current_step = 0 # how many times we have chosen an action
        self.memory_collection_size = 6000 # how many experiences we will store before performing gradient descent
        self.maxlen_memory = 70000 # max length of memory
        
        self.batch_size = 64 # batch size for experience replay
        self.gamma = 0.95
        self.sync_period = 10000 # how many times we update the target network
        
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=0.00025, momentum=0.95, eps=1e-5)
        self.loss = torch.nn.MSELoss()
        self.memory: PriorityQueue = PriorityQueue(self.maxlen_memory)
        
        self.episode_rewards = []
        self.moving_average_episode_rewards = []
        self.current_episode_reward = 0.0

    def log_episode(self):
        self.episode_rewards.append(self.current_episode_reward)
        self.current_episode_reward = 0.0

    def log_period(self, episode, epsilon, step, checkpoint_period):
        self.moving_average_episode_rewards.append(np.round(
            np.mean(self.episode_rewards[-checkpoint_period:]), 3))
        print(f"Episode {episode} - Step {step} - Epsilon {epsilon} "
              f"- Mean Reward {self.moving_average_episode_rewards[-1]}")
        plt.plot(self.moving_average_episode_rewards)
        filename = os.path.join(self.save_directory, "episode_rewards_plot.png")
        if exists(filename):
            plt.savefig(filename, format="png")
        with open(filename, "w"):
            plt.savefig(filename, format="png")
        plt.clf()

    def compute_td_error(self, state, next_state, reward):
        state_pred = state.cuda().unsqueeze(0)
        next_state_pred = next_state.cuda().unsqueeze(0)
        
        target = self.net(next_state_pred, model="target")
        online = self.net(state_pred, model="online")

        td_error = reward + self.gamma * torch.max(target) - torch.max(online)
        
        return td_error.cpu().detach().numpy()

    def remember(self, state, next_state, action, reward, done):
        
        state = torch.tensor(state.__array__())
        next_state = torch.tensor(next_state.__array__())
        
        td_error = self.compute_td_error(state, next_state, reward)
        
        print(td_error.ndim)
        
        if td_error.ndim == 0:
            td_error = np.atleast_1d(td_error)
            
        priority  = np.abs(td_error[0]) + 1e-7

        item = Experience(state, next_state, action, reward, done, td_error)
        self.memory.put((priority, item))
        
    def recall_td_error_only(self):
        experiences = [self.memory.get()[1] for _ in range(self.batch_size)]
        td_error = list(map(lambda x: x.td_error, experiences))
        return torch.tensor(td_error, dtype=torch.float32)
    
    def recall(self):
        # change to sampling instead of priority queue
        experiences = [self.memory.get()[1] for _ in range(self.batch_size)]
        
        state = list(map(lambda x: x.state, experiences))
        next_state = list(map(lambda x: x.next_state, experiences))
        action = list(map(lambda x: x.action, experiences))
        reward = list(map(lambda x: x.reward, experiences))
        done = list(map(lambda x: x.done, experiences))
        td_error = list(map(lambda x: x.td_error, experiences))
  
        state = torch.cat(state).view(self.batch_size, 4, 84, 84)
        next_state = torch.cat(next_state).view(self.batch_size, 4, 84, 84)
        
        return state, next_state, torch.tensor(action, dtype=torch.int64), torch.tensor(reward, dtype=torch.float32), torch.tensor(done, dtype=torch.bool), torch.tensor(td_error, dtype=torch.float32)

    def experience_replay(self, step_reward):
        self.current_episode_reward += step_reward
        if (self.current_step % self.sync_period) == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())
            
        if self.memory.qsize() < self.memory_collection_size:
            return

        # self.recall()[5] is the TD error
        #loss = weights * loss 
        loss = self.loss(self.recall_td_error_only())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            action_values = self.net(torch.tensor(state.__array__()).cuda().unsqueeze(0), model="online")
            action = torch.argmax(action_values, dim=1).item()
            
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.current_step += 1
        return action

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model'])
        self.exploration_rate = checkpoint['exploration_rate']

    def save_checkpoint(self):
        filename = os.path.join(self.save_directory, 'checkpoint.pth')
        torch.save(dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))

def setup_environment():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, [["right"], ["right", "A"]])
    env = FrameStack(ResizeObservation(GrayScaleObservation(
    SkipFrame(env, skip=4)), shape=84), num_stack=4)
    env.seed(42)
    env.action_space.seed(42)
    
    return env

def sweat():
    env = setup_environment()
    episode = 0
    checkpoint_period = 50
    replay_period = 100
    save_directory = "checkpoints"
    load_checkpoint = 'checkpoint.pth'
    agent = DDQNAgent(action_dim=env.action_space.n, save_directory=save_directory)
    if load_checkpoint is not None:
        agent.load_checkpoint(save_directory + "/" + load_checkpoint)

    while True:
        state = env.reset()
        while True:
            action = agent.act(state)
            env.render()
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, next_state, action, reward, done)
            
            if agent.current_step % replay_period == 0:
                agent.experience_replay(reward)
                
            state = next_state
            if done:
                episode += 1
                agent.log_episode()
                if episode % checkpoint_period == 0:
                    agent.save_checkpoint()
                    agent.log_period(
                        episode=episode,
                        epsilon=agent.exploration_rate,
                        step=agent.current_step,
                        checkpoint_period=checkpoint_period
                    )
                break


def play():
    env=setup_environment()
    save_directory = "checkpoints"
    load_checkpoint = "checkpoint.pth"
    agent = DDQNAgent(action_dim=env.action_space.n, save_directory=save_directory)
    if load_checkpoint is not None:
        agent.load_checkpoint(save_directory + "/" + load_checkpoint)

    while True:
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)

            next_state, _, done, _ = env.step(action)
            state = next_state
            env.render()


if __name__ == "__main__":
    sweat()
