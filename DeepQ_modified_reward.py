import copy
import os
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
        self.current_step = 0
        self.maxlen_memory = 70000
        self.memory = deque(maxlen=self.maxlen_memory)
        self.batch_size = 64
        self.gamma = 0.95
        self.sync_period = 1e4
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025, eps=1e-4)
        self.loss = torch.nn.SmoothL1Loss()
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

    def remember(self, state, next_state, action, reward, done):
        self.memory.append((torch.tensor(state.__array__()), torch.tensor(next_state.__array__()),
                            torch.tensor([action]), torch.tensor([reward]), torch.tensor([done])))

    def experience_replay(self, step_reward):
        self.current_episode_reward += step_reward
        if (self.current_step % self.sync_period) == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())
        if self.batch_size > len(self.memory):
            return
        state, next_state, action, reward, done = self.recall()
        q_estimate = self.net(state.cuda(), model="online")[np.arange(0, self.batch_size), action.cuda()]
        with torch.no_grad():
            best_action = torch.argmax(self.net(next_state.cuda(), model="online"), dim=1)
            next_q = self.net(next_state.cuda(), model="target")[np.arange(0, self.batch_size), best_action]
            q_target = (reward.cuda() + (1 - done.cuda().float()) * self.gamma * next_q).float()
        loss = self.loss(q_estimate, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def recall(self):
        state, next_state, action, reward, done = map(torch.stack,
                                                      zip(*random.sample(self.memory, self.batch_size)))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

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



def sweat():
    env = setup_environment()
    episode = 0
    checkpoint_period = 50
    save_directory = "checkpoints"
    load_checkpoint = None
    agent = DDQNAgent(action_dim=env.action_space.n, save_directory=save_directory)
    if load_checkpoint is not None:
        agent.load_checkpoint(save_directory + "/" + load_checkpoint)

    while True:
        state = env.reset()
        prev_score = 0
        while True:
            action = agent.act(state)
            env.render()
            next_state, reward, done, info = env.step(action)
            
            # modify reward
            reward += (info["score"] - prev_score) / 100
            prev_score = info["score"]
            
            agent.remember(state, next_state, action, reward, done)
            agent.experience_replay(info["x_pos"]) # log average x_pos
            
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
    env = setup_environment()
    save_directory = "mario_ql"
    load_checkpoint = "checkpoint.pth"
    agent = DDQNAgent(action_dim=env.action_space.n, save_directory=save_directory)
    if load_checkpoint is not None:
        agent.load_checkpoint('mario_ql\checkpoint_7400.pth')

    while True:
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)

            next_state, reward, done, info = env.step(action)
            state = next_state
            env.render()


if __name__ == "__main__":
    sweat()
