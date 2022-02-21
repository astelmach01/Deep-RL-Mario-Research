import copy
import os
import random
from collections import deque
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from gym.wrappers import *
from nes_py.wrappers import JoypadSpace
from torch import nn
from torch.distributions import *

from util import *

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
        if input.ndim == 3:
            input = input.unsqueeze(0)
        return self.online(input) if model == "online" else self.target(input)


class DDQNAgent:
    def __init__(self, action_dim, save_directory):
        self.action_dim = action_dim
        self.save_directory = save_directory
        self.net = DDQNSolver(self.action_dim).cuda()
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.999
        self.exploration_rate_min = 0.01
        self.current_step = 0  # how many times we have chosen an action
        self.a = 0.6
        self.b = 0.4

        self.batch_size = 64  # batch size for experience replay
        self.gamma = 0.95
        self.sync_period = 10000  # how many times we update the target network
        self.memory_collection_size = self.batch_size  # how many experiences we will store before performing gradient descent
        self.maxlen_memory = 60000  # max length of memory

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025, eps=1e-4)
        self.loss = torch.nn.MSELoss()
        self.memory = deque(maxlen=self.maxlen_memory)
        self.weights = deque(maxlen=self.maxlen_memory)
        self.dummy_memory = deque(maxlen=self.maxlen_memory)

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

        state = torch.tensor(state.__array__())
        next_state = torch.tensor(next_state.__array__())

        priority = None

        if done:
            priority = reward

        else:
            q_estimate, q_target = self.compute_td_error(state, next_state, action, reward, done)
            td_error = q_estimate - q_target
            priority = torch.abs(td_error) + 1e-5

        if len(self.memory) == 0:
            self.weights.append(np.array([1]))
        else:
            temp = (priority.item() ** self.a) / np.sum(self.weights)
            self.weights.append(temp)

        to_add = state, next_state, torch.tensor([action]), torch.tensor([reward]), torch.tensor([done])
        self.memory.append(to_add)
        self.dummy_memory.append(hash(to_add))

    def recall(self):
        state, next_state, action, reward, done = map(torch.stack,
                                                      zip(*random.choice(self.memory, weights=self.weights,
                                                                         k=self.batch_size)))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    # these args should not be of batch size
    def compute_td_error(self, state, next_state, action, reward, done):

        q_estimate = self.net(state.cuda(), model="online").squeeze()[action]

        with torch.no_grad():
            best_action = torch.argmax(self.net(next_state.cuda(), model="online"))
            next_q = self.net(next_state.cuda(), model="target").squeeze()[best_action]
            q_target = (reward + (1 - done) * self.gamma * next_q).float()

        return q_estimate, q_target

    def experience_replay(self, step_reward):
        self.current_episode_reward += step_reward
        if (self.current_step % self.sync_period) == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        if len(self.memory) < self.memory_collection_size:
            return

        s = np.sum(self.weights)[0]
        self.weights = deque(self.weights / s, maxlen=self.maxlen_memory)

        weight = 0
        td = 0
        q_est, q_t = 0, 0

        # TODO: optimize with vectorization
        for _ in range(self.batch_size):
            # so we dont choose the same
            temp = list(zip(self.memory, self.weights))
            random.shuffle(temp)
            self.memory, self.weights = zip(*temp)
            self.memory = deque(self.memory, maxlen=self.maxlen_memory)
            self.weights = deque(self.weights, maxlen=self.maxlen_memory)

            x = random.choices(self.dummy_memory, weights=self.weights, k=1)
            j = self.dummy_memory.index(x[0])
            state, next_state, action, reward, done = self.memory[j]

            # get index of this experience and its corresponding weight
            p_j = self.weights[j]

            # compute importance sampling weight
            importance = ((len(self.memory) * p_j) ** -min(self.b, 1)) / np.max(self.weights)

            # compute td error
            q_estimate, q_target = self.compute_td_error(state, next_state, action.squeeze(), reward.squeeze(),
                                                         done.squeeze())
            td_error = torch.abs(q_target - q_estimate).cpu().detach().numpy()

            # update transition priority
            self.weights[j] = td_error

            # accumulate weight change
            weight += importance * td_error
            td += td_error

            q_est += q_estimate
            q_t += q_target

        self.optimizer.zero_grad()

        loss = self.loss(q_est, q_t) * torch.cuda.FloatTensor(np.array(weight))
        loss.backward()

        self.optimizer.step()

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            action_values = self.net(torch.tensor(state.__array__()).cuda(), model="online")
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


def train():
    env = setup_environment()
    episode = 0
    checkpoint_period = 30
    save_directory = "checkpoints"
    load_checkpoint = None
    agent = DDQNAgent(action_dim=env.action_space.n, save_directory=save_directory)
    if load_checkpoint is not None and exists(save_directory + "/" + load_checkpoint):
        agent.load_checkpoint(save_directory + "/" + load_checkpoint)

    while True:
        state = env.reset()
        while True:
            action = agent.act(state)
            
            if episode > 10000:
                env.render()

            next_state, reward, done, info = env.step(action)

            done = 1 if done else 0

            agent.remember(state, next_state, action, reward, done)
            agent.experience_replay(reward)

            state = next_state
            if done:
                if agent.b < 1:
                    agent.b += 1 / 100000
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

                print("B: " + str(agent.b))
                break


def play():
    env = setup_environment()
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
    train()
