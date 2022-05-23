import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import *
from stable_baselines3 import DQN
import os
import random
from collections import deque
from os.path import exists

import matplotlib.pyplot as plt
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.app.play_human import play_human
from torch import nn
from resnet import *

from util import *

torch.manual_seed(42)
torch.random.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDQNSolver(nn.Module):
    def __init__(self, output_dim, resnet):
        super().__init__()

        if not resnet:
            self.online = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=32,
                          kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64,
                          kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(7744, 512),
                nn.ReLU(),
                nn.Linear(512, output_dim),
            )

        else:
            self.online = ResNet(
                4, ResBlock, [2, 2, 2, 2], useBottleneck=False, outputs=output_dim)

        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False
            
        print("DDQN model initialized")
        print("Online model:")
        print(self.online)

    def forward(self, input, model):
        return self.online(input) if model == "online" else self.target(input)


class DDQNAgent:
    def __init__(self, action_dim, save_directory, resnet=False):
        self.action_dim = action_dim
        self.save_directory = save_directory
        self.net = DDQNSolver(self.action_dim, resnet).to(device)
        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.999
        self.exploration_rate_min = 0.01
        self.current_step = 0
        self.maxlen_memory = 60000
        self.memory = Memory(self.maxlen_memory)
        self.batch_size = 100
        self.gamma = 0.95
        self.sync_period = 1e4
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=0.0005, eps=1e-4)
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
        filename = os.path.join(self.save_directory,
                                "episode_rewards_plot.png")
        if exists(filename):
            plt.savefig(filename, format="png")
        with open(filename, "w"):
            plt.savefig(filename, format="png")
        plt.clf()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.net.load_state_dict(checkpoint['model'])
        self.exploration_rate = checkpoint['exploration_rate']

    def save_checkpoint(self):
        filename = os.path.join(self.save_directory, 'checkpoint.pth')
        torch.save(dict(model=self.net.state_dict(),
                        exploration_rate=self.exploration_rate), f=filename)
        print('Checkpoint saved to \'{}\''.format(filename))

    def remember(self, state, next_state, action, reward, done):
        done = 1 if done else 0
        state = torch.tensor(state.__array__())
        sample = (state, torch.tensor(next_state.__array__()),
                  torch.tensor([action]), torch.tensor([reward]), torch.tensor([done]))

        with torch.no_grad():
            pred = self.net(state.unsqueeze(dim=0).to(device), "online")
            error = torch.abs(pred[0, action.to(device)] - reward.to(device))
        
        self.memory.append(error, sample)

    def gradient_descent(self):
        idxs, is_weights, sampled = self.recall()

        state, next_state, action, reward, done = sampled

        q_estimate = self.net(state.cuda(), model="online")[
            np.arange(0, self.batch_size), action.cuda()]
        with torch.no_grad():
            best_action = torch.argmax(
                self.net(next_state.cuda(), model="online"), dim=1)
            next_q = self.net(next_state.cuda(), model="target")[
                np.arange(0, self.batch_size), best_action]
            q_target = (reward.cuda() + (1 - done.cuda().float())
                        * self.gamma * next_q).float()

        # update priorities
        error = torch.abs(q_estimate - q_target)
        for i in range(self.batch_size):
            self.memory.update(idxs[i], error[i])

        self.optimizer.zero_grad()

        # loss incorporates importance sampling weights
        loss = self.loss(q_estimate, q_target) * (error * is_weights.cuda()).mean()
        loss.backward()
        
        self.optimizer.step()

    def experience_replay(self, step_reward):
        self.current_episode_reward += step_reward
        if (self.current_step % self.sync_period) == 0:
            self.net.target.load_state_dict(self.net.online.state_dict())

        if len(self.memory) < self.batch_size:
            return

        self.gradient_descent()

    def recall(self):
        batch, idxs, is_weight = self.memory.sample(self.batch_size)

        idxs = torch.tensor(idxs)
        is_weight = torch.tensor(is_weight)

        state = torch.stack([sample[0] for sample in batch])
        next_state = torch.stack([sample[1] for sample in batch])
        action = torch.stack([sample[2] for sample in batch])
        reward = torch.stack([sample[3] for sample in batch])
        done = torch.stack([sample[4] for sample in batch])
        return idxs, is_weight, (state, next_state, action.squeeze(), reward.squeeze(), done.squeeze())

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            action_values = self.net(torch.tensor(
                state.__array__()).cuda().unsqueeze(0), model="online")
            action = torch.argmax(action_values, dim=1).item()
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(
            self.exploration_rate_min, self.exploration_rate)
        self.current_step += 1
        return action


def loop(env, agent, num_episodes=40000, checkpoint_period=20, render=False):
    episode = 0
    for _ in range(num_episodes):
        try:
            state = env.reset()
            done = False
            reward_per_episode = 0
            while not done:  # what happens during every episode

                action = agent.act(state)

                if render:
                    env.render()

                next_state, reward, done, info = env.step(action)

                agent.remember(state, next_state, action, reward, done)
                agent.experience_replay(reward)

                state = next_state
                reward_per_episode += reward

                if done:
                    agent.log_episode()
                    episode += 1

                    if episode % checkpoint_period == 0:
                        agent.save_checkpoint()
                        agent.log_period(
                            episode, agent.exploration_rate, agent.current_step, checkpoint_period)

        except KeyboardInterrupt:
            render = not render


def train_with_demonstration(resnet=True):
    env = setup_environment(actions=SIMPLE_MOVEMENT, skip=3, second=False)

    # collect replay experiences
    buffer = play_human(env)

    agent = DDQNAgent(action_dim=env.action_space.n,
                      save_directory="checkpoints", resnet=resnet)
    
    for experience in buffer:
        agent.remember(experience[0], experience[1], experience[2], experience[3], experience[4])

    k = len(agent.memory)
    print(k)

    for _ in range(k):
        agent.gradient_descent()

    print("Demonstration finished")
    sweat(agent=agent)


def sweat(agent=None):
    env = setup_environment(actions=SIMPLE_MOVEMENT, skip=3, second=False)

    checkpoint_period = 20
    save_directory = "checkpoints"
    load_checkpoint = None

    if agent is None:
        agent = DDQNAgent(action_dim=env.action_space.n,
                          save_directory=save_directory, resnet=True)
    if load_checkpoint is not None and exists(save_directory + "/" + load_checkpoint):
        agent.load_checkpoint(save_directory + "/" + load_checkpoint)

    num_episodes = 40000

    loop(env, agent, num_episodes, checkpoint_period, render=True)


def play():
    env = setup_environment(actions=SIMPLE_MOVEMENT, skip=3, second=False)
    save_directory = "checkpoints"
    load_checkpoint = "checkpoint.pth"
    agent = DDQNAgent(action_dim=env.action_space.n,
                      save_directory=save_directory)
    if load_checkpoint is not None and exists(save_directory + "/" + load_checkpoint):
        agent.load_checkpoint(save_directory + "/" + load_checkpoint)

    x_s = list()

    for _ in range(10000):

        state = env.reset()
        done = False

        while not done:
            action = agent.act(state)
            env.render()

            next_state, _, done, _ = env.step(action)
            state = next_state

            x_s.append(state)

    # write x_s to file
    np.save("x_s.npy", x_s)


def stable_baselines():
    env = setup_environment()
    env = DummyVecEnv([lambda: env])

    policy_kwargs = dict(net_arch=[2048, 2048])
    model = DQN.load("dqn_model")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()

        if done:
            print(info)
            break


if __name__ == "__main__":
    # changed so no individual stages
    train_with_demonstration()
