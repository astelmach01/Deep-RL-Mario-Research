
from DeepQLearningAgent import *
episodes = 100


def play_q(env, args, actions):
    """Play the game using the Q-learning agent."""
    agent: DDQNAgent = DDQNAgent(env, actions)
    agent.valueIteration()
    for _ in range(episodes):

        environment = None
        if actions is None:
            actions = env.action_space.n
        else:
            environment = JoypadSpace(gym.make(args.env), actions)
            environment.reset()

        done = False
        state, _, _, info, = environment.step(0)
        while not done:
            if done:
                _ = environment.reset()

            action = agent.get_action(state)
            state, _, done, _ = environment.step(action)
            environment.render()

        # close the environment
        env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Breakout-v0')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    env = gym.make(args.env)
    actions = None
    if args.load is not None:
        actions = env.action_space.n
    play_q(env, args, actions)