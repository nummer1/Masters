import tensorflow as tf
import gym


class RandomAgent:
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


if __name__ == '__main__':
    # env = gym.make('Copy-v0')
    env = gym.make('FrozenLake-v0')
    env.reset()

    agent = RandomAgent(env.action_space)
    for i in range(10):
        action = agent.act(env, 0, False)
        ob, reward, done, info = env.step(action)
        print(ob, reward, done, info)
        if done:
            break

    env.render(mode='human')
