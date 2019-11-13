# Gossip based actor
# Distral
# Attentive Network
# A3C


import tensorflow as tf
import gym

import argparse
import sys
from gym import wrappers, logger

# import deepmind_lab
import numpy as np


class RandomAgent:
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def print_environment(env):
    print("action space:", env.action_space)
    print("observation space:", env.observation_space)
    print("    upper bound:", env.observation_space.high)
    print("    lower bound", env.observation_space.low)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    for i_episode in range(20):
        observation = env.reset()
        reward = 0
        done = False
        for t in range(100):
            # env.render()
            agent = RandomAgent(env.action_space)
            print(observation)
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()


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


def main():
    pass
