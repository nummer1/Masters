import gym
import ray
import ray.rllib.agents.impala as impala
import tensorflow as tf
import numpy as np

import config


ray.init()
config = config.get_config_impala()

trainer = impala.ImpalaAgent(config=config, env="GuessingGame-v0")
trainer.restore("/home/kasparov/ray_results/IMPALA_GuessingGame-v0_16_16/checkpoint_30/checkpoint-30")
policy = trainer.get_policy()

# print(type(policy))
# print(type(policy.model))
# print(dir(policy))
# print(dir(policy.model))
# print(policy.model.last_layer)

env = gym.make('GuessingGame-v0')
from copy import copy

PRINT_SA = False
for i in range(20):
    ob = env.reset()
    cum_reward = 0
    discount = config["gamma"]
    # rnn_state is decided by rnn network size
    rnn_size = config["model"]["lstm_cell_size"]
    rnn_state = state=[[0 for i in range(rnn_size)], [0 for i in range(rnn_size)]]

    while True:
        action, rnn_state, logits_dictionary = trainer.compute_action(ob, state=rnn_state, prev_action=None, prev_reward=None, full_fetch=False)
        ob, reward, done, info = env.step(action)
        cum_reward += (reward * discount)
        discount *= discount

        if PRINT_SA:
            print("action:", action)
            try:
                env.render(mode='human')
            except NotImplementedError as e:
                print(ob)

        if done:
            break

    # print("final action:", action)
    # print("final observation:", ob)
    print("cumulative reward:", cum_reward)
