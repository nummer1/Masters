import gym
import ray
import ray.rllib.agents.impala as impala
from ray.rllib.models import ModelCatalog
import tensorflow as tf
import numpy as np

import config
import lstm_model


ray.init()
ModelCatalog.register_custom_model("lstm_model", lstm_model.LSTMCustomModel)
config = config.get_config_impala()

trainer = impala.ImpalaAgent(config=config, env="GuessingGame-v0")
trainer.restore("/home/kasparov/ray_results/IMPALA_GuessingGame-v0_2019-11-14_16-52-21j8tihlrj/checkpoint_10/checkpoint-10")
policy = trainer.get_policy()

# print(type(policy))
# print(type(policy.model))
# print(dir(policy))
# print(dir(policy.model))
# print(policy.model.last_layer)

env = gym.make('GuessingGame-v0')
from copy import copy

PRINT_SA = True
for i in range(20):
    ob = env.reset()
    cum_reward = 0
    discount = config["gamma"]
    # rnn_state is decided by rnn network size
    rnn_size = config["model"]["lstm_cell_size"]
    rnn_size = 8
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
