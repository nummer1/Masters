import gym
import ray
import sys

from ray.rllib.agents import impala
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray import tune

import config
import models_custom


alg = sys.argv[1]
# memory is total amount of memory available
if alg == "test":
    ray.init()
else:
    ray.init(memory=64*1024*1024*1024, object_store_memory=20*1024*1024*1024)
        # driver_object_store_memory=1*1024*1024*1024)
ModelCatalog.register_custom_model("lstm_model", models_custom.LSTMCustomModel)
ModelCatalog.register_custom_model("transformer_model", models_custom.TransformerCustomModel)

ModelCatalog.register_custom_preprocessor("procgen_preproc", models_custom.ProcgenPreprocessor)

config_impala = config.get_config_impala()
config_ppo = config.get_config_ppo()
config_appo = config.get_config_appo()
config_apex = config.get_config_apex()
config_rainbow = config.get_config_rainbow()
config_test = config.get_simple_test_config()


# NOTE: base policy is a dense network
# TODO: find good stopping point


checkpoint_freq = 10
checkpoint_at_end = True
max_failures = 0
# stop = {"training_iteration": 1}
stop = {"timesteps_total": int(2e8)}

if alg == 'impala':
    analysis = tune.run(
        "IMPALA",
        name="impala",
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=checkpoint_at_end,
        max_failures=0,   # will restart x times from last ceckpoint after crash
        reuse_actors=True,
        stop=stop,
        config=config_impala
    )
elif alg == 'ppo':
    analysis = tune.run(
        "PPO",
        name="ppo",
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=checkpoint_at_end,
        max_failures=0,   # will restart x times from last ceckpoint after crash
        reuse_actors=True,
        stop=stop,
        config=config_ppo
    )
elif alg == 'appo':
    analysis = tune.run(
        "APPO",
        name="appo",
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=checkpoint_at_end,
        max_failures=0,   # will restart x times from last ceckpoint after crash
        reuse_actors=True,
        stop=stop,
        config=config_appo
    )
elif alg == 'apex':
    analysis = tune.run(
        "APEX",
        name="apex",
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=checkpoint_at_end,
        max_failures=0,   # will restart x times from last ceckpoint after crash
        reuse_actors=True,
        stop=stop,
        config=config_apex
    )
elif alg == 'rainbow':
    analysis = tune.run(
        "DQN",
        name="dqn",
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=checkpoint_at_end,
        max_failures=0,   # will restart x times from last ceckpoint after crash
        reuse_actors=True,
        stop=stop,
        config=config_rainbow
    )
else:
    print("!!! no algorithm selected, running simple test")
    analysis = tune.run(
        "IMPALA",
        name="delete",
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=checkpoint_at_end,
        max_failures=0,   # will restart x times from last ceckpoint after crash
        reuse_actors=True,
        stop={"training_iteration": 1},
        config=config_test
    )


print(analysis.trials)
