import gym
import ray
import sys

from ray.rllib.agents import impala
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray import tune

import environments
import config
import models_custom


alg = sys.argv[1]
model = sys.argv[2]
is_single = True if sys.argv[3] == 't' else False
env_id = int(sys.argv[4])
num_levels = int(sys.argv[5])
use_generated_assets = True if sys.argv[6] == 't' else False

# memory is total amount of memory available
if alg == "test":
    ray.init()
else:
    ray.init(memory=90*1024*1024*1024, object_store_memory=60*1024*1024*1024)
        # driver_object_store_memory=1*1024*1024*1024)

ModelCatalog.register_custom_model("lstm_model", models_custom.LSTMCustomModel)
ModelCatalog.register_custom_model("transformer_model", models_custom.TransformerCustomModel)
ModelCatalog.register_custom_preprocessor("procgen_preproc", models_custom.ProcgenPreprocessor)

config_dict = {
    'impala': config.get_config_impala,
    'ppo': config.get_config_ppo,
    'appo': config.get_config_appo,
    'rainbow': config.get_config_rainbow,
    'apex': config.get_config_apex,
    'test': config.get_simple_test_config
}

alg_dict = {
    'impala': "IMPALA",
    'ppo': "PPO",
    'appo': "APPO",
    'rainbow': "DQN",
    'apex': "APEX",
    'test': "IMPALA"
}

conf = config_dict[alg]()

if model == 'lstm':
    config.set_lstm(conf)
elif model == 'transformer':
    config.set_transformer(conf)

config.set_env(conf, is_single, env_id, num_levels, use_generated_assets)


checkpoint_freq = 10
checkpoint_at_end = True
max_failures = 100
reuse_actors = True
stop = {"training_iteration": 2} if alg == "test" else {"timesteps_total": int(2e8)}

analysis = tune.run(
    alg_dict[alg],
    name=alg,
    checkpoint_freq=checkpoint_freq,
    checkpoint_at_end=checkpoint_at_end,
    max_failures=max_failures,  # will restart x times from last ceckpoint after crash
    reuse_actors=reuse_actors,
    stop=stop,
    config=conf
)

print(analysis.trials)
