import gym
import ray
import sys

from ray.rllib.models import ModelCatalog
from ray import tune

import environments
import config
import models_custom


alg = sys.argv[1]
model = sys.argv[2]
dist = sys.argv[3]
is_single = True if sys.argv[4] == 't' else False
env_id = int(sys.argv[5])
num_levels = int(sys.argv[6])
use_generated_assets = True if sys.argv[7] == 't' else False
buffer = True if sys.argv[8] == 't' else False

# memory is total amount of memory available
if alg == "test":
    ray.init()
else:
    # ray.init()
    ray.init(num_gpus=1, memory=25*1024*1024*1024, object_store_memory=60*1024*1024*1024)
        # driver_object_store_memory=10*1024*1024*1024)

ModelCatalog.register_custom_model("lstm_model", models_custom.LSTMCustomModel)
ModelCatalog.register_custom_model("transformer_model", models_custom.TransformerCustomModel)
ModelCatalog.register_custom_model("simple_model", models_custom.SimpleCustomModel)
ModelCatalog.register_custom_model("dense_model", models_custom.DenseCustomModel)
# ModelCatalog.register_custom_preprocessor("procgen_preproc", models_custom.ProcgenPreprocessor)

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

conf = config_dict[alg](buffer)
config.set_model(conf, model)
config.set_env(conf, is_single, env_id, num_levels, use_generated_assets, dist)


checkpoint_freq = 10
checkpoint_at_end = True
max_failures = 10  # TODO: increase this for memory environment
reuse_actors = True  # TODO: setting to True might break
stop = {"timesteps_total": int(2.5e7)} if dist == "easy" else {"timesteps_total": int(2e8)}
if alg == "test":
    stop = {"training_iteration": 2}

name = alg + "_" + model + "_" + dist + ("_single" if is_single else "_multi") + \
        (("_" + str(env_id)) if is_single else "") + "_" + str(num_levels) + \
        ("_genassets" if use_generated_assets else "") + \
        ("_buffer" if buffer else "")


# TODO: use num_samples to run multiple experiments in parallell
analysis = tune.run(
    alg_dict[alg],
    name=name,
    checkpoint_freq=checkpoint_freq,
    checkpoint_at_end=checkpoint_at_end,
    max_failures=max_failures,  # will restart x times from last ceckpoint after crash
    reuse_actors=reuse_actors,
    stop=stop,
    config=conf
)

print(analysis.trials)
