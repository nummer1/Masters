import gym
import ray
from ray.rllib.agents import impala
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

import config
import lstm_model


# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# version_path = os.path.join(SCRIPT_DIR, "version.txt")
# __version__ = open(version_path).read()
#
# from .env import ProcgenEnv
# from .gym_registration import register_environments
#
# register_environments()
#
# __all__ = ["ProcgenEnv"]


ray.init()
ModelCatalog.register_custom_model("lstm_model", lstm_model.LSTMCustomModel)
# ModelCatalog.register_custom_model("lstm_model", lstm_model.TransformerCustomModel)

config_impala = config.get_config_impala()
# config_ppo = config.get_config_ppo()
# config_apex = config.get_config_apex()

# procgen_env = gym.make("procgen:procgen-coinrun-v0")

# trainer_impala = impala.ImpalaAgent(config=config_impala, env="GuessingGame-v0")
# trainer_PPO = PPOTrainer(config=config_ppo, env="GuessingGame-v0")
# trainer_apex = ApexTrainer(config=config_apex, env="GuessingGame-v0")
trainer = impala.ImpalaAgent(config=config_impala, env="procgen:procgen-coinrun-v0")
policy = trainer.get_policy()
policy.model.base_model.summary()

# TODO: base policy is a dense network. This is not good


# TODO: run one trainer, then run script multiple times for each job at Idun
# for x in [trainer_impala, trainer_PPO, trainer_apex]:
# TODO: find good stopping point

for i in range(100):
   result = trainer.train()
   print(pretty_print(result))

   # TODO: do data collection here or in config functions
   if i % 1 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
