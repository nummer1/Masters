# TODO: use custom preprocessing
# TODO: use custom action distribution


import ray
# from ray import tune
import ray.rllib.agents.impala as impala
from ray.tune.logger import pretty_print
from ray.rllib.models import ModelCatalog

import config
import lstm_model


ray.init()
ModelCatalog.register_custom_model("lstm_model", lstm_model.LSTMCustomModel)
config = config.get_config_impala()
trainer = impala.ImpalaAgent(config=config, env="GuessingGame-v0")

for i in range(100):
   # Perform one iteration of training the policy with IMPALA
   result = trainer.train()
   print(pretty_print(result))

   if i % 1 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
