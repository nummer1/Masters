import ray
import ray.rllib.agents.impala as impala
from ray.tune.logger import pretty_print
import config


ray.init()
config = config.get_config_impala()

trainer = impala.ImpalaAgent(config=config, env="CartPole-v0")

for i in range(51):
   # Perform one iteration of training the policy with IMPALA
   result = trainer.train()
   print(pretty_print(result))

   if i % 10 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
