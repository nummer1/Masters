import ray
import ray.rllib.agents.impala as impala
from ray.tune.logger import pretty_print
from ray import tune

import config


ray.init()
config = config.get_config_impala()
trainer = impala.ImpalaAgent(config=config, env="CartPole-v0")

# tune.run(
#     "PPO",
#     stop={"episode_reward_mean": 150},
#     config={
#         "env": "CartPole-v0",
#         "num_gpus": 0,
#         "num_workers": 1,
#         "lr": tune.grid_search([0.01, 0.001, 0.0001]),
#         "eager": False,
#     },
# )

for i in range(10):
   # Perform one iteration of training the policy with IMPALA
   result = trainer.train()
   print(pretty_print(result))

   if i % 1 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
