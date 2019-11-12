import ray
import ray.rllib.agents.impala as impala
from ray.tune.logger import pretty_print


ray.init()
config = impala.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 2
config["eager"] = False
trainer = impala.ImpalaAgent(config=config, env="FrozenLake-v0")

# Can optionally call trainer.restore(path) to load a checkpoint.

for i in range(10):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 1 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)


trainer.restore("/home/kasparov/ray_results/IMPALA_FrozenLake-v0_2019-11-12_16-38-51pva_0683/checkpoint_10/checkpoint-10")
env = gym.make('FrozenLake-v0')
env.reset()
agent = Agent(env.action_space)
for i in range(10):
    action = agent.act(env, 0, False)
    ob, reward, done, info = env.step(action)
    print(ob, reward, done, info)
    if done:
        break

env.render(mode='human')
