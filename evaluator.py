import gym
import ray
import config
import ray.rllib.agents.impala as impala


ray.init()
config = config.get_config_impala()

trainer = impala.ImpalaAgent(config=config, env="CartPole-v0")
trainer.restore("/home/kasparov/ray_results/IMPALA_CartPole-v0_expert/checkpoint_51/checkpoint-51")
env = gym.make('CartPole-v0')

for i in range(2):
    ob = env.reset()
    while True:
        env.render(mode='human')
        action = trainer.compute_action(ob)
        ob, reward, done, info = env.step(action)
        print(ob, reward, done, info)
        if done:
            break
        env.render(mode='human')
