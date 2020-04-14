import gym
import random

from ray.tune.registry import register_env


# env_name - Name of environment, or comma-separate list of environment names to instantiate
#   as each env in the VecEnv.
# num_levels - The number of unique levels that can be generated. Set to 0 to use unlimited levels.
# start_level - The lowest seed that will be used to generated levels. 'start_level' and 'num_levels'
#   fully specify the set of possible levels.
#   !!! seed must be between -2 ** 31 < v < 2 ** 31
# paint_vel_info - Paint player velocity info in the top left corner. Only supported by certain games.
# use_generated_assets - Use randomly generated assets in place of human designed assets.
# debug_mode - A useful flag that's passed through to procgen envs. Use however you want during debugging.
# center_agent - Determines whether observations are centered on the agent or display the full level.
#   Override at your own risk.
# use_sequential_levels - When you reach the end of a level, the episode is ended and a new level is selected. If use_sequential_levels is set to True, reaching the end of a level does not end the episode, and the seed for the new level is derived from the current level seed. If you combine this with start_level=<some seed> and num_levels=1, you can have a single linear series of levels similar to a gym-retro or ALE game.
# distribution_mode - What variant of the levels to use, the options are "easy", "hard", "extreme",
#   "memory", "exploration". All games support "easy" and "hard", while other options are game-specific.
#   The default is "hard". Switching to "easy" will reduce the number of timesteps required to solve each
#   game and is useful for testing or when working with limited compute resources.

# Environments that support distribution_mode=memory:
#   CoinRun, CaveFlyer, Dodgeball, Miner, Jumper, Maze, and Heist.
#   CoinRun might not support it, thought it says so it in the paper

# -2 ** 31 + 1, 2 ** 31 - 1
# start_level_seed = random.randint(1, 2 ** 31 - 1)
start_level_seed = 1387712432
possible_levels = 2**31-1

# names = ["CoinRun", "StarPilot", "CaveFlyer", "Dodgeball", "FruitBot", "Chaser", "Miner",
#         "Jumper", "Leaper", "Maze", "BigFish", "Heist", "Climber", "Plunder", "Ninja", "Bossfight"]
# hard = [(5., 10.), (1.5, 35.), (2., 13.4), (1.5, 19.), (-.5, 27.2), (.5, 14.2), (1.5, 20.),
#         (1., 10.), (1.5, 10.), (4., 10.), (0., 40.), (2., 10.), (1., 12.6), (3., 30.), (2., 10.), (.5, 13.)]
env_list = ["procgen:procgen-caveflyer-v0", "procgen:procgen-dodgeball-v0", "procgen:procgen-miner-v0",
        "procgen:procgen-jumper-v0", "procgen:procgen-maze-v0", "procgen:procgen-heist-v0"]
# norm_const_hard = [(2.0, 13.4), (1.5, 19.0), (1.5, 20.0), (1.0, 10.0), (4.0, 10.0), (2.0, 10.0)]
# norm_const_memory = [(0.0, 13.4), (0.0, 19.0), (0.0, 20.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]

# TODO: use gym wrapper instead
# class EnvWrapper(gym.Env):
#     """
#     wrapped to normalise rewards
#     """
#     def __init__(self, id, num_levels, start_level, use_generated_assets):
#         name = env_list[id]
#         self.norm = norm_const_memory[id]
#         print("!!", name, "Created")
#         self.env = gym.make(name, num_levels=num_levels, start_level=start_level,
#                 use_generated_assets=use_generated_assets, distribution_mode="memory")
#         self.action_space = self.env.action_space
#         self.observation_space = self.env.observation_space
#
#     def reset(self):
#         return self.env.reset()
#
#     def step(self, action):
#         obs, reward, done, info = self.env.step(action)
#         reward = (reward - self.norm[0])/(self.norm[1] - self.norm[0])
#         return obs, reward, done, info


def set_seeds(env_config):
    num_levels = env_config["num_levels"]
    if env_config["is_eval"]:
        # if eval, test on all levels that are not trained on
        start_level = start_level_seed + num_levels
        if num_levels != 0:
            num_levels = possible_levels - num_levels
    else:
        start_level = start_level_seed
    return num_levels, start_level


# TODO
def guessing(env_config):
    env = gym.make("GuessingGame-v0")
    return env


def multi_task(env_config):
    num_levels, start_level = set_seeds(env_config)
    env_id = env_config.vector_index % 6
    name = env_list[env_id]

    env = gym.make(name, num_levels=num_levels, start_level=start_level,
            use_generated_assets=env_config["use_generated_assets"],
            distribution_mode=env_config["dist"])
    return env


def single_task(env_config):
    # train on one environment
    num_levels, start_level = set_seeds(env_config)
    env_id = env_config["env_id"]
    if env_id == 7:
        name = "procgen:procgen-coinrun-v0"
    else:
        name = env_list[env_id]

    env = gym.make(name, num_levels=num_levels, start_level=start_level,
                use_generated_assets=env_config["use_generated_assets"],
                distribution_mode=env_config["dist"])
    return env


register_env("multi_task", multi_task)
register_env("single_task", single_task)
