import gym
from procgen import ProcgenEnv
from ray.tune.registry import register_env

# import gym
# env = gym.make("procgen:procgen-coinrun-v0", start_level=0, num_levels=1)
# For the vectorized environment:
#
from procgen import ProcgenEnv
# venv = ProcgenEnv(num_envs=1, env_name="coinrun", start_level=0, num_levels=1)

# env_name - Name of environment, or comma-separate list of environment names to instantiate
#   as each env in the VecEnv.
# num_levels - The number of unique levels that can be generated. Set to 0 to use unlimited levels.
# start_level - The lowest seed that will be used to generated levels. 'start_level' and 'num_levels'
#   fully specify the set of possible levels.
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


# TODO: make evaluation config
# TODO: make shure env runs on all environments in multi-task
def multi_task_memory(env_config):
    # train on multiple environments
    if env_config.vector_index % 6==0:
        print("!! CAVEFLYER CREATED")
        env = gym.make("procgen:procgen-caveflyer-v0", num_levels=env_config["num_levels"],
                use_generated_assets=env_config["use_generated_assets"], distribution_mode="memory")
    elif env_config.vector_index % 6==1:
        print("!! DODGEBALL CREATED")
        env = gym.make("procgen:procgen-dodgeball-v0", num_levels=env_config["num_levels"],
                use_generated_assets=env_config["use_generated_assets"], distribution_mode="memory")
    elif env_config.vector_index % 6==2:
        print("!! MINER CREATED")
        env = gym.make("procgen:procgen-miner-v0", num_levels=env_config["num_levels"],
                use_generated_assets=env_config["use_generated_assets"], distribution_mode="memory")
    elif env_config.vector_index % 6==3:
        print("!! JUMPER CREATED")
        env = gym.make("procgen:procgen-jumper-v0", num_levels=env_config["num_levels"],
                use_generated_assets=env_config["use_generated_assets"], distribution_mode="memory")
    elif env_config.vector_index % 6==4:
        print("!! MAZE CREATED")
        env = gym.make("procgen:procgen-maze-v0", num_levels=env_config["num_levels"],
                use_generated_assets=env_config["use_generated_assets"], distribution_mode="memory")
    else:
        print("!! HEIST CREATED")
        env = gym.make("procgen:procgen-heist-v0", num_levels=env_config["num_levels"],
                use_generated_assets=env_config["use_generated_assets"], distribution_mode="memory")
    # env = ProcgenEnv(env_name="caveflyer",
    #         num_envs=1, num_levels=500, use_generated_assets=False, distribution_mode="memory")
    return env


def single_task_memory(env_config):
    # train on one environment
    env = gym.make("procgen:procgen-coinrun-v0", num_levels=env_config["num_levels"],
            use_generated_assets=env_config["use_generated_assets"], distribution_mode="memory")
    # env = ProcgenEnv(env_name=["coinrun", "caveflyer", "dodgeball", "miner", "jumper", "maze", "heist"],
    #         num_envs=6, num_levels=0, use_generated_assets=False, distribution_mode="memory")
    return env


register_env("memory_multi_task", multi_task_memory)
register_env("memory_single_task", single_task_memory)
