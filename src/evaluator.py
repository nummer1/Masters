import subprocess
import glob
import matplotlib.pyplot as plt

from ray.rllib.models import ModelCatalog
from ray import tune

import environments
import config
import models_custom


ModelCatalog.register_custom_model("lstm_model", models_custom.LSTMCustomModel)
ModelCatalog.register_custom_model("transformer_model", models_custom.TransformerCustomModel)
ModelCatalog.register_custom_model("simple_model", models_custom.SimpleCustomModel)


"""python rollout.py ~/Documents/Masters/results/ray_results_B_1/impala_simple_easy_multi_0/IMPALA_multi_task_0_2020-04-08_15-17-12up4fmd11/checkpoint_737/checkpoint-737 --run IMPALA --env procgen:procgen-caveflyer-v0 --episodes 100"""


directory_name = "ray_results_B_1/impala_simple_easy_multi_0/IMPALA_multi_task_0_2020-04-08_15-17-12up4fmd11/"
directory_name = "/home/kasparov/Documents/Masters/results/" + directory_name

checkpoint_list = glob.glob(directory_name + "checkpoint*")
checkpoint_list = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1]))
for i, c in enumerate(checkpoint_list):
    checkpoint_list[i] += "/checkpoint-" + str(c.split("/")[-1].split("_")[-1])

env_list = ["procgen:procgen-caveflyer-v0", "procgen:procgen-dodgeball-v0", "procgen:procgen-miner-v0",
        "procgen:procgen-jumper-v0", "procgen:procgen-maze-v0", "procgen:procgen-heist-v0"]

for i, env in enumerate(env_list):
    reward_list = []
    for checkpoint in checkpoint_list:
        result = subprocess.run(["python", "rollout.py", checkpoint,
                "--run", "IMPALA", "--env", env, "--episodes", "100", "--no-render"],
                stdout=subprocess.PIPE)

        result = result.stdout.decode("utf-8")
        result = result.split('\n')
        reward = 0
        for line in result:
            line = line.split()
            if len(line) > 0 and line[0] == "Episode":
                reward += float(line[3])
        reward /= 100
        reward_list.append(reward)

    plt.figure(i + 1)
    plt.ylabel('reward')
    plt.xlabel('iterations (10)')
    plt.plot([j for j in range(reward_list)], reward_list)

plt.show()
