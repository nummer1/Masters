import ray
import ray.rllib.agents.impala as impala
import numpy as np


# can log metrics using these functions
def on_episode_start(info):
    # print(info.keys())  # -> "env", 'episode"
    pass

def on_episode_step(info):
    pass

def on_episode_end(info):
    pass

def on_train_result(info):
    pass

def on_postprocess_traj(info):
    pass


def get_config_impala():
    # https://ray.readthedocs.io/en/latest/rllib-training.html#common-parameters
    # https://ray.readthedocs.io/en/latest/rllib-algorithms.html#importance-weighted-actor-learner-architecture-impala

    config = impala.DEFAULT_CONFIG.copy()
    # vtrace
    config["vtrace"] = False
    config["vtrace_clip_rho_threshold"] = 1.0
    config["vtrace_clip_pg_rho_threshold"] = 1.0

    config["num_gpus"] = 0
    config["num_workers"] = 1
    config["num_envs_per_worker"] = 1
    # TODO: crashes when eager is off
    config["eager"] = False
    config["eager_tracing"] = False  # setting to true greatly improves performance in eager mode
    config["gamma"] = 0.99  # discount for MDP

    # config["sample_batch_size"] = 50
    # config["train_batch_size"] = 500
    # config["min_iter_time_s"] = 10
    # config["learner_queue_timeout"] = 300

    # learning rate parameters IMPALA
    config["grad_clip"] = 40.0
    config["opt_type"] = "rmsprop"
    config["lr"] = 0.0005
    config["lr_schedule"] = None
    config["decay"] = 0.99
    config["momentum"] = 0.0
    config["epsilon"] = 0.01

    # balancing the three losses
    config["vf_loss_coeff"] = 0.5
    config["entropy_coeff"] = 0.01

    # logging
    config["log_level"] = "INFO"
    config["callbacks"] = {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
            "on_train_result": on_train_result,
            "on_postprocess_traj": on_postprocess_traj,
    }

    # custom model options
    config["model"]["custom_preprocessor"] = None
    config["model"]["custom_model"] = "lstm_model"
    config["model"]["custom_action_dist"] = None
    config["model"]["custom_options"] = {}

    # model parameters
    # https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py
    # https://ray.readthedocs.io/en/latest/rllib-models.html#built-in-model-parameters
    # # fc = fully connected
    # config["model"]["fcnet_activation"] = "tanh"
    # config["model"]["fcnet_hiddens"] = [16]
    #
    # # lstm model parameters
    config["model"]["max_seq_len"] = 100
    # config["model"]["use_lstm"] = True
    # config["model"]["lstm_cell_size"] = 16
    # Whether to feed a_{t-1}, r_{t-1} to LSTM
    # config["model"]["lstm_use_prev_action_reward"] = True
    # # When using modelv1 models with a modelv2 algorithm, you may have to define the state shape here (e.g., [256, 256]).
    # config["model"]["state_shape"] = None

    return config
