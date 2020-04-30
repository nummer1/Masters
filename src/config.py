import ray
from ray.rllib.agents import impala
from ray.rllib.agents import ppo
from ray.rllib.agents.dqn import dqn
from ray.rllib.agents.dqn import apex

import numpy as np


#  episode:
# 'add_extra_batch', 'agent_rewards', 'batch_builder', 'custom_metrics', 'episode_id',
# 'last_action_for', 'last_info_for', 'last_observation_for', 'last_pi_info_for',
# 'last_raw_obs_for', 'length', 'new_batch_builder', 'policy_for', 'prev_action_for',
# 'prev_reward_for', 'rnn_state_for', 'soft_reset', 'total_reward', 'user_data'

# env:
# 'action_space', 'cur_dones', 'cur_infos', 'cur_rewards', 'get_unwrapped',
# 'new_obs', 'num_envs', 'observation_space', 'poll', 'send_actions', 'stop',
# 'to_base_env', 'try_reset', 'vector_env'


# from ray.rllib.evaluation.rollout_worker import get_global_worker
# # You can use this from any callback to get a reference to the
# # RolloutWorker running in the process, which in turn has references to
# # all the policies, etc: see rollout_worker.py for more info.
# rollout_worker = get_global_worker()

# Callbacks that will be run during various phases of training. These all
# take a single "info" dict as an argument. For episode callbacks, custom
# metrics can be attached to the episode by updating the episode object's
# custom metrics dict (see examples/custom_metrics_and_callbacks.py). You
# may also mutate the passed in batch data in your callback.

# can log metrics using these functions
def on_episode_start(info):
    # "on_episode_start": None, # arg: {"env": .., "episode": ...}
    # episode = info["episode"]
    # episode.user_data["test"] = []
    pass


def on_episode_step(info):
    # "on_episode_step": None, # arg: {"env": .., "episode": ...}
    # episode = info["episode"]
    # episode.user_data["test"].append(1)
    pass


def on_episode_end(info):
    # "on_episode_end": None, # arg: {"env": .., "episode": ...}
    # episode = info["episode"]
    # test = np.mean(episode.user_data["test"])
    # episode.custom_metrics["test"] = test
    pass


def on_sample_end(info):
    # "on_sample_end": None, # arg: {"samples": .., "worker": ...}
    pass


def on_train_result(info):
    # "on_train_result": None, # arg: {"trainer": ..., "result": ...}
    pass


def on_postprocess_traj(info):
    # "on_postprocess_traj": None, # arg: {
    # "agent_id": ..., "episode": ...,
    # "pre_batch": (before processing),
    # "post_batch": (after processing),
    # "all_pre_batches": (other agent ids),
    # }
    pass



####################
# configurations   #
####################

def set_model(config, model_name):
    model_name = model_name + "_model"
    config["model"]["custom_model"] = model_name
    config["model"]["custom_action_dist"] = None
    config["model"]["custom_options"] = {}
    config["model"]["custom_preprocessor"] = None


def get_env_config(is_eval, env_id, num_levels, use_generated_assets, dist):
    env_config = {
        "is_eval": is_eval,
        "env_id": env_id,
        "num_levels": num_levels,
        "use_generated_assets": use_generated_assets,
        "dist": dist  # easy, hard, memory
    }
    return env_config


def set_env(config, is_single, env_id, num_levels, use_generated_assets, dist):
    # if is_single is false, num envs per worker must be a multiple of 6 and env_id is unused
    config["env"] = "single_task" if is_single else "multi_task"
    config["env_config"] = get_env_config(False, env_id, num_levels, use_generated_assets, dist)

    config["evaluation_interval"] = 10
    config["evaluation_num_episodes"] = 100
    env_config = get_env_config(True, env_id, num_levels, use_generated_assets, dist)
    if "vtrace" in config:
        config["evaluation_config"] = {
            # set is_eval to true and keep everything else the same
            "env_config": env_config,
            "num_envs_per_worker": 12,
            "vtrace": False
        }
    else:
        config["evaluation_config"] = {
            # set is_eval to true and keep everything else the same
            "env_config": env_config,
            "num_envs_per_worker": 12
        }
    # TODO set "explore": False for rainbow in evaluation_config


def set_common_config(config):
    # config["num_workers"] = 5  # one base worker is created in addition
    # config["num_envs_per_worker"] = 12  # must be a multiple of 6 if multi_task
    # config["sample_batch_size"] = 256
    # config["train_batch_size"] = 256*12  # train_batch_size > num_envs_per_worker * rollout_fragment_length
    # # Whether to rollout "complete_episodes" or "truncate_episodes" to
    # config["batch_mode"] = "truncate_episodes"
    #
    # # can be fraction
    # config["num_gpus"] = 1
    #
    # config["gamma"] = 0.999
    # config["lr"] = 5e-4

    config["log_level"] = "INFO"
    config["callbacks"] = {
        "on_episode_start": on_episode_start,
        "on_episode_step": on_episode_step,
        "on_episode_end": on_episode_end,
        "on_sample_end": on_sample_end,
        "on_train_result": on_train_result,
        "on_postprocess_traj": on_postprocess_traj,
    }

    config["explore"] = False
    config["num_gpus"] = 1

    # config["ignore_worker_failures"] = False
    # config["log_sys_usage"] = True
    # config["metrics_smoothing_episodes"] = 100
    # config["eager"] = False


def set_dqn_config(config):
    config["timesteps_per_iteration"] = 25000
    config["target_network_update_freq"] = 50000  # Update the target network every `target_network_update_freq` steps.

    # === Optimization ===
    config["lr"] = 2.5e-4
    config["lr_schedule"] = None
    config["adam_epsilon"] = 1e-8  # Adam epsilon hyper parameter
    config["grad_norm_clipping"] = 40  # If not None, clip gradients during optimization at this value
    config["learning_starts"] = 50000  # How many steps of the model to sample before learning starts

    # === Replay buffer ===
    config["buffer_size"] = 1000000  # is async_updates, each worker will have own replay buffer
    config["prioritized_replay"] = True  # If True prioritized replay buffer will be used.
    config["prioritized_replay_alpha"] = 0.6  # Alpha parameter for prioritized replay buffer.
    config["prioritized_replay_beta"] = 0.4  # Beta parameter for sampling from prioritized replay buffer.
    config["final_prioritized_replay_beta"] = 1.0  # Final value of beta (by default, we use constant beta=0.4).
    config["prioritized_replay_beta_annealing_timesteps"] = 20000  # Time steps over which the beta parameter is annealed.
    config["prioritized_replay_eps"] = 1e-6  # Epsilon to add to the TD errors when updating priorities.
    config["compress_observations"] = True  # Whether to LZ4 compress observations

    # set to -1 to not get error from key_checker
    config["schedule_max_timesteps"] = -1
    config["exploration_final_eps"] = -1
    config["exploration_fraction"] = -1
    config["beta_annealing_fraction"] = -1
    config["per_worker_exploration"] = -1
    config["softmax_temp"] = -1
    config["soft_q"] = -1


def set_apex_config(config):
    # set_dqn_config(config)
    # config["optimizer"]["max_weight_sync_delay"] = 400
    # config["optimizer"]["num_replay_buffer_shards"] = 4
    # config["optimizer"]["debug"] = False
    #
    # config["n_step"] = 3
    # config["buffer_size"] = 2000000
    # config["exploration_config"] = {"type": "PerWorkerEpsilonGreedy"}
    # config["worker_side_prioritization"] = True
    # config["min_iter_time_s"] = 10

    config["double_q"] = False
    config["dueling"] = False
    config["num_atoms"] = 1
    config["noisy"] = False
    config["n_step"] = 3
    config["lr"] = .0001
    config["adam_epsilon"] = .00015
    config["hiddens"] = [512]
    config["buffer_size"] = 1000000
    config["exploration_config"]["final_epsilon"] = 0.01
    config["exploration_config"]["epsilon_timesteps"] = 200000
    config["prioritized_replay_alpha"] = 0.5
    config["final_prioritized_replay_beta"] = 1.0
    config["prioritized_replay_beta_annealing_timesteps"] = 2000000

    config["num_gpus"] = 1

    # APEX
    config["num_workers"] = 8
    config["num_envs_per_worker"] = 8
    config["rollout_fragment_length"] = 20
    config["train_batch_size"] = 512
    config["target_network_update_freq"] = 50000
    config["timesteps_per_iteration"] = 25000


def set_rainbow_config(config):
    set_dqn_config(config)
    config["num_atoms"] = 51  # rainbow: "num_atoms": [more than 1]
    config["v_min"] = 0  # expected returns should be between 0 and 1 since they're normalized
    config["v_max"] = 20.0
    config["noisy"] = True  # rainbow: noisy = True
    config["sigma0"] = 0.5  # control the initial value of noisy nets
    config["dueling"] = True
    config["double_q"] = True
    # Postprocess model outputs with these hidden layers to compute the
    # state and action values. See also the model config in catalog.py.
    config["hiddens"] = [256]
    config["n_step"] = 3  # rainbow: "n_step": [between 1 and 10]

    # === Exploration Settings (Experimental) ===
    config["exploration_config"] = {
        # The Exploration class to use.
        "type": "EpsilonGreedy",
        # Config for the Exploration class' constructor:
        "initial_epsilon": 1.0,
        "final_epsilon": 0.02,
        "epsilon_timesteps": 10000  # Timesteps over which to anneal epsilon.
    }

    config["parameter_noise"] = False  # parameter noise for exploration

    # === Parallelism ===
    config["worker_side_prioritization"] = False  # Whether to compute priorities on workers.
    config["min_iter_time_s"] = 1  # Prevent iterations from going lower than this time span


def set_ppo_config(config):
    # config["use_critic"] = True  # required for GAE, use critic as baseline
    # config["use_gae"] = True
    # config["lambda"] = 0.95  # The GAE(lambda) parameter.
    # config["kl_coeff"] = 0.5  # Initial coefficient for KL divergence.
    # config["kl_target"] = 0.01 # Target value for KL divergence.
    #
    # config["sgd_minibatch_size"] = 8
    # config["shuffle_sequences"] = True  # Whether to shuffle sequences in the batch when training (recommended).
    # config["num_sgd_iter"] = 3
    #
    # config["vf_share_layers"] = True
    # config["vf_loss_coeff"] = 0.5  # IMPORTANT: you must tune this if vf_share_layers = True
    # config["entropy_coeff"] = 0.01
    # config["entropy_coeff_schedule"] = None
    # config["clip_param"] = 0.2
    #
    # config["vf_clip_param"] = 10.0  # Sensitive to the scale of the rewards. If your expected V is large, increase this.
    # config["grad_clip"] = None  # If specified, clip the global norm of gradients by this amount.
    #
    # config["batch_mode"] = "truncate_episodes"

    config["lambda"] = 0.95
    config["kl_coeff"] = 0.5
    config["clip_rewards"] = True
    config["clip_param"] = 0.1
    config["vf_clip_param"] = 10.0
    config["entropy_coeff"] = 0.01
    config["train_batch_size"] = 5000
    config["rollout_fragment_length"] = 100
    config["sgd_minibatch_size"] = 500
    config["num_sgd_iter"] = 10
    config["num_workers"] = 7
    config["num_envs_per_worker"] = 12
    config["batch_mode"] = "truncate_episodes"
    # config["observation_filter"] = "NoFilter"
    config["vf_share_layers"] = True
    config["num_gpus"] = 1


def set_impala_config(config, buffer):
    # config["vtrace"] = True
    # config["vtrace_clip_rho_threshold"] = 1.0
    # config["vtrace_clip_pg_rho_threshold"] = 1.0
    #
    # config["num_data_loader_buffers"] = 1  # larger number goes faster but uses more GPU memory
    # config["minibatch_buffer_size"] = 1  # number of train batches to  retain for minibatching, only effect if num_sgd_iter > 1
    # config["num_sgd_iter"] = 3  # number of passes over each train batch
    #
    # if buffer:
    #     config["replay_proportion"] = 0.8  # set to > 0 to use replay buffer
    #     config["replay_buffer_num_slots"] = 30000  # number of sample batches to store for replay
    # else:
    #     config["replay_proportion"] = 0  # set to > 0 to use replay buffer
    #     config["replay_buffer_num_slots"] = 0  # number of sample batches to store for replay
    #
    # # TODO: might crash if learner_queue_size is not 1
    # config["learner_queue_size"] = 10  # training batches in queue to learner
    # config["learner_queue_timeout"] = 600
    # config["max_sample_requests_in_flight_per_worker"] = 2
    # config["broadcast_interval"] = 1  # max number of workers to broadcast one set of weights to
    #
    # # Learning params.
    # config["grad_clip"] = 40.0
    # # either "adam" or "rmsprop"
    # config["opt_type"] = "adam"
    #
    # # only used if rmsprop
    # config["decay"] = 0.99
    # config["momentum"] = 0.0
    # config["epsilon"] = 0.1
    #
    # # balancing the three losses
    # config["vf_loss_coeff"] = 0.5
    # config["entropy_coeff"] = 0.01
    # config["entropy_coeff_schedule"] = None

    #### 1 ####
    # config["rollout_fragment_length"] = 50
    # config["train_batch_size"] = 500
    # config["num_workers"] = 7
    # config["num_envs_per_worker"] = 12
    # config["clip_rewards"] = True
    # config["lr_schedule"] = [[0, 0.0005],[20000000, 0.000000000001],]
    #### 1 ####

    #### 2 ####
    config["gamma"] = 0.999
    config["lr"] = 0.0005
    config["num_workers"] = 5
    config["num_envs_per_worker"] = 32
    config["rollout_fragment_length"] = 256
    config["train_batch_size"] = 256 * 32 * 8
    config["num_sgd_iter"] = 3
    config["entropy_coeff"] = 0.01

    if buffer:
        config["replay_proportion"] = 0.8  # set to > 0 to use replay buffer
        config["replay_buffer_num_slots"] = 10  # number of sample batches to store for replay
    else:
        config["replay_proportion"] = 0  # set to > 0 to use replay buffer
        config["replay_buffer_num_slots"] = 0  # number of sample batches to store for replay
    #### 2 ####


def set_appo_config(config, buffer):
    set_impala_config(config, buffer)
    # config["vtrace"] = True  # v-trace of GAE advantages
    #
    # # only used if v_trace is False
    # config["use_critic"] = True
    # config["use_gae"] = True
    # config["lambda"] = 0.95
    #
    # config["clip_param"] = 0.2
    # config["use_kl_loss"] = False
    # config["kl_coeff"] = 0.5
    # config["kl_target"] = 0.01

    config["lambda"] = 0.95
    config["kl_coeff"] = 0.5
    config["clip_param"] = 0.1


def get_config_apex(buffer):
    config = apex.APEX_DEFAULT_CONFIG.copy()
    set_common_config(config)
    set_apex_config(config)
    return config


def get_config_rainbow(buffer):
    config = dqn.DEFAULT_CONFIG.copy()
    set_common_config(config)
    set_rainbow_config(config)
    return config


def get_config_appo(buffer):
    config = impala.DEFAULT_CONFIG.copy()
    set_common_config(config)
    set_appo_config(config, buffer)
    return config


def get_config_ppo(buffer):
    config = ppo.DEFAULT_CONFIG.copy()
    set_common_config(config)
    set_ppo_config(config)
    return config


def get_config_impala(buffer):
    config = impala.DEFAULT_CONFIG.copy()
    set_common_config(config)
    set_impala_config(config, buffer)
    return config


def get_simple_test_config(buffer):
    # used to check for bugs
    config = get_config_impala(buffer)

    # config["preprocessor_pref"] = None  # Does nothing
    # config["model"]["max_seq_len"] = 20

    config["num_workers"] = 1
    config["num_envs_per_worker"] = 6
    # config["sample_batch_size"] = 256
    config["train_batch_size"] = 6*256
    config["num_gpus"] = 0
    config["evaluation_config"]["num_envs_per_worker"] = 6
    config["batch_mode"] = "truncate_episodes"
    config["minibatch_buffer_size"] = 1  # number of train batches to retain for minibatching, only effect if num_sgd_iter > 1
    config["num_sgd_iter"] = 3

    config["num_data_loader_buffers"] = 2  # larger number goes faster but uses more GPU memory
    config["minibatch_buffer_size"] = 1  # number of train batches to  retain for minibatching, only effect if num_sgd_iter > 1
    config["num_sgd_iter"] = 1  # number of passes over each train batch
    config["replay_proportion"] = 0.5  # set to > 0 to use replay buffer
    config["replay_buffer_num_slots"] = 6*500  # number of sample batches to store for replay
    config["learner_queue_size"] = 1  # training batches in queue to learner

    # config["num_atoms"] = 51  # rainbow: "num_atoms": [more than 1]
    # config["v_min"] = 0  # expected returns should be between 0 and 1 since they're normalized
    # config["v_max"] = 20.0
    # config["noisy"] = True  # rainbow: noisy = True
    # config["sigma0"] = 0.5  # control the initial value of noisy nets
    # config["dueling"] = True
    # config["double_q"] = True
    # # Postprocess model outputs with these hidden layers to compute the
    # # state and action values. See also the model config in catalog.py.
    # config["hiddens"] = [256]
    # config["n_step"] = 3  # rainbow: "n_step": [between 1 and 10]

    config["eager"] = False
    config["log_level"] = "INFO"

    config["evaluation_interval"] = 1
    config["evaluation_num_episodes"] = 10

    return config
