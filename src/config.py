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

def set_lstm(config):
    config["model"]["custom_model"] = "lstm_model"
    config["model"]["custom_action_dist"] = None
    config["model"]["custom_options"] = {}
    config["model"]["custom_preprocessor"] = None


def set_transformer(config):
    config["model"]["custom_model"] = "transformer_model"
    config["model"]["custom_action_dist"] = None
    config["model"]["custom_options"] = {}
    config["model"]["custom_preprocessor"] = None


def get_env_config(is_eval, env_id, num_levels, use_generated_assets):
    env_config = {
        "is_eval": is_eval,
        "env_id": env_id,
        "num_levels": num_levels,
        "use_generated_assets": use_generated_assets
    }
    return env_config


def set_env(config, is_single, env_id, num_levels, use_generated_assets):
    # if is_single is false, num envs per worker must be a multiple of 6 and env_id is unused
    config["env"] = "memory_single_task" if is_single else "memory_multi_task"
    config["env_config"] = get_env_config(False, env_id, num_levels, use_generated_assets)

    config["evaluation_interval"] = 10
    config["evaluation_num_episodes"] = 100
    env_config = get_env_config(True, env_id, num_levels, use_generated_assets)
    if "vtrace" in config:
        config["evaluation_config"] = {
            # set is_eval to true and keep everything else the same
            "env_config": env_config,
            "num_envs_per_worker": 30,
            "vtrace": False
        }
    else:
        config["evaluation_config"] = {
            # set is_eval to true and keep everything else the same
            "env_config": env_config,
            "num_envs_per_worker": 30
        }
    # TODO set "explore": False for rainbow in evaluation_config


def set_common_config(config):
    config["num_workers"] = 5  # one base worker is created in addition
    config["num_envs_per_worker"] = 30  # must be a multiple of 6 if multi_task
    config["rollout_fragment_length"] = 500
    config["train_batch_size"] = 15000  # train_batch_size > num_envs_per_worker * rollout_fragment_length
    # Whether to rollout "complete_episodes" or "truncate_episodes" to
    config["batch_mode"] = "truncate_episodes"

    # can be fraction
    config["num_gpus"] = 1

    config["gamma"] = 0.999
    config["lr"] = 5e-4

    config["log_level"] = "INFO"
    config["callbacks"] = {
        "on_episode_start": on_episode_start,
        "on_episode_step": on_episode_step,
        "on_episode_end": on_episode_end,
        "on_sample_end": on_sample_end,
        "on_train_result": on_train_result,
        "on_postprocess_traj": on_postprocess_traj,
    }

    config["ignore_worker_failures"] = False


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
    set_dqn_config(config)
    config["optimizer"]["max_weight_sync_delay"] = 400
    config["optimizer"]["num_replay_buffer_shards"] = 4
    config["optimizer"]["debug"] = False

    config["n_step"] = 3
    config["buffer_size"] = 2000000
    config["exploration_config"] = {"type": "PerWorkerEpsilonGreedy"}
    config["worker_side_prioritization"] = True
    config["min_iter_time_s"] = 10


def set_rainbow_config(config):
    set_dqn_config(config)
    config["num_atoms"] = 51  # rainbow: "num_atoms": [more than 1]
    config["v_min"] = -0.25  # expected returns should be between 0 and 1 since they're normalized
    config["v_max"] = 1.0
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
    config["use_critic"] = True  # required for GAE, use critic as baseline
    config["use_gae"] = True
    config["lambda"] = 0.95  # The GAE(lambda) parameter.
    config["kl_coeff"] = 0.5  # Initial coefficient for KL divergence.
    config["kl_target"] = 0.01 # Target value for KL divergence.

    config["sgd_minibatch_size"] = 100
    config["shuffle_sequences"] = True  # Whether to shuffle sequences in the batch when training (recommended).
    config["num_sgd_iter"] = 30

    config["vf_share_layers"] = True
    config["vf_loss_coeff"] = 0.5  # IMPORTANT: you must tune this if vf_share_layers = True
    config["entropy_coeff"] = 0.01
    config["entropy_coeff_schedule"] = None
    config["clip_param"] = 0.2

    config["vf_clip_param"] = 10.0  # Sensitive to the scale of the rewards. If your expected V is large, increase this.
    config["grad_clip"] = None  # If specified, clip the global norm of gradients by this amount.

    config["batch_mode"] = "truncate_episodes"


def set_impala_config(config):
    config["vtrace"] = True
    config["vtrace_clip_rho_threshold"] = 1.0
    config["vtrace_clip_pg_rho_threshold"] = 1.0

    config["num_data_loader_buffers"] = 1  # larger number goes faster but uses more GPU memory
    config["minibatch_buffer_size"] = 1  # number of train batches to  retain for minibatching, only effect if num_sgd_iter > 1
    config["num_sgd_iter"] = 3  # number of passes over each train batch
    config["replay_proportion"] = 0.8  # set to > 0 to use replay buffer
    config["replay_buffer_num_slots"] = 30000  # number of sample batches to store for replay
    config["learner_queue_size"] = 10  # training batches in queue to learner

    # Learning params.
    config["grad_clip"] = 40.0
    # either "adam" or "rmsprop"
    config["opt_type"] = "adam"

    # only used if rmsprop
    config["decay"] = 0.99
    config["momentum"] = 0.0
    config["epsilon"] = 0.1

    # balancing the three losses
    config["vf_loss_coeff"] = 0.5
    config["entropy_coeff"] = 0.01
    config["entropy_coeff_schedule"] = None


def set_appo_config(config):
    set_impala_config(config)
    config["vtrace"] = True  # v-trace of GAE advantages

    # only used if v_trace is False
    config["use_critic"] = True
    config["use_gae"] = True
    config["lambda"] = 0.95

    config["clip_param"] = 0.2
    config["use_kl_loss"] = False
    config["kl_coeff"] = 0.5
    config["kl_target"] = 0.01


def get_config_apex():
    config = apex.APEX_DEFAULT_CONFIG.copy()
    set_common_config(config)
    set_apex_config(config)
    return config


def get_config_rainbow():
    config = dqn.DEFAULT_CONFIG.copy()
    set_common_config(config)
    set_rainbow_config(config)
    return config


def get_config_appo():
    config = impala.DEFAULT_CONFIG.copy()
    set_common_config(config)
    set_appo_config(config)
    return config


def get_config_ppo():
    config = ppo.DEFAULT_CONFIG.copy()
    set_common_config(config)
    set_ppo_config(config)
    return config


def get_config_impala():
    config = impala.DEFAULT_CONFIG.copy()
    set_common_config(config)
    set_impala_config(config)
    return config


def get_simple_test_config():
    # used to check for bugs
    config = get_config_impala()

    # config["preprocessor_pref"] = None  # Does nothing
    # config["model"]["max_seq_len"] = 20

    config["num_workers"] = 1
    config["num_envs_per_worker"] = 6
    config["rollout_fragment_length"] = 500
    config["train_batch_size"] = 6*500
    config["num_gpus"] = 0
    config["evaluation_config"]["num_envs_per_worker"] = 6
    config["batch_mode"] = "truncate_episodes"
    config["minibatch_buffer_size"] = 1  # number of train batches to  retain for minibatching, only effect if num_sgd_iter > 1
    config["num_sgd_iter"] = 6

    config["num_data_loader_buffers"] = 1  # larger number goes faster but uses more GPU memory
    config["minibatch_buffer_size"] = 1  # number of train batches to  retain for minibatching, only effect if num_sgd_iter > 1
    config["num_sgd_iter"] = 1  # number of passes over each train batch
    config["replay_proportion"] = 0.5  # set to > 0 to use replay buffer
    config["replay_buffer_num_slots"] = 6*500  # number of sample batches to store for replay
    config["learner_queue_size"] = 1  # training batches in queue to learner

    config["eager"] = False
    config["log_level"] = "INFO"

    config["evaluation_interval"] = 1
    config["evaluation_num_episodes"] = 10

    return config
