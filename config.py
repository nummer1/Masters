import ray
import ray.rllib.agents.impala as impala
from ray.rllib.agents import ppo
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


# TODO:
# config["env_config"] = {
#     "distrubution_mode": "memory"
# }


def set_common_config(config):
    config["model"]["custom_model"] = "lstm_model"
    config["model"]["custom_action_dist"] = None
    config["model"]["custom_options"] = {}
    config["model"]["custom_preprocessor"] = "procgen_preproc"

    config["env"] = "procgen:procgen-coinrun-v0"
    config["env_config"] = {}  # TODO

    # === Settings for Rollout Worker processes ===
    # Number of rollout worker actors to create for parallel sampling. Setting
    # this to 0 will force rollouts to be done in the trainer actor.
    config["num_workers"] = 7
    # Number of environments to evaluate vectorwise per worker. This enables
    # model inference batching, which can improve performance for inference
    # bottlenecked workloads.
    config["num_envs_per_worker"] = 64
    # Default sample batch size (unroll length). Batches of this size are
    # collected from rollout workers until train_batch_size is met. When using
    # multiple envs per worker, this is multiplied by num_envs_per_worker.
    #
    # For example, given sample_batch_size=100 and train_batch_size=1000:
    #   1. RLlib will collect 10 batches of size 100 from the rollout workers.
    #   2. These batches are concatenated and we perform an epoch of SGD.
    #
    # If we further set num_envs_per_worker=5, then the sample batches will be
    # of size 5*100 = 500, and RLlib will only collect 2 batches per epoch.
    #
    # The exact workflow here can vary per algorithm. For example, PPO further
    # divides the train batch into minibatches for multi-epoch SGD.
    config["sample_batch_size"] = 50
    # Whether to rollout "complete_episodes" or "truncate_episodes" to
    # `sample_batch_size` length unrolls. Episode truncation guarantees more
    # evenly sized batches, but increases variance as the reward-to-go will
    # need to be estimated at truncation boundaries.
    config["batch_mode"] = "truncate_episodes"

    # === Settings for the Trainer process ===
    # Number of GPUs to allocate to the trainer process. Note that not all
    # algorithms can take advantage of trainer GPUs. This can be fractional
    # (e.g., 0.3 GPUs).
    config["num_gpus"] = 0
    # Training batch size, if applicable. Should be >= sample_batch_size.
    # Samples batches will be concatenated together to a batch of this size,
    # which is then passed to SGD.
    config["train_batch_size"] = 10000


    config["gamma"] = 0.999
    config["lr"] = 5.0 * (10 ** -4)

    config["log_level"] = "INFO"
    config["callbacks"] = {
        "on_episode_start": on_episode_start,
        "on_episode_step": on_episode_step,
        "on_episode_end": on_episode_end,
        "on_sample_end": on_sample_end,
        "on_train_result": on_train_result,
        "on_postprocess_traj": on_postprocess_traj,
    }

    # TODO: below crashes when activated
    # config["evaluation_interval"] = 10
    # config["evaluation_num_episodes"] = 10

    # # === Exploration Settings ===
    # config["explore"] = True
    # # Provide a dict specifying the Exploration object's config.
    # config["exploration_config"] = {
    #     # The Exploration class to use. In the simplest case, this is the name
    #     # (str) of any class present in the `rllib.utils.exploration` package.
    #     # You can also provide the python class directly or the full location
    #     # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
    #     # EpsilonGreedy").
    #     "type": "StochasticSampling",
    #     # Add constructor kwargs here (if any).
    # },


def get_config_rainbow():
    config = None
    set_common_config(config)

    # config num_workers = 8
    # learning rate = 2.5 * 10^-4

    return config


def get_config_ppo():
    config = ppo.DEFAULT_CONFIG.copy()
    set_common_config(config)

    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    config["use_critic"] = True
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    config["use_gae"] = True
    # The GAE(lambda) parameter.
    config["lambda"] = 0.95
    # Initial coefficient for KL divergence.
    config["kl_coeff"] = 0.5
    # Size of batches collected from each worker.
    config["sample_batch_size"] = 200
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    config["train_batch_size"] = 40000
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    config["sgd_minibatch_size"] = 128
    # Whether to shuffle sequences in the batch when training (recommended).
    config["shuffle_sequences"] = True
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    config["num_sgd_iter"] = 30
    # Stepsize of SGD.
    config["lr"] = 5e-4
    # Learning rate schedule.
    config["lr_schedule"] = None
    # Share layers for value function. If you set this to True, it's important
    # to tune vf_loss_coeff.
    config["vf_share_layers"] = False
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers: True.
    config["vf_loss_coeff"] = 1.0
    # Coefficient of the entropy regularizer.
    config["entropy_coeff"] = 0.01
    # Decay schedule for the entropy regularizer.
    config["entropy_coeff_schedule"] = None
    # PPO clip parameter.
    config["clip_param"] = 0.2
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    config["vf_clip_param"] = 10.0
    # If specified, clip the global norm of gradients by this amount.
    config["grad_clip"] = None
    # Target value for KL divergence.
    config["kl_target"] = 0.01
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    config["batch_mode"] = "truncate_episodes"
    # Uses the sync samples optimizer instead of the multi-gpu one. This is
    # usually slower, but you might want to try it if you run into issues with
    # the default optimizer.
    config["simple_optimizer"] = False

    return config


def get_config_impala():
    config = impala.DEFAULT_CONFIG.copy()
    set_common_config(config)

    # V-trace params (see vtrace.py).
    config["vtrace"] = True
    config["vtrace_clip_rho_threshold"] = 1.0
    config["vtrace_clip_pg_rho_threshold"] = 1.0

    # System params.
    #
    # == Overview of data flow in IMPALA ==
    # 1. Policy evaluation in parallel across `num_workers` actors produces
    #    batches of size `sample_batch_size * num_envs_per_worker`.
    # 2. If enabled, the replay buffer stores and produces batches of size
    #    `sample_batch_size * num_envs_per_worker`.
    # 3. If enabled, the minibatch ring buffer stores and replays batches of
    #    size `train_batch_size` up to `num_sgd_iter` times per batch.
    # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
    #    on batches of size `train_batch_size`.
    #
    # config["sample_batch_size"] = 50
    # config["train_batch_size"] = 8192
    # config["num_workers"] = 7

    # set >1 to load data into GPUs in parallel. Increases GPU memory usage
    # proportionally with the number of buffers.
    config["num_data_loader_buffers"] = 1
    # how many train batches should be retained for minibatching. This conf
    # only has an effect if `num_sgd_iter > 1`.
    config["minibatch_buffer_size"] = 1
    # number of passes to make over each train batch
    config["num_sgd_iter"] = 3
    # set >0 to enable experience replay. Saved samples will be replayed with
    # a p:1 proportion to new data samples.
    config["replay_proportion"] = 0.8
    # number of sample batches to store for replay. The number of transitions
    # saved total will be (replay_buffer_num_slots * sample_batch_size).
    config["replay_buffer_num_slots"] = 1024
    # max queue size for train batches feeding into the learner
    config["learner_queue_size"] = 16


    # Learning params.
    config["grad_clip"] = 40.0
    # either "adam" or "rmsprop"
    config["opt_type"] = "adam"
    config["lr"] = 5e-4
    config["lr_schedule"] = None

    # rmsprop considered
    config["decay"] = 0.99
    config["momentum"] = 0.0
    config["epsilon"] = 0.1

    # balancing the three losses
    config["vf_loss_coeff"] = 0.5
    config["entropy_coeff"] = 0.01
    config["entropy_coeff_schedule"] = None

    return config
