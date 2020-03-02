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


def get_common_config():
    config = None

    COMMON_CONFIG = {
    # === Settings for Rollout Worker processes ===
    # Number of rollout worker actors to create for parallel sampling. Setting
    # this to 0 will force rollouts to be done in the trainer actor.
    "num_workers": 2,
    # Number of environments to evaluate vectorwise per worker. This enables
    # model inference batching, which can improve performance for inference
    # bottlenecked workloads.
    "num_envs_per_worker": 1,
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
    "sample_batch_size": 200,
    # Whether to rollout "complete_episodes" or "truncate_episodes" to
    # `sample_batch_size` length unrolls. Episode truncation guarantees more
    # evenly sized batches, but increases variance as the reward-to-go will
    # need to be estimated at truncation boundaries.
    "batch_mode": "truncate_episodes",

    # === Settings for the Trainer process ===
    # Number of GPUs to allocate to the trainer process. Note that not all
    # algorithms can take advantage of trainer GPUs. This can be fractional
    # (e.g., 0.3 GPUs).
    "num_gpus": 0,
    # Training batch size, if applicable. Should be >= sample_batch_size.
    # Samples batches will be concatenated together to a batch of this size,
    # which is then passed to SGD.
    "train_batch_size": 200,
    # Arguments to pass to the policy model. See models/catalog.py for a full
    # list of the available model options.
    "model": MODEL_DEFAULTS,
    # Arguments to pass to the policy optimizer. These vary by optimizer.
    "optimizer": {},

    # === Environment Settings ===
    # Discount factor of the MDP.
    "gamma": 0.99,
    # Number of steps after which the episode is forced to terminate. Defaults
    # to `env.spec.max_episode_steps` (if present) for Gym envs.
    "horizon": None,
    # Calculate rewards but don't reset the environment when the horizon is
    # hit. This allows value estimation and RNN state to span across logical
    # episodes denoted by horizon. This only has an effect if horizon != inf.
    "soft_horizon": False,
    # Don't set 'done' at the end of the episode. Note that you still need to
    # set this if soft_horizon=True, unless your env is actually running
    # forever without returning done=True.
    "no_done_at_end": False,
    # Arguments to pass to the env creator.
    "env_config": {},
    # Environment name can also be passed via config.
    "env": None,
    # Unsquash actions to the upper and lower bounds of env's action space
    "normalize_actions": False,
    # Whether to clip rewards prior to experience postprocessing. Setting to
    # None means clip for Atari only.
    "clip_rewards": None,
    # Whether to np.clip() actions to the action space low/high range spec.
    "clip_actions": True,
    # Whether to use rllib or deepmind preprocessors by default
    "preprocessor_pref": "deepmind",
    # The default learning rate.
    "lr": 0.0001,

    # === Debug Settings ===
    # Whether to write episode stats and videos to the agent log dir. This is
    # typically located in ~/ray_results.
    "monitor": False,
    # Set the ray.rllib.* log level for the agent process and its workers.
    # Should be one of DEBUG, INFO, WARN, or ERROR. The DEBUG level will also
    # periodically print out summaries of relevant internal dataflow (this is
    # also printed out once at startup at the INFO level). When using the
    # `rllib train` command, you can also use the `-v` and `-vv` flags as
    # shorthand for INFO and DEBUG.
    "log_level": "WARN",
    # Callbacks that will be run during various phases of training. These all
    # take a single "info" dict as an argument. For episode callbacks, custom
    # metrics can be attached to the episode by updating the episode object's
    # custom metrics dict (see examples/custom_metrics_and_callbacks.py). You
    # may also mutate the passed in batch data in your callback.
    "callbacks": {
        "on_episode_start": None,     # arg: {"env": .., "episode": ...}
        "on_episode_step": None,      # arg: {"env": .., "episode": ...}
        "on_episode_end": None,       # arg: {"env": .., "episode": ...}
        "on_sample_end": None,        # arg: {"samples": .., "worker": ...}
        "on_train_result": None,      # arg: {"trainer": ..., "result": ...}
        "on_postprocess_traj": None,  # arg: {
                                      #   "agent_id": ..., "episode": ...,
                                      #   "pre_batch": (before processing),
                                      #   "post_batch": (after processing),
                                      #   "all_pre_batches": (other agent ids),
                                      # }
    },
    # Whether to attempt to continue training if a worker crashes. The number
    # of currently healthy workers is reported as the "num_healthy_workers"
    # metric.
    "ignore_worker_failures": False,
    # Log system resource metrics to results. This requires `psutil` to be
    # installed for sys stats, and `gputil` for GPU metrics.
    "log_sys_usage": True,

    # === Framework Settings ===
    # Use PyTorch (instead of tf). If using `rllib train`, this can also be
    # enabled with the `--torch` flag.
    # NOTE: Some agents may not support `torch` yet and throw an error.
    "use_pytorch": False,

    # Enable TF eager execution (TF policies only). If using `rllib train`,
    # this can also be enabled with the `--eager` flag.
    "eager": False,
    # Enable tracing in eager mode. This greatly improves performance, but
    # makes it slightly harder to debug since Python code won't be evaluated
    # after the initial eager pass.
    "eager_tracing": False,
    # Disable eager execution on workers (but allow it on the driver). This
    # only has an effect if eager is enabled.
    "no_eager_on_workers": False,

    # === Exploration Settings ===
    # Default exploration behavior, iff `explore`=None is passed into
    # compute_action(s).
    # Set to False for no exploration behavior (e.g., for evaluation).
    "explore": True,
    # Provide a dict specifying the Exploration object's config.
    "exploration_config": {
        # The Exploration class to use. In the simplest case, this is the name
        # (str) of any class present in the `rllib.utils.exploration` package.
        # You can also provide the python class directly or the full location
        # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
        # EpsilonGreedy").
        "type": "StochasticSampling",
        # Add constructor kwargs here (if any).
    },
    # === Evaluation Settings ===
    # Evaluate with every `evaluation_interval` training iterations.
    # The evaluation stats will be reported under the "evaluation" metric key.
    # Note that evaluation is currently not parallelized, and that for Ape-X
    # metrics are already only reported for the lowest epsilon workers.
    "evaluation_interval": None,
    # Number of episodes to run per evaluation period. If using multiple
    # evaluation workers, we will run at least this many episodes total.
    "evaluation_num_episodes": 10,
    # Internal flag that is set to True for evaluation workers.
    "in_evaluation": False,
    # Typical usage is to pass extra args to evaluation env creator
    # and to disable exploration by computing deterministic actions.
    # IMPORTANT NOTE: Policy gradient algorithms are able to find the optimal
    # policy, even if this is a stochastic one. Setting "explore=False" here
    # will result in the evaluation workers not using this optimal policy!
    "evaluation_config": {
        # Example: overriding env_config, exploration, etc:
        # "env_config": {...},
        # "explore": False
    },
    # Number of parallel workers to use for evaluation. Note that this is set
    # to zero by default, which means evaluation will be run in the trainer
    # process. If you increase this, it will increase the Ray resource usage
    # of the trainer since evaluation workers are created separately from
    # rollout workers.
    "evaluation_num_workers": 0,
    # Customize the evaluation method. This must be a function of signature
    # (trainer: Trainer, eval_workers: WorkerSet) -> metrics: dict. See the
    # Trainer._evaluate() method to see the default implementation. The
    # trainer guarantees all eval workers have the latest policy state before
    # this function is called.
    "custom_eval_function": None,

    # === Advanced Rollout Settings ===
    # Use a background thread for sampling (slightly off-policy, usually not
    # advisable to turn on unless your env specifically requires it).
    "sample_async": False,
    # Element-wise observation filter, either "NoFilter" or "MeanStdFilter".
    "observation_filter": "NoFilter",
    # Whether to synchronize the statistics of remote filters.
    "synchronize_filters": True,
    # Configures TF for single-process operation by default.
    "tf_session_args": {
        # note: overriden by `local_tf_session_args`
        "intra_op_parallelism_threads": 2,
        "inter_op_parallelism_threads": 2,
        "gpu_options": {
            "allow_growth": True,
        },
        "log_device_placement": False,
        "device_count": {
            "CPU": 1
        },
        "allow_soft_placement": True,  # required by PPO multi-gpu
    },
    # Override the following tf session args on the local worker
    "local_tf_session_args": {
        # Allow a higher level of parallelism by default, but not unlimited
        # since that can cause crashes with many concurrent drivers.
        "intra_op_parallelism_threads": 8,
        "inter_op_parallelism_threads": 8,
    },
    # Whether to LZ4 compress individual observations
    "compress_observations": False,
    # Wait for metric batches for at most this many seconds. Those that
    # have not returned in time will be collected in the next train iteration.
    "collect_metrics_timeout": 180,
    # Smooth metrics over this many episodes.
    "metrics_smoothing_episodes": 100,
    # If using num_envs_per_worker > 1, whether to create those new envs in
    # remote processes instead of in the same worker. This adds overheads, but
    # can make sense if your envs can take much time to step / reset
    # (e.g., for StarCraft). Use this cautiously; overheads are significant.
    "remote_worker_envs": False,
    # Timeout that remote workers are waiting when polling environments.
    # 0 (continue when at least one env is ready) is a reasonable default,
    # but optimal value could be obtained by measuring your environment
    # step / reset and model inference perf.
    "remote_env_batch_wait_ms": 0,
    # Minimum time per train iteration
    "min_iter_time_s": 0,
    # Minimum env steps to optimize for per train call. This value does
    # not affect learning, only the length of train iterations.
    "timesteps_per_iteration": 0,
    # This argument, in conjunction with worker_index, sets the random seed of
    # each worker, so that identically configured trials will have identical
    # results. This makes experiments reproducible.
    "seed": None,

    # === Advanced Resource Settings ===
    # Number of CPUs to allocate per worker.
    "num_cpus_per_worker": 1,
    # Number of GPUs to allocate per worker. This can be fractional. This is
    # usually needed only if your env itself requires a GPU (i.e., it is a
    # GPU-intensive video game), or model inference is unusually expensive.
    "num_gpus_per_worker": 0,
    # Any custom Ray resources to allocate per worker.
    "custom_resources_per_worker": {},
    # Number of CPUs to allocate for the trainer. Note: this only takes effect
    # when running in Tune. Otherwise, the trainer runs in the main program.
    "num_cpus_for_driver": 1,
    # You can set these memory quotas to tell Ray to reserve memory for your
    # training run. This guarantees predictable execution, but the tradeoff is
    # if your workload exceeeds the memory quota it will fail.
    # Heap memory to reserve for the trainer process (0 for unlimited). This
    # can be large if your are using large train batches, replay buffers, etc.
    "memory": 0,
    # Object store memory to reserve for the trainer process. Being large
    # enough to fit a few copies of the model weights should be sufficient.
    # This is enabled by default since models are typically quite small.
    "object_store_memory": 0,
    # Heap memory to reserve for each worker. Should generally be small unless
    # your environment is very heavyweight.
    "memory_per_worker": 0,
    # Object store memory to reserve for each worker. This only needs to be
    # large enough to fit a few sample batches at a time. This is enabled
    # by default since it almost never needs to be larger than ~200MB.
    "object_store_memory_per_worker": 0,

    # === Offline Datasets ===
    # Specify how to generate experiences:
    #  - "sampler": generate experiences via online simulation (default)
    #  - a local directory or file glob expression (e.g., "/tmp/*.json")
    #  - a list of individual file paths/URIs (e.g., ["/tmp/1.json",
    #    "s3://bucket/2.json"])
    #  - a dict with string keys and sampling probabilities as values (e.g.,
    #    {"sampler": 0.4, "/tmp/*.json": 0.4, "s3://bucket/expert.json": 0.2}).
    #  - a function that returns a rllib.offline.InputReader
    "input": "sampler",
    # Specify how to evaluate the current policy. This only has an effect when
    # reading offline experiences. Available options:
    #  - "wis": the weighted step-wise importance sampling estimator.
    #  - "is": the step-wise importance sampling estimator.
    #  - "simulation": run the environment in the background, but use
    #    this data for evaluation only and not for learning.
    "input_evaluation": ["is", "wis"],
    # Whether to run postprocess_trajectory() on the trajectory fragments from
    # offline inputs. Note that postprocessing will be done using the *current*
    # policy, not the *behavior* policy, which is typically undesirable for
    # on-policy algorithms.
    "postprocess_inputs": False,
    # If positive, input batches will be shuffled via a sliding window buffer
    # of this number of batches. Use this if the input data is not in random
    # enough order. Input is delayed until the shuffle buffer is filled.
    "shuffle_buffer_size": 0,
    # Specify where experiences should be saved:
    #  - None: don't save any experiences
    #  - "logdir" to save to the agent log dir
    #  - a path/URI to save to a custom output directory (e.g., "s3://bucket/")
    #  - a function that returns a rllib.offline.OutputWriter
    "output": None,
    # What sample batch columns to LZ4 compress in the output data.
    "output_compress_columns": ["obs", "new_obs"],
    # Max output file size before rolling over to a new file.
    "output_max_file_size": 64 * 1024 * 1024,

    # === Settings for Multi-Agent Environments ===
    "multiagent": {
        # Map from policy ids to tuples of (policy_cls, obs_space,
        # act_space, config). See rollout_worker.py for more info.
        "policies": {},
        # Function mapping agent ids to policy ids.
        "policy_mapping_fn": None,
        # Optional whitelist of policies to train, or None for all policies.
        "policies_to_train": None,
    },
    }

    return config


def get_config_ppo():
    config = ppo.DEFAULT_CONFIG.copy()

    return config


def get_config_impala():
    # https://ray.readthedocs.io/en/latest/rllib-training.html#common-parameters
    # https://ray.readthedocs.io/en/latest/rllib-algorithms.html#importance-weighted-actor-learner-architecture-impala

    # DEFAULT_CONFIG = with_common_config({
    # # V-trace params (see vtrace.py).
    # "vtrace": True,
    # "vtrace_clip_rho_threshold": 1.0,
    # "vtrace_clip_pg_rho_threshold": 1.0,
    # # System params.
    # #
    # # == Overview of data flow in IMPALA ==
    # # 1. Policy evaluation in parallel across `num_workers` actors produces
    # #    batches of size `sample_batch_size * num_envs_per_worker`.
    # # 2. If enabled, the replay buffer stores and produces batches of size
    # #    `sample_batch_size * num_envs_per_worker`.
    # # 3. If enabled, the minibatch ring buffer stores and replays batches of
    # #    size `train_batch_size` up to `num_sgd_iter` times per batch.
    # # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
    # #    on batches of size `train_batch_size`.
    # #
    # "sample_batch_size": 50,
    # "train_batch_size": 500,
    # "min_iter_time_s": 10,
    # "num_workers": 2,
    # # number of GPUs the learner should use.
    # "num_gpus": 1,
    # # set >1 to load data into GPUs in parallel. Increases GPU memory usage
    # # proportionally with the number of buffers.
    # "num_data_loader_buffers": 1,
    # # how many train batches should be retained for minibatching. This conf
    # # only has an effect if `num_sgd_iter > 1`.
    # "minibatch_buffer_size": 1,
    # # number of passes to make over each train batch
    # "num_sgd_iter": 1,
    # # set >0 to enable experience replay. Saved samples will be replayed with
    # # a p:1 proportion to new data samples.
    # "replay_proportion": 0.0,
    # # number of sample batches to store for replay. The number of transitions
    # # saved total will be (replay_buffer_num_slots * sample_batch_size).
    # "replay_buffer_num_slots": 0,
    # # max queue size for train batches feeding into the learner
    # "learner_queue_size": 16,
    # # wait for train batches to be available in minibatch buffer queue
    # # this many seconds. This may need to be increased e.g. when training
    # # with a slow environment
    # "learner_queue_timeout": 300,
    # # level of queuing for sampling.
    # "max_sample_requests_in_flight_per_worker": 2,
    # # max number of workers to broadcast one set of weights to
    # "broadcast_interval": 1,
    # # use intermediate actors for multi-level aggregation. This can make sense
    # # if ingesting >2GB/s of samples, or if the data requires decompression.
    # "num_aggregation_workers": 0,
    #
    # # Learning params.
    # "grad_clip": 40.0,
    # # either "adam" or "rmsprop"
    # "opt_type": "adam",
    # "lr": 0.0005,
    # "lr_schedule": None,
    # # rmsprop considered
    # "decay": 0.99,
    # "momentum": 0.0,
    # "epsilon": 0.1,
    # # balancing the three losses
    # "vf_loss_coeff": 0.5,
    # "entropy_coeff": 0.01,
    # "entropy_coeff_schedule": None,
    #
    # # use fake (infinite speed) sampler for testing
    # "_fake_sampler": False,
    # })

    config = impala.DEFAULT_CONFIG.copy()

    # environment
    config["env"] = "procgen:procgen-coinrun-v0"
    config["env_config"] = {
        "distrubution_mode": "memory"
    }

    # vtrace
    config["vtrace"] = False
    config["vtrace_clip_rho_threshold"] = 1.0
    config["vtrace_clip_pg_rho_threshold"] = 1.0

    # hardware config
    config["num_gpus"] = 0  # TODO: change this for cluster

    # workser config
    config["num_workers"] = 1
    config["num_envs_per_worker"] = 1
    config["eager"] = False  # Eager modes crashes (at least with many trainable parameters)
    config["eager_tracing"] = False  # setting to True, greatly improves performance in eager mode
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
    config["monitor"] = False
    config["log_level"] = "INFO"
    config["callbacks"] = {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
            "on_sample_end": on_sample_end,
            "on_train_result": on_train_result,
            "on_postprocess_traj": on_postprocess_traj,
    }

    # custom model options
    config["model"]["custom_preprocessor"] = "procgen_preproc"
    config["model"]["custom_model"] = "transformer_model"  # "lstm_model"
    config["model"]["custom_action_dist"] = None
    config["model"]["custom_options"] = {}

    return config
