import ray
import ray.rllib.agents.impala as impala


def get_config_impala():
    config = impala.DEFAULT_CONFIG.copy()
    config["num_gpus"] = 0
    config["num_workers"] = 4
    config["eager"] = False
    config["eager_tracing"] = False  # setting to true greatly improves performance in eager mode
    config["gamma"] = 0.99

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
    config["entropy_coeff_schedule"] = None
    return config
