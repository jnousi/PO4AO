config = {
    'RL': {
        'iterations':  51,
        'episode_length': 500,
        'warmup_episodes':  20,
        'max_sigma':  0.015,
        'min_sigma':0.0,
        'loss_function_penalty':0.15
    },
    'training': {
        'dynamics_grad_steps':  20,
        'policy_grad_steps':  10,
        'dynamics_grad_steps_warmup':  300,
        'policy_grad_steps_warmup':  150,
    },
    'MDP': {
        'n_history':32,
        'planning_horizon': 4,
        'data_shape': 24, # set by the DM
        'control_delay' : 1
    },
    'replay_buffers': {
        'replay_size': 160,
        'warmup_memory': 20,
        'train_warmup_percent': 0.2,
    },
    'integrator':{
        'gain': 0.5, # only for the warm up
        'leak': 0.99, # also for RL
        'n_modes': 275,
        'integrator': False # use the integrator as policy
    },
    'NN_models':{
        'filters_per_layer':64,
        'training_batch':32,
        'initial_std':0.01,
        'initial_mean':0,
    },
    'save_and_load':{
        'save_models_pretrained': False,
        'load_models_pretrained': False,
        'save_warmup_buffer': False,
        'load_warmup_buffer': False,
    }
}
