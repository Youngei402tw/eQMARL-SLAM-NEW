import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# eqmarl imports TensorFlow and initializes the GPU, so configure its allocator first.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import eqmarl
import tensorflow.keras as keras
from datetime import datetime
import yaml
from importlib import import_module
import gymnasium as gym
from typing import Union
import argparse
import copy
import random
import numpy as np
import tensorflow as tf


def load_obj_from_dotpath(path: str):
    """Load object from within module. 
    
    The path should be `.` delimited and with fully specified package names (e.g., `numpy.sum`).
    """
    module, obj = path.rsplit(".", maxsplit=1)
    m = import_module(module)
    return getattr(m, obj)


def load_env(config: dict) -> gym.Env:
    env_func = load_obj_from_dotpath(config['func'])
    env_params = config['params']
    env = env_func(env_params)
    return env


def load_model(config: dict) -> keras.Model:
    init_func = load_obj_from_dotpath(config['init_func'])
    init_params = config['init_params']
    model: keras.Model = init_func(**init_params)
    model.build(config['build_shape'])
    return model


def load_optimizer(config: Union[dict,list[dict]]) -> Union[keras.optimizers.Optimizer, list[keras.optimizers.Optimizer]]:
    # List of optimizers, one for each trainable variable.
    if isinstance(config, list):
        optimizers: list[keras.optimizers.Optimizer] = []
        for opt_dict in config:
            opt_func = load_obj_from_dotpath(opt_dict['func'])
            optimizer = opt_func(**opt_dict['params'])
            optimizers.append(optimizer)
        return optimizers
    # One optimizer for the entire model.
    else:
        opt_dict = config
        opt_func = load_obj_from_dotpath(opt_dict['func'])
        optimizer: keras.optimizers.Optimizer = opt_func(**opt_dict['params'])
        return optimizer


def load_experiment(config: dict, flag_print_model_summary: bool = False) -> dict:
    
    config_exp = config['experiment']
    roots = config_exp['roots']

    # Seed before constructing models so weight initialization is reproducible.
    seed = config_exp['algorithm']['init_params'].get('seed', None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    
    # Load the algorithm.
    config_algo = config['experiment']['algorithm']
    algo_init_func = load_obj_from_dotpath(config_algo['init_func'])
    algo_init_params = config_algo['init_params']
    if 'episode_metrics_callback' in algo_init_params:
        algo_init_params['episode_metrics_callback'] = load_obj_from_dotpath(algo_init_params['episode_metrics_callback'])

    
    # Environment.
    config_env = algo_init_params['env']
    env = load_env(config_env)
    algo_init_params['env'] = env # Overwrite the config.
    
    # Models.
    model_keys = [k for k in algo_init_params.keys() if 'model' in k]
    for key in model_keys:
        model_config = algo_init_params[key]
        model = load_model(model_config)
        algo_init_params[key] = model # Overwrite the config.
        if flag_print_model_summary:
            print(model.summary())

    # Optimizers.
    optimizer_keys = [k for k in algo_init_params.keys() if 'optimizer' in k]
    for key in optimizer_keys:
        optimizer_config = algo_init_params[key]
        optimizer = load_optimizer(optimizer_config)
        algo_init_params[key] = optimizer # Overwrite the config.
        
        
    algo = algo_init_func(**algo_init_params)
    
    # Load training parameters.
    train = config_exp['train']
    if 'callbacks' in train:
        cbs = []
        for cbd in train['callbacks']:
            cb_func = load_obj_from_dotpath(cbd['func'])
            cb_params = cbd['params']
            cb = cb_func(**cb_params)
            cbs.append(cb)
        train['callbacks'] = cbs # Overwrite the config.

    return dict(
        roots=roots,
        algorithm=algo,
        train=train,
        save=config_exp['save'],
    )


def set_round_seed(config: dict, round_index: int, stride: int = 100000):
    """Give each training round a reproducible, non-overlapping seed range."""
    params = config['experiment']['algorithm']['init_params']
    if 'seed' not in params:
        return
    round_seed = int(params['seed']) + int(round_index) * int(stride)
    params['seed'] = round_seed
    env_params = params.get('env', {}).get('params', {})
    if 'seed' in env_params:
        env_params['seed'] = round_seed


def apply_train_overrides(config: dict, n_episodes=None, max_steps_per_episode=None):
    """Apply validated command-line overrides to an experiment definition."""
    train = config['experiment']['train']
    if n_episodes is not None:
        if n_episodes <= 0:
            raise ValueError("--n-episodes must be positive")
        train['n_episodes'] = n_episodes
    if max_steps_per_episode is not None:
        if max_steps_per_episode <= 0:
            raise ValueError("--max-steps-per-episode must be positive")
        train['max_steps_per_episode'] = max_steps_per_episode
        env_params = config['experiment']['algorithm']['init_params'].get('env', {}).get('params', {})
        if 'time_limit' in env_params:
            env_params['time_limit'] = max_steps_per_episode


def apply_seed_override(config: dict, seed: int):
    """Set the algorithm and environment seed for a standalone cluster job."""
    params = config['experiment']['algorithm']['init_params']
    params['seed'] = int(seed)
    params.get('env', {}).get('params', {})['seed'] = int(seed)


def apply_pilot_monitor(config: dict):
    """Attach early stopping used only by multi-seed pilot jobs."""
    config['experiment']['train'].setdefault('callbacks', []).append(
        {
            'func': 'eqmarl.ActiveSLAMPilotMonitor',
            'params': {
                'window': 100,
                'turn_threshold': 0.85,
                'max_forward_fraction': 0.05,
                'coverage_drop': 0.10,
                'verbose': True,
            },
        }
    )


def apply_output_protocol(config: dict, protocol: str):
    """Isolate fast, pilot, and full outputs without duplicating configurations."""
    if protocol not in {'fast', 'pilot', 'full'}:
        raise ValueError(f"unsupported active-SLAM output protocol: {protocol}")
    namespaces = (
        'active_slam_faithful_',
        'active_slam_bounded_pose_',
        'active_slam_milestone99_',
    )

    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        if isinstance(value, str):
            for namespace in namespaces:
                value = value.replace(
                    f'{namespace}full_', f'{namespace}{protocol}_'
                )
        return value

    config['experiment'] = replace(config['experiment'])


def apply_fast_preset(config: dict):
    """Shrink active-SLAM models and environment for rapid iteration only."""
    experiment = config['experiment']
    train = experiment['train']
    params = experiment['algorithm']['init_params']
    env = params['env']['params']
    patch_size, n_beams = 7, 16
    local_dim = patch_size * patch_size * 3
    train.update(n_episodes=50, max_steps_per_episode=50)
    env.update(map_size=16, n_beams=n_beams, patch_size=patch_size, time_limit=50)
    for key, model_config in params.items():
        if not key.startswith('model_'):
            continue
        init_params = model_config['init_params']
        init_params['observation_dim'] = local_dim
        if 'n_layers' in init_params:
            init_params['n_layers'] = 2
            if 'encoder_units' in init_params:
                init_params['encoder_units'] = [32]
        elif 'units' in init_params:
            init_params['units'] = [32] if key == 'model_actor' else [16]
        shape = model_config['build_shape']
        model_config['build_shape'] = [None, local_dim] if len(shape) == 2 else [
            None, 2, local_dim
        ]





def main(
    config: str,
    n_train_rounds: int,
    flag_print_model_summary: bool = False,
    flag_dry_run: bool = False,
    ):

    # Time of training session start.
    datetime_session = datetime.now()
    print(f"Training session start at {datetime_session.isoformat()}")

    if n_train_rounds > 1:
        print(f'Training for {n_train_rounds} rounds')

    # Iteratively 
    for r in range(n_train_rounds):

        config_session = copy.deepcopy(config)
        set_round_seed(config_session, r)
        round_seed = config_session['experiment']['algorithm']['init_params'].get('seed', 0)
        session_dir = Path(config_session['experiment']['roots']['session_dir'].format(
            datetime_session=datetime_session, seed=round_seed
        ))
        if not flag_dry_run:
            session_dir.expanduser().mkdir(parents=True, exist_ok=True)
            with (session_dir / 'config.yml').open('w') as config_file:
                yaml.safe_dump(config_session, config_file, sort_keys=False)
        exp = load_experiment(config_session, flag_print_model_summary=flag_print_model_summary)
        algo: eqmarl.Algorithm = exp['algorithm']
        train_params = exp['train']
    
        # Terminate if dry run.
        if flag_dry_run:
            break
        
        # Save some of the session and round details within the algorithm so that callbacks and other entities will have access to them.
        algo.datetime_session = datetime_session
        algo.round = r

        round_start = datetime.now()
        if n_train_rounds > 1:
            print(f'Training round {r} start: {round_start}')

        # Train models using algorithm.
        reward_history, metrics_history = algo.train(
            **train_params,
            )

        # Save results to file if a metrics file was provided.
        metrics_file = exp['save'].get('metrics_file', None)
        if metrics_file is not None:
            metrics_file = metrics_file.format(
                datetime_session=datetime_session,
                round=r,
                seed=algo.seed,
            )
            algo.save_train_results(metrics_file, reward_history, metrics_history)
            print(f"Saved metrics file {metrics_file}")
        
        # Save models to file if filenames were provided.
        for d in exp['save'].get('model_files', []):
            model_file = d['filepath'].format(
                datetime_session=datetime_session,
                round=r,
                seed=algo.seed,
            )
            algo.save_model(d['name'], model_file, d['save_weights_only'])
            print(f"Saved model file {model_file}")
        
        # Print the round ending time and elapsed time.
        if n_train_rounds > 1:
            round_end = datetime.now()
            print(f'Training round {r} end: {round_end}')
            print(f'Training round {r} elapsed: {round_end - round_start}')
            print()
    
    # Print the ending time and how much time has elapsed.
    datetime_session_end = datetime.now()
    print(f"Training session end at {datetime_session_end.isoformat()} (elapsed {datetime_session_end-datetime_session})")


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('config',
        help='Experiment config file in YAML format.',
        )
    parser.add_argument('-r', '--n-train-rounds',
        type=int,
        default=1,
        help='Number of times to perform training.',
        )
    parser.add_argument('-p', '--print-model-summary',
        action='store_true',
        default=False,
        help='Print model summary.',
        )
    parser.add_argument('-d', '--dry-run',
        action='store_true',
        default=False,
        help='Dry run of experiment, only loads experiment files and preps for experiment to be run but does not actually train anything; useful for testing.',
        )
    parser.add_argument('--n-episodes',
        type=int,
        default=None,
        help='Override the number of training episodes from the experiment configuration.',
        )
    parser.add_argument('--max-steps-per-episode',
        type=int,
        default=None,
        help='Override the maximum episode length from the experiment configuration.',
        )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--fast',
        action='store_true',
        help='Use a smaller active-SLAM environment and model for rapid iteration.',
        )
    mode.add_argument('--pilot',
        action='store_true',
        help='Enable active-SLAM policy-collapse monitoring for a pilot run.',
        )
    parser.add_argument('--seed',
        type=int,
        default=None,
        help='Override the algorithm and environment seed.',
        )
    
    
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    # Get program options.
    opts = get_opts()
    
    # Load the YAML config file.
    print(f"Loading experiment: {opts.config}")
    config_path = Path(opts.config)
    assert config_path.exists(), f"experiment config file does not exist: {opts.config}"
    with open(config_path) as f:
        config = yaml.load(f, Loader=eqmarl.yaml.ConfigLoader)

    if opts.fast:
        apply_fast_preset(config)
        apply_output_protocol(config, 'fast')
    if opts.pilot:
        apply_pilot_monitor(config)
        apply_output_protocol(config, 'pilot')
    if opts.seed is not None:
        apply_seed_override(config, opts.seed)
    apply_train_overrides(
        config,
        n_episodes=opts.n_episodes,
        max_steps_per_episode=opts.max_steps_per_episode,
    )

    # Run the experiment.
    main(
        config=config,
        n_train_rounds=opts.n_train_rounds,
        flag_print_model_summary=opts.print_model_summary,
        flag_dry_run=opts.dry_run,
    )
