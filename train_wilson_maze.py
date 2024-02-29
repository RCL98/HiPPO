import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import yaml
import time
import copy
import pandas as pd
import numpy as np
import struct

from sklearn.model_selection import train_test_split

import wilson_maze_env
from wilson_maze_callback import WilsonMazeCallback

import torch
import gymnasium as gym

from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.ppo_recurrent import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from wandb.integration.sb3 import WandbCallback
import wandb

set_random_seed(32, True)


# # Custom actor (pi) and value function (vf) networks
# # of two layers of size 32 each with Relu activation function
# # Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
# policy_kwargs = dict(net_arch=dict(pi=[256, 64], vf=[256, 64]), activation_fn=torch.nn.ReLU)


def make_env(env_id: str, rank: int, x: np.ndarray, y: np.ndarray, seed: int = 0, **kwargs):
    """
        Utility function for multiprocessed env.

        :param env_id: the environment ID
        :param rank: index of the subprocess
        :param x: the input embedding data
        :param y: the target and coin data
        :param seed: the initial seed for RNG
    """

    def _init():
        target_id = (seed + rank) % kwargs.get('number_of_targets', 1)
        target_indexes = np.where(y[:, 0] == target_id)
        x_target = x[target_indexes]
        y_target = y[target_indexes][:, 1]

        # print('Target id: ', target_id)
        env = gym.make(env_id, target_id=target_id,
                       prompts=x_target, should_pickup_coins=y_target, **kwargs)

        return Monitor(env)

    set_random_seed(seed)
    return _init


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_input_data(dataset_path: str, embd_path: str, embd_size=4096) -> tuple[np.ndarray, np.ndarray]:
    """
        Load the input data from the dataset and the embeddings file.

        :param dataset_path: the path to the dataset file
        :param embd_path: the path to the embeddings file
        :param embd_size: the size of the embeddings
        :return: a tuple of two numpy arrays, the first one is the input embedding data and the second one is the target data
    """
    df = pd.read_csv(dataset_path, sep=',')
    targets = df['target'].to_numpy()
    coins = df['coin'].to_numpy()
    Y = np.stack([targets, coins], axis=1)

    with open(embd_path, 'rb') as f:
        data = f.read()
        embds = struct.unpack('f' * int(len(data) / 4), data)
        X = np.vstack([np.array(embds[i:i + embd_size]) for i in range(0, len(embds), embd_size)])

    return X, Y


def parse_policy_kwargs(in_policy_kwargs: dict) -> dict:
    """
        Parse the policy_kwargs dictionary and replace the string values with the actual classes.

        :param in_policy_kwargs: the policy_kwargs dictionary to parse
        :return: a new dictionary with the string values replaced with the actual classes
    """

    return {
        "net_arch": {
            "pi": [x for x in in_policy_kwargs['net_arch']['pi']],
            "vf": [x for x in in_policy_kwargs['net_arch']['vf']],
        },
        "activation_fn": torch.nn.ReLU if in_policy_kwargs['activation_fn'] == 'ReLU' else torch.nn.Tanh
    }


def transform_config(dictionary: dict) -> dict:
    """
        Convert the float values in the dictionary to integers and
        resolves power expressions in the dictionary.

        :param dictionary: the dictionary to convert
        :return: a new dictionary with the float values converted to integers
    """
    new_dict = {}
    for k, v in dictionary.items():
        if isinstance(v, str) and '^' in v:
            new_dict[k] = eval(v.replace('^', '**'))
        elif isinstance(v, float):
            new_dict[k] = int(v)
        else:
            new_dict[k] = v
    return new_dict


if __name__ == "__main__":
    with open('configs/llama_config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    run_config = transform_config(config['run_config'])
    policy_kwargs = parse_policy_kwargs(config['policy_kwargs'])
    env_config = config['env_config']

    embeds, targets = get_input_data(run_config['dataset_path'], run_config['embeddings_path'], run_config['embedding_size'])
    embeds, targets = embeds[:9000], targets[:9000]
    X_train, X_valid, y_train, y_valid = train_test_split(embeds, targets, test_size=0.3, 
                                                            random_state=run_config['random_state'], stratify=targets[:, 0])
    
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

    # run = dotdict({id: 'test'})
    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    assert run_config['n_envs'] % env_config[
        "number_of_targets"] == 0, "n_envs % number_of_targets == 0 if force_uniformity is True"

    vec_env = SubprocVecEnv(
        [make_env(rank=i, x=X_train, y=y_train, seed=0, **env_config)
         for i in range(run_config['n_envs'])])
    vec_env = VecNormalize(vec_env, norm_reward=False)

    if run_config['evaluate_on_validation_set']:
        eval_config = copy.deepcopy(env_config)
        eval_config['prompts'] = X_valid
        eval_config['labels'] = y_valid
        eval_config['id'] = eval_config['env_id']
        del eval_config['env_id']
    else:
        eval_config = None

    save_freq = max(int(run_config['total_timesteps'] // 10 // run_config['n_envs']), 1)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    callbacks = []
    if not run_config["evaluate_on_validation_set"]:
        # Save a checkpoint every save_freq // config['n_envs'] steps
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=f"{run_config['models_save_path']}/model-{timestamp}-{run.id}",
            name_prefix=f'checkpoint',
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        custom_callback = WilsonMazeCallback()
        callbacks.extend([checkpoint_callback, custom_callback])
    else:
        custom_callback = WilsonMazeCallback(evaluate=True,
                                             eval_config=eval_config,
                                             eval_freq=save_freq,
                                             logs_path=run_config['logs_save_path'] + f"/{run.id}",
                                             best_model_save_path=f"{run_config['models_save_path']}/model-{timestamp}-{run.id}",
                                             deterministic=run_config['deterministic'],
                                             use_action_masks=run_config['use_action_masks'],
                                             max_number_of_steps=run_config['max_number_of_steps'])
        callbacks.append(custom_callback)

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=f"{run_config['models_save_path']}/model-{timestamp}-{run.id}",
        verbose=2)
    callbacks.append(wandb_callback)

    model = MaskablePPO("MlpPolicy", vec_env, n_steps=run_config['n_steps'], batch_size=run_config['batch_size'],
                        vf_coef=0.5, ent_coef=0.1, verbose=1, device='cuda:0',
                        tensorboard_log=f"{run_config['logs_save_path']}/{run.id}",
                        policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=run_config['total_timesteps'], progress_bar=True,
                callback=callbacks)

    run.finish()
