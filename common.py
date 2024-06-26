import os
import torch.nn as nn
import struct
import numpy as np
import pandas as pd

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

def set_common_seed(seed: int):
    set_random_seed(seed, True)
    os.environ['PYTHONASHSEED'] = f'{seed}'

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
        prompts = x
        labels = y
        
        if not kwargs.get('variable_target', False):
            number_of_targets = len(np.unique(y[:, 0]))
            target_id = (seed + rank) % number_of_targets
            target_indexes = np.where(y[:, 0] == target_id)
            prompts = x[target_indexes]
            labels = y[target_indexes]
            

        # print('Target id: ', target_id)
        env = gym.make(env_id, prompts=prompts, labels=labels, **kwargs)

        return Monitor(env)

    return _init

def simplify_data(embeds: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
        Simplify the embeddings and targets by using only a single embedding for each target.

        :param embeds: the input embeddings data
        :param targets: the target and coin data
        :return: a tuple of two numpy arrays, the first one is the simplified input embeddings data and the second one is the simplified target data
    """
    chosen_embds = []
    uniqe_targets = set(targets[:,0].tolist())

    for target in uniqe_targets:
        for i in range(len(targets)):
            if targets[i][0] == target:
                chosen_embds.append(embeds[i])
                break

    for idx, target in enumerate(uniqe_targets):
        embeds[np.where(targets[:,0] == target)] = chosen_embds[idx]

    return embeds, targets

def get_np_input_data(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(data_path)
    a, b, c, d = X['arr_0'], X['arr_1'], X['arr_2'], X['arr_3']

    X = np.concatenate((a, b, c, d), axis=0)
    Y = np.concatenate((np.zeros(a.shape[0]), np.ones(b.shape[0]), np.ones(c.shape[0]) * 2, np.ones(d.shape[0]) * 3))

    return X, Y

def get_input_data(embd_path: str, dataset_path: str, embd_size=4096) -> tuple[np.ndarray, np.ndarray]:
    """
        Load the input data from the dataset and the embeddings file.
        
        :param embd_path: the path to the embeddings file
        :param dataset_path: the path to the dataset file
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
        X = np.vstack([np.array(embds[i:i + embd_size])
                      for i in range(0, len(embds), embd_size)])

    return X, Y

def get_npz_input_data(embd_path: str, dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
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

    X = np.load(embd_path)['X']

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
        "activation_fn": {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[in_policy_kwargs['activation_fn']]
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
        elif isinstance(v, float) and v.is_integer():
            new_dict[k] = int(v)
        else:
            new_dict[k] = v
    return new_dict
