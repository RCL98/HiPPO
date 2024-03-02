import torch
import struct
import numpy as np
import pandas as pd



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
        X = np.vstack([np.array(embds[i:i + embd_size])
                      for i in range(0, len(embds), embd_size)])

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
