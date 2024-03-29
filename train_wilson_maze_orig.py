import warnings

from common import get_input_data, get_np_input_data, parse_policy_kwargs, transform_config

warnings.filterwarnings("ignore", category=DeprecationWarning)

import yaml
import time
import copy
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from wilson_maze_callback import WilsonMazeCallback

import gymnasium as gym

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
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
        target_indexes = np.where(y == target_id)
        x_target = x[target_indexes]
        # y_target = y[target_indexes][:, 1]

        # print('Target id: ', target_id)
        env = gym.make(env_id, target_id=target_id,
                       prompts=x_target, should_pickup_coins=False, **kwargs)

        return Monitor(env)

    set_random_seed(seed)
    return _init


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    with open('configs/llama_config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    run_config = transform_config(config['run_config'])
    policy_kwargs = parse_policy_kwargs(config['policy_kwargs'])
    env_config = config['env_config']

    X_train, y_train = get_np_input_data('/Users/cranete/_workspace/big_dataset_rar/big_dataset/big_dataset/train_prompts_llama_v2.npz')
    X_valid, y_valid = get_np_input_data('/Users/cranete/_workspace/big_dataset_rar/big_dataset/big_dataset/valid_prompts_llama_v2.npz')
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    # embeds, targets = embeds[:9000], targets[:9000]
    # X_train, X_valid, y_train, y_valid = train_test_split(embeds, targets, test_size=0.3, 
    #                                                         random_state=run_config['random_state'], stratify=targets[:, 0])
    
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
                                             record_coins_behaviour=True,
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
