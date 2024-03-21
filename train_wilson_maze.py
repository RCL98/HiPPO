import warnings

from common import get_input_data, parse_policy_kwargs, simplify_data, transform_config

warnings.filterwarnings("ignore", category=DeprecationWarning)

import yaml
import time
import copy
from types import SimpleNamespace
import numpy as np

from sklearn.model_selection import train_test_split

from wilson_maze_callback import WilsonMazeCallback

import gymnasium as gym

from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from wandb.integration.sb3 import WandbCallback
import wandb

set_random_seed(32, True)


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

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    with open('configs/llama-2_config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)

    run_config = transform_config(config['run_config'])
    policy_kwargs = parse_policy_kwargs(config['policy_kwargs'])
    env_config = config['env_config']

    embeds, targets = get_input_data(run_config['dataset_path'], run_config['embeddings_path'], run_config['embedding_size'])
    embeds, targets = embeds[:9000], targets[:9000]  # embeds[:12800], targets[:12800]  
    # embeds, targets = simplify_data(embeds, targets)
    X_train, X_valid, y_train, y_valid = train_test_split(embeds, targets, test_size=0.20, 
                                                            random_state=run_config['random_state'], stratify=targets[:, 0])
    
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

    #run = SimpleNamespace(**({'id': 'test'}))
    #print(run.id)
    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    vec_env = SubprocVecEnv(
        [make_env(rank=i, x=X_train, y=y_train, seed=0, **env_config)
         for i in range(run_config['n_envs'])])
    vec_env = VecNormalize(vec_env, norm_reward=False)

    if run_config['evaluate_on_validation_set']:
        eval_config = copy.deepcopy(env_config)
        eval_config['prompts'] = X_valid
        eval_config['labels'] = y_valid
        eval_config['id'] = eval_config['env_id']
        eval_config['variable_target'] = False
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

    model = PPO("MlpPolicy", vec_env, n_steps=run_config['n_steps'], batch_size=run_config['batch_size'],
                        vf_coef=run_config['vf_coef'], ent_coef=run_config['ent_coef'], gamma=run_config['gamma'],
                        verbose=1, device='auto', tensorboard_log=f"{run_config['logs_save_path']}/{run.id}",
                        policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=run_config['total_timesteps'], progress_bar=True, callback=callbacks)

    run.finish()
