from itertools import repeat
import warnings

from common import get_input_data, make_env, parse_policy_kwargs, set_common_seed, transform_config

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

from wandb.integration.sb3 import WandbCallback
import wandb


def train_model(config_file_path: str, seed: int, eval_episodes=10, verbose=0):
    set_common_seed(seed)

    with open(config_file_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    run_config = transform_config(config['run_config'])
    policy_kwargs = parse_policy_kwargs(config['policy_kwargs'])
    env_config = config['env_config']

    embeds, targets = get_input_data(run_config['embeddings_path'], run_config['dataset_path'], run_config['embedding_size'])
    X_train, X_valid, y_train, y_valid = train_test_split(embeds, targets, test_size=0.25, 
                                                            random_state=run_config['random_state'], stratify=targets[:, 0])
    
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

    run = wandb.init(
        project="hippo-test",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    print(f'Started training the model with seed: {seed} - run.id: {run.id}')

        
    vec_env = DummyVecEnv(
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

    save_freq = max(int(run_config['total_timesteps'] // eval_episodes // run_config['n_envs']), 1)
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
                                             record_bad_behaviour=True,
                                             record_coins_behaviour=True,
                                             logs_path=run_config['logs_save_path'] + f"/{run.id}",
                                             best_model_save_path=f"{run_config['models_save_path']}/model-{timestamp}-{run.id}",
                                             deterministic=run_config['deterministic'],
                                             use_action_masks=run_config['use_action_masks'],
                                             max_number_of_steps=run_config['max_number_of_steps'],
                                             verbose=verbose
                                            )
        callbacks.append(custom_callback)

    wandb_callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=f"{run_config['models_save_path']}/model-{timestamp}-{run.id}",
        verbose=2)
    callbacks.append(wandb_callback)

    model = PPO("MlpPolicy", vec_env, n_steps=run_config['n_steps'], batch_size=run_config['batch_size'],
                        vf_coef=run_config['vf_coef'], ent_coef=run_config['ent_coef'], gamma=run_config['gamma'],
                         max_grad_norm=run_config['max_grad_norm'], clip_range=run_config['clip_range'], 
                         policy_kwargs=policy_kwargs, seed=seed, verbose=verbose, device='cuda',
                          tensorboard_log=f"{run_config['logs_save_path']}/{run.id}")

    model.learn(total_timesteps=run_config['total_timesteps'], progress_bar=False, callback=callbacks)

    run.finish()

    print(f'Finished training model with seed: {seed}\n')

    return 0



if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"

    config_file_path = 'configs/phi-2_config.yaml'
    seeds = [42, 16, 201, 67, 1082, 2021, 5, 3255, 7223, 10562]
    
    for seed in seeds:
        train_model(config_file_path=config_file_path, seed=seed)