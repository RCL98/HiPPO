import copy
import torch
import gymnasium as gym
import time

import wilson_maze_env
from wilson_maze_callback import WilsonMazeCallback

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

# Custom actor (pi) and value function (vf) networks
# of two layers of size 32 each with Relu activation function
# Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]), activation_fn=torch.nn.ReLU)


def make_env(env_id: str, rank: int, seed: int = 0, forced_uniformity=False, **kwargs):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param forced_uniformity: whether the target distribution is uniform or not
    :param rank: index of the subprocess
    """

    def _init():
        if forced_uniformity:
            target_id = (seed + rank) % kwargs.get('number_of_targets', 1)
            # print('Target id: ', target_id)
            env = gym.make(env_id, target_id=target_id, **kwargs)
        else:
            env = gym.make(env_id, **kwargs)
            env.reset(seed=seed + rank)
        return Monitor(env)

    set_random_seed(seed)
    return _init


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


if __name__ == "__main__":
    config = {
        "run_config": {
            "n_envs": 16,
            "policy_type": "MlpPolicy",
            "total_timesteps": 3e6,
            "n_steps": 2 ** 12,
            "batch_size": 2 ** 6,
            "forced_uniformity": True,
            "evaluate_on_validation_set": True,
            "max_number_of_steps": 15,
            "render_on_evaluation": False,
            "deterministic": True,
            "use_action_masks": True,
            "models_save_path": "./models/tests-5",
            "logs_save_path": "./logs/tensorboard/tests-5",
        },
        "env_config": {
            "env_id": "WilsonMaze-v0",
            "render_mode": "text",
            "size": 10,
            "timelimit": 30,
            "random_seed": 42,
            "variable_target": False,
            "number_of_targets": 4,
            "terminate_on_wrong_target": True,
            "prompts_file": "./prompts/big_dataset/train_prompts_llama_v1_mean_l1.npz",
            "prompt_size": 4096,
            "prompt_mean": False,
        }
    }

    run_config = config['run_config']
    env_config = config['env_config']

    if run_config['forced_uniformity']:
        assert run_config['n_envs'] % env_config[
            "number_of_targets"] == 0, "n_envs % number_of_targets == 0 if force_uniformity is True"

    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # run = dotdict({id: 'test'})

    if run_config['forced_uniformity']:
        vec_env = SubprocVecEnv(
            [make_env(rank=i, seed=0, forced_uniformity=True, **env_config)
             for i in range(run_config['n_envs'])])
        vec_env = VecNormalize(vec_env, norm_reward=False)
    else:
        vec_env = make_vec_env(config['env_id'], n_envs=config['n_envs'], seed=42, vec_env_cls=SubprocVecEnv,
                               env_kwargs=dict(list(config.items())[7:]))

    if run_config['evaluate_on_validation_set']:
        eval_config = copy.deepcopy(env_config)
        eval_config['prompts_file'] = eval_config['prompts_file'].replace('train', 'valid')
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
                        vf_coef=0.5, ent_coef=0.1, verbose=1,
                        tensorboard_log=f"{run_config['logs_save_path']}/{run.id}",
                        policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=run_config['total_timesteps'], progress_bar=True,
                callback=callbacks)

    run.finish()
