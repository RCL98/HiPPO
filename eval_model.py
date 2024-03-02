import os
from time import sleep
import pandas as pd

import yaml
import numpy as np
from sb3_contrib import MaskablePPO

from gymnasium import make
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import wilson_maze_env

from common import get_input_data



def try_model(config_file_path: str, prompt_id: int):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f.read())
    
    run_id = config_file_path.split('/')[-3].split('-')[-1]
    env_config = config['env_config']['value']
    run_config = config['run_config']['value']

    models_path = run_config['models_save_path'] + '/'

    model_path = None
    for model in os.listdir(models_path):
        if run_id in model:
            model_path = os.path.join(models_path, model)
            break
    
    if model_path is None:
        print('No model found for this run: ', run_id)
        return
    
    env_config['render_mode'] = 'human'
    prompt = pd.read_csv(run_config['dataset_path'], sep=',')['prompt'].tolist()[prompt_id]
    embeds, targets = get_input_data(run_config['dataset_path'], run_config['embeddings_path'], run_config['embedding_size'])
    vec_env = DummyVecEnv([lambda: make(**env_config, 
                                        prompts=embeds, 
                                        chosen_prompt=prompt_id,
                                        should_pickup_coins=targets[prompt_id][1],
                                        target_id=targets[prompt_id][0])])
    print(f'Running model {run_id} on target {targets[prompt_id][0]} and prompt: "{prompt}"')
    
    if os.path.isfile(model_path + '/best_model_vec_normalizer.pkl'):
        vec_env = VecNormalize.load(model_path + '/best_model_vec_normalizer.pkl', vec_env)
        vec_env.training = False

    model = MaskablePPO.load(model_path + '/best_model.zip', vec_env, device='cuda')

    obs = vec_env.reset()
    vec_env.render()
    wins = 0
    for _ in range(15):
        action, _state = model.predict(obs, deterministic=True, action_masks=vec_env.env_method("action_masks"))
        print(action)
        obs, rewards, dones, infos = vec_env.step(action)
        vec_env.render()
        sleep(1)
        if dones[0] and rewards[0] > 0:
            wins += 1
            break
        if infos[0]["TimeLimit.truncated"]:
            obs = infos[0]["terminal_observation"]
        vec_env.render()
    vec_env.close()
    print(wins)
    

def test_on_model(model_path, **config):
    env = make('WilsonMaze-v0', **config)
    env.training = False
    model = MaskablePPO.load(model_path, env, device='cuda')

    obs, info = env.reset()
    env.render()
    wins = 0
    for _ in range(15):
        action, _state = model.predict(obs, deterministic=True, action_masks=env.action_masks())
        print(action)
        obs, reward, terminated, truncated, info = env.step(action.item())
        env.render()
        if terminated and reward > 0:
            wins += 1
            break
        if truncated:
            obs, info = env.reset()
        env.render()
    env.close()
    print(wins)


def test_on_model_vec(model_path, **config):
    vec_env = DummyVecEnv([lambda: make('WilsonMaze-v0', **config)])
    vec_env = VecNormalize.load(
        './models/tests-2/model-20230806_195139-7hsn5ybo/checkpoint_vecnormalize_4500000_steps.pkl', vec_env)
    vec_env.training = False
    model = MaskablePPO.load(model_path, vec_env, device='cuda')

    obs = vec_env.reset()
    vec_env.render()
    wins = 0
    for _ in range(15):
        action, _state = model.predict(obs, deterministic=True, action_masks=vec_env.env_method("action_masks"))
        print(action)
        obs, rewards, dones, infos = vec_env.step(action)
        vec_env.render()
        if dones[0] and rewards[0] > 0:
            wins += 1
            break
        if infos[0]["TimeLimit.truncated"]:
            obs = infos[0]["terminal_observation"]
        vec_env.render()
    vec_env.close()
    print(wins)


def evaluate_model(model_path=None, model_class=None, normalizer_path=None,
                   deterministic=False, use_action_masks=False, max_number_of_steps=10, device='cpu', **config):
    assert model_path is not None, 'No model path provided'
    assert model_class is not None, 'No model class provided'
    assert config['prompts'] is not None, 'No prompts provided'
    assert config['labels'] is not None, 'No labels provided'
    assert config['id'] is not None, 'No env id provided'
    assert config["number_of_targets"] is not None, 'No number of targets provided'
    assert max_number_of_steps > 1, 'Max number of steps must be a natural number bigger than 0'

    prompts = config['prompts']
    labels = config['labels']
    del config['labels']
    del config['prompts']

    move_solved, coins_solved, total_size = [], [], 0
    for target_i in range(config["number_of_targets"]):
        move_wins, coin_wins = 0, 0
        
        targets_i = np.where(labels[:, 0] == target_i)
        target_prompts = prompts[targets_i]
        if config['add_coins']:
            coins = labels[targets_i][:, 1]
        
        for prompt_j in range(target_prompts.shape[0]):
            vec_env = DummyVecEnv([lambda: make(target_id=target_i,
                                                prompts=target_prompts,
                                                chosen_prompt=prompt_j,
                                                should_pickup_coins=coins[prompt_j], **config)])
            if normalizer_path is not None:
                vec_env = VecNormalize.load(normalizer_path, vec_env)
            vec_env.training = False
            if model_path is not None:
                model = model_class.load(model_path, vec_env, device=device)

            obs = vec_env.reset()
            if config['render_mode'] == 'human':
                vec_env.render()

            for _ in range(max_number_of_steps):
                if use_action_masks:
                    action, _ = model.predict(obs, deterministic=deterministic, 
                                              action_masks=vec_env.env_method("action_masks"))
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)

                obs, rewards, dones, infos = vec_env.step(action)

                if config['render_mode'] == 'human':
                    vec_env.render()

                if dones[0] and rewards[0] > 0:
                    if config['add_coins'] and rewards[0] >= 2.0:
                        coin_wins += 1
                    move_wins += 1 
                    break
                if infos[0]["TimeLimit.truncated"]:
                    obs = infos[0]["terminal_observation"]

            vec_env.close()

        move_solved.append(move_wins)
        coins_solved.append(coin_wins)
        total_size += target_prompts.shape[0]

    print()
    stats = {}
    for i in range(config["number_of_targets"]):
        stats[f'target_{i}'] = {'move_solved': move_solved[i],
                                'move_percentage': move_solved[i] / target_prompts.shape[0],
                                'total': target_prompts.shape[0]}
        if config['add_coins']:
            stats[f'target_{i}']['coins_solved'] = coins_solved[i]
            stats[f'target_{i}']['coins_percentage'] = coins_solved[i] / target_prompts.shape[0]

        print(f'For target {i}, move solved: {move_solved[i]}  {target_prompts.shape[0]} '
              f'or {stats[f"target_{i}"]["move_percentage"] * 100:2f}')
        if config['add_coins']:
            print(f'For target {i} coins solved: {coins_solved[i]}  {target_prompts.shape[0]} ' 
                    f'or {stats[f"target_{i}"]["coins_percentage"] * 100:2f}')

    total_move_solved = sum(move_solved)
    move_score = total_move_solved / total_size
    print(f'Solved: {total_move_solved} moves out of {total_size} or {move_score * 100:2f}')
    final_score = move_score

    if config['add_coins']:
        total_coins_solved = sum(coins_solved)
        coin_score = total_coins_solved / total_size
        print(f'Solved: {total_coins_solved} coins out of {total_size} or {coin_score * 100:2f}')
        final_score = (move_score + coin_score) / 2

    return final_score, stats


if __name__ == '__main__':
    try_model('wandb/run-20240302_211902-uzt68ca5/files/config.yaml', 10)