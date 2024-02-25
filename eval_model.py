from time import sleep

import numpy as np
from sb3_contrib import MaskablePPO

from gymnasium import make
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
import wilson_maze_env


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

    solved, total_size = [], 0
    for target_i in range(config["number_of_targets"]):
        wins = 0
        targets_i = np.where(labels[:, 0] == target_i)
        target_prompts = prompts[targets_i]
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
                    wins += 1
                    break
                if infos[0]["TimeLimit.truncated"]:
                    obs = infos[0]["terminal_observation"]

            vec_env.close()

        solved.append(wins)
        total_size += target_prompts.shape[0]

    print()
    stats = {}
    for i in range(config["number_of_targets"]):
        stats[f'target_{i}'] = {'solved': solved[i], 
                                'percentage': solved[i] /target_prompts.shape[0],  
                                'total': target_prompts.shape[0]}
        print(f'For target {i}, {solved[i]} out of {target_prompts.shape[0]} '
              f'or {stats[f"target_{i}"]["percentage"] * 100:2f}')

    total_solved = sum(solved)
    score = total_solved / total_size
    print(f'Solved: {total_solved} out of {total_size} or {score * 100:2f}')

    return score, stats


if __name__ == '__main__':
    valid_config = {
        "render_mode": "text",
        "size": 7,
        "timelimit": 30,
        "random_seed": 42,
        "variable_target": False,
        "number_of_targets": 4,
        "prompts_file": "./prompts/big_dataset/train_prompts_llama_v2.npz",
        "prompt_size": 512,
        "prompt_mean": False,
    }

    # evaluate_model_on_validation_set('./models/tests-2\model-20230809_190401-s6vwqh2a\checkpoint_4000000_steps.zip',
    #                                  './models/tests-2\model-20230809_190401-s6vwqh2a\checkpoint_vecnormalize_4000000_steps.pkl',
    #                                  **dict(list(valid_config.items())))

    test_config = {
        "render_mode": "human",
        "size": 7,
        "timelimit": 30,
        "random_seed": 42,
        "variable_target": False,
        "number_of_targets": 4,
        "prompts_file": "./prompts/big_dataset/valid_prompts.npz",
        "prompt_size": 512,
        "prompt_mean": False,
        "target_id": 2,
        "user_prompt": 12
    }

    # test_on_model_vec('./models/tests-2/model-20230806_195139-7hsn5ybo/checkpoint_4500000_steps.zip', **dict(list(test_config.items())))

    # fails = 0
    # for i in range(45):
    #     result = evaluate_model('prompts/prompts.npz', 'models/tests/ni1oju2n/checkpoint_5000000_steps.zip', 0, 'human',
    #                             i)
    #     print(f"command: {i} results: {result}")
    #     if result == 0:
    #         fails += 1
    # print(f'\n{fails} fails out of {45} prompts')
