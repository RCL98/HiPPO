import argparse
import json
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import wilson_maze_env

from get_llm_output import load_llama_cpp_model, load_hf_model_and_tokenizer, process_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='test_model_on_live_prompts.py',
        description='Test a model on live prompts',
        epilog='')

    parser.add_argument('--config', help='Path to configuration file', type=str, required=True)
    parser.add_argument('--model', help="Path to agent's weights file", type=str, required=True)
    parser.add_argument('--llm_model', help='Path to llm model weights file', type=str, required=True)
    parser.add_argument('--tokenizer', help='Path to tokenizer file', type=str)
    parser.add_argument('-lpp', '--llama_cpp', help='path to LlaMa C++ extractor executable', type=str)
    parser.add_argument('--vec_normalizer', help='Path to vector normalizer file', type=str)
    parser.add_argument('--method', help='method used to process hidden states', default='mean')
    parser.add_argument('--layers', help='number of hidden layers to extract output from', default=1)
    parser.add_argument('--cuda', help='use cuda or not', default=False)

    args = parser.parse_args()

    config = json.load(open(args.config, 'r'))

    run_config = config['run_config']
    env_config = config['env_config']

    assert run_config['model_class'] is not None, 'No model class provided'

    if args.llama_cpp_model is not None:
        use_llama_cpp = True
        tokenizer = None
        llm_model, embedding_size = load_llama_cpp_model(args.llama_cpp, args.llm_model, args.cuda)
    else:
        use_llama_cpp = False
        embedding_size = None
        llm_model, tokenizer = load_hf_model_and_tokenizer(args.llm_model, args.tokenizer, args.cuda)

    while True:
        # try:
        #     target_id = int(input("Enter a target id: "))
        #     if target_id < 0 or target_id > 3:
        #         raise ValueError
        # except ValueError:
        #     print('Target id must be an integer in [0, 1, 2, 3]')
        #     continue
        # try:
        #     max_number_of_steps = int(input("Enter a max number of steps: "))
        #     if max_number_of_steps < 1:
        #         raise ValueError
        # except ValueError:
        #     print('Max number of steps must be an integer greater than 0')
        #     continue

        # prompt = input("Enter a prompt: ")

        target_id = 0
        max_number_of_steps = 100
        prompt = 'Would you kindly go to the blue triangle?'

        if prompt == 'exit' or prompt == 'quit' or prompt == 'q' or prompt.strip() == '':
            if use_llama_cpp:
                llm_model.stdin.write(b'\n')
                llm_model.stdin.flush()
                llm_model.terminate()
            break
        
        hidden_states = process_prompt(prompt, tokenizer, llm_model, use_llama_cpp, embedding_size, args.layers,
                                        args.method, args.cuda)

        vec_env = DummyVecEnv([lambda: gym.make(target_id=target_id, user_prompt_value=hidden_states, **env_config)])
        if args.vec_normalizer is not None:
            vec_env = VecNormalize.load(args.vec_normalizer, vec_env)
        vec_env.training = False
    
        if run_config['model_class'] == 'maskable_ppo':
            model = MaskablePPO.load(args.model, vec_env, device='cpu')
        elif run_config['model_class'] == 'ppo':
            model = PPO.load(args.model, vec_env, device='cpu')
        
        obs = vec_env.reset()
        for _ in range(max_number_of_steps):
            if run_config['model_class'] == 'maskable_ppo' and run_config['use_action_masks']:
                action, _states = model.predict(obs, action_masks=vec_env.env_method("action_masks"))
            else:
                action, _states = model.predict(obs)

            obs, rewards, dones, infos = vec_env.step(action)
            if env_config['render_mode'] == 'human':
                vec_env.render()

            if dones[0]:
                print('Done!')
                break
            if infos[0]["TimeLimit.truncated"]:
                print('Time limit truncated!')
                obs = infos[0]["terminal_observation"]

        vec_env.close()
