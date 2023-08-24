import os
import re
import argparse
import glob
from sys import platform
from tqdm import tqdm
from typing import List, Tuple
import warnings
import numpy as np
import torch
from datetime import datetime
from subprocess import Popen, PIPE

from transformers import AutoTokenizer, LlamaModel


def extract_numbers(input_string: str) -> List[int]:
    numbers = re.findall(r'\d+', input_string)
    return [int(num) for num in numbers]

def get_time_delta(start: int, format: str):
    end = datetime.now().time().strftime(format)
    return (datetime.strptime(end, format) - datetime.strptime(start, format))

def move_to_gpu(object):
    if platform == "darwin":
        object.to(torch.device("mps"))
    else:
        object.to(torch.device("cuda"))

def load_hf_model_and_tokenizer(tokenizer_path: str, model_path: str, use_cuda=False) -> Tuple[AutoTokenizer, LlamaModel]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaModel.from_pretrained(
        model_path)

    if use_cuda:
        move_to_gpu(model)

    model.eval()

    return model, tokenizer

def load_llama_cpp_model(cpp_extractor_path: str, model_path: str, use_cuda=False) -> Tuple[Popen, int]:
    lcpp_args = [cpp_extractor_path, '-m', 
                 model_path, "--interactive", 
                            "--embeddings-all"]
    if use_cuda:
        lcpp_args.extend(['-ngl', '1'])

    model = Popen(lcpp_args, 
                shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    
    embed_size = None
    output = model.stdout.readline().decode('utf-8').strip()
    while output:
        if output.startswith('Size of embeddings = '):
            embed_size = extract_numbers(output)[0]
        elif output.startswith('Enter prompt'):
            break
        output = model.stdout.readline().decode('utf-8').strip()

    if embed_size is None:
        raise ValueError('Could not extract embeddings size from LlaMa C++ extractor')
    
    return model, embed_size 


def process_hidden_states(hidden_states, processing_method='mean', axis=0, use_numpy=False) -> np.ndarray | torch.Tensor:
    if processing_method == 'none' and isinstance(hidden_states, list) or isinstance(hidden_states, tuple):
        hidden_states =  torch.stack(hidden_states, axis=axis) if not use_numpy else np.stack(hidden_states, axis=axis)
    if processing_method == 'mean':
        hidden_states = torch.mean(hidden_states, axis=axis) if not use_numpy else np.mean(hidden_states, axis=axis)
    elif processing_method == 'sum':
        hidden_states = torch.sum(hidden_states, axis=axis) if not use_numpy else np.sum(hidden_states, axis=axis)
    elif processing_method == 'max':
        hidden_states = torch.max(hidden_states, axis=axis).values if not use_numpy else np.max(hidden_states, axis=axis)
    
    return hidden_states

def split_and_prompts(prompts, total, destination_path, folder_path, split_ratio=0.25):
    train_prompts, valid_prompts = [], []
    for i in range(total):
        train_size = prompts[i].shape[0] - int(prompts[i].shape[0] * split_ratio)
        print(f'{i}: train size {train_size} - validation size = {prompts[i].shape[0] - train_size}')
        train_prompts.append(prompts[i][:train_size])
        valid_prompts.append(prompts[i][train_size:])

    destination = f'{destination_path}/train_prompts.npz' if destination_path else f'{folder_path}/train_prompts.npz'
    np.savez_compressed(destination, *train_prompts)

    destination = f'{destination_path}/valid_prompts.npz' if destination_path else f'{folder_path}/valid_prompts.npz'
    np.savez_compressed(destination, *valid_prompts)

def extract_hidden_states_llama_cpp(model, batch, embed_size, processing_method='mean'):
    batch_hidden_states = []
    for prompt in batch:
        model.stdin.write(prompt.strip().encode() + b'\n')
        model.stdin.flush()

        output = model.stdout.readline().decode('utf-8').strip()
        if not output.startswith('Embeddings start'):
            raise ValueError('Could not extract embeddings from LlaMa C++ extractor: expected `Embeddings start` message.')
        
        embeddings = []
        output = model.stdout.readline().decode('utf-8').strip()
        while output and output.startswith("Token") and not output.startswith('Embeddings end'):

            output = model.stdout.readline().decode('utf-8').strip()
            if output == '':
                raise ValueError('Could not extract embeddings from LlaMa C++ extractor: expected embeddings.')
            
            embeddings.append(np.array([float(num) for num in output.split()]))
            if embeddings[-1].shape[0] != embed_size:
                raise ValueError('Could not extract embeddings from LlaMa C++ extractor: embeddings size does not match the expected size.')
            
            output = model.stdout.readline().decode('utf-8').strip()
        
        if len(embeddings) == 0:
            raise ValueError('Could not extract embeddings from LlaMa C++ extractor: no embeddings were extracted.')
        
        batch_hidden_states.append(process_hidden_states(np.vstack(embeddings), processing_method, axis=0, use_numpy=True))
        
    return np.vstack(batch_hidden_states)

def extract_hidden_states_hf(tokenizer, model, batch, num_layers=1, processing_method='mean', use_cuda=False):
    with torch.no_grad():
        inputs = tokenizer(batch, return_tensors="pt", padding=True)

        if use_cuda:
            move_to_gpu(inputs)

        output_hidden_states = model(inputs.input_ids, output_hidden_states=(num_layers != 1))
        if num_layers == 1:
            output_hidden_states = output_hidden_states.last_hidden_state.real
        else:
            output_hidden_states = output_hidden_states.hidden_states[-num_layers:]
            output_hidden_states = process_hidden_states(output_hidden_states, processing_method, axis=0)
        
        output_hidden_states = process_hidden_states(output_hidden_states, processing_method, axis=1)
        
        if use_cuda:
            output_hidden_states = output_hidden_states.cpu()

        return output_hidden_states.detach().numpy()

def process_folder(folder_path, tokenizer=None, model=None, use_llama_cpp=False, embed_size=-1,
                   batch_siez=32, num_layers=1, processing_method='mean', destination_path=None,
                   use_cuda=False, split=False, shuffle=False, split_ratio=0.25):
    
    assert not use_llama_cpp or embed_size > 0, 'Embeddings size must be greater than 0 when using LlaMa C++ extractor'

    time_format = r'%H:%M:%S:%f'
    start_time = datetime.now().strftime(time_format)
    print(f'Starting prompts processing:')

    files = list(filter(lambda x: x.endswith('txt'), os.listdir(folder_path)))
    prompts = [None for _ in range(len(files))]
    for i, prompt_file_path in enumerate(files):
        target_id = i

        print(f'Now working with raw file: {folder_path}/{prompt_file_path}')
        start = datetime.now().strftime(time_format)
        file_numbers = extract_numbers(prompt_file_path)
        if file_numbers:
            target_id = file_numbers[0]

        with open(f'{folder_path}/{prompt_file_path}', 'r') as prompts_file:
            lines = prompts_file.readlines()

            if shuffle:
                np.random.shuffle(lines)

            hidden_states, batch = [], []
            for prompt in tqdm(lines):
                if len(batch) == batch_siez:
                    if not use_llama_cpp:
                        batch_hidden_states = extract_hidden_states_hf(tokenizer, model, batch, num_layers, processing_method, use_cuda)
                    else:
                        batch_hidden_states = extract_hidden_states_llama_cpp(model, batch, embed_size, processing_method)
                
                    hidden_states.append(batch_hidden_states)
                    batch = []
                batch.append(prompt.strip())
            if len(batch) > 0:
                if not use_llama_cpp:
                    batch_hidden_states = extract_hidden_states_hf(tokenizer, model, batch, num_layers, processing_method, use_cuda)
                else:
                    batch_hidden_states = extract_hidden_states_llama_cpp(model, batch, embed_size, processing_method)

                hidden_states.append(batch_hidden_states)

        print(
            f'Finished working with raw file: {folder_path}/{prompt_file_path} - time: {get_time_delta(start, time_format)}')
        
        if processing_method != 'none':
            prompts[target_id] = np.vstack(hidden_states)
            print(f'Shape of prompts array: {prompts[target_id].shape}')
        else:
            print(f'Length of prompts array: {len(hidden_states)} - Shape: {hidden_states[0].shape}')
            print(f'Saving hidden states {i} to separate file.')
            destination = f'{destination_path}/prompts_{target_id}.npz' if destination_path else f'{folder_path}/prompts_{target_id}.npz'
            np.savez_compressed(destination, *hidden_states)

    print(f'Finished in: {get_time_delta(start_time, time_format)}')

    if split:
        split_and_prompts(prompts, len(prompts), destination_path, folder_path, split_ratio)
    elif processing_method != 'none':
        destination = f'{destination_path}/prompts.npz' if destination_path else f'{folder_path}/prompts.npz'
        np.savez_compressed(destination, *prompts)


def process_prompt(prompt, tokenizer=None, model=None, use_llama_cpp=False, embed_size=-1,
                   num_layers=1, processing_method='mean',
                   use_cuda=False, save=False, destination_path=None):
    assert num_layers > 0, 'Number of layers must be greater than 0'
    assert processing_method != 'none', 'Processing method none is not supported when processing a single prompt'

    time_format = r'%H:%M:%S:%f'
    start_time = datetime.now().strftime(time_format)
    print(f'Starting prompts processing:')

    if not use_llama_cpp:
        assert tokenizer is not None and model is not None, 'Tokenizer and model are required when not using LlaMa C++ extractor'
        output_hidden_states = extract_hidden_states_hf(tokenizer, model, [prompt.strip()], num_layers, processing_method, use_cuda)
    else:
        assert model is not None, 'Model is required when using LlaMa C++ extractor'
        assert embed_size > 0, 'Embeddings size must be greater than 0 when using LlaMa C++ extractor'
        if num_layers > 1:
            warnings.warn('LlaMa C++ extractor does not support extracting hidden states from multiple layers. Only the last layer will be extracted.')
        output_hidden_states = extract_hidden_states_llama_cpp(model, [prompt.strip()], embed_size, processing_method)

    print(f'Finished in: {get_time_delta(start_time, time_format)}')

    if save:
        destination = f'{destination_path}/prompt.npz' if destination_path else f'./prompt.npz'
        np.savez_compressed(f'{destination}/prompt.npz')

    return output_hidden_states

def process_hidden_states_files(hidden_states_path, processing_method='mean', num_layers=1, split=False, ratio=0.25, destination_path=None):
    npz_files = glob.glob(f'{hidden_states_path}/prompts_[0-9].npz')
    print(f'Found {len(npz_files)} prompts files: ', npz_files)

    prompts = [None for _ in range(len(npz_files))]
    for file_path in npz_files:

        file_numbers = extract_numbers(file_path)
        if file_numbers:
            target_id = file_numbers[0]

        hidden_states = np.load(file_path)

        full_hidden_states = []
        for k in hidden_states.files:
            hidden_states_k = hidden_states[k]
            if len(hidden_states_k.shape) == 4:
                hidden_states_k = hidden_states_k[-num_layers:]
                hidden_states_k  = process_hidden_states(hidden_states_k, processing_method, axis=0, use_numpy=True)
            hidden_states_k = process_hidden_states(hidden_states_k, processing_method, axis=1, use_numpy=True)
            full_hidden_states.append(hidden_states_k)

        prompts[target_id] = np.vstack(full_hidden_states)

    if split:
        split_and_prompts(prompts, len(prompts), destination_path, hidden_states_path, ratio)
    else:
        destination = f'{destination_path}/prompts.npz' if destination_path else f'{hidden_states}/prompts.npz'
        np.savez_compressed(destination, *prompts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='get_llm_output',
        description='Extract LLM hidden states for given prompts',
        epilog='')

    parser.add_argument('-t', '--tokenizer', help='path to tokenizer used by the LLM model', required=False)
    parser.add_argument('-m', '--model', help='path to LLM model weights', required=False)
    parser.add_argument('-lpp', '--llama_cpp_model', help='path to LlaMa C++ extractor executable', required=False)
    parser.add_argument('-f', '--folder', help='path to folder containing prompt text files', default='')
    parser.add_argument('-hs', '--hidden_states', help='path to file containing saved hidden states', default='')
    parser.add_argument('-d', '--destination', help='path to where the  hidden states will be saved', required=False)
    parser.add_argument('-p', '--prompt', help='single prompt given by the user', default='')
    parser.add_argument('-b', '--batch', help='batch size', default=1)
    parser.add_argument('-i', '--method', help='method used to process hidden states', default='none')
    parser.add_argument('-l', '--layers', help='number of hidden layers to extract output from', default=1)
    parser.add_argument('-c', '--cuda', help='use cuda or not', default=False)
    parser.add_argument('-s', '--split', help='whether to split into validation/test sets', default=False, type=bool)
    parser.add_argument('-sh', '--shuffle', help='whether to shuffle the dataset before the split', default=False, type=bool)
    parser.add_argument('-r', '--ratio', help='validation split ration', default=0.25, type=float)

    args = parser.parse_args()

    assert args.folder != '' or args.prompt != '' or args.hidden_states != '', 'Prompt input is required'
    assert args.destination is None or args.destination[-1] != '/' or args.destination[-1] != '\\'

    batch_size = int(args.batch)
    num_layers = int(args.layers)
    cuda = args.cuda == 'True' or args.cuda == True
    split = args.split == 'True' or args.split == True
    ratio = float(args.ratio)
    shuffle = args.shuffle == 'True' or args.shuffle == True

    assert num_layers >= 1
    assert not args.split or 0 < args.ratio < 1.0

    using_llama_cpp = False

    if args.model:
        if args.tokenizer:
            if args.method == 'none':
                if split:
                    warnings.warn('Splitting is not supported when processing method is none')
                    split = False

            model, tokenizer = load_hf_model_and_tokenizer(args.tokenizer, args.model, cuda)
        elif args.llama_cpp_model:
            assert args.method != 'none', 'Working with LlaMa C++ extractor does not yet supports processing method none'
            using_llama_cpp = True
            tokenizer = None
            model, embed_size = load_llama_cpp_model(args.llama_cpp_model, args.model, cuda)

        if args.folder:
            process_folder(folder_path=args.folder, 
                           tokenizer=tokenizer, 
                           model=model, 
                           use_llama_cpp=using_llama_cpp,
                           embed_size=embed_size,
                           batch_siez=batch_size, 
                           num_layers=num_layers, 
                           processing_method=args.method, 
                           destination_path=args.destination, 
                           use_cuda=cuda, 
                           split=split,
                           shuffle=shuffle, 
                           split_ratio=ratio)
        elif args.prompt:
            process_prompt(tokenizer, model, args.prompt, args.layers, args.method, True, args.destination)
        
        if using_llama_cpp:
            model.stdin.write(b'\n')
            model.stdin.flush()
            model.stdin.close()
            model.terminate()

    elif args.hidden_states:
        process_hidden_states_files(args.hidden_states, args.method, num_layers, split, ratio)

