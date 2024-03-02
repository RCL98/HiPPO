import copy
import os
import editdistance
from itertools import zip_longest

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
import pandas as pd


def write_new_line(line, w_file):
    """
        Write a new line to a file.
    """
    line = line if line[-1] == '\n' else line + '\n'
    w_file.write(line)

def grouper(iterable, n, fill_value=None):
    """
        Collect data into fixed-length chunks or blocks.
        Taken from https://stackoverflow.com/questions/434287/how-to-iterate-over-a-list-in-chunks
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)

def clear_to_similar_prompts(prompts: list[str], temp_file: str) -> None:
    final_prompts = [prompts[0]]
    with open(temp_file, 'w') as out:
        for prompt in prompts[1:]:
            min_distances = {p: editdistance.eval(prompt, p) for p in final_prompts}
            sorted_distances = sorted(min_distances.items(), key=lambda x: x[1])
            
            if sorted_distances[0][1] > 10 or sorted_distances[0][0][-1] != prompt[-1]:
                write_new_line(prompt, out)
                final_prompts.append(prompt)
            else:
                print('Skipping prompt:', prompt + ' Distance:', sorted_distances[0][1])


def call_openai_api(agent_context: str, user_context: str) -> ChatCompletion | Stream[ChatCompletionChunk] | None:
    """
        Call the OpenAI API to get the completions.

        Args:
            agent_context: The context of the agent.
            user_context: The context of the user.
        Returns:
            The completion of the API call.
    """
    try:
        client = OpenAI(api_key=API_KEY)
        completion = client.chat.completions.create(model=MODEL, messages=[
            {
                "role": "system",
                "content": agent_context
            },
            {"role": "user", "content": user_context}
        ])
    except Exception as e:
        # By this way we can know about the type of error occurring
        print("The error is: ", e)
        return None
    return completion

def generate_prompts(how_to_write: str = '', 
                     numder_of_prompt_pairs=10, 
                     with_rearrangements=True) -> list[str]:
    print('Generating prompts!')
    completion = call_openai_api(
        f"""
            You are an assistant tasked with generating synthetic text data needed for
            training a LLM agent.

            The synthetic data needs to come in form of prompts given to the agent.
            The prompt should tell the agent towards what target it needs to go to and
            if it needs to pickup up coins along the way, or not.

            TALK ABOUT THE COINS BEFORE TALKING ABOUT THE TARGET.

            The target must appear in the form of <adjective> and <noun> PLACEHOLDERS tuple.
            USE THE PLACEHOLDERS IN THE PROMPTS. DO NOT SEPARATE THE PROMPTS OR PROMPT PAIRS WITH EMPTY LINES.
            Please try to separate the two placeholders in the sentence from time to time.

            The prompts should be as diverse as possible, have at most 35 words, and each prompt that
            tells the agent to pick up the coins should have an equivalent one
            that tells it not to do that.

            {how_to_write}

            Each prompt should be followed by 1 if agent should pick up coins or 0 otherwise.
        """,
        f"Please generate at least f{numder_of_prompt_pairs} prompt pairs."
    )
    if completion is None:
        print('Got error while calling OpenAI API')
        return

    print('Prompts have been generated: ')
    with open(PROMPTS_FILE, 'r+') as file:
        existing_prompts = set([p.strip() for p in file.readlines()])

        file.write('\n')
        new_prompts = []
        for choice in completion.choices:
            for new_prompt in choice.message.content.split('\n'):
                new_prompt = new_prompt.strip()

                if new_prompt == "" or 'noun' not in new_prompt or 'adjective' not in new_prompt:
                    continue

                min_distance = min([editdistance.eval(new_prompt, p) for p in existing_prompts])

                if min_distance > 10:
                    print(new_prompt)
                    new_prompts.append(new_prompt)
                    existing_prompts.add(new_prompt)
                    write_new_line(new_prompt, file)
    
    if not with_rearrangements:
        return new_prompts
    
    new_prompts_plus = copy.deepcopy(new_prompts)
    print('Rearranging prompts!')
    re_prompts = rearrange_given_prompts(new_prompts, print_outut=True)
    values = [p[-1] for p in new_prompts]

    if re_prompts is None:
        print('Got error while rearranging prompts. Skipping this group.')
        return new_prompts

    if len(re_prompts) != len(new_prompts):
        print('The number of rearranged prompts does not match the number of prompts. Skipping this group.')
        return new_prompts

    with open(PROMPTS_FILE, 'a') as file:
        for prompt, value in zip(re_prompts, values):
            min_distance = min([editdistance.eval(prompt, p) for p in existing_prompts])
            if min_distance > 10:
                write_new_line(prompt + ' ' + value, file)
                existing_prompts.add(prompt)
                new_prompts_plus.append(prompt)

    
    return new_prompts_plus


def rearrange_given_prompts(prompts: list[str], print_outut=False) -> list[str]:
    target_prompts = " # ".join([p[:-1] for p in prompts])
    
    if print_outut:
        print('Target prompts:\n', target_prompts.replace('#', '\n'))
    
    completion = call_openai_api(
        """
            You are an assistant tasked with rearranging sentences in given prompts while preserving their meaning.
            You will receive a list of propmpts. The prompts will be separated by the # symbol. 
            You need to treat each prompt separately. Do not rearrange the list of prompts itself.
        """,
        f"Please rearrange the following prompts: {target_prompts}."
    )
    if completion is None:
        print('Got error while calling OpenAI API')
        return

    if print_outut:
        print('\nRearranged prompts: ')
    
    new_prompts = []
    for choice in completion.choices:
        for i, new_prompt in enumerate(choice.message.content.split('#')):
            new_prompt = new_prompt.strip()
            
            if new_prompt == "":
                continue
            
            if print_outut:
                print(new_prompt)
            
            new_prompts.append(new_prompt)

    return new_prompts

def rearrange_prompts(original_prompts: list[str], temp_file: str) -> None:
    with open(temp_file, 'w') as file:
        for group in grouper(original_prompts, 5):
            group = [p for p in group if p is not None]

            values = []
            for prompt in group:
                write_new_line(prompt, file)
                values.append(prompt[-1])
            rearranged_prompts = rearrange_given_prompts(group, print_outut=True)

            if rearrange_prompts is None:
                print('Got error while rearranging prompts. Skipping this group.')
                continue

            if len(rearranged_prompts) != len(group):
                print('The number of rearranged prompts does not match the number of prompts. Skipping this group.')
                continue

            for prompt, value in zip(rearranged_prompts, values):
                # If there is a prompt the original ones that is too similar to the new one, we skip it.
                min_distance = min([editdistance.eval(prompt, p) for p in original_prompts])
                if min_distance > 10:
                    write_new_line(prompt + ' ' + value, file)


def get_synonyms(target):
    completion = call_openai_api(
        "You are an assistant tasked with generating synonyms for a given target.",
        f"Please give some synonyms for {target}."
    )
    if completion is None:
        print('Got error while calling OpenAI API')
        return

    for choice in completion.choices:
        print(choice.message.content)

def generate_final_prompts():
    with open(PROMPTS_FILE, 'r') as file:
    
        final_prompts, target_values, coin_values = [], [], []
        for prompt in file.readlines():
            prompt = prompt.strip()
            value = prompt[-1]
            prompt = prompt[:-2].replace('"', '')

            for i, (target, synonyms) in enumerate(SYNONYMS.items()):
                for _target in synonyms + [target]:
                    adj = _target.split()[0]
                    noun = _target.split()[1]

                    new_prompt = copy.deepcopy(prompt)
                    new_prompt = new_prompt.replace('<adjective>', adj)
                    new_prompt = new_prompt.replace('<noun>', noun)

                    final_prompts.append(new_prompt)
                    target_values.append(i)
                    coin_values.append(value)
            
        df = pd.DataFrame(data={'prompt': final_prompts, 'target': target_values, 'coin': coin_values})
        df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv('./prompts/big_dataset/prompts.csv', index=False)

        prompts = df['prompt'].tolist()
        with open(TEMP_FILE, 'w') as file:
            for prompt in prompts:
                write_new_line(prompt, file)

if __name__ == "__main__":
    PROMPTS_FILE = './prompts/big_dataset/base_prompts.txt'
    TEMP_FILE = './prompts/big_dataset/temp_prompts.txt'
    API_KEY = os.environ.get("OPENAI_API_KEY")
    MODEL = "gpt-4" #-0125-preview"
    SYNONYMS = {
        'blue triangle': ['azure triangle', 'navy triangle', 'teal triangle'],
        'yellow square': ['golden square', 'amber square', 'lemon square'],
        'red circle': ['crimson circle', 'ruby circle', 'scarlet circle'],
        'purple diamond': ['violet diamond', 'magenta diamond', 'lilac diamond'],
    }

    # way_to_write = 'Write the prompts like you were a manager talking to his employees.'
    # prompts = generate_prompts(way_to_write, 5, True)

    # get_synonyms('yellow square')

    generate_final_prompts()


                