import random
import re
import editdistance

blue_synonyms = ['azure', 'teal']
purple_synonyms = ['mauve', 'violet']
yellow_synonyms = ['amber', 'golden']
red_synonyms = ['crimson', 'scarlet']

TARGET_SIZE = 3200

def reduce_list_to_size(input_list, target_size):
    current_size = len(input_list)

    if target_size >= current_size:
        return input_list

    while current_size > target_size:
        random_index = random.randint(0, current_size - 1)
        input_list.pop(random_index)
        current_size = len(input_list)

    return input_list


with open('prompts/raw_blue_prompts.txt', 'r') as inFile:
    prompts = inFile.readlines()
    clean_prompts = [prompts[0]]
    for prompt in prompts:
        if prompt.strip() != '' and 'triangle' in prompt:
            for p in clean_prompts:
                prompt = prompt.replace('\"', '')
                if editdistance.distance(p, prompt) > 5:
                    clean_prompts.append(prompt)
                    break

    new_prompts = []
    for prompt in clean_prompts:
        new_prompts.append(prompt)
        for synonym in blue_synonyms:
            new_prompt = prompt.replace('blue', synonym)
            if new_prompt != prompt:
                new_prompts.append(new_prompt)

    random.shuffle(new_prompts)

    if len(new_prompts) > TARGET_SIZE:
        new_prompts = reduce_list_to_size(new_prompts, TARGET_SIZE)

    with open('prompts/big_dataset/blue_triangle_0_target.txt', 'w') as outFile:
        for p in new_prompts:
            outFile.write(f'{p.strip()}\n')

    yellow_square_prompts = []
    for prompt in clean_prompts:
        yellow_prompt = prompt.replace('triangle', 'square').replace('Triangle', 'Square').replace('blue', 'yellow').replace('Blue', 'Yellow')
        yellow_square_prompts.append(yellow_prompt)
        for synonym in yellow_synonyms:
            new_prompt = yellow_prompt.replace('yellow', synonym)
            if new_prompt != yellow_prompt:
                yellow_square_prompts.append(new_prompt)

    random.shuffle(yellow_square_prompts)

    if len(yellow_square_prompts) > TARGET_SIZE:
        yellow_square_prompts = reduce_list_to_size(yellow_square_prompts, TARGET_SIZE)

    with open('prompts/big_dataset/yellow_square_1_target.txt', 'w') as outFile:
        for p in yellow_square_prompts:
            outFile.write(f'{p.strip()}\n')

    red_circle_prompts = []
    for prompt in clean_prompts:
        red_prompt = prompt.replace('triangle', 'circle').replace('Triangle', 'Circle').replace('blue',
                                                                                                'red').replace(
            'Blue', 'Red')
        red_circle_prompts.append(red_prompt)
        for synonym in red_synonyms:
            new_prompt = red_prompt.replace('red', synonym)
            if new_prompt != red_prompt:
                red_circle_prompts.append(new_prompt)

    random.shuffle(red_circle_prompts)

    if len(red_circle_prompts) > TARGET_SIZE:
        red_circle_prompts = reduce_list_to_size(red_circle_prompts, TARGET_SIZE)

    with open('prompts/big_dataset/red_circle_2_target.txt', 'w') as outFile:
        for p in red_circle_prompts:
            outFile.write(f'{p.strip()}\n')

    purple_diamond_prompts = []
    for prompt in clean_prompts:
        purple_prompt = prompt.replace('triangle', 'diamond').replace('Triangle', 'Diamond').replace('blue',
                                                                                                'purple').replace(
            'Blue', 'Purple')
        purple_diamond_prompts.append(purple_prompt)
        for synonym in purple_synonyms:
            new_prompt = purple_prompt.replace('purple', synonym)
            if new_prompt != purple_prompt:
                purple_diamond_prompts.append(new_prompt)

    random.shuffle(purple_diamond_prompts)

    if len(purple_diamond_prompts) > TARGET_SIZE:
        purple_diamond_prompts = reduce_list_to_size(purple_diamond_prompts, TARGET_SIZE)

    with open('prompts/big_dataset/purple_diamond_3_target.txt', 'w') as outFile:
        for p in purple_diamond_prompts:
            outFile.write(f'{p.strip()}\n')

