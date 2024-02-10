import os
from openai import OpenAI


def rearrange_prompt(prompt):
    value = prompt[-2]
    text = prompt[:-2].strip()

    comma = text.find(', ')
    dot = text.find('. ')
    conjunction = -1
    for conj in ['and', 'but', 'with', 'without', 'while', 'before']:
        conjunction = text.find(conj)
        if conjunction != -1:
            break

    p1, p2 = '', ''
    if comma != -1 and 0 > dot or dot == len(text) - 1:
        p1 = text[:comma]
        p2 = text[comma + 2:]
        p2 = p2[0].upper() + p2[1:]
    elif 0 < dot < len(text) - 1:
        p1 = text[:dot]
        p2 = text[dot + 2:]
        p2 = p2[0].upper() + p2[1:]
    elif conjunction != -1:
        p2 = text[conjunction:]
        split = p2.split(' ')
        conj, p2 = split[0], ' '.join(split[1:])
        p2 = p2[0].upper() + p2[1:-1]

        p1 = text[:conjunction]
        p1 = conj + ' ' + p1[0].lower() + p1[1:]

    if p1 != '' and p2 != '':
        return p2 + ' ' + p1 + '. ' + value

    return prompt


if __name__ == "__main__":
    PROMPTS_FILE = './prompts/big_dataset/base_prompts.txt'
    API_KEY = os.environ.get("OPENAI_API_KEY")
    MODEL = "gpt-4"  # -0125-preview"

    client = OpenAI(api_key=API_KEY)
    completion = client.chat.completions.create(model=MODEL, messages=[
        {"role": "system",
         "content": """
                    You are an assistant tasked with generating synthetic text data needed for 
                    training a LLM agent.

                    The synthetic data needs to come in form of prompts given to the agent. 
                    The prompt should tell the agent towards what target it needs to go to and
                    if it needs to pickup up coins along the way, or not. The order of the requests should vary.
                    
                    The target needs to appear in the form of <adjective> and <noun> PLACEHOLDERS tuple.
                    USE THE PLACEHOLDERS IN THE PROMPTS. DO NOT SEPARATE THE PROMPTS OR PROMPT PAIRS WITH EMPTY LINES.
                    Please try to separate the two placeholders in the sentence from time to time.

                    The prompts should be as diverse as possible, have at most 35 words, and each prompt that 
                    tells the agent to pick up the coins should have an equivalent one 
                    that tells it not to do that.
                    
                    Each prompt should be followed by 1 if agent should pick up coins or 0 otherwise.
                    """
         },
        {"role": "user", "content": f"Please generate at least 15 prompt pairs."}
    ])

    with open(PROMPTS_FILE, 'a') as file:
        file.write('\n')
        for choice in completion.choices:
            new_prompt = choice.message.content
            print(new_prompt)
            file.write(new_prompt)

            rearranged_prompt = rearrange_prompt(new_prompt)
            if rearranged_prompt != new_prompt:
                print(rearranged_prompt)
                file.write(rearranged_prompt)
