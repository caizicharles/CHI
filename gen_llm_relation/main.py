from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pickle
import itertools
import numpy as np
from tqdm import tqdm
import os

# Set the cache directory to the current directory
os.environ['HF_HOME'] = './hf_cache'
os.environ['TRANSFORMERS_CACHE'] = './hf_cache'
os.environ['HF_DATASETS_CACHE'] = './hf_cache'

path = './nodes.pickle'


def read_pickle_file(path: str):
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def save_response_to_file(response, idx):
    if not os.path.exists('./responses'):
        os.makedirs('./responses')
    with open(f'./responses/response_{idx}.txt', 'w') as f:
        f.write(response)


def load_responses():
    responses = []
    for filename in os.listdir('./responses'):
        if filename.startswith('response_') and filename.endswith('.txt'):
            with open(os.path.join('./responses', filename), 'r') as f:
                responses.append(f.read())
    return responses


def main():
    code_list = np.array(read_pickle_file(path))
    code_list = code_list[:, 1]
    code_pairs = list(itertools.permutations(code_list, 2))

    model_id = "meta-llama/Meta-Llama-3-70B"
    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                 device_map="auto",
                                                 torch_dtype=torch.float16,
                                                 cache_dir="./hf_cache")

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./hf_cache")

    for idx, pair in enumerate(tqdm(code_pairs)):
        code_1 = pair[0]
        code_2 = pair[1]

        prompt = f'''
        ### Instruction:

        Given a pair of medical codes (could be a diagnosis, procedure, or drug), extrapolate the most plausible relationships between the drugs.
        The relationships should be helpful for healthcare prediction (e.g., drug recommendation, mortality prediction, readmission prediction …).
        Each response should be exactly in format of [{{ENTITY 1}}, {{RELATIONSHIP}}, {{ENTITY 2}}]. The relationship is directed, so the order matters.
        ENTITY 1 can be either code 1 or code 2 and ENTITY 2 can be either code 1 or code 2, as long as ENTITY 1 and ENTITY 2 are different.
        Any RELATIONSHIP in [{{ENTITY 1}}, {{RELATIONSHIP}}, {{ENTITY 2}}] should be conclusive, make it as short as possible.
        Curly brackets {{}} must encapsulate any entity or relationship produces.
        If the RELATIONSHIP between the codes are largely indirect, output the entities and RELATIONSHIP = N/A like this: [{{ENTITY 1}}, {{N/A}}, {{ENTITY 2}}].

        Example 1:
        code pair: ['OPIOID ANALGESICS', 'ASPIRIN']
        response: [{{OPIOID ANALGESICS}}, {{HAS SIMILAR EFFECT AS}}, {{ASPIRIN}}]

        Example 2:
        code pair: ['Osteoporosis', 'Diverticulosis and diverticulitis']
        response: [{{Osteoporosis}}, {{N/A}}, {{Diverticulosis and diverticulitis}}]

        Prompt:
        code 1: {code_1}
        code 2: {code_2}

        ### Response:'''

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        output = model.generate(inputs=input_ids,
                                max_new_tokens=256,
                                pad_token_id=tokenizer.eos_token_id,
                                do_sample=False)
        response_text = tokenizer.decode(output[0])
        save_response_to_file(response_text, idx)

    all_responses = load_responses()
    concatenated_responses = "\n".join(all_responses)

    with open('./responses/concatenated_responses.txt', 'w') as f:
        f.write(concatenated_responses)


if __name__ == "__main__":
    main()
