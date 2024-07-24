from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pickle
import itertools
import numpy as np

path = '/home/engs2635/Desktop/caizi/graph_construction/nodes.pickle'


def read_pickle_file(path: str):

    with open(path, "rb") as f:
        file = pickle.load(f)

    return file


code_list = np.array(read_pickle_file(path))
code_list = code_list[:, 1]
code_pairs = list(itertools.permutations(code_list, 2))

model_id = "malhajar/meditron-70b-chat"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             torch_dtype=torch.float16,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_id)

response = []

for pair in code_pairs:

    code_1 = pair[0]
    code_2 = pair[1]

    prompt = f'''
    ### Instruction:

    Given a pair of medical codes (could be a diagnosis, procedure, or drug), extrapolate the most plausible relationships between the drugs.
    The relationships should be helpful for healthcare prediction (e.g., drug
    recommendation, mortality prediction, readmission prediction …)
    Each response should be exactly in format of [{{ENTITY 1}}, {{RELATIONSHIP}}, {{ENTITY 2}}]. The
    relationship is directed, so the order matters.
    ENTITY 1 can be either code 1 or code 2 and ENTITY 2 can be either code 1 or code 2, as long as ENTITY 1 and ENTITY 2 are different.
    Any RELATIONSHIP in [{{ENTITY 1}}, {{RELATIONSHIP}}, {{ENTITY 2}}] should be conclusive, make it as short as possible.
    Curly brackets {{}} must encapsulate any entity or relationship produces.

    Example:
    code pair: ['OPIOID ANALGESICS', 'ASPIRIN']
    response: [{{OPIOID ANALGESICS}}, {{HAS SIMILAR EFFECT AS}}, {{ASPIRIN}}]

    Prompt:
    code 1: {code_1}
    code 2: {code_2}

    ### Response:'''

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(inputs=input_ids,
                            max_new_tokens=512,
                            pad_token_id=tokenizer.eos_token_id,
                            top_k=50,
                            do_sample=True,
                            top_p=0.95)
    response.append(tokenizer.decode(output[0]))
