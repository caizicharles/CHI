import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from openai import OpenAI
from tqdm import tqdm
import re
import math

from utils.utils import *


def get_BERT_embeddings(strings: np.ndarray):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    inputs = tokenizer(strings.tolist(), return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :]

    return embedding


def verify_embeddings(triplets: np.ndarray):

    client = OpenAI(
        api_key=
        
    )
    MODEL = 'gpt-4o'
    call_triplet_num = 300

    triplets = triplets.tolist()
    for idx, triplet in enumerate(triplets):
        triplets[idx] = [f'{idx % call_triplet_num } '] + triplet

    full_output = []
    for idx in tqdm(range(math.ceil(len(triplets) / call_triplet_num))):
        start_idx = idx * call_triplet_num
        stop_idx = (idx + 1) * call_triplet_num
        if stop_idx > len(triplets):
            stop_idx = len(triplets)

        input = triplets[start_idx:stop_idx]

        completion = client.chat.completions.create(
            model=MODEL,
            seed=0,
            messages=[{
                "role":
                    "system",
                "content":
                    "You are a medical expert that helps to verify relationships between medical concepts (either diagnoses, procedure, or drug)."
            }, {
                "role":
                    "user",
                "content":
                    f'''Given a list of triplets in the form of [[1, ENTITY 1, RELATION, ENTITY 2], [2, ENTITY 1, RELATION, ENTITY 2],...]
                where number is the triplet index, ENTITY is either a medical diagnoses, procedure, or drug, RELATION is the connection from the first ENTITY to the ssecond.
                assertain whether each triplet is correct, record <triplet index>-True if this triplet valid otherwise
                record <triplet index>-False. 
                The total number of booleans in the output list should match the number of triplets in the input list.
                No explanation is needed in the output.
                Below is an example of input and output.

                Example:
                Input: [['0 ', 'ABDOMINAL HERNIA', 'CAN AGGRAVATE', 'ASTHMA'], ['1 ', 'ABDOMINAL HERNIA', 'CAN AGGRAVATE', 'ESOPHAGEAL DISORDERS']]
                Output: [{{0-True}}, {{1-True}}]
                
                Input: {input}
                Output:'''
            }])

        output = completion.choices[0].message.content
        print(output)
        output = re.findall(r'\{(.*?)\}', output)
        full_output.extend(output)

        assert len(input) == len(output), f'I/O shape mismatch: {len(input)} {len(output)}'

    mask = []
    for o in full_output:
        if 'True' in o:
            mask.append(True)
        elif 'False' in o:
            mask.append(False)
    mask = np.array(mask)

    assert len(mask) == len(triplets), f'Mask num {len(mask)} must match triplet num {len(triplets)}'

    return mask
