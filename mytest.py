import os
import os.path as osp
import re
import itertools
import numpy as np
import torch
import random

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from tqdm import tqdm
import time
from torch_geometric.data import Data, Batch
from torch_geometric.utils import subgraph
from torch_geometric.loader import DataLoader

from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
from utils.utils import *

import networkx as nx
from torch_geometric.utils import from_networkx

from copy import deepcopy
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

import random
import math
import torch_scatter

from sklearn.metrics import accuracy_score

from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

x = torch.tensor([[1],[1],[2],[2],[2]])
y = torch.tensor([[1, 1, 2, 2, 2]])

z = x == y
print(z)
exit()
def nt_xent_loss(embeddings, mask, temperature=0.5):
    """
    Compute NT-Xent contrastive loss.

    :param embeddings: A tensor of shape (N, D) where N is the number of samples and D is the embedding dimension.
    :param mask: A tensor of shape (N,) where each element is an integer label indicating the class of each sample.
    :param temperature: A scaling factor for the logits.
    :return: The computed NT-Xent loss.
    """
    # Normalize the embeddings
    embeddings = F.normalize(embeddings, dim=1)

    # Compute the pairwise cosine similarity (N, N)
    similarity_matrix = torch.matmul(embeddings, embeddings.T)

    # Scale by temperature
    similarity_matrix /= temperature

    # Mask to zero out diagonal (self-similarities)
    N = similarity_matrix.size(0)
    similarity_matrix.fill_diagonal_(float('-inf'))

    # Create label mask (positive pairs have the same label)
    label_mask = mask.unsqueeze(0) == mask.unsqueeze(1)
    label_mask.fill_diagonal_(False)

    # Compute the log-softmax of the similarity matrix
    logits = similarity_matrix.log_softmax(dim=1)

    # Extract positive logits using the label mask
    positive_logits = logits[label_mask]

    # Compute the NT-Xent loss by averaging the negative log-probabilities of the positive pairs
    loss = -positive_logits.mean()

    return loss


# Example usage:
N, D = 394, 128  # Example batch size and embedding dimension
embeddings = torch.randn(N, D)
labels = torch.randint(0, 5, (N,))  # Example labels for 5 classes

loss = nt_xent_loss(embeddings, labels)
print(f"NT-Xent Loss: {loss.item()}")
exit()

llm_response = read_pickle_file('/home/engs2635/Desktop/caizi/CHI/data/mimiciv', 'mimiciv_LLM_response.pickle')
node_id_to_name = read_pickle_file('/home/engs2635/Desktop/caizi/CHI/data/mimiciv', f'mimiciv_node_id_to_name.pickle')

node_name_to_id = read_pickle_file('/home/engs2635/Desktop/caizi/CHI/data/mimiciv', f'mimiciv_node_name_to_id.pickle')
visit_id_to_nodes = read_pickle_file('/home/engs2635/Desktop/caizi/CHI/data/mimiciv',
                                     f'mimiciv_visit_id_to_nodes.pickle')
diagnoses_map = read_csv_file('/home/engs2635/Desktop/caizi/CHI/data/mimiciv', 'diagnoses_code.csv')
procedures_map = read_csv_file('/home/engs2635/Desktop/caizi/CHI/data/mimiciv', 'procedures_code.csv')
prescriptions_map = read_csv_file('/home/engs2635/Desktop/caizi/CHI/data/mimiciv', 'prescriptions_code.csv')

# print(len(prescriptions_map))

# header = ['code', 'code_name']

# for idx, d in enumerate(prescriptions_map):
#     # print(d)
#     info = d['code_name']
#     info = convert_to_uppercase(info)
#     info = info.replace('"', '')
#     prescriptions_map[idx]['code_name'] = info
#     # print(prescriptions_map[idx])

# print(len(prescriptions_map))
# # exit()

# save_with_csv(prescriptions_map, header, '/home/engs2635/Desktop/caizi/CHI/data/mimiciv', 'prescriptions_code.csv')
# exit()

llm_response = np.array(llm_response)
llm_response = np.unique(llm_response, axis=0)

for idx, triplet in enumerate(llm_response):
    formatted_triplet = triplet

    for i, item in enumerate(triplet):
        formatted_item = convert_to_uppercase(item)
        formatted_item = formatted_item.replace('"', '')
        formatted_triplet[i] = formatted_item

    llm_response[idx] = formatted_triplet

src_dst = llm_response[:, [0, -1]]
stored_nodes = list(node_name_to_id.keys())
mask = np.ones(len(llm_response), dtype=bool)
indices_to_remove = []

for idx, pair in enumerate(src_dst):
    for node in pair:
        if node not in stored_nodes:
            # print(node)
            indices_to_remove.append(idx)
    # if idx == 200:
    #     exit()

mask[indices_to_remove] = False
llm_response = llm_response[mask]

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example list of strings
relations = llm_response[:, 1]
relations, rev_map = np.unique(relations, return_inverse=True)
store = deepcopy(relations)

# Tokenize the list of texts
inputs = tokenizer(relations.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512)

# Forward pass through the model
with torch.no_grad():
    outputs = model(**inputs)

cls_embeddings = outputs.last_hidden_state[:, 0, :]

kmeans = KMeans(n_clusters=1750, random_state=0)
kmeans.fit(cls_embeddings)
labels = kmeans.labels_

for cluster_idx in range(1750):
    mask = torch.zeros(len(labels), dtype=bool)
    idx_loc = labels == cluster_idx
    mask[idx_loc] = True

    cluster_edge_names = relations[mask]
    cluster_rep_name = cluster_edge_names[0]
    relations[mask] = cluster_rep_name

    cluster_edge_attr = cls_embeddings[mask]
    cluster_rep_attr = torch.mean(cluster_edge_attr, dim=0)
    cls_embeddings[mask] = cluster_rep_attr

print(cls_embeddings.shape)

full_relations = relations[rev_map]
full_edge_attr = cls_embeddings[rev_map]
llm_response[:, 1] = full_relations
llm_triplets, index = np.unique(llm_response, axis=0, return_index=True)
edge_attr = full_edge_attr[index]
print(llm_triplets.shape, edge_attr.shape)
exit()

# sort_inds = np.argsort(labels)
# labels = np.sort(labels).astype(str)
# relations = relations[sort_inds]
# relations = np.char.add(labels, relations)
# store = store[sort_inds]
# store = np.char.add(labels, store)
# save_txt(relations.tolist(), '/home/engs2635/Desktop/caizi/CHI/data/mimiciv', 'clustered.txt')
# save_txt(store.tolist(), '/home/engs2635/Desktop/caizi/CHI/data/mimiciv', 'original.txt')

# map = {}
# nap = {}
# for num in tqdm(range(500, 3000, 50)):
#     kmeans = KMeans(n_clusters=num, random_state=0)
#     kmeans.fit(cls_embeddings)
#     labels = kmeans.labels_
#     silhouette_avg = silhouette_score(cls_embeddings, labels)
#     ch_score = calinski_harabasz_score(cls_embeddings, labels)
#     map[silhouette_avg] = num
#     nap[ch_score] = num

# map_scores = list(map.keys())
# nap_scores = list(nap.keys())
# x = list(map.values())

# plt.plot(x, map_scores)
# plt.plot(x, nap_scores)
# plt.show()
# print(max(map_scores), map[max(map_scores)])
# print(max(nap_scores), nap[max(nap_scores)])
# exit()

# dist = torch.cdist(cls_embeddings, cls_embeddings)
# dist.diagonal().add_(dist.max())
# bottom_k, _ = torch.topk(dist.flatten(), k=800, largest=False)

# dist_mask = torch.ones_like(dist, dtype=bool)
# threshold = bottom_k.max()
# dist_mask[dist <= threshold] = False

# dist[dist_mask] = 0
# dist = torch.sum(dist, dim=0)
# inds = torch.where(dist > 0)[0]

# response_mask = np.zeros(len(texts), dtype=bool)
# response_mask[inds] = True
# response_mask = response_mask[rev_map]

# llm_triplets = llm_response[response_mask]
# print(llm_triplets[:, 1])
# exit()

# graph = nx.Graph()

# l = [1,2,3,4,5,6,7,8,9]
# e = [(1,2),(3,4),(4,7),(5,8),(7,1),(8,9)]

# for i in l:
#     graph.add_nodes_from([
#         (i, {'y': i, 'x': torch.zeros(8)})
#     ])

# for i in e:
#     graph.add_edge(*i)

# G = from_networkx(graph)
# print(G.edge_index, G.y)

# edge_mask1 = torch.tensor([True, True, True, False, False, False, False, False, False, False, False, False])
# mask_idx = torch.where(edge_mask1)[0]
# A = G.edge_subgraph(mask_idx)
# a = A.subgraph(torch.tensor([0,6]))
# print('---------------')
# print(A.edge_index, A.y)
# print(a.edge_index, a.y)

# edge_mask2 = torch.tensor([False, False, False, True, True, True, False, False, False, False, False, False])
# mask_idx = torch.where(edge_mask2)[0]
# B = G.edge_subgraph(mask_idx)
# b = B.subgraph(torch.tensor([2,3,6]))
# print('---------------')
# print(B.edge_index, B.y)
# print(b.edge_index, b.y)

# class Dataset(torch.utils.data.Dataset):

#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

# dataset = Dataset(dataset=[a, b, b, b])

# dl = DataLoader(dataset=dataset, batch_size=4)

# print('-------------')
# for data in dl:
#     print(data.y, data.edge_index, data.batch)
