# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:31:03 2020

@author: 63561
"""

import torch
import random

def reverse_batch(data, device):
    
    idx = [i for i in range(data.size(1)-1, -1, -1)]
    idx = torch.LongTensor(idx).to(device)
    return data.index_select(1, idx)

def log_seq_prob(log_prob, y, device):
    
    with torch.no_grad():
        mask = torch.zeros(log_prob.shape)
        dim1 = []
        dim2 = []
        dim3 = []
        for i in range(len(y)):
            for j in range(len(y[i])):
                dim1.append(i)
                dim2.append(j)
                dim3.append(y[i][j])
        mask = mask.index_put_([torch.LongTensor(dim1), torch.LongTensor(dim2), torch.LongTensor(dim3)],
                               torch.Tensor([1 for _ in dim1]))
        mask = mask.to(device)
        return (log_prob * mask).sum(-1).sum(-1)
    
def random_sample_from_unnormalized_prob(prob):
    
    threshold = [prob[0]]
    for p in prob[1:]:
        threshold.append(threshold[-1] + p)
    rand = random.uniform(0, threshold[-1])
    for i, p in enumerate(threshold):
        if rand <= p:
            return i
    return len(prob)-1