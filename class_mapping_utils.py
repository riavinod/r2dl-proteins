# https://arxiv.org/abs/2102.07325

import os
import data_utils
import json
import numpy as np
import datasets
from transformers import AutoTokenizer
import argparse
import torch.nn as nn
import torch


def get_mapped_logits(logits, class_mapping, h, theshold_f):
    """
    logits : Tensor of shape (batch_size, 1000) # imagenet class logits
    class_mapping: class_mapping[i] = list of image net labels for text class i
    theshold_f : aggregating label prob, max or mean
    h : label mapping
    """
    if h is None:
        mapped_logits = []
        for h_i in range(len(class_mapping)):
            if theshold_f == "max":
                class_logits, _ = torch.max(logits[:,class_mapping[h_i]], dim = 1) # batch size
            elif theshold_f == "mean":
                class_logits = torch.mean(logits[:,class_mapping[h_i]], dim = 1) # batch size
            else:
                pass
            mapped_logits.append(class_logits)
        return torch.stack(mapped_logits, dim = 1)
    else:
        scores = nn.Softmax(dim=-1)(logits)
        mapped_logits = h(scores)
        return mapped_logits

def create_label_mapping(num_classes, m_per_class, n, n_labels):
    if prot_task_labels is None:
        prot_task_labels = range(n_labels)

    class_mapping = [[] for i in range(num_classes)]

    idx = 0
    for _m in range(m_per_class):
        for _class_no in range(num_classes):
            class_mapping[_class_no].append(num_classes[idx])
            idx += 1
    return class_mapping
