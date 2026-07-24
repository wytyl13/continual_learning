#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/19 10:27
@Author  : weiyutao
@File    : dataset_base.py
"""

import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
from typing import Any

class PretrainDataset(Dataset):
    """
    Pretrained Dataset.
    max_length: 每个训练样本的序列长度，也就是上下文窗口大小
    stride: 滑动窗口的步长，决定了样本之间的重叠程度，stride=max_length说明无重叠
    stride=max_length // 2说明有50%的重叠
    注意构建的训练数据和推理时的区别
    """
    def __init__(self, txt, tokenizer: Any, max_length: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            # 构建训练数据：[0, 1, 2, 3]
            input_chunk = token_ids[i:i+max_length]
            # 构建训练数据对应的label：[1, 2, 3, 4]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(txt, tokenizer, max_length, stride, 
                      batch_size=4, shuffle=True, drop_last=True, num_workers=0):
    dataset = PretrainDataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last,
        num_workers=num_workers
    )
    

        