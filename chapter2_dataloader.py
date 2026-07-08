#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/05/10 08:54:13
@Author : weiyutao
@File : chapter2_dataloader.py
前向传播是一个很泛化的概念，不是必须进行数学运算才属于，embedding层进行单纯的查表操作也属于前向传播
查表的数学本质是one-hot向量相乘，在进行反向传播的时候，未被激活的索引在嵌入层的梯度为0，激活的索引在嵌入
层的梯度为1，然后从下游层反向传播来的提取乘以当前层的梯度就是当前层每个索引权重需要更新的值。

注意这里梯度为1，梯度为0和其他普通层梯度的区别，1说明当前层全量更新，0就是不更新，普通层比如线性映射层
因为输入X每个位置都有数值，所以每个W权重都会被更新，而嵌入层只有激活索引行会被更新。因为嵌入层的
权重是随机初始化的，其权重就是对应的每个输入词的嵌入数据，而嵌入层的输入是输入索引数组的one-hot编码
包含大量的输入为0，其实就类似于在上一层的输出加入了dropout层将本层的部分输入置为0了。但是dropout是随机的
嵌入层是固定的one-hot编码。可以想象使用embedding层这种稀疏更新机制去设计类脑神经网络，但是需要考虑
的是如何将多模态特征输入（连续数据）格式化为离散的数据？比如id，也就是如何把连续的信号，稳定合理地变成离散
的索引，去触发稀疏网络，一旦某个索引永远不会被处罚，他就会变成一个幽灵神经元，造成资源浪费。
torch使用的embedding层比如nn.embedding底层是索引的方式，是为了效率考虑，并不是进行的onehot编码和初始化权重进行的
矩阵运算。那么在反向传播的时候如何计算梯度呢？首先会初始化一个梯度矩阵，维度和嵌入层的矩阵维度一致，比如6*3
假设前向传播的tensor是torch.tensor([1, 3, 1])，对应的意思是要查表第1个索引，第三个索引和第1个索引，那么对应的反向传播回来的时候
会进行梯度的累加，反向传播回来的梯度维度是3*3，因为是3个矩阵对应的3个隐藏层，然后将第一行梯度累加给初始化梯度矩阵的第二行（索引1），第二行
梯度累加给初始化梯度矩阵的第四行（索引3），第三行梯度累加给初始化梯度矩阵的第二行（索引1）。以此形成前向和反向传播的操作。
"""


import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt, 
    batch_size=4, 
    max_length=256, 
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0    
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return data_loader




if __name__ == "__main__":
    with open("The_Verdict.txt", encoding="utf-8") as f:
        raw_text = f.read()
    
    """
    data_loader = create_dataloader_v1(
        txt=raw_text,
        batch_size=1, 
        max_length=4, 
        stride=1, 
        shuffle=False
    )
    
    data_iter = iter(data_loader)
    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)
    """
    
    data_loader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4)
    data_iter = iter(data_loader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)


    input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 6
    output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)

    print(embedding_layer(torch.tensor([3])))
    print(embedding_layer(input_ids))


