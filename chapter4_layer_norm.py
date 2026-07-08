#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/06/20 00:58:38
@Author : weiyutao
@File : chapter4.py
implementing a gpt model from scratch to generate text.

在计算方差的时候分母是使用n还是n-1？pytorch中一般是n-1，tensorflow中一般是n
无偏估计：比如投飞镖，靶心是总体的真实参数，投飞镖的技术可能不太稳定，有的偏左，有的偏右
有的偏上，有的偏下，但是如果投了无数次，所有飞镖落点的平均位置，刚好就在靶心上，意味着估计方法
在长期看来是没有系统性误差的。

有偏估计：你的飞镖总是习惯性地偏向某个方向。即使投了无数次，所有飞镖的平均落点依然不在靶心上。
而是偏离了靶心，这就说明在长期看来你的估计方法存在系统性误差。

为什么计算样本方差的时候，除以n是有偏的。在统计学中，方差是用来衡量数据分散程度的，
当我们拥有全部总体数据（数量为N）时，我们可以算出真实的总体平均值。此时总体真实方差的公式为：
Σ(x_i-μ)**2/N, 但是在现实中，我们通常只有一部分样本数据（数量为n），此时我们不知道真实的总体
平均值μ，我们智能算出来样本自己的平均值来计算方差，但是此时如果还是直接除以样本数量n，
得到的就是有偏方差（样本方差的有偏估计）Σ(x_i-x^)/n
为什么他是有偏的？核心在于样本数据天生就离样本均值x^更近，而不是离总体均值μ更近。比如
衡量一群人的身高差异，但是不知道全国的平均身高，只能以这群人自己的平均身高作为基准，因为这群人跟他们
自己内部的平均身高更接近，所以算出来的差异程序会比他们在全国范围内的真实差异程度要小。所以
除以n会系统性低低估真实方差，这就是它被称为有偏估计的原因。为什么除以n-1就变成了无偏估计？
自由度：手头有n个数据，已知样本均值x^，前n-1个数据是可以随便取得，但是第n个数据的值就被锁定了。因此
真正自由变化提供新信息的数据点只有n-1个，这就叫做自由度。所以在计算方差时，应该除以有效自由度，
而不是简单的数量。但是这个偏差在随着样本数量逐渐增加而递减的，LLM中每个词的嵌入维度都是很大的
，比如768，所以除以768和767基本是没有偏差的。




"""

"""
Total number of parameters: 163,009,536
Token embedding layer shape: torch.Size([50257, 768])
Output layer shape: torch.Size([50257, 768])
Number of trainable parametersconsidering weight trying: 124,412,160
Total size of the model: 621.83 MB
"""
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


"""
Total number of parameters: 406,212,608
Token embedding layer shape: torch.Size([50257, 1024])
Output layer shape: torch.Size([50257, 1024])
Number of trainable parametersconsidering weight trying: 354,749,440
Total size of the model: 1549.58 MB
"""
GPT_CONFIG_MEDIUM = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "drop_rate": 0.1,
    "qkv_bias": False
}



"""
Total number of parameters: 838,220,800
Token embedding layer shape: torch.Size([50257, 1280])
Output layer shape: torch.Size([50257, 1280])
Number of trainable parametersconsidering weight trying: 773,891,840
Total size of the model: 3197.56 MB
"""
GPT_CONFIG_LARGE = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 1280,
    "num_heads": 20,
    "num_layers": 36,
    "drop_rate": 0.1,
    "qkv_bias": False
}



"""
Total number of parameters: 1,637,792,000
Token embedding layer shape: torch.Size([50257, 1600])
Output layer shape: torch.Size([50257, 1600])
Number of trainable parametersconsidering weight trying: 1,557,380,800
Total size of the model: 6247.68 MB
"""
GPT_CONFIG_XL = {
    "vocab_size": 50257,
    "context_length": 1024,
    "embedding_dim": 1600,
    "num_heads": 25,
    "num_layers": 48,
    "drop_rate": 0.1,
    "qkv_bias": False
}


import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tokenizer_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.drop_embedding = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["num_layers"])])        
        self.final_norm = DummyLayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_length = in_idx.shape
        token_embeddings = self.tokenizer_embedding(in_idx)
        position_embeddings = self.position_embedding(torch.arange(seq_length, device=in_idx.device))
        x = token_embeddings + position_embeddings
        x = self.drop_embedding(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    


class DummyTransformerBlock(nn.Module):
    
    def __init__(self, cfg):
        super().__init__()
    
    
    def forward(self, x):
        return x
    
    
class DummyLayerNorm(nn.Module):
    """
    Normalizing activations with layer normalization.
    Training deep nueral networks with many layers can sometimes prove challenging due to problems like vanishing or exploding gradients.
    These problems lead to unstable training dynamics and make it difficult for the network to effectively adjust its weights, which means
    the learning process struggles to find a set of parameters (weights) for the neural network that minimizes the loss function.
    The main idea behind layer normalization is to adjust the activation (outputs) of a neural network layer to have a mean of 0 and a variance of 1, also
    known as unit variance. This adjustment speeds up the convergence to effective weights and ensures consistent, reliable training. In GPT-2 and modern
    transformer architectures, layer normalization is typically applied before and after the multi-head attention module.
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    
    def forward(self, x):
        return x
    
    

class LayerNorm(nn.Module):
    """
    The difference between layer normalization and batch layer normalization.
    假设有32个矿石，每个矿石的特征维度是204
    批次归一化：
        对于每一个波段维度，在当前32个批次内计算它的均值和方差，然后调整。比如先看第一个波段，然后在整个批次中调整
        所有矿石第一个波段的数值，依次类推第二个波段，等等。
    层归一化：
        对于每块矿石，在当前矿石的204个波段内计算均值和方差，然后调整。比如第一个矿石，计算当前矿石的所有波段的均值和方差，
        然后调整当前矿石的所有波段。
        
    批次归一化是全班同学考同一门数学课，看自己第二步数学排名。层归一化是一个人考了所有科目，看自己的数学成绩在自己总分里的排名。
    批次归一化适用于batch较大且批次数据有较强的代表性的图像网络。层归一化适用于序列长度不确定的NLP任务。
    
    """
    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift




if __name__ == "__main__":
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)
    
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape: ", logits.shape)
    print(logits)
    
    
    """
    layer normal example.
    """
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)

    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Varaince:\n", var)

    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Normalized layer outputs:\n", out_norm)
    print("Mean:\n", mean)
    print("Variance:\n", var)


    torch.set_printoptions(sci_mode=False)
    print("Mean:\n", mean)
    print("Variance:\n", var)


    layer_norm = LayerNorm(embedding_dim=5)
    out_layer_norm = layer_norm(batch_example)
    mean = out_layer_norm.mean(dim=-1, keepdim=True)
    var = out_layer_norm.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variace:\n", var)