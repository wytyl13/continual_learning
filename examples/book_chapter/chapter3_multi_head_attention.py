#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/06/19 12:04
@Author  : weiyutao
@File    : chapter3_multi_head_attention.py
The difference between torch.view and torch.transpose is that torch.view is used to reshape a tensor without changing its data, 
while torch.transpose is used to permute the dimensions of a tensor. torch.view can only be used to change the shape of a tensor, 
but it cannot change the order of the dimensions. For example, if you have a tensor of shape (2, 3) and you want to reshape it to (3, 2), you can use torch.view.
tensor = torch.randn(2, 3)
reshaped_tensor = tensor.view(3, 2)     
However, if you want to permute the dimensions of the tensor, you can use torch.transpose.
tensor = torch.randn(2, 3)
transposed_tensor = tensor.transpose(0, 1)  # This will change the shape                
view是按照底层物理顺序无脑重新切块分装，不改变数据原本的先后顺序，而transpose是在数学逻辑上真正交换维度身份，会彻底改变数据的读取顺序。

self.out_proj = torch.nn.Linear(hidden_dim, hidden_dim)
x = batch, num_tokens, hidden_dim
hidden_dim = num_heads * head_dim

W_q = d_in, d_out
W_k = d_in, d_out
W_v = d_in, d_out

d_in = d_out = hidden_dim = num_heads * head_dim


queries = x @ W_q = batch, num_tokens, hidden_dim
keys = x @ W_k = batch, num_tokens, hidden_dim
values = x @ W_v = batch, num_tokens, hidden_dim

queries = keys.view(batch, num_tokens, num_heads, head_dim)
keys = keys.view(batch, num_tokens, num_heads, head_dim)
values = values.view(batch, num_tokens, num_heads, head_dim)

keys.transpose(1, 2) = batch, num_heads, num_tokens, head_dim
queries.transpose(1, 2) = batch, num_heads, num_tokens, head_dim
values.transpose(1, 2) = batch, num_heads, num_tokens, head_dim

attention_scores = queries @ keys.transpose(2, 3) = batch, num_heads, num_tokens, head_dim @ batch, num_heads, head_dim, num_tokens = batch, num_heads, num_tokens, num_tokens

attention_weights = batch, num_heads, num_tokens, num_tokens
context_vectors = attention_weights @ values = batch, num_heads, num_tokens, num_tokens @ batch, num_heads, num_tokens, head_dim = batch, num_heads, num_tokens, head_dim
context_vectors.transpose(1, 2) = batch, num_tokens, num_heads, head_dim
context_vectors.view(batch, num_tokens, hidden_dim) = batch, num_tokens, num_heads * head_dim
context_vectors = self.out_proj(context_vectors) = batch, num_tokens, hidden_dim
"""
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, 
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    
    def forward(self, x):
        batch, num_tokens, d_in = x.shape

        queries = self.W_query(x) # batch, num_tokens, hidden_dim
        keys = self.W_key(x)
        values = self.W_value(x)
        
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2) # batch, num_heads, num_tokens, head_dim
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3) # batch, num_heads, num_tokens, head_dim @ batch, num_heads, head_dim, num_tokens = batch, num_heads, num_tokens, num_tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1) # batch, num_heads, num_tokens, num_tokens
        attention_weights = self.dropout(attention_weights)
        
        context_vectors = attention_weights @ values # batch, num_heads, num_tokens, num_tokens @ batch, num_heads, num_tokens, head_dim = batch, num_heads, num_tokens, head_dim
        context_vectors = context_vectors.transpose(1, 2)
         
        # Because transpose() is a lazy operation that only alters the metadata (strides) resulting in a non-contiguous tensor, and view() strictly requires
        # the underlying physical memory to be contiguous, we must use contiguous() to force memory reallocation and realign the data.
        context_vectors = context_vectors.contiguous().view(batch, num_tokens, self.d_out) # batch, num_tokens, d_out
        
        context_vectors = self.out_proj(context_vectors) # batch, num_tokens, d_out
        return context_vectors




if __name__ == "__main__":
    
    inputs = torch.tensor([
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55]
    ])
    
    batch = torch.stack((inputs, inputs), dim=0)
    print(batch.shape)
    print(batch)

    torch.manual_seed(123)
    
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    
    mha = MultiHeadAttention(
        d_in=d_in, 
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=2
    )
    context_vectors = mha(batch)
    
    print(context_vectors)
    print("context_vectors.shape: ", context_vectors.shape)
        
        
        