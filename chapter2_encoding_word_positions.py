#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/05/10 12:39:56
@Author : weiyutao
@File : chapter2_encoding_word_positions.py
    In principle, the deterministic, position-independent embedding of the token ID
is good for reproducibility purpose. However, since the self-attention mechanism of
LLMs itself is also position-agnostic, it is helpful to inject additional position information
into the LLM.
    To achieve this, we can use two broad categories of postion-aware embeddings: relative positional embeddings
and absolute positional embeddings. Absolute positional embeddings are directly associated with
specific positions in a sequence. For each position in the input sequence, a unique embedding is added to the token's
embedding to convey its exact location. For instance, the first token will have a specific postional
embedding, the second token another distinct embedding.add()
    Why query, key, and values? The terms "key", "query", and "value" in the context of attention mechanisms are
borrowed from the domain of information retrieval and databases, where similar concepts are used to store, 
and retrieval information.
    A query is analogous to a search query in a database. It represents the current item, the model focuses on 
or tries to understand. The query is used to probe the other parts of the input sequence to determine how much attention
to pay to them.
    The key is like a database key used for indexing and searching. In the attention mechanism, each item in the input sequence
has an associated key. These keys are used to match the query.
    The value in this context is similar to the value in a key-value pair in database. It represents the actual content or representation
of the input items. Once the model determines which keys are most relevant to the query, it retrieves the corresponding vlaue.
"""

import torch

from chapter2_dataloader import create_dataloader_v1

with open("The_Verdict.txt", encoding="utf-8") as f:
        raw_text = f.read()

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
data_loader = create_dataloader_v1(
    txt=raw_text,
    batch_size=8,
    max_length=max_length,
    stride=max_length,
    shuffle=True
)

data_iter = iter(data_loader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)


token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)


input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)




import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attention_scores = queries @ keys
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vector = attention_weights @ values
        return context_vector
        

        
        