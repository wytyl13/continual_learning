#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/07 12:23
@Author  : weiyutao
@File    : chapter4_gpt_model.py
Coding the gpt model.
"""

import torch
import torch.nn as nn

from examples.book_chapter.chapter4_layer_norm import GPT_CONFIG_124M, GPT_CONFIG_MEDIUM, GPT_CONFIG_LARGE, GPT_CONFIG_XL
from examples.book_chapter.chapter4_transformer_block import TransformerBlock
from examples.book_chapter.chapter4_layer_norm import LayerNorm

class GPTModel(nn.Module):
    """
    input.
    token embedding layer.
    position embedding layer.
    dropout
    
    -------------- transformer block 12 times--------------
    layer norm 1.
    multihead attention layer.
    dropout.
    shortcut connection.

    layer norm 2.
    feed forward layer.
    dropout.
    shortcut connection.
    -------------- transformer block 12 times--------------
    
    final layer norm.
    linear output layer.
    """
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.drop_embedding = nn.Dropout(cfg["drop_rate"])

        self.transformer_block=  nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["embedding_dim"])
        
        # 注意全连接层的权重在内存中存储是转置存储的。这样保证计算性能，因为直接按照行取值相乘即可。不需要取前者的行和后者的列了
        # X(b, seq_len, embedding_dim) @ W_out_head(vocab_size, embedding_dim).T
        # 因为线性代数理论上矩阵相乘是前一个的行*后一个的列，这在计算机实际运算的时候效率不高
        # 为了让前一个的行*后一个的行等效于前一个的行*后一个的列，直接使用转置后的矩阵在内存中存储，然后再使用公式的时候需要对权重也就是后者进行转置。
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embedding = self.token_embedding(in_idx)

        position_embedding = self.position_embedding(torch.arange(seq_len, device=in_idx.device))
        x = token_embedding + position_embedding
        x = self.drop_embedding(x)
        x = self.transformer_block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # idx: (batch, n_token)
        idx_cond = idx[:, -context_size:] # the context length (input seq_length + output seq_length) is context_size.
        with torch.no_grad():
            logits = model(idx_cond) # (batch, n_token, vocab_size)
        logits = logits[:, -1, :] # (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx



if __name__ == "__main__":
    
    import tiktoken
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    out = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape: ", out.shape)
    print(out)
    
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    
    """
    We initialized a 124-million-parameter GPT model, why is the actual number of parameters 163 million?
    The reason is a concept called weight trying （权重绑定）, which was used in original GPT-2 architecthre.
    It means that the original GPT-2 architecture reuses the weights from the token embedding layer in its output 
    layer. We can see the number of the embedding layer and out_head layer.
    """ 
    print("Token embedding layer shape:", model.token_embedding.weight.shape)
    
    print("Output layer shape:", model.out_head.weight.shape)
    
    
    total_params_gpt2 = (
        total_params - sum(p.numel() for p in model.out_head.parameters())
    )
    
    print(f"Number of trainable parameters"
          f"considering weight trying: {total_params_gpt2:,}"
          )
    """
    As we can see, the model is now only 124 million parameters large, matching the original size of the GPT-2 model.
    Weight trying reduces the overall memory footprint and computational complexity of the model. However, in my experience,
    using separate token embedding and output layers results in better training and model performance; hence, we use separate
    laters in our GPTModel implementation. The same is true for modern LLMs. However, we will revisit and implement the weigth
    trying concept later when we load the pretrained weigths from OpenAI.
    """
    
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")
    
    print("================= start generate text =================")
    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape: ", encoded_tensor.shape)

    model.eval()
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output:", out)
    print("Output length:", len(out[0]))
    
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
    
    """
    注意力机制和嵌入层中不涉及归一化处理
    嵌入层和注意力机制层和残差之前都涉及dropout处理
    在进行注意力机制和前馈网络计算之前都要进行归一化处理
    在进行计算输出层之前也要进行归一化处理
    并且在计算完注意力机制和前馈网络之后都要先进行dropout，然后进行残差连接
    每一层先注意力机制计算，再进行feed_forward计算
    TransformerBlock包含：注意力机制计算和前馈网络计算
    
    GPT网络由：嵌入层 + TransformerBlock + 输出层
    嵌入层包含：嵌入+位置编码+dropout
    TransformerBlock包含：多头注意力计算+前馈网络
    输出层包含：层归一化和线性映射（全连接网络）
    前馈网络包含：全连接升维度（4*embedding_dimension），激活函数GELU，全连接降维
    
    注意归一化一般在每一层的最开始，残差一般在每一层的最结尾，而dropout一般在残差之前，如果没有残差则一般在每一层的最后。
    注意力权重计算出来首先进行dropout操作。
    """

    
