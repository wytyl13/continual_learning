#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/10 10:15
@Author  : weiyutao
@File    : chapter5_load_model.py
"""
import torch
import tiktoken
import numpy as np

from examples.book_chapter.chapter4_gpt_model import GPTModel
from examples.book_chapter.chapter5_train import GPT_CONFIG_124M, text_to_token_ids, token_ids_to_text
from examples.book_chapter.chapter5_decoding_strategies import generate
from examples.book_chapter.chapter4_transformer_block import TransformerBlock
from examples.book_chapter.chapter3_multi_head_attention import MultiHeadAttention
from examples.book_chapter.chapter4_feed_forward import FeedForward

model_config = {
    "gpt2-small (124M)": {"embedding_dim": 768, "num_layers": 12, "num_heads": 12},
    "gpt2-medium (355M)": {"embedding_dim": 1024, "num_layers": 24, "num_heads": 16},
    "gpt2-large (774M)": {"embedding_dim": 1280, "num_layers": 36, "num_heads": 20},
    "gpt2-xl (1558M)": {"embedding_dim": 1600, "num_layers": 48, "num_heads": 25}
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_config[model_name])
NEW_CONFIG.update({"context_length": 1024})
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()



def assign(left_tensor, right_array):
    """
    We will first define a small assign utility function that checks whether two tensors or arrays
    (left and right) have the same dimensions or shape and returns the right tensor as trainable PyTorch parameters.
    """
    if left_tensor.shape != right_array.shape:
        raise ValueError(f"shape mismatch. Left: {left_tensor.shape}, "
                         "Right: {right_array.shape}"
        )
    # return torch.nn.Parameter(torch.tensor(right_array))

    with torch.no_grad():
        left_tensor.copy_(torch.tensor(right_array))
    return left_tensor



def load_weights_into_gpt(gpt: GPTModel, params):
    """
    Next, we define a load_weigts_into_gpt function that loads the weights from the params dictionary into a GPTModel instance gpt.

    # GPT architecture
        transformer block是核心：主要分为multihead_attention和feed_forward，这两个模块的最开始记录shortcut，末尾进行shortcut_connection操作
        dopout适用于：嵌入层、注意力层的注意力权重、TransformerBlock层的multihead_attention层输出和feed_forward层的输出
        shortcut_connection在transformer_block的multihead_attention和feed_forward得末尾（也是dropout之后）
        除了dropout、gelu没有可训练参数，其余层全有训练参数
        layer_norm层是scale和shift这两个训练参数，维度是embedding_dim
        其余层（W_query, W_key, W_value, out_proj, layers_linear1, layers_linear2, out_head_linear1）的可训练参数：weight和bias
        token_embedding、position_embedding层的可训练参数weight，无bias
    GPTModel
        token_embedding
        position_embedding
        drop_embedding
        
        TransformerBlock
            shortcut=x
            layer_norm1
            MultiHeadAttention
                W_query
                W_key
                W_value
                dropout_attention_weight
                out_proj
            dropout_attention_out
            x=shortcut_connection
            
            shortcut=x
            layer_norm2
            FeedForward
                layers
                    linear1(embedding_dim, 4*embedding_dim)
                    gelu
                    linear2(4*embedding_dim, embedding_dim)
            dropout_feed_forward_out
            x=short_cut_connection
        final_norm
        
        out_head
            linear(embedding_dim, vocab_size)
    """
    # the GPT-2 architecture is consist of 1.embedding layer, 2.tranformer block layer, 
    # 3.final normalization layer and 4.out_head layer.
    
    # 1. The weight of token embedding and position embedding layers.
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params['wpe'])
    
    # token_embedding: nn.Embedding(vocab_size, embedding_dim), which weight shape is (vocab_size, embedding_dim)
    gpt.token_embedding.weight = assign(gpt.token_embedding.weight, params['wte'])
    
    
    # 2. The transformer_block is consist of multihead_attention block and 
    # feed_forward block and layer_norm1 and layer_norm2.
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        transformer_block: TransformerBlock = gpt.transformer_block[b]
        multi_head_attention: MultiHeadAttention = transformer_block.attention
        feed_forward: FeedForward = transformer_block.feed_forward
        
        # 2.1 The multihead_attention block
        multi_head_attention.W_query.weight = assign(
            multi_head_attention.W_query.weight, q_w.T
        )
        multi_head_attention.W_key.weight = assign(
            multi_head_attention.W_key.weight, k_w.T
        )
        multi_head_attention.W_value.weight = assign(
            multi_head_attention.W_value.weight, v_w.T
        )
        
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        multi_head_attention.W_query.bias = assign(
            multi_head_attention.W_query.bias, q_b
        )
        multi_head_attention.W_key.bias = assign(
            multi_head_attention.W_key.bias, k_b
        )
        multi_head_attention.W_value.bias = assign(
            multi_head_attention.W_value.bias, v_b
        )
        
        multi_head_attention.out_proj.weight = assign(
            multi_head_attention.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T
        )

        multi_head_attention.out_proj.bias = assign(
            multi_head_attention.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"]
        )
        
        
        # 2.2 The feed_froward block
        feed_forward.layers[0].weight = assign(
            feed_forward.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        
        feed_forward.layers[0].bias = assign(
            feed_forward.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        
        feed_forward.layers[2].weight = assign(
            feed_forward.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        
        feed_forward.layers[2].bias = assign(
            feed_forward.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"]
        )
        
        
        # 2.3 The layer normalization.
        transformer_block.norm1.scale = assign(
            transformer_block.norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        
        transformer_block.norm1.shift = assign(
            transformer_block.norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )

        transformer_block.norm2.scale = assign(
            transformer_block.norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        
        transformer_block.norm2.shift = assign(
            transformer_block.norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )
        
        
    # 3. The final normalization layer.
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    
    # 4. The out head layer.
    # The original GPT-2 model by OpenAI reused the token embedding weights in the output layer 
    # to reduce the total number of parameters, which is a concept known as weight tying.
    # nn.Linear(embedding_dim, vocab_size), which weight shape is (vocab_size, embedding_dim)
    # because x @ W.T = (batch, num_tokens, embedding_dim) @ (embedding_dim, vocab_size) = (batch, num_tokens, vocab_size)
    # so the shape of weight in PyTorch must be (vocab_size, embedding_dim), which is different
    # from nn.Embedding. Notice the weight shape is negative from the Linear code description.
    # The linear1 = nn.Linear(input_dim, out_dim), which the weight of linear1 shape is (out_dim, in_dim)
    # The token_embedding = nn.Embedding(vocab_size, embedding_dim), which the weight of token_embedding shape
    # is (vocab_size, embedding_dim), because the nn.Embedding Module execute the index operation not
    # the matrix multipy operation. Weight tying enables backpropagated gradients to simultaneously
    # update both input and output representations, forcing the models's semantic understanding and 
    # expression to strictly align on the same high-dimensional mainfold, thereby minimizing local oprima
    # and accelerating training.
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"]) 
        
        


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    model = GPTModel(GPT_CONFIG_124M)
    
    checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    """
    
    """
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Evert effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    """
    
    """
    # download gpt code
    import urllib.request 
    url = ( 
           "https://raw.githubusercontent.com/rasbt/" 
           "LLMs-from-scratch/main/ch05/" 
           "01_main-chapter-code/gpt_download.py" 
    ) 
    filename = url.split('/')[-1] 
    urllib.request.urlretrieve(url, filename)
    """
    
    """
    # download gpt model.
    """
    from examples.book_chapter.gpt_download import download_and_load_gpt2
    import os
    # os.environ["http_proxy"] = "http://127.0.0.1:7890"
    # os.environ["https_proxy"] = "http://127.0.0.1:7890"
    settings, params = download_and_load_gpt2(
        model_size="124M", models_dir="/home/weiyutao/ai/continual_learning/gpt2"
    )
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())
    
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    torch.manual_seed(123)
    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5
    )
    print("Output rext:\n", token_ids_to_text(token_ids, tokenizer))
    
    
    
    