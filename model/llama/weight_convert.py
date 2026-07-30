#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/07/29 18:26:23
@Author : weiyutao
@File : weight_convert.py

================================================================================
自定义LLAMA格式：
================================================================================
LLamaForCausalLM(
  (model): LLamaModel(
    (embed_tokens): Embedding(32000, 2048)
    (drop_embeddings): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-21): 22 x DecoderLayer(
        (input_layernorm): RMSNorm()
        (self_attn): Attention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=256, bias=False)
          (v_proj): Linear(in_features=2048, out_features=256, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): RMSNorm()
        (mlp): SwiGLU(
          (g_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (norm): RMSNorm()
  )
  (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
)
"""

import torch
from typing import Dict, Any

def convert_safetensors_to_custom(hf_state_dict: dict, n_layers: int, tie_word_embeddings: bool) -> dict:
    """
    将HuggingFace格式的state_dict转换为自定义LLamaModel格式

    支持两种输入格式：
    1. HuggingFace下载的safetensors（无前缀）: wte.weight, h.0.ln_1.weight
    2. Transformers训练保存的（有transformer.前缀）: transformer.wte.weight, transformer.h.0.ln_1.weight

    自动检测格式并处理
    官方模型架构
    model = LlamaForCausalLM.from_pretrained("/mnt/wsl/fast_disk/continual_learning/source/hf/llama/tiny_llama")
    print(model)
    
    LlamaForCausalLM(
        (model): LlamaModel(
            (embed_tokens): Embedding(32000, 2048)
            (layers): ModuleList(
            (0-21): 22 x LlamaDecoderLayer(
                (self_attn): LlamaAttention(
                (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
                (k_proj): Linear(in_features=2048, out_features=256, bias=False)
                (v_proj): Linear(in_features=2048, out_features=256, bias=False)
                (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
                )
                (mlp): LlamaMLP(
                (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
                (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
                (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
                (act_fn): SiLU()
                )
                (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
                (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
            )
            )
            (norm): LlamaRMSNorm((2048,), eps=1e-05)
            (rotary_emb): LlamaRotaryEmbedding()
        )
        (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
    )
    
    差异：
    1、drop_embeddings：dropout没有可学习参数，不涉及转换，可能存在官方有些位置没有使用dropout的情况。
    2、(rotary_emb): LlamaRotaryEmbedding() 官方有rotary_emb，这个模块在官方的 state_dict 中会留下一个叫作 rotary_emb.inv_freq 的张量
    自定义模型架构中，旋转位置编码的频率矩阵（freqs_cos 和 freqs_sin）是通过 precompute_freqs_cis 函数动态计算并注册为 buffer 的，所以舍弃官方的rotary_emb字典
    其余结构一致。
    """

    sd = {}
    for k, v in hf_state_dict.items():
        # 1. 舍弃官方的位置编码缓存
        if "rotary_emb.inv_freq" in k:
            continue
        sd[k] = v
            
    # 词嵌入绑定逻辑
    if tie_word_embeddings and "model.embed_tokens.weight" in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
            
    return sd


def convert_custom_to_hf(custom_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 自定义格式的 state_dict 转换回 HuggingFace 官方格式
    用于将我们自己训练/微调后的模型，导出给开源社区或推理引擎(如vLLM)使用。
    """
    hf_sd = {}
    
    for k, v in custom_state_dict.items():
        # HF 的 LlamaForCausalLM 会在模型内部重新注册或计算这些值
        if "freqs_cos" in k or "freqs_sin" in k or "mask" in k:
            continue
            
        # 2. 其余所有参数直接原样拷贝
        hf_sd[k] = v
        
    return hf_sd

if __name__ == "__main__":
    from model.llama.model import LLamaForCausalLM
    from model.llama.config import LLamaConfig
    from transformers import LlamaForCausalLM
    config = LLamaConfig.tiny_llama()
    from config import SOURCE_DIR, OUT_DIR
    """
    # 自定义model结构
    """
    model = LLamaForCausalLM(config)
    print(model)
    
    """
    # HF官方模型架构
    model = LlamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/tiny_llama")
    print(model)
    """
    
    
    
    
    
    