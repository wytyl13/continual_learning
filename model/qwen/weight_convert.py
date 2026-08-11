#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/08/11
@Author : weiyutao
@File : weight_convert.py

Qwen2/2.5 权重格式转换
复用 LLaMA 的 permute 逻辑，额外处理 QKV bias
================================================================================
自定义 Qwen 格式：
================================================================================
QwenForCausalLM(
  (model): QwenModel(
    (embed_tokens): Embedding(151936, 2048)
    (drop_embeddings): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-35): 36 x DecoderLayer(
        (input_layernorm): RMSNorm()
        (self_attn): Attention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)  # Qwen 特有
          (k_proj): Linear(in_features=2048, out_features=256, bias=True)   # Qwen 特有
          (v_proj): Linear(in_features=2048, out_features=256, bias=True)   # Qwen 特有
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): RMSNorm()
        (mlp): SwiGLU(
          (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (norm): RMSNorm()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)

HuggingFace Qwen2 官方格式：
================================================================================
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 2048)
    (layers): ModuleList(
      (0-35): 36 x Qwen2DecoderLayer(
        (self_attn): Qwen2Attention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=256, bias=True)
          (v_proj): Linear(in_features=2048, out_features=256, bias=True)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((2048,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
"""

import torch
from typing import Dict, Any


def permute_for_llama2c(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    将 HuggingFace split-half 格式转换为 llama2.c 相邻成对格式
    view(n_heads, 2, hd//2, dim2).transpose(1,2).reshape(...)
    """
    dim1, dim2 = w.shape
    return (
        w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
         .transpose(1, 2)
         .reshape(dim1, dim2)
    )


def permute_llama2c_to_hf(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    将 llama2.c 相邻成对格式转换回 HuggingFace split-half 格式
    view(n_heads, hd//2, 2, dim2).transpose(1,2).reshape(...)
    """
    dim1, dim2 = w.shape
    return (
        w.view(n_heads, dim1 // n_heads // 2, 2, dim2)
         .transpose(1, 2)
         .reshape(dim1, dim2)
    )


def convert_safetensors_to_custom(
    hf_state_dict: dict,
    n_layers: int,
    tie_word_embeddings: bool,
    n_heads: int = 16,
    n_kv_heads: int = 2,
) -> dict:
    """
    将 HuggingFace Qwen2/2.5 格式的 state_dict 转换为自定义格式

    与 LLaMA 的主要区别：
    1. Qwen 的 q_proj/k_proj/v_proj 有 bias，需要保留
    2. RMSNorm 的 eps 是 1e-6（LLaMA 是 1e-5）
    3. RoPE theta 是 1000000.0（LLaMA2 是 10000.0）

    其余转换逻辑与 LLaMA 完全相同
    """
    sd = {}
    for k, v in hf_state_dict.items():
        # 1. 舍弃官方的位置编码缓存
        if "rotary_emb.inv_freq" in k:
            continue

        # 2. Q/K 权重 permute：HF split-half → llama2.c 相邻成对
        if k.endswith("q_proj.weight"):
            v = permute_for_llama2c(v, n_heads)
        elif k.endswith("k_proj.weight"):
            v = permute_for_llama2c(v, n_kv_heads)

        # 3. Qwen 特有：QKV bias 也需要 permute
        elif k.endswith("q_proj.bias"):
            v = permute_for_llama2c(v.unsqueeze(-1), n_heads).squeeze(-1)
        elif k.endswith("k_proj.bias"):
            v = permute_for_llama2c(v.unsqueeze(-1), n_kv_heads).squeeze(-1)
        # v_proj.bias 不需要 permute，直接保留

        sd[k] = v

    # 词嵌入绑定逻辑（小模型 ≤3B）
    if tie_word_embeddings and "model.embed_tokens.weight" in sd:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"]

    return sd


def convert_custom_to_hf(
    custom_state_dict: Dict[str, Any],
    n_heads: int = 16,
    n_kv_heads: int = 2,
) -> Dict[str, Any]:
    """
    将自定义格式的 state_dict 转换回 HuggingFace 官方 Qwen2/2.5 格式
    用于导出模型给开源社区或推理引擎使用
    """
    hf_sd = {}
    for k, v in custom_state_dict.items():
        # 跳过 buffer（freqs_cos, freqs_sin, mask）
        if "freqs_cos" in k or "freqs_sin" in k or "mask" in k:
            continue

        # Q/K 权重逆 permute：llama2.c 相邻成对 → HF split-half
        if k.endswith("q_proj.weight"):
            v = permute_llama2c_to_hf(v, n_heads)
        elif k.endswith("k_proj.weight"):
            v = permute_llama2c_to_hf(v, n_kv_heads)

        # QKV bias 逆 permute
        elif k.endswith("q_proj.bias"):
            v = permute_llama2c_to_hf(v.unsqueeze(-1), n_heads).squeeze(-1)
        elif k.endswith("k_proj.bias"):
            v = permute_llama2c_to_hf(v.unsqueeze(-1), n_kv_heads).squeeze(-1)

        hf_sd[k] = v
    return hf_sd


if __name__ == "__main__":
    from model.qwen.model import QwenForCausalLM
    from model.qwen.config import QwenConfig
    from config import SOURCE_DIR

    # 测试权重转换
    config = QwenConfig.qwen2_5_0_5b()

    # 自定义模型架构
    model = QwenForCausalLM(config)
    print(model)

    # HF 官方模型架构（需要安装 transformers 并下载 Qwen2.5-3B）
    # from transformers import Qwen2ForCausalLM
    # hf_model = Qwen2ForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/qwen/Qwen2.5-3B")
    # print(hf_model)
