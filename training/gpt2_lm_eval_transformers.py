#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/22
@Author  : weiyutao
@File    : gpt2_lm_eval_transformers.py
"""

import lm_eval
import torch
from transformers import GPT2LMHeadModel, GPT2Config

from training import GPT2LM
from model.gpt2.weight_convert import convert_custom_to_hf
from config import SOURCE_DIR, OUT_DIR

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_pretrained_mode = 1  # 1: scratch, 2: load_pretrained, 3: load_continual_learning
    # 随机初始化的模型产生近乎随机的 token 概率，困惑度极高（约等于词表大小 ~50257）如果随机初始化，
    # 在100次 bootstrap 迭代的方差计算中可能导致 float64 溢出
    bootstrap_iters = 0 
    
    
    # 初始化模型
    cfg = GPT2Config(vocab_size=50257, n_positions=1024, n_embd=768, n_layer=12, n_head=12)
    model = GPT2LMHeadModel(cfg)
    
    # 加载权重
    if load_pretrained_mode == 3:
        # 加载持续学习权重（需要转换格式）
        custom_sd = torch.load(f"{OUT_DIR}/train_simple_20260722/pytorch_model.bin", weights_only=True) 
        hf_sd = convert_custom_to_hf(custom_sd, num_layers=12)
        model.load_state_dict(hf_sd)
        bootstrap_iters=100
    elif load_pretrained_mode == 2:
        # 加载官方预训练权重
        model = GPT2LMHeadModel.from_pretrained(f"{SOURCE_DIR}/hf/gpt2/124M")
        bootstrap_iters=100

    # 评估（使用 tiktoken，与自定义评估保持一致）
    lm = GPT2LM(model=model, device=device, context_length=128)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["lambada_openai", "wikitext"],
        batch_size=2,
        bootstrap_iters=bootstrap_iters
    )

    print(results["results"])
