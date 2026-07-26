#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/22
@Author  : weiyutao
@File    : eval.py

GPT-2 模型评估脚本
统一支持自定义模型与Transformers模型，支持不同权重来源的 LM Evaluation Harness 评测
"""

import lm_eval
import torch
from enum import Enum
from tokenizers import Tokenizer
from safetensors.torch import load_file

from model.gpt2.model import GPTConfig, GPTForCausalLM
from transformers import GPT2LMHeadModel, GPT2Config
from training import GPT2LM
from config import SOURCE_DIR, OUT_DIR
from model.gpt2.weight_convert import convert_custom_to_hf

from utils.enums import ModelType, LoadMode


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模式选择
    load_pretrained_mode = LoadMode.CONTINUAL
    model_type = ModelType.TRANSFORMERS
    
    # 随机初始化的模型困惑度极高，在 bootstrap 迭代中可能导致 float64 溢出
    bootstrap_iters = 0 
    
    # 1. 基础配置与 Tokenizer 初始化
    gpt_cfg = GPTConfig.gpt2_small()
    tokenizer = Tokenizer.from_pretrained("gpt2") 
    
    # 2. 实例化模型
    if model_type == ModelType.TRANSFORMERS:
        hf_cfg = GPT2Config(**gpt_cfg.to_transformers_dict())
        model = GPT2LMHeadModel(hf_cfg)
    else:
        model = GPTForCausalLM(gpt_cfg)
        
    model.to(device)
    
    # 3. 加载权重 (与训练脚本保持一致)
    if load_pretrained_mode == LoadMode.PRETRAINED:
        bootstrap_iters = 100
        tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/gpt2/124M/tokenizer.json")
        if model_type == ModelType.CUSTOM:
            model = GPTForCausalLM.from_pretrained(
                f"{SOURCE_DIR}/hf/gpt2/124M", 
                source="hf",
                map_location=device
            )
            # 使用本地对应的 tokenizer.json
        elif model_type == ModelType.TRANSFORMERS:
            model = GPT2LMHeadModel.from_pretrained(f"{SOURCE_DIR}/hf/gpt2/124M").to(device)
            
    elif load_pretrained_mode == LoadMode.CONTINUAL:
        bootstrap_iters = 100
        tokenizer = Tokenizer.from_file(f"{OUT_DIR}/train_simple_20260722/tokenizer.json")
        if model_type == ModelType.CUSTOM:
            # 注意: 这里你原本用的是 source="pt"
            model = GPTForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260722", source="local", map_location=device)  
        elif model_type == ModelType.TRANSFORMERS:
            try:
                custom_sd = torch.load(f"{OUT_DIR}/train_simple_20260722/pytorch_model.bin", weights_only=True) 
            except Exception:
                custom_sd = load_file(f"{OUT_DIR}/train_simple_20260722/model.safetensors")
            if "model.embed_tokens.weight" in custom_sd:
                # 自定义格式 -> HF 格式
                hf_sd = convert_custom_to_hf(custom_sd, num_layers=gpt_cfg.n_layer)
                model.load_state_dict(hf_sd)
            else:
                # 已经是 HF 格式，直接加载
                model.load_state_dict(custom_sd)

    # 4. 初始化评估 Wrapper 
    # (确保统一传入 tokenizer，保持自定义评估和 Transformers 评估的一致性)
    lm = GPT2LM(model=model, device=device, context_length=128, tokenizer=tokenizer)

    # 5. 执行评估
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["wikitext"],
        # tasks=["lambada_openai", "wikitext"],
        batch_size=2,
        bootstrap_iters=bootstrap_iters
    )
    
    print(results["results"])