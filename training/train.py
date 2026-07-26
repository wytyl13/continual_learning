#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/20
@Author  : weiyutao
@File    : gpt2_train.py

GPT-2 自定义模型训练脚本 和 transformers类加载模型训练脚本
支持从头训练（transformers库加载、自定义模型加载）
支持加载官方预训练权重（transformers库加载、自定义模型加载）
支持加载持续训练权重（transformer类预训练和自定义模型预训练）继续训练
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import time
from tokenizers import Tokenizer
from safetensors.torch import load_file

from model.gpt2.model import GPTConfig, GPTForCausalLM

from transformers import GPT2LMHeadModel, GPT2Config
from training import create_dataloader
from training import train_model
from config import SOURCE_DIR, OUT_DIR
from model.gpt2.weight_convert import convert_custom_to_hf

from utils.logger import get_logger
from utils.enums import ModelType, LoadMode

logger = get_logger(__name__)

if __name__ == "__main__":
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用枚举进行模式选择
    load_pretrained_mode = LoadMode.CONTINUAL 
    model_type = ModelType.TRANSFORMERS
    
    gpt_cfg = None
    tokenizer = Tokenizer.from_pretrained("gpt2") # 只下载hf格式的tokenizer.json文件
    context_length = 256
    gpt_cfg: GPTConfig = GPTConfig.gpt2_small() 
    if model_type == ModelType.TRANSFORMERS:
        gpt_cfg = GPT2Config(**gpt_cfg.to_transformers_dict())
    model = GPTForCausalLM(gpt_cfg) if model_type == ModelType.CUSTOM else GPT2LMHeadModel(gpt_cfg)
    model.to(device)
    # 加载模型权重
    if load_pretrained_mode == LoadMode.PRETRAINED:
        # model = GPTForCausalLM.from_pretrained(
        #     f"{SOURCE_DIR}/trf/gpt2/124M", 
        #     source="openai",
        #     map_location=device
        # )
        
        if model_type == ModelType.CUSTOM:
            # tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/gpt2/124M/tokenizer.json") # 效果等同于Tokenizer.from_pretrained("gpt2")
            model = GPTForCausalLM.from_pretrained(
                f"{SOURCE_DIR}/hf/gpt2/124M", 
                source="hf",
                map_location=device
            )
        elif model_type == ModelType.TRANSFORMERS:
            model = GPT2LMHeadModel.from_pretrained(f"{SOURCE_DIR}/hf/gpt2/124M").to(device)
    elif load_pretrained_mode == LoadMode.CONTINUAL:
        if model_type == ModelType.CUSTOM:
            # tokenizer = Tokenizer.from_file(f"{OUT_DIR}/train_simple_20260722/tokenizer.json") # 如果没有新增或删除，效果等同于Tokenizer.from_pretrained("gpt2")
            model = GPTForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260722", source="local", map_location=device)    
        elif model_type == ModelType.TRANSFORMERS:
            try:
                custom_sd = torch.load(f"{OUT_DIR}/train_simple_20260722/pytorch_model.bin", weights_only=True) 
            except Exception as e:
                custom_sd = load_file(f"{OUT_DIR}/train_simple_20260722/model.safetensors")
            if "model.embed_tokens.weight" in custom_sd:
                # 自定义格式 -> HF 格式
                hf_sd = convert_custom_to_hf(custom_sd, num_layers=gpt_cfg.n_layer)
                model.load_state_dict(hf_sd)
            else:
                # 已经是 HF 格式，直接加载
                model.load_state_dict(custom_sd)
    # 初始化训练数据
    with open(f"{SOURCE_DIR}/the-verdict.txt") as f:
        text = f.read()
    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]
    
    
    # 创建dataloader训练集和测试集
    train_loader = create_dataloader(
        train_text, 
        tokenizer, 
        max_length=context_length,
        stride=context_length // 2,
        batch_size=2
    )
    
    val_loader = create_dataloader(
        val_text,
        tokenizer,
        max_length=context_length,
        stride=context_length // 2,
        batch_size=2
    )
    
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    
    
    # 执行训练
    train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=10,
        eval_freq=50,
        eval_iter=5,
        start_context="Every effort moves you",
        context_size=context_length,
        tokenizer=tokenizer,
        generate_sample_temperature=0.8,
        generate_sample_top_k=50,
        patience=3,
        save_dir=f"{OUT_DIR}/train_simple_20260722",
        model_type=model_type
    )



