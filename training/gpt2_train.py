#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/20
@Author  : weiyutao
@File    : gpt2_train.py

GPT-2 模型训练脚本
支持从头训练和加载预训练权重继续训练
"""

import sys
import os
import torch
import tiktoken
import time
import tiktoken

from model.gpt2.model import GPTModel, GPTConfig, GPTForCausalLM
from training import create_dataloader
from training import train_model
from training import get_logger


logger = get_logger(__name__)

if __name__ == "__main__":
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_pretrained_mode = 3 # 1: scratch, 2: load_pretrained, 3: load_continual_learning
    gpt_cfg = None
    context_length = 256
    gpt_cfg: GPTConfig = GPTConfig.gpt2_small()
    model = GPTForCausalLM(gpt_cfg)
    model.to(device)
    
    # 加载模型权重
    if load_pretrained_mode == 2:
        # model = GPTForCausalLM.from_pretrained(
        #     "/mnt/wsl/fast_disk/continual_learning/source/trf/gpt2/124M", 
        #     source="openai",
        #     map_location=device
        # )
        
        model = GPTForCausalLM.from_pretrained(
            "/mnt/wsl/fast_disk/continual_learning/source/hf/gpt2/124M", 
            source="hf",
            map_location=device
        )
    elif load_pretrained_mode == 3:
        model = GPTForCausalLM.from_pretrained("/mnt/wsl/fast_disk/continual_learning/out/train_simple_20260722", source="pt", map_location=device)    


    # 初始化训练数据
    with open("/mnt/wsl/fast_disk/continual_learning/source/The_Verdict.txt") as f:
        text = f.read()
    split = int(len(text) * 0.9)
    train_text, val_text = text[:split], text[split:]
    tokenizer = tiktoken.get_encoding("gpt2")
    
    
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
        save_dir="/mnt/wsl/fast_disk/continual_learning/out/train_simple_20260722"
    )



