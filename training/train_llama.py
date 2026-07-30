#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/20
@Author  : weiyutao
@File    : train_llama.py
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import time
from tokenizers import Tokenizer
from safetensors.torch import load_file


from model.llama.model import LLamaModel, LLamaConfig, LLamaForCausalLM
from training import create_dataloader
from training import train_model
from config import SOURCE_DIR, OUT_DIR

from utils.logger import get_logger
from utils.enums import ModelType, LoadMode

logger = get_logger(__name__)

if __name__ == "__main__":
    batch_size = 2
    lr=5e-5
    weight_decay=0.1
    # ==========================================
    # DeepSpeed 优化器卸载配置
    # ==========================================
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size, # 必须与你 dataloader 的 batch_size 一致
        "gradient_accumulation_steps": 1,    # 如果需要模拟更大 batch，可调大此值
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "weight_decay": weight_decay
            }
        },
        "bf16": {
            "enabled": True  # 强制半精度训练
        },
        "zero_optimization": {
            "stage": 2,      # 开启 ZeRO-2
            "offload_optimizer": {
                "device": "cpu",   # 将优化器状态（约 8.8GB）卸载到 CPU 主存
                "pin_memory": True # 锁页内存加速拷贝
            },
            "overlap_comm": True,
            "contiguous_gradients": True
        }
    }
    
    
    import deepspeed
    from model.llama.model import LLamaForCausalLM
    from transformers import LlamaForCausalLM, LlamaConfig
    from model.llama.weight_convert import convert_custom_to_hf
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用枚举进行模式选择
    load_pretrained_mode = LoadMode.CONTINUAL 
    model_type = ModelType.CUSTOM
    DEEPSPEED_FLAG = 0
    tokenizer = Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    context_length = 256
    # llama_cfg: LLamaConfig = LLamaConfig.tiny_llama() 
    llama_cfg: LLamaConfig = LLamaConfig.llama2_7b() 
    if model_type == ModelType.TRANSFORMERS:
        llama_cfg = LlamaConfig(**llama_cfg.to_transformers_dict())
    with torch.device(device):
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        if model_type == ModelType.CUSTOM:
            model = LLamaForCausalLM(llama_cfg)
        else:
            model = LlamaForCausalLM(llama_cfg)
        torch.set_default_dtype(old_dtype)
    
    
    # 加载模型权重
    if load_pretrained_mode == LoadMode.PRETRAINED:
        if model_type == ModelType.CUSTOM:
            # tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/tiny_llama/tokenizer.json") # 效果等同于Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
            model = LLamaForCausalLM.from_pretrained(
                # f"{SOURCE_DIR}/hf/llama/tiny_llama", 
                f"{SOURCE_DIR}/hf/llama/llama-2-7b", 
                source="hf",
                map_location=device
            )
        elif model_type == ModelType.TRANSFORMERS:
            # model = LlamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/tiny_llama", torch_dtype=torch.bfloat16, device_map=device)
            model = LlamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/llama-2-7b", torch_dtype=torch.bfloat16, device_map=device)
    elif load_pretrained_mode == LoadMode.CONTINUAL:
        if model_type == ModelType.CUSTOM:
            # tokenizer = Tokenizer.from_file(f"{OUT_DIR}/train_simple_20260730_tiny_llama/tokenizer.json") # 如果没有新增或删除，效果等同于Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
            # model = LLamaForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260730_tiny_llama", source="local", map_location=device)    
            model = LLamaForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260730_llama2_7b", source="local", map_location=device)    
        elif model_type == ModelType.TRANSFORMERS:
            try:
                # custom_sd = torch.load(f"{OUT_DIR}/train_simple_20260730_tiny_llama/pytorch_model.bin", weights_only=True, map_location=device) 
                custom_sd = torch.load(f"{OUT_DIR}/train_simple_20260730_llama2_7b/pytorch_model.bin", weights_only=True, map_location=device) 
            except Exception as e:
                # custom_sd = load_file(f"{OUT_DIR}/train_simple_20260730_tiny_llama/model.safetensors", device=str(device))
                custom_sd = load_file(f"{OUT_DIR}/train_simple_20260730_llama2_7b/model.safetensors", device=str(device))
            hf_sd = convert_custom_to_hf(custom_sd)
            model.load_state_dict(hf_sd)
    
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
        batch_size=batch_size
    )
    
    val_loader = create_dataloader(
        val_text,
        tokenizer,
        max_length=context_length,
        stride=context_length // 2,
        batch_size=batch_size
    )
    
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if DEEPSPEED_FLAG:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
    
    # 执行训练
    train_losses, val_losses, tokens_seen = train_model(
        model=model_engine if DEEPSPEED_FLAG else model, # model
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
        # save_dir=f"{OUT_DIR}/train_simple_20260730_tiny_llama",
        save_dir=f"{OUT_DIR}/train_simple_20260730_llama2_7b",
        model_type=model_type
    )



