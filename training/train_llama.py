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
from accelerate.utils import DeepSpeedPlugin
from accelerate import Accelerator

from model.llama.model import LLamaModel, LLamaConfig, LLamaForCausalLM
from training import create_dataloader
from training import train_model
from config import SOURCE_DIR, OUT_DIR

from utils.logger import get_logger
from utils.enums import ModelType, LoadMode

logger = get_logger(__name__)

if __name__ == "__main__":
    # cpu初始化模型权重参数，由accelerate prepare自动分配显存
    batch_size = 1
    lr=5e-5
    weight_decay=0.1
    from model.llama.model import LLamaForCausalLM
    from transformers import LlamaForCausalLM, LlamaConfig
    from model.llama.weight_convert import convert_custom_to_hf
    # 初始化模型
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用枚举进行模式选择
    load_pretrained_mode = LoadMode.SCRATCH 
    model_type = ModelType.TRANSFORMERS
    tokenizer = Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    # llama_cfg: LLamaConfig = LLamaConfig.tiny_llama() 
    llama_cfg: LLamaConfig = LLamaConfig.llama2_7b() 
    context_length = 256
    torch.manual_seed(123)
    if model_type == ModelType.TRANSFORMERS:
        llama_cfg = LlamaConfig(**llama_cfg.to_transformers_dict())
    
    # 加载模型权重
    if load_pretrained_mode == LoadMode.PRETRAINED:
        if model_type == ModelType.CUSTOM:
            # tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/tiny_llama/tokenizer.json") # 效果等同于Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
            # model = LLamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/tiny_llama", source="hf", map_location="cpu")
            model = LLamaForCausalLM.from_pretrained(
                f"{SOURCE_DIR}/hf/llama/llama-2-7b",
                source="hf",
                map_location="cpu"
            )
        elif model_type == ModelType.TRANSFORMERS:
            # model = LlamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/tiny_llama", torch_dtype=torch.bfloat16)
            model = LlamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/llama-2-7b", torch_dtype=torch.bfloat16)
    elif load_pretrained_mode == LoadMode.CONTINUAL:
        if model_type == ModelType.CUSTOM:
            # tokenizer = Tokenizer.from_file(f"{OUT_DIR}/train_simple_20260730_tiny_llama/tokenizer.json") # 如果没有新增或删除，效果等同于Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
            # model = LLamaForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260730_tiny_llama", source="local", map_location="cpu")    
            model = LLamaForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260730_llama2_7b", source="local", map_location="cpu")    
        elif model_type == ModelType.TRANSFORMERS:
            with torch.device("cpu"):
                old_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                model = LlamaForCausalLM(llama_cfg)
                torch.set_default_dtype(old_dtype)
            try:
                # custom_sd = torch.load(f"{OUT_DIR}/train_simple_20260730_tiny_llama/pytorch_model.bin", weights_only=True, map_location="cpu") 
                custom_sd = torch.load(f"{OUT_DIR}/train_simple_20260730_llama2_7b/pytorch_model.bin", weights_only=True, map_location="cpu") 
            except Exception as e:
                # custom_sd = load_file(f"{OUT_DIR}/train_simple_20260730_tiny_llama/model.safetensors", device="cpu")
                custom_sd = load_file(f"{OUT_DIR}/train_simple_20260730_llama2_7b/model.safetensors", device="cpu")
            hf_sd = convert_custom_to_hf(custom_sd)
            model.load_state_dict(hf_sd)
    else:
        # 随机初始化，cpu 上建好再由 Accelerate prepare 分配（训练路径不能用 device_map="auto"）
        if model_type == ModelType.CUSTOM:
            model = LLamaForCausalLM.from_empty(llama_cfg, map_location="cpu")
        else:
            with torch.device("cpu"):
                old_dtype = torch.get_default_dtype()
                torch.set_default_dtype(torch.bfloat16)
                model = LlamaForCausalLM(llama_cfg)
                torch.set_default_dtype(old_dtype)
    
    
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
        batch_size=batch_size,
        shuffle=False,    # 验证集不需要打乱
        drop_last=False   # 样本少时不丢弃，防止分布式后 loader 为空
    )
    
    # 自动读取 --config_file 指定的 YAML
    accelerator = Accelerator()
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 可以在这里设置梯度检查点，梯度检查点节省的显存大小和batch, 
    # max_position_embeddings有直接关系。
    # Accelerate 接管设备管理
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    device = accelerator.device
    
    # 执行训练
    train_losses, val_losses, tokens_seen = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=2,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        context_size=context_length,
        tokenizer=tokenizer,
        generate_sample_temperature=0.8,
        generate_sample_top_k=50,
        patience=1,
        # save_dir=f"{OUT_DIR}/train_simple_20260730_tiny_llama",
        save_dir=f"{OUT_DIR}/train_simple_20260730_llama2_7b",
        model_type=model_type,
        accelerator=accelerator
    )



