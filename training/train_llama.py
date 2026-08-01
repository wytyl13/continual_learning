#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/20
@Author  : weiyutao
@File    : train_llama.py
"""

import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
from tokenizers import Tokenizer
from safetensors.torch import load_file
from accelerate import Accelerator, init_empty_weights

from model.llama.model import (
    LLamaConfig, 
    LLamaForCausalLM
)
from training import create_dataloader
from training import train_model
from config import SOURCE_DIR, OUT_DIR

from model.llama.model import LLamaForCausalLM
from transformers import LlamaForCausalLM, LlamaConfig
from model.llama.weight_convert import convert_custom_to_hf
from model.llama.config import MODEL_CONFIGS

from utils.model_loader import resolve_checkpoint_files
from utils.logger import log_stage
from utils.enums import (
    ModelType, 
    LoadMode, 
    ModelName
)

from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")

if __name__ == "__main__":
    
    # 自动读取 --config_file 指定的 YAML
    # 在 get_logger 使用前初始化该实例
    accelerator = Accelerator()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    # cpu初始化模型权重参数，由accelerate prepare自动分配显存
    batch_size = 1
    lr=5e-5
    weight_decay=0.1
    
    # 加载模式
    load_pretrained_mode = LoadMode.SCRATCH 
    model_type = ModelType.CUSTOM
    model_name = ModelName.LLAMA2_7B.value
    
    # tokenizer = Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/{model_name}/tokenizer.json")
    llama_cfg: LLamaConfig = MODEL_CONFIGS[model_name] 
    context_length = 256
    torch.manual_seed(123)
    if model_type == ModelType.TRANSFORMERS:
        llama_cfg = LlamaConfig(**llama_cfg.to_transformers_dict())
    
    # 加载模型权重
    if load_pretrained_mode == LoadMode.PRETRAINED:
        if model_type == ModelType.CUSTOM:
            # tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/{model_name}/tokenizer.json") # 效果等同于Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                # 磁盘读取权重 → CPU内存 → GPU↑ 这一步无法绕过，文件必须先经过CPU内存
                model = LLamaForCausalLM.from_pretrained(
                    f"{SOURCE_DIR}/hf/llama/{model_name}",
                    source="hf",
                    map_location="cpu"
                )
        elif model_type == ModelType.TRANSFORMERS:
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                    model = LlamaForCausalLM.from_pretrained(
                    f"{SOURCE_DIR}/hf/llama/{model_name}", 
                    torch_dtype=torch.bfloat16
                )
    elif load_pretrained_mode == LoadMode.CONTINUAL:
        if model_type == ModelType.CUSTOM:
            # tokenizer = Tokenizer.from_file(f"{OUT_DIR}/train_simple_20260730_{model_name}/tokenizer.json") # 如果没有新增或删除，效果等同于Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                # 磁盘读取权重 → CPU内存 → GPU↑ 这一步无法绕过，文件必须先经过CPU内存
                model = LLamaForCausalLM.from_pretrained(
                    f"{OUT_DIR}/train_simple_20260730_{model_name}", 
                    source="local", 
                    map_location="cpu"
                )    
        elif model_type == ModelType.TRANSFORMERS:
            # 磁盘读取权重 → CPU内存 → GPU↑ 这一步无法绕过，文件必须先经过CPU内存
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                # 需要将权重全部加载到cpu中，所以较为缓慢。
                with init_empty_weights():
                    model = LlamaForCausalLM(llama_cfg)
                # resolve_checkpoint_files 统一处理 safetensors/bin、单文件/分片，
                # 逐 shard 转换后合并，避免硬编码文件名和格式 try/except
                shard_files, use_safetensors = resolve_checkpoint_files(
                    f"{OUT_DIR}/train_simple_20260730_{model_name}"
                )
                for path in shard_files:
                    raw = (load_file(path, device="cpu") if use_safetensors
                        else torch.load(path, map_location="cpu", weights_only=True))
                    partial_sd = convert_custom_to_hf(raw)
                    del raw
                    model.load_state_dict(partial_sd, strict=False, assign=True)
                    del partial_sd
                model.tie_weights()
    else:   
        # 随机初始化，两个选择：
        # 1、在cpu上创建再由accelerate传递到指定GPU上 CPU分配13GB →随机初始化 → PCIe搬到GPU↑这整个过程是纯浪费
        # 2、直接初始化到accelerate所在的GPU设备上 快
        if model_type == ModelType.CUSTOM:
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                model = LLamaForCausalLM.from_empty(llama_cfg, map_location=accelerator.device)
        else:
            # 所以可以直接初始化到accelerate指定的GPU设备
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                with torch.device(accelerator.device):
                    old_dtype = torch.get_default_dtype()
                    torch.set_default_dtype(torch.bfloat16)
                    model = LlamaForCausalLM(llama_cfg) # _init_weights() 直接在 GPU 上跑
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
        save_dir=f"{OUT_DIR}/train_simple_20260730_{model_name}",
        model_type=model_type,
        accelerator=accelerator
    )



