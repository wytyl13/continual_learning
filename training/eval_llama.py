#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/22
@Author  : weiyutao
@File    : eval_llama.py

GPT-2 模型评估脚本
统一支持自定义模型与Transformers模型，支持不同权重来源的 LM Evaluation Harness 评测
"""

import lm_eval
import torch

from enum import Enum
from tokenizers import Tokenizer
from safetensors.torch import load_file
from accelerate import init_empty_weights
            
from model.llama.model import LLamaConfig, LLamaForCausalLM
from transformers import LlamaForCausalLM, LlamaConfig
from training import GPT2LM
from config import SOURCE_DIR, OUT_DIR
from model.llama.weight_convert import convert_custom_to_hf

from utils.model_loader import (
    build_auto_device_map, 
    materialize_meta_model, 
    fast_load_weights
)

from utils.enums import (
    ModelType, 
    LoadMode, 
    ModelName
)

from utils.logger import get_logger, log_stage
from utils.model_loader import resolve_checkpoint_files
from model.llama.config import MODEL_CONFIGS

logger = get_logger(__name__)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 模式选择
    load_pretrained_mode = LoadMode.PRETRAINED
    model_type = ModelType.TRANSFORMERS
    model_name = ModelName.LLAMA3_8B.value
    
    # 随机初始化的模型困惑度极高，在 bootstrap 迭代中可能导致 float64 溢出
    bootstrap_iters = 0 
    tokenizer = Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/{model_name}/tokenizer.json")
    
    llama_cfg: LLamaConfig = MODEL_CONFIGS[model_name] 
    if model_type == ModelType.TRANSFORMERS:
        llama_cfg = LlamaConfig(**llama_cfg.to_transformers_dict())
    
    # 3. 加载权重 (与训练脚本保持一致)
    if load_pretrained_mode == LoadMode.PRETRAINED:
        # 本地权重加载，肯定经过cpu传输到gpu
        bootstrap_iters = 100
        tokenizer = Tokenizer.from_file(
            f"{SOURCE_DIR}/hf/llama/{model_name}/tokenizer.json"
        )
        if model_type == ModelType.CUSTOM:
            # 直接调用from_pre_pretrianed，自动计算device_map
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                model = LLamaForCausalLM.from_pretrained(
                    f"{SOURCE_DIR}/hf/llama/{model_name}", 
                    source="hf",
                    device_map="auto"
                )
        elif model_type == ModelType.TRANSFORMERS:
            # 先自行计算device_map，from_pretrained 传入 device_map 时，transformers 内部会自动调用
            # 也可以直接传递device_map="auto"
            # 无参初始化 -> 设计 device_map -> 直接传递device_map到transformers库初始化
            with init_empty_weights():
                _empty = LlamaForCausalLM(llama_cfg)
            device_map = build_auto_device_map(_empty)
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                model = LlamaForCausalLM.from_pretrained(
                    f"{SOURCE_DIR}/hf/llama/{model_name}",
                    torch_dtype=torch.bfloat16,
                    device_map=device_map
                )
            # from_pretrained 的 device_map 已安装 hooks，输入设备固定为 cuda:0
            device = next(model.parameters()).device
    elif load_pretrained_mode == LoadMode.CONTINUAL:
        bootstrap_iters = 100
        tokenizer = Tokenizer.from_file(
            f"{OUT_DIR}/train_simple_20260730_{model_name}/tokenizer.json"
        )
        if model_type == ModelType.CUSTOM:
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                model = LLamaForCausalLM.from_pretrained(
                    f"{OUT_DIR}/train_simple_20260730_{model_name}",
                    source="local",
                    device_map="auto"
                )
        elif model_type == ModelType.TRANSFORMERS:
            # 无参初始化 -> 设计device_map -> 加载本地权重 -> 逐shard解析并传输到指定map
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                # 1. meta 模型计算均衡 device_map（不占真实显存）
                with init_empty_weights():
                    model = LlamaForCausalLM(llama_cfg)
                device_map = build_auto_device_map(model, no_split_classes=["LlamaDecoderLayer"])
                # 2. 解析权重路径
                shard_files, use_safetensors = resolve_checkpoint_files(
                    f"{OUT_DIR}/train_simple_20260730_{model_name}",
                )
                # 3. convert_fn：自定义格式 → HF格式
                def convert_fn(shard):
                    return convert_custom_to_hf(
                        shard,
                        n_heads=llama_cfg.num_attention_heads,
                        n_kv_heads=llama_cfg.num_key_value_heads,
                    )
                # 4. 逐shard加载，直接落到目标GPU，CPU峰值 = 单shard大小
                fast_load_weights(model, shard_files, use_safetensors, device_map, convert_fn)
                device = next(model.parameters()).device
    else:
        # 仅测试架构（不加载权重），与 PRETRAINED/CONTINUAL 同步走 device_map="auto"
        if model_type == ModelType.CUSTOM:
            # 直接初始化到指定的gpu设备，不需要经过cpu
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                model = LLamaForCausalLM.from_empty(llama_cfg, device_map="auto")
        else:
            # 直接初始化到指定的gpu设备，不需要经过cpu
            with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}, {model_type}"):
                with init_empty_weights():
                    torch.set_default_dtype(torch.bfloat16)
                    model = LlamaForCausalLM(llama_cfg)
                    torch.set_default_dtype(torch.float32)
                device_map = build_auto_device_map(model, no_split_classes=["LlamaDecoderLayer"])
                materialize_meta_model(model, device_map)
                device = next(model.parameters()).device

    # 4. 初始化评估 Wrapper 
    # (确保统一传入 tokenizer，保持自定义评估和 Transformers 评估的一致性)
    lm = GPT2LM(
        model=model, 
        device=device, 
        context_length=llama_cfg.max_position_embeddings, 
        tokenizer=tokenizer,
        max_gen_toks=1024,
        vocab_size=llama_cfg.vocab_size,
        eot_token_id=llama_cfg.eos_token_id
    )

    # 5. 执行评估
    results = lm_eval.simple_evaluate(
        model=lm,
        # tasks=["wikitext"],
        tasks=["lambada_openai", "wikitext"],
        batch_size=32,
        bootstrap_iters=bootstrap_iters
    )
    
    print(results["results"])