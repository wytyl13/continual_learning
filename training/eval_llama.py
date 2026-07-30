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

from model.llama.model import LLamaConfig, LLamaForCausalLM
from transformers import LlamaForCausalLM, LlamaConfig
from training import GPT2LM
from config import SOURCE_DIR, OUT_DIR
from model.llama.weight_convert import convert_custom_to_hf

from utils.enums import ModelType, LoadMode


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模式选择
    load_pretrained_mode = LoadMode.CONTINUAL
    model_type = ModelType.CUSTOM
    
    # 随机初始化的模型困惑度极高，在 bootstrap 迭代中可能导致 float64 溢出
    bootstrap_iters = 0 
    # tokenizer = Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
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
    
    # 3. 加载权重 (与训练脚本保持一致)
    if load_pretrained_mode == LoadMode.PRETRAINED:
        bootstrap_iters = 100
        # tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/tiny_llama/tokenizer.json")
        tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/llama-2-7b/tokenizer.json")
        if model_type == ModelType.CUSTOM:
            model = LLamaForCausalLM.from_pretrained(
                # f"{SOURCE_DIR}/hf/llama/tiny_llama", 
                f"{SOURCE_DIR}/hf/llama/llama-2-7b", 
                source="hf",
                map_location=device
            )
            # 使用本地对应的 tokenizer.json
        elif model_type == ModelType.TRANSFORMERS:
            # model = LlamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/tiny_llama").to(device)
            model = LlamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/llama-2-7b").to(device)
            
    elif load_pretrained_mode == LoadMode.CONTINUAL:
        bootstrap_iters = 100
        # tokenizer = Tokenizer.from_file(f"{OUT_DIR}/train_simple_20260730_tiny_llama/tokenizer.json")
        tokenizer = Tokenizer.from_file(f"{OUT_DIR}/train_simple_20260730_llama2_7b/tokenizer.json")
        if model_type == ModelType.CUSTOM:
            # model = LLamaForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260730_tiny_llama", source="local", map_location=device)  
            model = LLamaForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260730_llama2_7b", source="local", map_location=device)  
        elif model_type == ModelType.TRANSFORMERS:
            try:
                # custom_sd = torch.load(f"{OUT_DIR}/train_simple_20260730_tiny_llama/pytorch_model.bin", weights_only=True) 
                custom_sd = torch.load(f"{OUT_DIR}/train_simple_20260730_llama2_7b/pytorch_model.bin", weights_only=True) 
            except Exception:
                # custom_sd = load_file(f"{OUT_DIR}/train_simple_20260730_tiny_llama/model.safetensors")
                custom_sd = load_file(f"{OUT_DIR}/train_simple_20260730_llama2_7b/model.safetensors")
            hf_sd = convert_custom_to_hf(custom_sd, n_heads=llama_cfg.num_attention_heads, n_kv_heads=llama_cfg.num_key_value_heads)
            model.load_state_dict(hf_sd)

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