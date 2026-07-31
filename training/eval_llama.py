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
from utils.model_loader import resolve_checkpoint_files

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 模式选择
    load_pretrained_mode = LoadMode.SCRATCH
    model_type = ModelType.TRANSFORMERS
    
    # 随机初始化的模型困惑度极高，在 bootstrap 迭代中可能导致 float64 溢出
    bootstrap_iters = 0 
    tokenizer = Tokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    
    # llama_cfg: LLamaConfig = LLamaConfig.tiny_llama() 
    llama_cfg: LLamaConfig = LLamaConfig.llama2_7b() 
    # llama_cfg: LLamaConfig = LLamaConfig.llama3_8b() 
    if model_type == ModelType.TRANSFORMERS:
        llama_cfg = LlamaConfig(**llama_cfg.to_transformers_dict())
    
    # 3. 加载权重 (与训练脚本保持一致)
    if load_pretrained_mode == LoadMode.PRETRAINED:
        bootstrap_iters = 100
        tokenizer = Tokenizer.from_file(
            # f"{SOURCE_DIR}/hf/llama/tiny_llama/tokenizer.json"
            f"{SOURCE_DIR}/hf/llama/llama-2-7b/tokenizer.json"
            # f"{SOURCE_DIR}/hf/llama/llama-3-8b/tokenizer.json"
        )
        if model_type == ModelType.CUSTOM:
            model = LLamaForCausalLM.from_pretrained(
                # f"{SOURCE_DIR}/hf/llama/tiny_llama", 
                f"{SOURCE_DIR}/hf/llama/llama-2-7b", 
                # f"{SOURCE_DIR}/hf/llama/llama-3-8b", 
                source="hf",
                device_map="auto"
            )
        elif model_type == ModelType.TRANSFORMERS:
            # from_pretrained 传入 device_map 时，transformers 内部会自动调用
            # dispatch_model，AlignDevicesHook 已安装，无需手动处理。
            from utils.model_loader import build_auto_device_map
            from accelerate import init_empty_weights
            with init_empty_weights():
                _empty = LlamaForCausalLM(llama_cfg)
            device_map = build_auto_device_map(_empty, no_split_classes=["LlamaDecoderLayer"])
            model = LlamaForCausalLM.from_pretrained(
                # f"{SOURCE_DIR}/hf/llama/tiny_llama",
                f"{SOURCE_DIR}/hf/llama/llama-2-7b",
                # f"{SOURCE_DIR}/hf/llama/llama-3-8b",
                torch_dtype=torch.bfloat16,
                device_map=device_map
            )
            # from_pretrained 的 device_map 已安装 hooks，输入设备固定为 cuda:0
            device = next(model.parameters()).device
    elif load_pretrained_mode == LoadMode.CONTINUAL:
        bootstrap_iters = 100
        tokenizer = Tokenizer.from_file(
            # f"{OUT_DIR}/train_simple_20260730_tiny_llama/tokenizer.json"
            f"{OUT_DIR}/train_simple_20260730_llama2_7b/tokenizer.json"
            # f"{OUT_DIR}/train_simple_20260730_llama3_8b/tokenizer.json"
        )
        if model_type == ModelType.CUSTOM:
            model = LLamaForCausalLM.from_pretrained(
                # f"{OUT_DIR}/train_simple_20260730_tiny_llama",
                f"{OUT_DIR}/train_simple_20260730_llama2_7b",
                # f"{OUT_DIR}/train_simple_20260730_llama3_8b",
                source="local",
                device_map="auto"
            )
        elif model_type == ModelType.TRANSFORMERS:
            from accelerate import init_empty_weights, dispatch_model
            from utils.model_loader import build_auto_device_map, fast_load_weights
            # 1. meta 模型计算均衡 device_map（不占真实显存）
            with init_empty_weights():
                model = LlamaForCausalLM(llama_cfg)
            device_map = build_auto_device_map(model, no_split_classes=["LlamaDecoderLayer"])
            # 2. 解析权重路径
            shard_files, use_safetensors = resolve_checkpoint_files(
                # f"{OUT_DIR}/train_simple_20260730_tiny_llama",
                f"{OUT_DIR}/train_simple_20260730_llama2_7b",
                # f"{OUT_DIR}/train_simple_20260730_llama3_8b",
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
            # 5. fast_load_weights 内部已调用 dispatch_model，hooks 已安装
            # 多卡模型的输入设备固定为 cuda:0，hooks 会自动处理跨卡搬运
            device = next(model.parameters()).device
    else:
        # 仅测试架构（不加载权重），与 PRETRAINED/CONTINUAL 同步走 device_map="auto"
        if model_type == ModelType.CUSTOM:
            model = LLamaForCausalLM.from_empty(llama_cfg, device_map="auto")
        else:
            from accelerate import init_empty_weights, dispatch_model
            from utils.model_loader import build_auto_device_map
            with init_empty_weights():
                torch.set_default_dtype(torch.bfloat16)
                model = LlamaForCausalLM(llama_cfg)
                torch.set_default_dtype(torch.float32)
            device_map = build_auto_device_map(model, no_split_classes=["LlamaDecoderLayer"])
            # 1. meta tensor → 真实空 tensor（dispatch_model 不会自动实例化）
            for mod_name, mod in model.named_modules():
                tgt = device_map.get(mod_name, device_map.get("", "cpu"))
                for param_name, p in mod.named_parameters(recurse=False):
                    if p.is_meta:
                        mod._parameters[param_name] = torch.nn.Parameter(
                            torch.empty(p.shape, dtype=p.dtype, device=tgt),
                            requires_grad=p.requires_grad,
                        )
                for buf_name, b in mod.named_buffers(recurse=False):
                    if b.is_meta:
                        mod._buffers[buf_name] = torch.empty(b.shape, dtype=b.dtype, device=tgt)
            # 2. 装 AlignDevicesHook，多卡间自动搬运张量
            model = dispatch_model(model, device_map=device_map)
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