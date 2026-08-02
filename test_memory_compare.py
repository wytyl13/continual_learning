#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比自定义模型和Transformers模型的显存占用"""

import torch
from transformers import LlamaForCausalLM
from model.llama.model import LLamaForCausalLM
from config import SOURCE_DIR
from utils.enums import ModelName

model_name = ModelName.LLAMA3_8B.value

print("="*70)
print("测试1: 自定义模型 (LLamaForCausalLM)")
print("="*70)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device=0)

custom_model = LLamaForCausalLM.from_pretrained(
    f"{SOURCE_DIR}/hf/llama/{model_name}",
    source="hf",
    device_map="cuda:0"
)

allocated_custom = torch.cuda.memory_allocated(0) / 1024**3
reserved_custom = torch.cuda.memory_reserved(0) / 1024**3
max_allocated_custom = torch.cuda.max_memory_allocated(0) / 1024**3

print(f"已分配: {allocated_custom:.2f}GB")
print(f"已保留: {reserved_custom:.2f}GB")
print(f"峰值分配: {max_allocated_custom:.2f}GB")

# 统计参数量
total_params = sum(p.numel() for p in custom_model.parameters())
print(f"总参数量: {total_params:,}")
print(f"理论大小(bf16): {total_params * 2 / 1024**3:.2f}GB")

del custom_model
torch.cuda.empty_cache()

print("\n" + "="*70)
print("测试2: Transformers模型 (LlamaForCausalLM)")
print("="*70)
torch.cuda.reset_peak_memory_stats(device=0)

hf_model = LlamaForCausalLM.from_pretrained(
    f"{SOURCE_DIR}/hf/llama/{model_name}",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0"
)

allocated_hf = torch.cuda.memory_allocated(0) / 1024**3
reserved_hf = torch.cuda.memory_reserved(0) / 1024**3
max_allocated_hf = torch.cuda.max_memory_allocated(0) / 1024**3

print(f"已分配: {allocated_hf:.2f}GB")
print(f"已保留: {reserved_hf:.2f}GB")
print(f"峰值分配: {max_allocated_hf:.2f}GB")

# 统计参数量
total_params_hf = sum(p.numel() for p in hf_model.parameters())
print(f"总参数量: {total_params_hf:,}")
print(f"理论大小(bf16): {total_params_hf * 2 / 1024**3:.2f}GB")

print("\n" + "="*70)
print("差异分析")
print("="*70)
print(f"显存差异: {allocated_custom - allocated_hf:.2f}GB")
print(f"参数量是否相同: {total_params == total_params_hf}")
