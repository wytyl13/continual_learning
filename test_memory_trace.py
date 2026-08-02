#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""详细追踪显存占用的变化"""

import torch
from model.llama.model import LLamaForCausalLM
from config import SOURCE_DIR
from utils.enums import ModelName
import gc

model_name = ModelName.LLAMA3_8B.value

def print_memory(stage):
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    print(f"[{stage}] 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")

print("="*70)
print("追踪自定义模型加载过程的显存变化")
print("="*70)

torch.cuda.empty_cache()
gc.collect()
print_memory("初始状态")

# 开始加载
print("\n开始加载模型...")
custom_model = LLamaForCausalLM.from_pretrained(
    f"{SOURCE_DIR}/hf/llama/{model_name}",
    source="hf",
    device_map="cuda:0"
)

print_memory("加载完成")

# 统计实际参数
print("\n统计参数信息:")
total_params = sum(p.numel() for p in custom_model.parameters())
print(f"总参数量: {total_params:,}")
print(f"理论大小(bf16): {total_params * 2 / 1024**3:.2f}GB")

# 检查是否有重复的参数（权重绑定）
param_ids = {}
for name, param in custom_model.named_parameters():
    param_id = id(param.data)
    if param_id in param_ids:
        print(f"发现共享权重: {name} 与 {param_ids[param_id]} 共享")
    else:
        param_ids[param_id] = name

unique_params = len(param_ids)
print(f"\n唯一参数对象数: {unique_params}")
print(f"参数名称总数: {len(list(custom_model.named_parameters()))}")

if unique_params != len(list(custom_model.named_parameters())):
    print(f"有 {len(list(custom_model.named_parameters())) - unique_params} 个参数被共享")

# 检查 buffers
total_buffers = 0
for name, buf in custom_model.named_buffers():
    if buf is not None:
        total_buffers += buf.numel() * buf.element_size()

print(f"\nBuffers 占用: {total_buffers / 1024**3:.2f}GB")

# 检查是否有 accelerate hooks
hooks_count = 0
for name, module in custom_model.named_modules():
    if hasattr(module, '_hf_hook'):
        hooks_count += 1

print(f"Accelerate hooks 数量: {hooks_count}")

print("\n" + "="*70)
print("尝试手动清理...")
gc.collect()
torch.cuda.empty_cache()
print_memory("清理后")
