#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 device_map 是否正确分配模型到不同GPU"""

import torch
from model.llama.model import LLamaForCausalLM
from config import SOURCE_DIR
from utils.enums import ModelName

model_name = ModelName.LLAMA3_8B.value

print(f"CUDA设备数: {torch.cuda.device_count()}")
print(f"当前可用GPU: {[f'cuda:{i}' for i in range(torch.cuda.device_count())]}")

# 测试加载两个模型到不同GPU
models = []
for gpu_id in range(torch.cuda.device_count()):
    print(f"\n{'='*60}")
    print(f"正在加载模型 {gpu_id+1} 到 cuda:{gpu_id}...")
    m = LLamaForCausalLM.from_pretrained(
        f"{SOURCE_DIR}/hf/llama/{model_name}",
        source="hf",
        device_map=f"cuda:{gpu_id}"
    )
    models.append(m)

    # 检查模型参数实际在哪个设备
    param_devices = {}
    for name, param in m.named_parameters():
        device_str = str(param.device)
        param_devices[device_str] = param_devices.get(device_str, 0) + 1

    print(f"模型 {gpu_id+1} 的参数分布:")
    for device, count in sorted(param_devices.items()):
        print(f"  {device}: {count} 个参数")

    # 检查第一个参数的设备
    first_param_device = next(m.parameters()).device
    print(f"  第一个参数设备: {first_param_device}")

# 显存使用情况
print(f"\n{'='*60}")
print("显存使用情况:")
for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / 1024**3
    reserved = torch.cuda.memory_reserved(i) / 1024**3
    print(f"  GPU {i}: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
