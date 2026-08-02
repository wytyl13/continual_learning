#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试权重加载过程"""

import torch
import json
import os
from accelerate import init_empty_weights
from safetensors.torch import load_file
from model.llama.model import LLamaForCausalLM
from model.llama.config import LLamaConfig
from model.llama.weight_convert import convert_safetensors_to_custom
from utils.model_loader import resolve_checkpoint_files, auto_device_map
from config import SOURCE_DIR
from utils.enums import ModelName

model_name = ModelName.LLAMA3_8B.value
model_path = f"{SOURCE_DIR}/hf/llama/{model_name}"

print("="*70)
print("步骤1: 加载配置")
print("="*70)
with open(os.path.join(model_path, "config.json")) as f:
    config = LLamaConfig.from_dict(json.load(f))
print(f"配置: {config.num_hidden_layers} 层, vocab_size={config.vocab_size}")

print("\n" + "="*70)
print("步骤2: 创建空模型")
print("="*70)
with init_empty_weights():
    model = LLamaForCausalLM(config)
print(f"模型参数名称数: {len(list(model.named_parameters()))}")

print("\n" + "="*70)
print("步骤3: 解析权重文件")
print("="*70)
shard_files, use_safetensors = resolve_checkpoint_files(model_path)
print(f"Shard 文件: {len(shard_files)} 个")
for i, f in enumerate(shard_files):
    print(f"  Shard {i+1}: {os.path.basename(f)}")

print("\n" + "="*70)
print("步骤4: 加载第一个 shard 并转换")
print("="*70)
raw = load_file(shard_files[0], device="cpu")
print(f"原始 shard 键数: {len(raw)}")
print(f"前5个键: {list(raw.keys())[:5]}")

converted = convert_safetensors_to_custom(
    raw, config.num_hidden_layers,
    tie_word_embeddings=config.tie_word_embeddings,
    n_heads=config.num_attention_heads,
    n_kv_heads=config.num_key_value_heads,
)
print(f"转换后键数: {len(converted)}")
print(f"前5个键: {list(converted.keys())[:5]}")

# 检查是否有重复的张量对象
tensor_ids = {}
for k, v in converted.items():
    tid = id(v)
    if tid in tensor_ids:
        print(f"警告: {k} 和 {tensor_ids[tid]} 共享同一个张量对象！")
    else:
        tensor_ids[tid] = k

print(f"唯一张量对象数: {len(tensor_ids)}")

print("\n" + "="*70)
print("步骤5: 检查 device_map")
print("="*70)
device_map = auto_device_map(model, "cuda:0", "cpu")
print(f"Device map: {device_map}")

print("\n完成调试")
