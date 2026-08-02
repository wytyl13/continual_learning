#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""完整测试权重加载流程，找出权重共享的根源"""

import torch
import json
import os
from accelerate import init_empty_weights, dispatch_model
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
print("完整加载流程测试")
print("="*70)

# 1. 配置
with open(os.path.join(model_path, "config.json")) as f:
    config = LLamaConfig.from_dict(json.load(f))

# 2. 创建空模型
with init_empty_weights():
    model = LLamaForCausalLM(config)

print(f"步骤1: 创建空模型，参数名称数: {len(list(model.named_parameters()))}")

# 3. device_map
device_map = auto_device_map(model, "cuda:0", "cpu")
print(f"步骤2: device_map = {device_map}")

# 4. 加载权重
shard_files, use_safetensors = resolve_checkpoint_files(model_path)

def _param_device(name):
    for prefix in sorted(device_map, key=len, reverse=True):
        if name.startswith(prefix) or prefix == "":
            return device_map[prefix]
    return "cpu"

full_sd = {}
for i, path in enumerate(shard_files):
    print(f"\n步骤3.{i+1}: 加载 shard {i+1}/{len(shard_files)}")
    raw = load_file(path, device="cpu")
    converted = convert_safetensors_to_custom(
        raw, config.num_hidden_layers,
        tie_word_embeddings=config.tie_word_embeddings,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
    )
    print(f"  原始键数: {len(raw)}, 转换后键数: {len(converted)}")

    for key, tensor in converted.items():
        full_sd[key] = tensor.to(device=_param_device(key), dtype=torch.bfloat16)

    del raw, converted
    torch.cuda.empty_cache()

print(f"\n步骤4: 累积的 state_dict 键数: {len(full_sd)}")
print(f"  GPU 显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")

# 检查 full_sd 中是否有重复的张量对象
tensor_ids_in_sd = {}
for k, v in full_sd.items():
    tid = id(v)
    if tid in tensor_ids_in_sd:
        print(f"  警告: full_sd 中 {k} 和 {tensor_ids_in_sd[tid]} 共享张量！")
    else:
        tensor_ids_in_sd[k] = k

print(f"  full_sd 唯一张量数: {len(tensor_ids_in_sd)}")

# 5. load_state_dict
print(f"\n步骤5: 调用 load_state_dict (assign=True)")
model.load_state_dict(full_sd, strict=False, assign=True)
print(f"  GPU 显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")

# 检查模型参数是否有共享
print(f"\n步骤6: 检查模型参数共享情况")
param_ids = {}
shared_count = 0
for name, param in model.named_parameters():
    param_id = id(param.data)
    if param_id in param_ids:
        shared_count += 1
        if shared_count <= 5:  # 只打印前5个
            print(f"  共享: {name} <-> {param_ids[param_id]}")
    else:
        param_ids[param_id] = name

print(f"  唯一参数对象数: {len(param_ids)}")
print(f"  参数名称总数: {len(list(model.named_parameters()))}")
print(f"  共享参数数: {shared_count}")

# 6. tie_weights
print(f"\n步骤7: 调用 tie_weights")
if hasattr(model, 'tie_weights'):
    model.tie_weights()

# 再次检查
param_ids_after_tie = {}
shared_after_tie = 0
for name, param in model.named_parameters():
    param_id = id(param.data)
    if param_id in param_ids_after_tie:
        shared_after_tie += 1
    else:
        param_ids_after_tie[param_id] = name

print(f"  唯一参数对象数: {len(param_ids_after_tie)}")
print(f"  共享参数数: {shared_after_tie}")

# 7. dispatch_model
print(f"\n步骤8: 调用 dispatch_model")
print(f"  GPU 显存 (调用前): {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
dispatch_model(model, device_map=device_map)
print(f"  GPU 显存 (调用后): {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")

# 最终检查
param_ids_final = {}
shared_final = 0
for name, param in model.named_parameters():
    param_id = id(param.data)
    if param_id in param_ids_final:
        shared_final += 1
        if shared_final <= 10:  # 打印前10个
            print(f"  共享: {name} <-> {param_ids_final[param_id]}")
    else:
        param_ids_final[param_id] = name

print(f"\n最终结果:")
print(f"  唯一参数对象数: {len(param_ids_final)}")
print(f"  参数名称总数: {len(list(model.named_parameters()))}")
print(f"  共享参数数: {shared_final}")
print(f"  GPU 显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
