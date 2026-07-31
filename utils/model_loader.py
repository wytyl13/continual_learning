#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/07/31 19:55:05
@Author : weiyutao
@File : model_loader.py
"""

import os, json
import torch
from typing import Callable, Optional
from safetensors.torch import load_file


def resolve_checkpoint_files(model_dir: str) -> tuple[list[str], bool]:
    """
    解析权重目录，返回 (shard文件路径列表, 是否safetensors)。
    优先级：safetensors分片 > safetensors单文件 > bin分片 > bin单文件
    可复用于任何HuggingFace格式模型。
    """
    def _index(name): return os.path.join(model_dir, name)

    if os.path.exists(_index("model.safetensors.index.json")):
        with open(_index("model.safetensors.index.json")) as f:
            index = json.load(f)
        files = [os.path.join(model_dir, s) for s in sorted(set(index["weight_map"].values()))]
        return files, True
    elif os.path.exists(_index("model.safetensors")):
        return [_index("model.safetensors")], True
    elif os.path.exists(_index("pytorch_model.bin.index.json")):
        with open(_index("pytorch_model.bin.index.json")) as f:
            index = json.load(f)
        files = [os.path.join(model_dir, s) for s in sorted(set(index["weight_map"].values()))]
        return files, False
    elif os.path.exists(_index("pytorch_model.bin")):
        return [_index("pytorch_model.bin")], False
    raise FileNotFoundError(f"No weight files found in {model_dir}")
  
  
def build_auto_device_map(
    model: torch.nn.Module,
    no_split_classes: list[str],
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    均衡分配多卡device_map，行为与HF from_pretrained(device_map="auto")一致。
    可复用于任何模型，只需传入 no_split_classes（不可跨卡切分的层类名）。
    """
    from accelerate import infer_auto_device_map
    from accelerate.utils import get_balanced_memory

    # 在计算设备映射前，务必先绑定权重，否则 accelerate 会将共享权重的层分到不同的 GPU
    if hasattr(model, 'tie_weights'):
        model.tie_weights()

    max_memory = get_balanced_memory(
        model, no_split_module_classes=no_split_classes,
        dtype=dtype, low_zero=False,
    )
    return infer_auto_device_map(
        model, max_memory=max_memory,
        no_split_module_classes=no_split_classes,
    )
      

def fast_load_weights(
    model: torch.nn.Module,
    shard_files: list[str],
    use_safetensors: bool,
    device_map: dict,
    convert_fn: Callable[[dict], dict],
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """
    逐shard加载权重，convert后直接落到目标GPU，原地替换meta tensor（assign=True）。
    CPU峰值内存 = 单个shard大小，不再需要全量中转。

    参数：
        model        : init_empty_weights 初始化的meta模型
        shard_files  : resolve_checkpoint_files 返回的路径列表
        use_safetensors: 同上
        device_map   : build_auto_device_map 返回的映射
        convert_fn   : 接收 {key: tensor} →返回转换后的 {key: tensor}，模型专属逻辑放这里
        dtype        : 目标dtype，默认bfloat16
    """
    def _param_device(name):
        for prefix in sorted(device_map, key=len, reverse=True):
            if name.startswith(prefix) or prefix == "":
                return device_map[prefix]
        return "cpu"

    full_sd = {}
    for path in shard_files:
        raw = load_file(path, device="cpu") if use_safetensors else torch.load(path, map_location="cpu", weights_only=False)
        converted = convert_fn(raw)
        for key, tensor in converted.items():
            full_sd[key] = tensor.to(device=_param_device(key), dtype=dtype)
        del raw, converted  # 立即释放，CPU只保留一个shard

    # assign=True: meta tensor直接被替换，无额外拷贝（需PyTorch >= 2.1）
    model.load_state_dict(full_sd, strict=False, assign=True)
    # tie_weights 必须在 dispatch_model 之前调用，否则 infer_auto_device_map 报错
    if hasattr(model, 'tie_weights'):
        model.tie_weights()
    # 安装 accelerate hooks，使模型行为与 from_pretrained(device_map=...) 一致
    from accelerate import dispatch_model
    dispatch_model(model, device_map=device_map)
    

def resolve_local_checkpoint(model_name_or_path: str) -> tuple[str, str, bool]:
    """
    解析本地训练保存的模型路径，返回 (config_file, weight_file, use_safetensors)。
    支持两种传参方式：- 目录路径：自动在目录下查找 config.json 和权重文件
    - 直接文件路径：自动推导同目录下的 config 文件
    """
    import os

    if os.path.isdir(model_name_or_path):
        config_file = os.path.join(model_name_or_path, "config.json")
        safetensors_file = os.path.join(model_name_or_path, "model.safetensors")
        bin_file = os.path.join(model_name_or_path, "pytorch_model.bin")

        if os.path.exists(safetensors_file):
            weight_file, use_safetensors = safetensors_file, True
        elif os.path.exists(bin_file):
            weight_file, use_safetensors = bin_file, False
        else:
            raise FileNotFoundError(f"找不到权重文件: {safetensors_file} 或 {bin_file}")
    else:
        config_file = (model_name_or_path
                        .replace(".pt", "_config.json")
                        .replace(".bin", "_config.json")
                        .replace(".safetensors", "_config.json"))
        weight_file = model_name_or_path
        use_safetensors = weight_file.endswith(".safetensors")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    return config_file, weight_file, use_safetensors