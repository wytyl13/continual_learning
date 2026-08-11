#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen 模型系列

支持 Qwen2 / Qwen2.5 / Qwen2.5-Coder / Qwen2.5-Math 等所有变体
架构完全相同，仅超参数配置不同
"""

from model.qwen.config import QwenConfig, MODEL_CONFIGS
from model.qwen.model import QwenModel, QwenForCausalLM
from model.qwen.weight_convert import convert_safetensors_to_custom, convert_custom_to_hf

__all__ = [
    "QwenConfig",
    "MODEL_CONFIGS",
    "QwenModel",
    "QwenForCausalLM",
    "convert_safetensors_to_custom",
    "convert_custom_to_hf",
]
