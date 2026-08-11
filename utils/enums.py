#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/07/26 22:08:44
@Author : weiyutao
@File : enums.py
"""

from enum import Enum


class LoadMode(str, Enum):
    SCRATCH = "scratch"
    PRETRAINED = "pretrained"
    CONTINUAL = "continual"


class ModelType(str, Enum):
    CUSTOM = "custom"            # 自定义类加载模型
    TRANSFORMERS = "transformers" # Transformers加载模型
    
class ModelName(str, Enum):
    TINY_LLAMA = "tiny_llama"
    LLAMA2_7B = "llama-2-7b"
    LLAMA3_8B = "llama-3-8b"
    QWEN2_0_5B = "Qwen2-0.5B"
    QWEN2_1_5B = "Qwen2-1.5B"
    QWEN2_7B = "Qwen2-7B"
    QWEN2_72B = "Qwen2-72B"
    QWEN2_5_0_5B = "Qwen2.5-0.5B"
    QWEN2_5_1_5B = "Qwen2.5-1.5B"
    QWEN2_5_3B = "Qwen2.5-3B"
    QWEN2_5_7B = "Qwen2.5-7B"
    QWEN2_5_14B = "Qwen2.5-14B"
    QWEN2_5_32B = "Qwen2.5-32B"
    QWEN2_5_72B = "Qwen2.5-72B"

    
    