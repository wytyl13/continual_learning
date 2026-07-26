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