#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/24 14:59
@Author  : weiyutao
@File    : config.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# 服务器特定的根目录配置
BASE_DIR = Path(os.getenv("BASE_DIR"))
SOURCE_DIR = BASE_DIR / "source"
OUT_DIR = BASE_DIR / "out"