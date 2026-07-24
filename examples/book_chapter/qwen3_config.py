#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/14 09:46
@Author  : weiyutao
@File    : qwen3_config.py
"""

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from examples.book_chapter.chapter4_layer_norm import GPT_CONFIG_124M

from examples.book_chapter.chapter4_gpt_model import GPTModel

model_name = "Qwen/Qwen3-0.6B-Base"
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
print(config.num_hidden_layers)
print(config)

print(f"正在加载 {model_name} Tokenizer ....")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)


print(model)



# gpt_model = GPTModel(GPT_CONFIG_124M)
# print(gpt_model)
