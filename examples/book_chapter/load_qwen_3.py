#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/07/12 08:42:31
@Author : weiyutao
@File : load_qwen.py
"""

import os
import torch
for proxy in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(proxy, None)

# WSL2 下 transformers 5.x 的 caching_allocator_warmup 会尝试预分配大块连续显存
# 但 WSL2 的虚拟化 GPU 不支持（分配上限约 256MB），替换为空操作绕过此步骤
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *args, **kwargs: None

from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


model_name = "Qwen/Qwen3-0.6B-Base"
print(f"正在加载 {model_name} Tokenizer ....")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("正在加载原生Base模型（CPU+GPU 混合，embed/lm_head 留 CPU 绕过 WSL2 486MB 连续显存限制）")


# Qwen3.5-0.8B 共 24 层；embed_tokens/lm_head 单张量 ~486MB 超出 WSL2 CUDA 的 ~256MB 连续分配上限
# 将这两个大张量留在 CPU，其余全部放 GPU，推理时 embedding lookup 在 CPU，前向在 GPU


device_map = {
    "model.embed_tokens": "cpu",
    "lm_head":            "cpu",
    "model.norm":         "cuda:0",
    "model.rotary_emb":   "cuda:0",
    **{f"model.layers.{i}": "cuda:0" for i in range(28)},
}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=True,
)

print("\n模型加载成功！当前模型结构如下：")
print(model)

print("-" * 50)

prompt = "农业专家指出，苹果树常见的病害有"

# 跨设备模型：第一个操作是 embed_tokens（在 CPU），所以 input_ids 留在 CPU
inputs = tokenizer(prompt, return_tensors="pt")  # 不 .to() 任何设备，默认 cpu



print(f"输入文本：{prompt}")
print("正在生成回答...")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.3)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== 生成结果 ===")
print(result)