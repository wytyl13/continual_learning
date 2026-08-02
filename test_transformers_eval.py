#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试 Transformers 模型的 Data Parallelism 是否会OOM"""

import torch
from transformers import LlamaForCausalLM
from tokenizers import Tokenizer
from config import SOURCE_DIR
from utils.enums import ModelName
from training import GPT2LM
import lm_eval

model_name = ModelName.LLAMA3_8B.value

print(f"CUDA设备数: {torch.cuda.device_count()}")

# 使用 Transformers 加载两个模型（Data Parallelism）
print("\n" + "="*60)
print("使用 Transformers 加载模型到双卡（Data Parallelism）...")
models = []
for gpu_id in range(torch.cuda.device_count()):
    print(f"加载模型 {gpu_id+1} 到 cuda:{gpu_id}...")
    m = LlamaForCausalLM.from_pretrained(
        f"{SOURCE_DIR}/hf/llama/{model_name}",
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{gpu_id}"
    )
    models.append(m)

    # 检查显存
    allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
    print(f"  GPU {gpu_id} 已分配: {allocated:.2f}GB")

print("\n" + "="*60)
print("初始化 GPT2LM wrapper...")
tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/{model_name}/tokenizer.json")

lm = GPT2LM(
    model=models,
    device=torch.device("cuda:0"),
    context_length=8192,
    tokenizer=tokenizer,
    max_gen_toks=1024,
    vocab_size=128256,
    eot_token_id=128001,
    batch_size=64,  # 原始的batch_size
)

print("\n" + "="*60)
print("开始评估...")
results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["lambada_openai"],
    batch_size=64,
    bootstrap_iters=0,
    limit=100,  # 只测试100个样本
)

print("\n" + "="*60)
print("评估完成！")
print(results["results"])
