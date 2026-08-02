#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 generate_until 批量化的加速效果
"""
import torch
import lm_eval
import time
from transformers import LlamaForCausalLM, LlamaConfig
from tokenizers import Tokenizer
from training.common.gpt_lm import GPT2LM
from model.llama.config import MODEL_CONFIGS
from utils.enums import ModelName
from config import SOURCE_DIR

model_name = ModelName.LLAMA3_8B.value
llama_cfg = MODEL_CONFIGS[model_name]
llama_cfg = LlamaConfig(**llama_cfg.to_transformers_dict())

print('加载模型 (单卡)...')
model = LlamaForCausalLM.from_pretrained(
    f'{SOURCE_DIR}/hf/llama/{model_name}',
    torch_dtype=torch.bfloat16,
    device_map='cuda:0'
)
tokenizer = Tokenizer.from_file(f'{SOURCE_DIR}/hf/llama/{model_name}/tokenizer.json')

# 测试不同 batch_size
for batch_size in [1, 4, 8]:
    print(f'\n{"="*60}')
    print(f'测试 batch_size={batch_size}')
    print(f'{"="*60}')

    lm = GPT2LM(
        model=model,
        device=torch.device('cuda:0'),
        tokenizer=tokenizer,
        context_length=llama_cfg.max_position_embeddings,
        vocab_size=llama_cfg.vocab_size,
        eot_token_id=llama_cfg.eos_token_id,
        batch_size=batch_size,
        max_gen_toks=32  # 限制生成长度加快测试
    )

    start = time.time()
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=['hellaswag'],
        limit=100,  # 只测试100个样本
        bootstrap_iters=0
    )
    cost = time.time() - start

    print(f"\n结果: {results['results']['hellaswag']['acc,none']:.4f}")
    print(f"耗时: {cost:.2f}s")
    print(f"吞吐: {100/cost:.2f} samples/s")
