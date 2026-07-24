#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/21 18:37
@Author  : weiyutao
@File    : gpt2_lm_eval.py
"""

import lm_eval
import torch
from tokenizers import Tokenizer

from model.gpt2.model import GPTModel, GPTConfig, GPTForCausalLM
from training  import GPT2LM
from config import SOURCE_DIR, OUT_DIR

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_pretrained_mode = 1 # 1: scratch, 2: load_pretrained, 3: load_continual_learning
    bootstrap_iters = 0 
    cfg = GPTConfig.gpt2_small()
    model = GPTForCausalLM(cfg)
    tokenizer = Tokenizer.from_pretrained("gpt2") # 只下载hf格式的tokenizer.json文件
    if load_pretrained_mode == 2:
        # model = GPTForCausalLM.from_pretrained(
        #     f"{SOURCE_DIR}/trf/gpt2/124M", 
        #     source="openai",
        #     map_location=device
        # )
        
        model = GPTForCausalLM.from_pretrained(
            f"{SOURCE_DIR}/hf/gpt2/124M", 
            source="hf",
            map_location=device
        )
        tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/gpt2/124M/tokenizer.json")
        bootstrap_iters = 100
    elif load_pretrained_mode == 3:
        model = GPTForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260722", source="pt", map_location=device)  
        tokenizer = Tokenizer.from_file(f"{OUT_DIR}/train_simple_20260722/tokenizer.json")
        bootstrap_iters = 100
    
    lm = GPT2LM(model=model, device=device, context_length=128, tokenizer=tokenizer)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["lambada_openai", "wikitext"],
        batch_size=2,
        bootstrap_iters=bootstrap_iters
    )
    
    print(results["results"])