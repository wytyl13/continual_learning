#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/21 18:37
@Author  : weiyutao
@File    : gpt2_lm_eval.py
"""

import lm_eval
import torch

from model.gpt2.model import GPTModel, GPTConfig, GPTForCausalLM
from training  import GPT2LM


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_pretrained_mode = 1 # 1: scratch, 2: load_pretrained, 3: load_continual_learning
    bootstrap_iters = 0 
    cfg = GPTConfig.gpt2_small()
    model = GPTForCausalLM(cfg)
    if load_pretrained_mode == 2:
        # model = GPTForCausalLM.from_pretrained(
        #     "/mnt/wsl/fast_disk/continual_learning/source/trf/gpt2/124M", 
        #     source="openai",
        #     map_location=device
        # )
        
        model = GPTForCausalLM.from_pretrained(
            "/mnt/wsl/fast_disk/continual_learning/source/hf/gpt2/124M", 
            source="hf",
            map_location=device
        )
        bootstrap_iters = 100
    elif load_pretrained_mode == 3:
        model = GPTForCausalLM.from_pretrained("/mnt/wsl/fast_disk/continual_learning/out/train_simple_20260722", source="pt", map_location=device)  
        bootstrap_iters = 100
    
    lm = GPT2LM(model=model, device=device, context_length=128)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["lambada_openai", "wikitext"],
        batch_size=2,
        bootstrap_iters=bootstrap_iters
    )
    
    print(results["results"])