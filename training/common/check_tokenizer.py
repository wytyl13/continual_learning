#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/24 14:23
@Author  : weiyutao
@File    : check_tokenizer.py
"""


if __name__ == "__main__":
    import tiktoken
    from transformers import AutoTokenizer
    from config import SOURCE_DIR

    # gpt2
    enc = tiktoken.get_encoding("gpt2")
    print(enc.eot_token) # 50256

    """
    # hugginface tokenizer
    """
    # gpt2
    tokenizer = AutoTokenizer.from_pretrained(f"{SOURCE_DIR}/hf/gpt2/124M") 
    print(tokenizer.eos_token_id)
    print(tokenizer.eos_token)
    
    
    