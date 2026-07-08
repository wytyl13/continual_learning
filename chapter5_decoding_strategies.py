#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/08 16:45
@Author  : weiyutao
@File    : chapter5_decoding_strategies.py
Let's look at text generation strategies to generate more original text. First, we will briefly revisit the
generate_text_simple function that we used inside generate_and_print_sample earlier. Then we will cover two
techniques, temperature scaling and top-k sampling, to improve this function.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import tiktoken
import torch
from chapter4_gpt_model import GPTModel
from chapter5_train import GPT_CONFIG_124M, train_model_simple, text_to_token_ids, token_ids_to_text
from chapter2_dataloader import create_dataloader_v1
from chapter4_gpt_model import generate_text_simple

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = "The_Verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    
    
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    print(len(train_loader))
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1
    )
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, 
        start_context="Every effort moves you", tokenizer=tokenizer
    )
    
    # model.to("cpu")
    # model.eval()
    
    # token_ids = generate_text_simple(
    #     model=model,
    #     idx=text_to_token_ids("Every effort moves you", tokenizer),
    #     max_new_tokens=25,
    #     context_size=GPT_CONFIG_124M["context_length"]
    # )
    # print("Output text:\n", token_ids_to_text(token_ids, tokenizer))