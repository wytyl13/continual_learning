#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试所有模型加载路径的兼容性
"""
import torch
from model.gpt2.model import GPTConfig, GPTForCausalLM
from transformers import GPT2LMHeadModel, GPT2Config
from config import SOURCE_DIR, OUT_DIR

def test_loading():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("测试1: 从HuggingFace下载的模型加载到自定义模型")
    print("=" * 80)
    try:
        model = GPTForCausalLM.from_pretrained(
            f"{SOURCE_DIR}/hf/gpt2/124M",
            source="hf",
            map_location=device
        )
        print(f"✓ 成功加载")
        print(f"  Config有pad_token_id: {hasattr(model.cfg, 'pad_token_id')}")
        if hasattr(model.cfg, 'pad_token_id'):
            print(f"  pad_token_id={model.cfg.pad_token_id}, eos_token_id={model.cfg.eos_token_id}")
    except Exception as e:
        print(f"✗ 失败: {e}")

    print("\n" + "=" * 80)
    print("测试2: 从自定义训练保存的模型加载")
    print("=" * 80)
    try:
        model = GPTForCausalLM.from_pretrained(
            f"{OUT_DIR}/train_simple_20260722",
            source="pt",
            map_location=device
        )
        print(f"✓ 成功加载")
        print(f"  Config有pad_token_id: {hasattr(model.cfg, 'pad_token_id')}")
        if hasattr(model.cfg, 'pad_token_id'):
            print(f"  pad_token_id={model.cfg.pad_token_id}, eos_token_id={model.cfg.eos_token_id}")
    except Exception as e:
        print(f"✗ 失败: {e}")

    print("\n" + "=" * 80)
    print("测试3: 测试generate接口兼容性")
    print("=" * 80)
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_pretrained("gpt2")

        # 自定义模型
        model1 = GPTForCausalLM.from_pretrained(
            f"{SOURCE_DIR}/hf/gpt2/124M",
            source="hf",
            map_location=device
        )

        # Transformers模型
        model2 = GPT2LMHeadModel.from_pretrained(f"{SOURCE_DIR}/hf/gpt2/124M").to(device)

        text = "Hello, I am"
        encoded = torch.tensor(tokenizer.encode(text).ids).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(encoded)

        # 测试自定义模型的generate
        print("测试自定义模型generate...")
        with torch.no_grad():
            output1 = model1.generate(
                input_ids=encoded,
                attention_mask=attention_mask,
                max_new_tokens=10,
                temperature=0.8,
                top_k=50,
                do_sample=True
            )
        print(f"✓ 自定义模型generate成功，输出shape: {output1.shape}")

        # 测试Transformers模型的generate
        print("测试Transformers模型generate...")
        with torch.no_grad():
            output2 = model2.generate(
                input_ids=encoded,
                attention_mask=attention_mask,
                max_new_tokens=10,
                pad_token_id=50256,
                eos_token_id=50256,
                do_sample=True,
                temperature=0.8,
                top_k=50
            )
        print(f"✓ Transformers模型generate成功，输出shape: {output2.shape}")

    except Exception as e:
        print(f"✗ 失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("所有测试完成")
    print("=" * 80)

if __name__ == "__main__":
    test_loading()
