#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示 attention_mask 的作用
"""
import torch

# 模拟一个批次的数据，使用左padding对齐
# 假设我们有3个句子：
# 句子1: [1, 2, 3, 4, 5]  (5个token)
# 句子2: [6, 7, 8]        (3个token)
# 句子3: [9, 10]          (2个token)

# 左padding后（padding token用0表示，对齐到最长5）：
input_ids = torch.tensor([
    [1, 2, 3, 4, 5],   # 句子1：无padding
    [0, 0, 6, 7, 8],   # 句子2：前面2个padding
    [0, 0, 0, 9, 10],  # 句子3：前面3个padding
])

# attention_mask 告诉模型哪些位置是真实token（1），哪些是padding（0）
attention_mask = torch.tensor([
    [1, 1, 1, 1, 1],   # 句子1：全是真实token
    [0, 0, 1, 1, 1],   # 句子2：前2个是padding，后3个是真实token
    [0, 0, 0, 1, 1],   # 句子3：前3个是padding，后2个是真实token
])

print("=" * 60)
print("批量推理中的左padding问题")
print("=" * 60)
print("\n输入 token IDs (左padding):")
print(input_ids)
print("\nAttention Mask (0=padding, 1=真实token):")
print(attention_mask)

print("\n" + "=" * 60)
print("为什么需要 attention_mask？")
print("=" * 60)
print("""
1. 批量推理需要对齐长度：
   - 不同句子长度不同，需要padding到同一长度才能组成batch

2. 为什么用左padding而不是右padding？
   - 因为我们要计算的是续写token的概率，续写永远在末尾
   - 左padding让所有句子的"结尾"对齐，方便统一索引

3. attention_mask 的作用：
   - 告诉模型：padding位置的token（0）应该被忽略
   - 在计算注意力分数时，padding位置不应该影响真实token
   - 如果不用mask，padding token也会参与注意力计算，污染结果

举例：
   句子2是 [6, 7, 8]，左padding后变成 [0, 0, 6, 7, 8]

   WITHOUT attention_mask:
   - token 6 会看到前面的 [0, 0, 6]
   - token 7 会看到前面的 [0, 0, 6, 7]
   - 这些padding token (0, 0) 会影响注意力计算！❌

   WITH attention_mask:
   - token 6 只看到 [6]（padding被mask掉）
   - token 7 只看到 [6, 7]（padding被mask掉）
   - padding不参与注意力计算 ✅
""")

print("\n" + "=" * 60)
print("在 lm_eval 评测中的实际使用场景")
print("=" * 60)
print("""
eval_llama.py 中，GPT2LM 类在评测时会：

1. 收集一个batch的请求（比如64个）
2. 每个请求长度不同，需要左padding对齐
3. 构造 attention_mask 来标记哪些是padding

示例代码（来自 gpt_lm.py:186-200）：

    max_len = max(len(x) for x in batch_input_ids)
    padded = []
    attn_masks = []
    for ids in batch_input_ids:
        pad_len = max_len - len(ids)
        padded.append([pad_id] * pad_len + ids)      # 左padding
        attn_masks.append([0] * pad_len + [1] * len(ids))  # mask标记

    input_tensor = torch.tensor(padded, dtype=torch.long).to(device)
    attn_tensor = torch.tensor(attn_masks, dtype=torch.long).to(device)

    # ❌ 你的自定义模型不支持这个参数，所以报错！
    out = model(input_tensor, attention_mask=attn_tensor)

官方 transformers 的 LlamaForCausalLM 支持这个参数，
但你的自定义实现不支持，所以需要添加。
""")

print("\n" + "=" * 60)
print("官方 LlamaForCausalLM 的 forward 签名")
print("=" * 60)
print("""
def forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,  ← 支持这个参数！
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    ...
) -> CausalLMOutputWithPast:

attention_mask 说明：
    - shape: (batch_size, sequence_length)
    - 值为 0 或 1
    - 0 = 这个位置是padding，忽略它
    - 1 = 这个位置是真实token，参与计算
""")
