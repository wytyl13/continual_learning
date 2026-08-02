#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示有无 attention_mask 对注意力计算的影响
"""
import torch
import torch.nn.functional as F

print("=" * 70)
print("演示：attention_mask 如何影响注意力计算")
print("=" * 70)

# 假设我们有一个简化的注意力计算场景
# Q, K, V shape: (batch=2, seq_len=5, hidden=4)
batch_size = 2
seq_len = 5
hidden = 4

# 创建简单的 Q, K, V
torch.manual_seed(42)
Q = torch.randn(batch_size, seq_len, hidden)
K = torch.randn(batch_size, seq_len, hidden)
V = torch.randn(batch_size, seq_len, hidden)

# 句子1: 完整5个token [1, 2, 3, 4, 5]
# 句子2: 只有3个token，左padding 2个 [PAD, PAD, 3, 4, 5]
attention_mask = torch.tensor([
    [1, 1, 1, 1, 1],  # 句子1：无padding
    [0, 0, 1, 1, 1],  # 句子2：前2个是padding
])

print("\nAttention Mask:")
print(attention_mask)
print("(0 = padding位置应该被忽略, 1 = 真实token)")

# ============================================================
# 情况1: 不使用 attention_mask（错误的做法）
# ============================================================
print("\n" + "=" * 70)
print("情况1: 不使用 attention_mask（你当前的自定义模型）")
print("=" * 70)

scores_without_mask = torch.matmul(Q, K.transpose(-2, -1)) / (hidden ** 0.5)
print("\n未mask的注意力分数 (batch=0, 只看第一个句子的第一行):")
print(scores_without_mask[0, 0])  # 第一个句子的第一个token对所有位置的注意力分数

# 使用causal mask（因果掩码，只能看到之前的token）
causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
scores_with_causal = scores_without_mask + causal_mask
attn_weights_without_mask = F.softmax(scores_with_causal, dim=-1)

print("\n应用causal mask后的注意力权重 (batch=1, 句子2):")
print(attn_weights_without_mask[1])
print("\n⚠️ 问题：句子2的第3个位置（索引2，真实的第一个token）")
print("   它的注意力权重分布在 [位置0, 位置1, 位置2]")
print("   但位置0和位置1是padding！不应该参与注意力计算！")
print(f"   注意力权重: {attn_weights_without_mask[1, 2]}")

# ============================================================
# 情况2: 使用 attention_mask（正确的做法）
# ============================================================
print("\n" + "=" * 70)
print("情况2: 使用 attention_mask（官方 LlamaForCausalLM 的做法）")
print("=" * 70)

# 将 attention_mask 转换为注意力分数的mask
# 0 -> -inf (完全屏蔽), 1 -> 0 (不影响)
# shape: (batch, 1, 1, seq_len) 方便广播
attn_mask_expanded = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9

scores_with_mask = scores_without_mask + causal_mask + attn_mask_expanded
attn_weights_with_mask = F.softmax(scores_with_mask, dim=-1)

print("\n应用 attention_mask 后的注意力权重 (batch=1, 句子2):")
print(attn_weights_with_mask[1])
print("\n✅ 正确：句子2的第3个位置（索引2，真实的第一个token）")
print("   它的注意力权重只在 [位置2]（padding被完全屏蔽）")
print(f"   注意力权重: {attn_weights_with_mask[1, :, 2]}")
print("   前两个位置（padding）的权重为0！")

# ============================================================
# 对比输出结果
# ============================================================
print("\n" + "=" * 70)
print("对比：有无 attention_mask 的输出差异")
print("=" * 70)

output_without_mask = torch.matmul(attn_weights_without_mask, V)
output_with_mask = torch.matmul(attn_weights_with_mask, V)

print("\n句子2的第3个位置（真实第1个token）的输出向量：")
print(f"WITHOUT mask: {output_without_mask[1, 2]}")
print(f"WITH mask:    {output_with_mask[1, 2]}")
print(f"\n差异（L2距离）: {torch.norm(output_without_mask[1, 2] - output_with_mask[1, 2]).item():.4f}")

print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print("""
1. 官方 LlamaForCausalLM 支持 attention_mask 参数：
   ✅ forward(input_ids, attention_mask=...)

2. 你的自定义模型不支持：
   ❌ forward(idx)  # 只接受 idx，没有 attention_mask

3. 后果：
   - 在批量推理时（lm_eval评测），左padding的数据会被错误处理
   - padding token会参与注意力计算，污染结果
   - 评测分数会不准确

4. 解决方案：
   需要在你的自定义模型中添加 attention_mask 支持：
   - LLamaForCausalLM.forward 添加参数
   - LLamaModel.forward 添加参数并传递
   - Attention.forward 添加参数并在计算注意力时使用
""")
