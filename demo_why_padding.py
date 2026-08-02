#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
演示为什么批量推理必须padding，不能直接用不同长度
"""
import torch
import torch.nn as nn

print("=" * 70)
print("为什么批量推理必须padding？")
print("=" * 70)

# 假设我们有3个不同长度的句子
sentences = [
    [1, 2, 3, 4, 5],      # 长度 5
    [6, 7, 8],            # 长度 3
    [9, 10],              # 长度 2
]

print("\n我们有3个不同长度的句子：")
for i, s in enumerate(sentences):
    print(f"  句子{i+1}: {s} (长度 {len(s)})")

print("\n" + "=" * 70)
print("方案1: 直接用不同长度（❌ 不可行）")
print("=" * 70)

print("""
问题1: PyTorch 的 tensor 必须是矩形的
--------------------------------------
tensor 是多维数组，每个维度的大小必须一致。

尝试创建不同长度的 tensor：
""")

try:
    # 这会失败！
    batch = torch.tensor(sentences)
    print(f"成功: {batch}")
except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {e}")

print("""
为什么会失败？
因为 PyTorch tensor 要求所有行长度相同：
  [[1, 2, 3, 4, 5],
   [6, 7, 8, ?, ?],     ← 这里缺2个元素
   [9, 10, ?, ?, ?]]    ← 这里缺3个元素

tensor 不支持"参差不齐"的结构！
""")

print("\n" + "=" * 70)
print("方案2: 分别处理每个句子（✅ 可行但低效）")
print("=" * 70)

print("""
可以逐个句子单独处理：

for sentence in sentences:
    input_tensor = torch.tensor(sentence).unsqueeze(0)  # (1, seq_len)
    output = model(input_tensor)
    # 处理输出...

这样可行，但是：
❌ 效率极低！完全没有利用 GPU 的并行计算能力
❌ 3个句子需要调用3次 forward，而不是1次
❌ GPU 利用率低（batch_size=1）

在 lm_eval 评测中：
- 可能有5000+条评测数据
- 如果 batch_size=1，需要调用5000次 forward
- 如果 batch_size=64，只需要调用 ~78 次 forward
- 速度差距：几十倍甚至上百倍！
""")

print("\n" + "=" * 70)
print("方案3: Padding 对齐（✅ 正确且高效）")
print("=" * 70)

print("""
通过 padding 将所有句子对齐到相同长度：

左padding（用于评测）：
  [[1, 2, 3, 4, 5],
   [0, 0, 6, 7, 8],     ← 前面补2个0
   [0, 0, 0, 9, 10]]    ← 前面补3个0

右padding（用于训练）：
  [[1, 2, 3, 4, 5],
   [6, 7, 8, 0, 0],     ← 后面补2个0
   [9, 10, 0, 0, 0]]    ← 后面补3个0

优点：
✅ 可以组成规则的 tensor
✅ 批量处理，充分利用 GPU 并行能力
✅ 一次 forward 处理多个句子
""")

# 演示左padding
pad_id = 0
max_len = max(len(s) for s in sentences)
padded = []
for s in sentences:
    pad_len = max_len - len(s)
    padded.append([pad_id] * pad_len + s)  # 左padding

batch_tensor = torch.tensor(padded)
print(f"\n左padding后的 tensor:\n{batch_tensor}")
print(f"Shape: {batch_tensor.shape}  ← 规则的 (3, 5) 矩阵")

print("\n" + "=" * 70)
print("为什么是左padding而不是右padding？")
print("=" * 70)

print("""
在评测任务（loglikelihood）中：
--------------------------------
我们要计算的是：给定上下文，续写的概率

例如：
  上下文: "The cat sat on the"
  续写:   "mat"

拼接后: "The cat sat on the mat"
分词后: [464, 3797, 3332, 319, 262, 2603]
                                    ^^^^
                                    这是续写部分

模型前向传播：
  logits = model(input_ids)
  # logits shape: (batch, seq_len, vocab_size)

  # 我们要取续写部分对应的 logits
  # 续写在末尾，所以用负索引：logits[:, -len(continuation)-1:-1]

如果用右padding：
  原始:     [464, 3797, 3332, 319, 262, 2603]
  右padding: [464, 3797, 3332, 319, 262, 2603, 0, 0, 0]
                                            ^^^^^^^^^^^^^^
                                            续写被推到中间了！

  问题：续写不在末尾了，索引变复杂！
  每个样本的续写位置都不同，无法统一索引

如果用左padding：
  原始:     [464, 3797, 3332, 319, 262, 2603]
  左padding: [0, 0, 0, 464, 3797, 3332, 319, 262, 2603]
             ^^^^^^^^^^
             padding在前面
                                        ^^^^
                                        续写永远在末尾！

  优点：续写位置对齐，可以用统一的负索引：
  logits[:, -len(continuation)-1:-1]  ← 所有样本都适用
""")

print("\n" + "=" * 70)
print("实际代码示例（来自 gpt_lm.py）")
print("=" * 70)

print("""
# 批量处理64个请求
batch_size = 64
requests = [...]  # 64个评测请求

# 1. 左padding对齐
max_len = max(len(x) for x in batch_input_ids)
padded = []
attn_masks = []
for ids in batch_input_ids:
    pad_len = max_len - len(ids)
    padded.append([pad_id] * pad_len + ids)          # 左padding
    attn_masks.append([0] * pad_len + [1] * len(ids))  # mask标记

# 2. 转成 tensor
input_tensor = torch.tensor(padded).to(device)       # (64, max_len)
attn_tensor = torch.tensor(attn_masks).to(device)    # (64, max_len)

# 3. 一次 forward 处理64个
out = model(input_tensor, attention_mask=attn_tensor)
logits = out.logits  # (64, max_len, vocab_size)

# 4. 统一索引续写部分（因为左padding，续写永远在末尾相同位置）
for j, cont_ids in enumerate(batch_cont_ids):
    cont_len = len(cont_ids)
    cont_logits = logits[j, -cont_len-1:-1, :]  # ← 负索引，适用所有样本
    # 计算概率...

对比：如果不padding，需要循环64次
for req in requests:
    input_tensor = torch.tensor(req.ids).unsqueeze(0)  # (1, len)
    out = model(input_tensor)
    # 处理...
# 慢64倍！
""")

print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print("""
1. 为什么必须padding？
   - PyTorch tensor 要求矩形结构，不支持不同长度
   - 批量处理需要固定的 shape

2. 为什么是左padding？
   - 评测任务的续写永远在末尾
   - 左padding让续写位置对齐，方便统一索引

3. 为什么需要 attention_mask？
   - 告诉模型哪些是padding，哪些是真实token
   - 避免padding污染注意力计算

4. 不padding可以吗？
   - 理论上可以，但需要逐个处理（batch_size=1）
   - 效率极低，评测时间从几分钟变成几小时
   - 完全浪费了 GPU 的并行能力
""")
