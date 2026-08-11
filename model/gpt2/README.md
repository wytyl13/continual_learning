# GPT-2 模型课件

## 0. 整体架构概览 (预计时长: 8-10 mins)

### 内容与理解

GPT-2 是基于 Transformer Decoder 的自回归语言模型，通过因果注意力机制实现从左到右的序列建模。整体架构采用 Pre-Norm + 残差连接设计，由 Embedding 层、N 层 Transformer Block、最终归一化层和语言模型头组成。

**完整数据流（model/gpt2/model.py:377-455）：**

```
输入: Token IDs (batch, seq_len)
  ↓
Embedding 层
  ├─ Token Embedding: (batch, seq_len) → (batch, seq_len, hidden_size)
  ├─ Positional Embedding: (seq_len,) → (seq_len, hidden_size)
  └─ 相加 + Dropout → (batch, seq_len, hidden_size)
  ↓
Transformer Block × N 层
  ├─ LayerNorm → Multi-Head Attention → Residual
  └─ LayerNorm → Feed-Forward Network → Residual
  每层输入输出: (batch, seq_len, hidden_size)
  ↓
Final LayerNorm
  (batch, seq_len, hidden_size) → (batch, seq_len, hidden_size)
  ↓
Language Model Head
  (batch, seq_len, hidden_size) @ (vocab_size, hidden_size).T
  → (batch, seq_len, vocab_size)
  ↓
输出: Logits (batch, seq_len, vocab_size)
```

**关键模块说明：**

| 模块 | 输入形状 | 输出形状 | 参数量 (GPT-2 124M) | 作用 |
|------|---------|---------|-------------------|------|
| Token Embedding | (batch, seq_len) | (batch, seq_len, 768) | 50257 × 768 = 38.6M | 将 token id 映射为语义向量 |
| Position Embedding | (seq_len,) | (seq_len, 768) | 1024 × 768 = 0.8M | 注入绝对位置信息 |
| Transformer Block × 12 | (batch, seq_len, 768) | (batch, seq_len, 768) | ~7M × 12 = 84M | 自注意力 + FFN，捕获上下文依赖 |
| LM Head | (batch, seq_len, 768) | (batch, seq_len, 50257) | 不共享权重时 38.6M | 预测下一个 token 的概率分布 |

**架构特点：**

1. **Pre-Norm 结构**：LayerNorm 在子层之前，训练更稳定
   ```python
   x = x + Attention(LayerNorm(x))
   x = x + MLP(LayerNorm(x))
   ```

2. **因果注意力掩码**：上三角矩阵遮掉未来 token，保证自回归特性
   ```python
   mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
   attn_scores.masked_fill(mask, float("-inf"))
   ```

3. **残差连接**：每个子层后都有残差连接，缓解梯度消失

**测试案例：**

```python
import torch
from model.gpt2.model import GPTForCausalLM
from model.gpt2.config import GPTConfig
from tokenizers import Tokenizer

# 初始化 GPT-2 模型
cfg = GPTConfig.gpt2_small()  # 124M 参数
model = GPTForCausalLM(cfg)
tokenizer = Tokenizer.from_pretrained("gpt2")

# 输入文本
text = "Hello, I am"
token_ids = tokenizer.encode(text).ids  # [15496, 11, 314, 716]
idx = torch.tensor([token_ids])  # (1, 4)

print(f"输入文本: {text}")
print(f"Token IDs: {token_ids}")
print(f"输入形状: {idx.shape}")  # (1, 4)

# 前向传播
logits = model(idx)
print(f"\n输出形状: {logits.shape}")  # (1, 4, 50257)

# 预测下一个 token
next_token_logits = logits[0, -1, :]  # 取最后一个位置
next_token_id = torch.argmax(next_token_logits).item()
next_token = tokenizer.decode([next_token_id])
print(f"\n预测的下一个 token: '{next_token}' (id={next_token_id})")

# 参数量统计
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,}")  # ~124M
```

**预期输出示例：**
```
输入文本: Hello, I am
Token IDs: [15496, 11, 314, 716]
输入形状: torch.Size([1, 4])

输出形状: torch.Size([1, 4, 50257])

预测的下一个 token: 'a' (id=257)

总参数量: 124,439,808
```

**形状变化追踪：**

```python
# 在 model.py 中插入打印语句观察中间层形状
def forward(self, idx):
    x = self.model.embed_tokens(idx) + self.model.embed_pos_tokens(...)
    print(f"After Embedding: {x.shape}")  # (1, 4, 768)

    for i, layer in enumerate(self.model.layers):
        x = layer(x)
        print(f"After Block {i}: {x.shape}")  # (1, 4, 768)

    x = self.model.norm(x)
    print(f"After Final Norm: {x.shape}")  # (1, 4, 768)

    logits = self.lm_head(x)
    print(f"After LM Head: {logits.shape}")  # (1, 4, 50257)
    return logits
```

### 模块回顾与总结

- GPT-2 采用 Embedding + N×Transformer Block + LM Head 的堆叠架构，通过因果注意力实现自回归语言建模
- Pre-Norm 结构和残差连接保证深层网络训练稳定性，整个前向过程中 hidden_size 维度保持不变（768）
- 下一步将深入学习 Embedding 层，了解如何将离散 token 映射为连续向量表示

### 思维拓展

1. 为什么 GPT-2 的每层输出都保持 (batch, seq_len, hidden_size) 的形状不变？这种设计有何优势？
2. 如果将 Pre-Norm 改为 Post-Norm，对训练稳定性和梯度流有何影响？
3. 参数量中 Embedding 和 LM Head 占比约 62%（38.6M × 2 / 124M），在持续学习场景下如何减少这部分显存占用？
4. 因果注意力掩码在训练时实现并行计算，但推理时需要逐 token 生成，如何通过 KV Cache 优化推理速度？
5. 如果在持续学习中只更新部分层（如最后 2 层 Transformer Block），对模型表达能力和遗忘程度有何影响？

---

## 1. 模块：Embedding 词嵌入与位置编码 (预计时长: 8-10 mins)

### 内容与理解

Embedding 层是 Transformer 架构的输入接口，将离散的 token id 映射为连续的高维向量，并注入位置信息。GPT-2 采用词嵌入（token embedding）+ 绝对位置编码（positional embedding）的可学习方案。

**nn.Embedding 查找机制（model/gpt2/model.py:406-407）：**

`nn.Embedding` 本质是一个可学习的查找表，存储形状为 (vocab_size, hidden_size) 的权重矩阵。给定 token id，通过索引操作返回对应的行向量。

```python
# 词嵌入表：w(vocab_size, hidden_size)
self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

# 位置嵌入表：w(max_position_embeddings, hidden_size)
self.embed_pos_tokens = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
```

查找逻辑等价于：
```python
# 输入：token_ids = [15496, 11, 314]
# 等价于：embed_tokens.weight[[15496, 11, 314], :]
# 输出：(3, hidden_size) 的嵌入矩阵
```

**Forward 完整流程（model/gpt2/model.py:414-432）：**

```python
def forward(self, idx: torch.Tensor) -> torch.Tensor:
    """
    Args:
        idx: (batch, seq_len) token id 序列
    Returns:
        x: (batch, seq_len, hidden_size) 嵌入向量
    """
    batch, seq_len = idx.shape

    # 步骤1：词嵌入查找
    # idx: (batch, seq_len) -> tok_emb: (batch, seq_len, hidden_size)
    tok_emb = self.embed_tokens(idx)

    # 步骤2：位置嵌入查找
    # 位置索引: [0, 1, 2, ..., seq_len-1]
    # pos_emb: (seq_len, hidden_size)
    pos_emb = self.embed_pos_tokens(torch.arange(seq_len, device=idx.device))

    # 步骤3：向量相加（广播机制自动扩展 pos_emb 到 batch 维度）
    # tok_emb: (batch, seq_len, hidden_size)
    # pos_emb: (seq_len, hidden_size) -> 广播为 (batch, seq_len, hidden_size)
    # 输出: (batch, seq_len, hidden_size)
    x = self.drop_embedding(tok_emb + pos_emb)
    return x
```

**数学原理：**

$$\text{tok\_emb}[i] = \mathbf{E}_{\text{token}}[\text{token\_id}_i], \quad \mathbf{E}_{\text{token}} \in \mathbb{R}^{V \times d}$$

$$\text{pos\_emb}[t] = \mathbf{E}_{\text{pos}}[t], \quad \mathbf{E}_{\text{pos}} \in \mathbb{R}^{L \times d}$$

$$\mathbf{h}_0 = \text{tok\_emb} + \text{pos\_emb}$$

其中 $V$ 是词表大小（50257），$d$ 是隐藏维度（768/1024/1280/1600），$L$ 是最大序列长度（1024）。

**测试案例：**

```python
import torch
import torch.nn as nn
from tokenizers import Tokenizer

# 初始化 GPT-2 tokenizer 和 Embedding 层
tokenizer = Tokenizer.from_pretrained("gpt2")
vocab_size = 50257
hidden_size = 768
max_position_embeddings = 1024

embed_tokens = nn.Embedding(vocab_size, hidden_size)
embed_pos_tokens = nn.Embedding(max_position_embeddings, hidden_size)

# 输入文本
text = "Hello, I am"
token_ids = tokenizer.encode(text).ids  # [15496, 11, 314, 716]
idx = torch.tensor([token_ids])  # (1, 4)

print(f"输入文本: {text}")
print(f"Token IDs: {token_ids}")
print(f"输入形状: {idx.shape}")  # (1, 4)

# 词嵌入
tok_emb = embed_tokens(idx)
print(f"\n词嵌入形状: {tok_emb.shape}")  # (1, 4, 768)
print(f"Token 'Hello'(id=15496) 的前5维: {tok_emb[0, 0, :5]}")

# 位置嵌入
seq_len = idx.shape[1]
pos_emb = embed_pos_tokens(torch.arange(seq_len))
print(f"\n位置嵌入形状: {pos_emb.shape}")  # (4, 768)
print(f"位置0的前5维: {pos_emb[0, :5]}")
print(f"位置1的前5维: {pos_emb[1, :5]}")

# 最终嵌入
final_emb = tok_emb + pos_emb
print(f"\n最终嵌入形状: {final_emb.shape}")  # (1, 4, 768)
print(f"位置0 ('Hello') 最终嵌入的前5维: {final_emb[0, 0, :5]}")
```

**预期输出示例：**
```
输入文本: Hello, I am
Token IDs: [15496, 11, 314, 716]
输入形状: torch.Size([1, 4])

词嵌入形状: torch.Size([1, 4, 768])
Token 'Hello'(id=15496) 的前5维: tensor([-0.0321,  0.8472, -1.2341,  0.5621, -0.3456])

位置嵌入形状: torch.Size([4, 768])
位置0的前5维: tensor([ 0.1234, -0.5678,  0.9012, -0.3456,  0.7890])
位置1的前5维: tensor([-0.2345,  0.6789, -0.1234,  0.4567, -0.8901])

最终嵌入形状: torch.Size([1, 4, 768])
位置0 ('Hello') 最终嵌入的前5维: tensor([ 0.0913,  0.2794, -0.3329,  0.2165,  0.4434])
```

### 模块回顾与总结

- Embedding 层通过两个可学习查找表将离散 token 序列转换为连续向量表示，词嵌入编码语义，位置嵌入编码顺序
- GPT-2 的位置编码是绝对位置编码，每个位置对应一个固定的可学习向量，序列长度受限于预训练时的 max_position_embeddings（1024）
- Embedding 层的输出将进入多头自注意力机制，通过 Query-Key-Value 计算实现 token 间的信息交互

### 思维拓展

1. 为什么词嵌入和位置嵌入是相加而不是拼接？相加会不会导致语义信息和位置信息混淆？
2. GPT-2 的绝对位置编码是可学习的，这与 Transformer 原论文中的正弦位置编码有何差异？哪种方案在长序列外推时表现更好？
3. 如果在持续学习场景中需要扩展词表（如添加新领域专有词汇），应该如何初始化新增的词嵌入向量？随机初始化会不会破坏原有知识？
4. Embedding 层的参数量占整个模型的比例是多少（vocab_size × hidden_size）？在算力受限时，是否可以通过词表剪枝或共享嵌入来减少显存占用？
5. 位置嵌入的梯度更新频率是否均匀？序列开头的位置（如 pos=0）是否比末尾位置（如 pos=1023）更新更频繁，从而导致位置编码的"老化"问题？
