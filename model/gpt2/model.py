#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/14
@Author  : weiyutao
@File    : model.py

持续学习

前置条件：
    熟练GPT模型架构、熟练每个模块前向传播以及反向传播、熟练每个模块的权重情况、熟练开源模型的架构配置、
    复现开源模型架构并加载开源模型权重、熟练对开源模型进行不同维度评估、熟练各种开源数据、熟练开源模型训练数据处理方式
    熟练构建pytorch优化器和训练脚本、熟练各种数据预处理方法、熟练使用pytorch构建训练数据集
    熟练各种微调数据集格式、熟练各种优化器、熟练构建新的模型架构、熟练优化现有的模型架构
    
    熟练使用transformers库复现各种开源模型架构
    熟练使用transformers库进行训练、推理和微调

熟练掌握持续学习理论、会结合神经科学最新研究理论优化神经网络持续学习、如何不新增训练数据情况下进行数据回放训练（也就是在持续学习过程中模型内部自动实现等效数据回放）
如何实现局部参数更新、如何结合非量化规则（也即模型内部支持简单的规则和细粒度量化规则两种，前者类似前额叶，后者类似海马体，只有这样模型内部才能实现数据回放）
如何减少灾难性遗忘




GPT 架构完整实现。
    整体结构：
        GPTConfig          配置类（词表大小、层数、维度等）
        LayerNorm          层归一化
        GELU               激活函数
        GPTAttention       多头自注意力（MHA + causal mask）
        GPTMLP             前馈网络（Linear → GELU → Linear）
        GPTBlock           Transformer 块（Pre-Norm + 残差）
        GPTModel           完整 GPT 模型

    与 LLaMA 的关键差异（打开 llama1_model.py 对比）：
        归一化  LayerNorm（本文件）       vs  RMSNorm（llama1）
        位置编码 绝对位置编码 nn.Embedding  vs  RoPE（llama1）
        激活函数 GELU（本文件）           vs  SwiGLU（llama1）
        注意力   MHA 无 bias-free（本文件）vs  MHA bias-free（llama1）

"""
import os
import json

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from safetensors.torch import load_file

from training import get_logger
from model.gpt2.weight_convert import convert_safetensors_to_custom, convert_openai_to_custom


logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPTConfig:
    """GPT 模型配置。

    设计参考 transformers 库的 PretrainedConfig，但简化为 dataclass：
    - 所有超参数都有默认值，对应 GPT-2 Small（124M）
    - 通过类方法快速创建各尺寸预设

    使用方式：
        cfg = GPTConfig()                    # GPT-2 Small（默认）
        cfg = GPTConfig.gpt2_medium()        # GPT-2 Medium
        cfg = GPTConfig(num_layers=6)        # 自定义
    """
    vocab_size: int = 50257       # GPT-2 tokenizer 词表大小
    hidden_size: int = 768      # 词向量维度，同时也是所有层的宽度
    intermediate_size: Optional[int] = None # 默认None，表示自动计算
    max_position_embeddings: int = 1024    # 最大序列长度（位置编码的上限）
    num_attention_heads: int = 12           # 注意力头数，head_dim = hidden_size / num_attention_heads = 64
    num_key_value_heads: int = 12
    num_hidden_layers: int = 12          # Transformer 块堆叠层数
    layer_norm_epsilon: float = 1e-5
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    activation_function: str = "gelu_new"
    tie_word_embeddings: bool = True
    qkv_bias: bool = True        # Q/K/V 投影是否带 bias（原始 GPT-2 为 True，现代模型倾向 False）

    # ── 预设配置 ──────────────────────────────────────────────────────────────

    def __post_init__(self):
          """实例化后自动计算intermediate_size"""
          if self.intermediate_size is None:
              self.intermediate_size = self.hidden_size * 4

    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        """124M 参数（不含 output head 权重绑定）"""
        return cls()

    @classmethod
    def gpt2_medium(cls) -> "GPTConfig":
        """355M 参数"""
        return cls(max_position_embeddings=1024, num_attention_heads=16, num_key_value_heads=16, num_hidden_layers=24)

    @classmethod
    def gpt2_large(cls) -> "GPTConfig":
        """774M 参数"""
        return cls(max_position_embeddings=1280, num_attention_heads=20, num_key_value_heads=20, num_hidden_layers=36)

    @classmethod
    def gpt2_xl(cls) -> "GPTConfig":
        """1.5B 参数"""
        return cls(max_position_embeddings=1600, num_attention_heads=25, num_key_value_heads=25, num_hidden_layers=48)


# ─────────────────────────────────────────────────────────────────────────────
# 基础组件（GPT 专用）
# ─────────────────────────────────────────────────────────────────────────────

class LayerNorm(nn.Module):
    """层归一化（Layer Normalization）。

    GPT 使用 LayerNorm（跨特征维度归一化），而 LLaMA 改用 RMSNorm（去掉均值中心化，更快）。
    LayerNorm 对每个 token 独立归一化其 embedding_dim 维度：
        均值 → 0，方差 → 1，再通过可学习的 scale/shift 恢复表达能力。

    为什么 unbiased=False（除以 n 而非 n-1）？
        embedding_dim 通常很大（768/1024/...），n 和 n-1 差异极小，
        且在 LLM 的每次 forward 中计算的是"当前 token 特征的内部统计"，
        不需要作为总体方差的无偏估计，所以直接除以 n。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        
        # w(embedding_dim,)
        self.scale = nn.Parameter(torch.ones(hidden_size))   # 可学习缩放，恢复被归一化压缩的方差

        # w(embedding_dim,)
        self.shift = nn.Parameter(torch.zeros(hidden_size))  # 可学习平移，恢复均值自由度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # x_norm(batch, seq_len, embedding_dim)
        # scale(embedding_dim,)
        # shift(embedding_dim,)
        # scale(embedding_dim,) * (batch, seq_len, embedding_dim) + (embedding_dim,) -> (batch, seq_len, embedding_dim)
        return self.scale * x_norm + self.shift


class GELUScratch(nn.Module):
    """GELU 激活函数（Gaussian Error Linear Unit）。

    GPT 使用 GELU，LLaMA 改用 SwiGLU（门控结构，表达能力更强）。
    GELU 假设神经网络的输入数据服从标准正态分布N(0, 1)。它给每个输入 x 赋予一个“保留概率”，
    这个概率等于标准正态分布中随机变量 X 小于等于当前输入 x 的累积概率，即 P(X ≤ x)
    GELU(x) = x * P(X ≤ x)
    当 x 是一个很大的正数（比如 x=3），它在正态分布中击败了 99.8% 的数据，所以它的保留概率 P(X ≤ 3) ≈ 1。此时 GELU(x) ≈ x * 1 = x。
    当 x 是一个很大的负数（比如 x=-3），它在正态分布中垫底，保留概率 P(X ≤ -3) ≈ 0。此时 GELU(x) ≈ x * 0 = 0。
    当 x = 0 时，它处于正态分布正中间，击败了 50% 的数据，概率为 0.5。此时 GELU(0) = 0 * 0.5 = 0。
    最精妙的地方在于 x 处于 [-1, 0] 之间时： 此时概率是一个很小的值，相乘之后，函数图像会往下拉，产生一个微小的负数区域。
    这就给了模型在负数区间进行微调和容错的空间，而不会像 ReLU 那样直接把梯度抹杀成 0。
    正态分布的累积概率函数 P(X ≤ x) 在数学上记作 Φ(x)。它是一个积分公式，没有简单的基础代数表达式（不能只用加减乘除算出来）。

    精确公式（误差函数）：
        P(X ≤ x) = Φ(x) = 0.5 * [1 + erf(x / √2)]
        GELU(x) = 0.5 * x * [1 + erf(x / √2)]
        如何算 erf？底层 C++ 或 CUDA 是通过泰勒展开式、切比雪夫多项式等极高精度的数值算法，经过十几次甚至几十次浮点运算去逼近这个积分值的

    近似公式（Hendrycks & Gimpel 2016）：
        如果使用 GPU 早就优化到底层的 tanh 函数，再加上一个巧妙的三次多项式x^3，画出来的曲线跟精确的erf曲线几乎一模一样。最大误差在0.0003左右 
        x^3用来微调曲线的弯曲程度，使其完贴合真实的正态分布概率曲线。0.044715是一个最佳拟合参数。    
        GELU_new(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    GELU 相比 ReLU 的优势：
        ReLU 对所有负值硬截断为 0，造成死神经元（梯度永久为 0）。
        GELU 在负值区域保留了一个小的平滑梯度（约 x * Φ(x)），
        使得负激活神经元仍有机会通过梯度更新被"救活"。

        
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
        ))


class GELU(nn.Module):
    """
    F.gelu比纯python拼接算子效率高
    approximate=none是精确的数学积分计算误差函数
    approximate=tanh是多项式模拟积分的曲线形状，提高运算效率。
    """
    def __init__(self, activation_function: str = "gelu_new"):
        super().__init__()
        self.approximate = "tanh" if activation_function == "gelu_new" else "none"
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return F.gelu(x, approximate=self.approximate)
    


# ─────────────────────────────────────────────────────────────────────────────
# GPT 核心模块
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """多头自注意力（Multi-Head Attention，MHA）+ causal mask。
    input(batch, max_position_embeddings, hidden_size)
    output(batch, max_position_embeddings, hidden_size)
    通过因果注意力，output中max_position_embeddings索引从小到大，每一个token的嵌入向量都包含了自己和前面所有token
    的注意力加权平均，比如index=3的token，其加权注意力是index=0, 1, 2, 3所有token按照因果注意力
    加权平均求和之后的结果，而RNN逐步传递信息越传越衰减的，这里是按需取用任意位置的内容，距离再远也不会产生衰减。
    
    
    GPT 使用标准 MHA（每个 head 都有完整的 Q/K/V）。
    LLaMA 2 引入 GQA（Grouped Query Attention），多个 Q head 共享同一组 K/V，
    大幅减少 KV Cache 显存占用。

    Causal Mask：
        上三角矩阵遮掉未来 token 的注意力分数（填 -inf，softmax 后趋近 0），
        保证自回归生成时第 t 个 token 只能看到 t 及之前的 token。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_attention_heads == 0, \
            "hidden_size 必须能被 num_attention_heads 整除"

        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        self.num_key_value_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.max_position_embeddings = cfg.max_position_embeddings
        self.d_out = cfg.hidden_size

        # w(num_attention_heads * head_dim, hidden_size), bias(num_attention_heads * head_dim,)
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=cfg.qkv_bias)
        
        # w(num_key_value_heads * head_dim, hidden_size), bias(num_key_value_heads * head_dim,)
        self.k_proj   = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=cfg.qkv_bias)

        # w(num_key_value_heads * head_dim, hidden_size), bias(num_key_value_heads * head_dim,)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=cfg.qkv_bias)

        # w(hidden_size, self.num_attention_heads * self.head_dim), bias(hidden_size,)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size)
        self.dropout = nn.Dropout(cfg.attn_pdrop)

        # causal mask：上三角为 1，forward 时用来遮掉未来位置
        # persistent=False 表示 mask 不会被保存到 state_dict 中（因为它是固定的，不需要训练），这样做目的是适配后续的load_state_dict_from_openai函数
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(self.max_position_embeddings, self.max_position_embeddings), diagonal=1),
            persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, max_position_embeddings, _ = x.shape

        # input(batch, max_position_embeddings, hidden_size) @ w.T(num_attention_heads * head_dim, hidden_size) -> out(batch, max_position_embeddings, hidden_size)
        q = self.q_proj(x)
        
        # input(batch, max_position_embeddings, hidden_size) @ w.T(num_key_value_heads * head_dim, hidden_size) -> out(batch, max_position_embeddings, hidden_size)
        k = self.k_proj(x)
        
        # input(batch, max_position_embeddings, hidden_size) @ w.T(num_key_value_heads * head_dim, hidden_size) -> out(batch, max_position_embeddings, hidden_size)
        v = self.v_proj(x)

        # input(batch, max_position_embeddings, hidden_size) -> view(batch, max_position_embeddings, num_attention_heads, head_dim) -> out(batch, num_attention_heads, max_position_embeddings, head_dim)
        q = q.view(batch, max_position_embeddings, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # input(batch, max_position_embeddings, hidden_size) -> view(batch, max_position_embeddings, num_key_value_heads, head_dim) -> out(batch, num_key_value_heads, max_position_embeddings, head_dim)
        k = k.view(batch, max_position_embeddings, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # input(batch, max_position_embeddings, hidden_size) -> view(batch, max_position_embeddings, num_key_value_heads, head_dim) -> out(batch, num_key_value_heads, max_position_embeddings, head_dim)
        v = v.view(batch, max_position_embeddings, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # q(batch, num_attention_heads, max_position_embeddings, head_dim)
        # k.transpose(2, 3)(batch, num_key_value_heads, head_dim, max_position_embeddings)
        # (batch, num_attention_heads, max_position_embeddings, head_dim) @ (batch, num_key_value_heads, head_dim, max_position_embeddings) -> out(batch, num_attention_heads, max_position_embeddings, max_position_embeddings)
        attn_scores = q @ k.transpose(2, 3) / math.sqrt(self.head_dim)

        # 应用 causal mask，遮掉未来 token
        # mask(seq_len, seq_len) -> (seq_len, seq_len)
        mask = self.mask.bool()[:max_position_embeddings, :max_position_embeddings]
        
        # (batch, num_attention_heads, seq_len, seq_len) -> (batch, num_attention_heads, seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        # (batch, num_attention_heads, seq_len, seq_len) -> (batch, num_attention_heads, seq_len, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # atten_weights(batch, num_attention_heads, seq_len, seq_len) @ v(batch, num_attention_heads, seq_len, head_dim) -> out(batch, num_attention_heads, seq_len, head_dim)
        context = attn_weights @ v
        
        # input(batch, num_attention_heads, max_position_embeddings, head_dim) -> out(batch, max_position_embeddings, num_attention_heads, head_dim))
        context = context.transpose(1, 2).contiguous()
        
        # input(batch, max_position_embeddings, num_attention_heads, head_dim) -> out(batch, max_position_embeddings, hidden_size)
        context = context.view(batch, max_position_embeddings, self.d_out)
        
        # input(batch, max_position_embeddings, hidden_size) -> out(batch, max_position_embeddings, hidden_size)
        return self.o_proj(context)


class MLP(nn.Module):
    """前馈网络（Feed-Forward Network，FFN）。
    input(batch, max_position_embeddings, hidden_size)
    output(batch, max_position_embeddings, hidden_size) 
    MLP层是模型存储事实知识的地方。
    假设 Paris is the captial of _.
    attention处理完后，_这个位置的向量已经融合了paris、captial of的信息。
    MLP拿到这个向量，从参数里查出这个组合，应该输出france相关的表示。
    
    GPT 的 FFN 结构：Linear(d → 4d) → GELU → Linear(4d → d)
    升维到 4d 的目的：给模型更大的"工作空间"来学习复杂的特征变换，
    GELU 负责非线性过滤，然后降维再把有效信息压缩回原始维度。

    LLaMA 改用 SwiGLU：Linear(d → 8/3*d) * sigmoid gate → Linear(8/3*d → d)
    三个矩阵而非两个，但参数量通过调整维度保持近似等效。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        
        # w(intermediate_size, hidden_size), bias(intermediate_size,)
        self.up_proj  = nn.Linear(cfg.hidden_size, cfg.intermediate_size)

        self.act_fn  = GELU(cfg.activation_function)
        
        # w(hidden_size, intermediate_size), bias(hidden_size,)
        self.down_proj  = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # self.up_proj: input(batch, max_position_embeddings, hidden_size) @ w.T(hidden_size, intermediate_size) -> out(batch, max_position_embeddings, intermediate_size)
        # self.down_proj: input(batch, max_position_embeddings, intermediate_size) @ w.T(intermediate_size, hidden_size) -> out(batch, max_position_embeddings, hidden_size)
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DecoderLayer(nn.Module):
    """GPT Transformer 块（Pre-LayerNorm 结构）。

    结构（Pre-Norm，归一化在子层之前）：
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    原始 Transformer 论文用 Post-Norm（归一化在残差之后），
    GPT-2 改为 Pre-Norm，训练更稳定，梯度流更顺畅。
    LLaMA 继承了 Pre-Norm，但把 LayerNorm 换成了 RMSNorm。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        
        self.input_layernorm = LayerNorm(cfg.hidden_size)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = LayerNorm(cfg.hidden_size)
        self.mlp = MLP(cfg)
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力子层
        # self.input_layernorm: input(batch, max_position_embeddings, hidden_size) -> # output(batch, max_position_embeddings, hidden_size)
        # self.self_attn: input(batch, max_position_embeddings, hidden_size) -> # output(batch, max_position_embeddings, hidden_size)
        x = x + self.dropout(self.self_attn(self.input_layernorm(x)))
        
        # FFN 子层
        # self.post_attention_layernorm(x): input(batch, max_position_embeddings, hidden_size) -> # output(batch, max_position_embeddings, hidden_size)
        # self.mlp: input(batch, max_position_embeddings, hidden_size) -> # output(batch, max_position_embeddings, hidden_size)
        x = x + self.dropout(self.mlp(self.post_attention_layernorm(x)))
        
        # output(batch, max_position_embeddings, hidden_size)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 完整模型
# ─────────────────────────────────────────────────────────────────────────────

class GPTModel(nn.Module):
    """GPT 完整模型。

    输入 token id → 词嵌入 + 位置嵌入 → N × GPTBlock → LayerNorm → 线性输出层
    注意训练的时候使用因果掩码矩阵构建并行训练的环境，比如输入数据是token_ids = [0, 1, 2, 3] shape为(batch, max_position_embeddings) = (1, 4)，
    构造的ground_truth是[1, 2, 3, 4]，shape为(batch, max_position_embeddings)。在实际训练中，因果注意力掩码会构造每一个当前token后面的注意力分数为-inf，
    最终设置每一个token及前面token的注意力分为之和为1，之后的token的注意力分数为0，这样在注意力模块输出中，shape为(batch, max_position_embeddings, hidden_size)
    每一个当前token对应的embedding_dim都包含了其与之前所有token的加权求和，然后再进入MLP进行知识的存储（数据维度缩放并还原），
    最后输出shape为(batch, max_position_embeddings, hidden_size)，最后输入到out_head中进行next_token预测，输出shape为(batch, max_position_embeddings, vocab_size)
    
    在推理的时候则是逐步进行解码，用户输入[0]，然后前向传播计算当前token的query, key, value, 计算当前token的加权求和hidden_size，然后经过
    MLP进行知识索引，然后经过out_head层进行next_token预测。在预测完之后进行token_ids的拼接得到[0, 1]，然后继续前向传播，计算1这个token_id对应的
    query, key, value，然后使用该query和拼接后的key(拼接之前计算0的key和当前key)去计算注意力分数，然后和拼接后的value(拼接之前计算0的value和当前value)
    去计算注意力层的最终输出，然后继续MLP层知识索引和out_head层的next_token预测，依次类推进行推理计算。

    位置编码：GPT 使用可学习的绝对位置嵌入（nn.Embedding）。
    LLaMA 改用 RoPE（旋转位置编码），不需要单独的位置嵌入层，
    而是在每层 Attention 计算 Q/K 时动态注入位置信息，外推性更好。

    输出层与词嵌入权重共享（weight tying）：
        out_head.weight ← token_embedding.weight
        可将参数量从 163M 降至 124M，但实践中分开训练效果往往更好。
        此处不做权重绑定，保持与现代 LLM 一致的做法。
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens    = nn.Embedding(cfg.vocab_size, cfg.hidden_size) # w(vocab_size, hidden_size)
        self.embed_pos_tokens = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size) # w(max_position_embeddings, hidden_size)
        self.drop_embedding     = nn.Dropout(cfg.embd_pdrop) 

        # self.layers   = nn.Sequential(*[DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm     = LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_epsilon)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (batch, seq_len) token id 序列
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, max_position_embeddings = idx.shape
        assert max_position_embeddings <= self.cfg.max_position_embeddings, \
            f"序列长度 {max_position_embeddings} 超过 context_length {self.cfg.max_position_embeddings}"


        # input(batch, seq_len, vocab_size) @ w(vocab_size, embedding_dim) -> out(batch, seq_len, embedding_dim)
        tok_emb = self.embed_tokens(idx) 
        
        # input(batch, max_position_embeddings, max_position_embeddings) @ w(max_position_embeddings, hidden_size) -> out(batch, max_position_embeddings, hidden_size)
        pos_emb = self.embed_pos_tokens(torch.arange(max_position_embeddings, device=idx.device))  
        
        x = self.drop_embedding(tok_emb + pos_emb) # (batch, max_position_embeddings, hidden_size)
        for layer in self.layers:
            x = layer(x) # (batch, max_position_embeddings, hidden_size)
        x = self.norm(x) # (batch, max_position_embeddings, hidden_size)
        
        return x


class GPTForCausalLM(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.model = GPTModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False) # w(vocab_size, embedding_dim)
        
        if cfg.tie_word_embeddings:
          self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # shape(batch, max_position_embeddings)
        hidden_states = self.model(idx)
        # input(batch, max_position_embeddings, hidden_size) @ w.T(hidden_size, vocab_size) -> (batch, max_position_embeddings, vocab_size)
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """自回归文本生成。

        Args:
            idx:            (batch, seq_len) 初始 token 序列
            max_new_tokens: 最多生成多少个新 token
            temperature:    温度，<1 更保守，>1 更随机
            top_k:          只从概率最高的 top_k 个 token 中采样（None 则不限制）
        """
        for _ in range(max_new_tokens):
            # 裁剪到 context_length，避免位置编码越界
            # input(1, 3)
            idx_cond = idx[:, -self.cfg.max_position_embeddings:]
            
            # input(1, 3) -> out(1, 3, vocab_size)
            logits = self(idx_cond)
            
            # input(1, 3, vocab_size) -> out(1, vocab_size)
            """
            “农业 是 重要 的”，token id 为 [10, 20, 30, 40]
            第1步: 输入 [10]→ 输出 (1,1,vocab_size) → 取最后位置 → argmax=20(是)
            第2步: 输入 [10,20]       → 输出 (1,2,vocab_size) → 取最后位置 → argmax=30(重要)
            第3步: 输入 [10,20,30]    → 输出 (1,3,vocab_size) → 取最后位置 → argmax=40(的)
            第4步: 输入 [10,20,30,40] → 输出 (1,4,vocab_size) → 取最后位置 → argmax=50(<end>)

            第4步处理 [10,20,30,40] 时：
            - 10 的 K、V 已经算过3次了
            - 20 的 K、V 已经算过2次了
            - 30 的 K、V 已经算过1次了
            K 和 V 都是由输入 x 线性变换得到的。对于已经出现过的 token，它的位置不变、内容不变，所以它的 K 和 V 永远不会变。
            因此kv cache可以显著减少推理过程中的显存占用。
            也就是在预测完第index=2的token之后（根据第index=1的token embedding去预测index=2的），需要预测第index=3的token时候，
            需要计算第index=2的query, key和value，然后使用新计算的index=2的query和index=0, 1的key和index=2的key拼接之后的key
            去计算注意力分数，然后和拼接后的value去计算得到注意力层最终的输出。这是推理时候的显存优化，在训练的时候是并行计算的，所以
            不涉及这个问题。
            """
            
            logits = logits[:, -1, :]

            if top_k is not None:
                # 将 top_k 之外的 logits 设为 -inf
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            logits = logits / temperature
            
            # (1, vocab_size)
            probs = F.softmax(logits, dim=-1)

            # (1, 1)
            # multinomial -> argmax when temperature=0.0
            # multinomial -> random sampling when temperature>0.0
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # (1, 3) cat (1, 1) -> (1, 4)
            idx = torch.cat([idx, idx_next], dim=-1)

        # (1, 13)
        return idx
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "124M",
        source: str = "hf",
        model_dir: str = "/home/weiyutao/ai/continual_learning/gpt2",
        map_location: Optional[Union[str, torch.device]] = "cpu"
    ) -> "GPTModel":
        """加载 OpenAI 官方 GPT-2 预训练权重。

        Args:
            model_name_or_path: 模型名称或路径
            source: "hf" | "openai" | "pt"
            map_location: 设备映射位置，默认 "cpu"。可以是 "cuda", "cuda:0", torch.device 对象等
        """
        if source == "hf":
            return cls._from_hf(model_name_or_path, map_location)
        elif source == "openai":
            return cls._from_openai(model_name_or_path, model_dir, map_location)
        elif source == "pt":
            return cls._from_pt(model_name_or_path, map_location)
        
    @classmethod
    def _from_hf(cls, model_name_or_path: str, map_location: Optional[Union[str, torch.device]] = "cpu"):
        
        """
        # gpt2 config for huggingface model config.
        {
            'activation_function': 'gelu_new', 
            'architectures': ['GPT2LMHeadModel'], 
            'attn_pdrop': 0.1, 
            'bos_token_id': 50256, 
            'embd_pdrop': 0.1, 
            'eos_token_id': 50256, 
            'initializer_range': 0.02, 
            'layer_norm_epsilon': 1e-05, 
            'model_type': 'gpt2', 
            'n_ctx': 1024, 
            'n_embd': 768, 
            'n_head': 12, 
            'n_layer': 12, 
            'n_positions': 1024, 
            'resid_pdrop': 0.1, 
            'summary_activation': None, 
            'summary_first_dropout': 0.1, 
            'summary_proj_to_labels': True, 
            'summary_type': 'cls_index', 
            'summary_use_proj': True, 
            'task_specific_params': {'text-generation': {'do_sample': True, 'max_length': 50}}, 
            'vocab_size': 50257
        }
        
        从Hugginface safetensor格式加载
              
        Args:
            model_name_or_path: 
                - 规格名称: "124M", "355M", "774M", "1558M"
                - 或模型目录路径: "/path/to/gpt2/124M"
        """
        config_path = os.path.join(model_name_or_path, "config.json")
        safetensors_path = os.path.join(model_name_or_path, "model.safetensors")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        if not os.path.exists(safetensors_path):
            raise FileNotFoundError(f"Safetensors file not found at {safetensors_path}")

        # 2. 读取 Hugging Face 模型的 config
        with open(config_path, "r", encoding="utf-8") as f:
            hf_config_dict = json.load(f)
            
        # logger.info(hf_config_dict)

        # config 映射
        config = GPTConfig(
            vocab_size=hf_config_dict["vocab_size"],
            hidden_size=hf_config_dict["n_embd"],
            intermediate_size=hf_config_dict["n_embd"] * 4,
            max_position_embeddings=hf_config_dict["n_positions"],
            num_attention_heads=hf_config_dict["n_head"],
            num_key_value_heads=hf_config_dict["n_head"],
            num_hidden_layers=hf_config_dict["n_layer"],
            layer_norm_epsilon=hf_config_dict["layer_norm_epsilon"],
            embd_pdrop=hf_config_dict["embd_pdrop"],
            resid_pdrop=hf_config_dict["resid_pdrop"],
            attn_pdrop=hf_config_dict["attn_pdrop"],
            activation_function=hf_config_dict["activation_function"],
            qkv_bias=True,
        )
        
        # 加载huggingface权重
        device_str = str(map_location) if map_location else "cpu"
        hf_state_dict = load_file(safetensors_path, device=device_str)
        # logger.info(print_state_dict_keys(hf_state_dict))

        # 转换为自定义格式
        state_dict = convert_safetensors_to_custom(hf_state_dict, config.num_hidden_layers, config.tie_word_embeddings)

        # 创建模型并加载权重
        model = cls(config)
        model.load_state_dict(state_dict)
        model.to(map_location)
        
        return model

    @classmethod
    def _from_openai(
        cls,
        model_name_or_path: str,
        models_dir: str = "/home/weiyutao/ai/continual_learning/gpt2",
        map_location: Optional[Union[str, torch.device]] = "cpu"
    ):
        """从OpenAI TensorFlow格式加载
      
        Args:
            model_name_or_path: 
                - 规格名称: "124M", "355M", "774M", "1558M"
                - 或模型目录路径: "/path/to/gpt2/124M"
            models_dir: 当使用规格名称时的缓存目录
        
        """
        from model.gpt2.gpt_download import download_and_load_gpt2
        
        # 规格名称 → 配置映射
        size_map = {
            "124M":  GPTConfig.gpt2_small,
            "355M":  GPTConfig.gpt2_medium,
            "774M":  GPTConfig.gpt2_large,
            "1558M": GPTConfig.gpt2_xl,
        }
        
        # ── 情况1: 传入的是规格名称（如 "124M"）──────────────
        if model_name_or_path in size_map:
            # 从预设配置获取
            config = size_map[model_name_or_path]()
            config.qkv_bias = True

            # 下载并加载权重
            _, params = download_and_load_gpt2(model_name_or_path, models_dir)
        
        # ── 情况2: 传入的是模型路径 ──────────────────────────
        else:
            model_path = model_name_or_path

            # 2.1 读取 hparams.json 获取配置
            config_file = os.path.join(model_path, "hparams.json")
            if not os.path.exists(config_file):
                raise FileNotFoundError(
                    f"配置文件不存在: {config_file}\n"
                    f"请确保 {model_path} 是OpenAI格式的模型目录"
                )

            with open(config_file) as f:
                hparams = json.load(f)

            # 2.2 将hparams转换为GPTConfig
            """
            # hparams config json
            {
                "n_vocab": 50257,
                "n_ctx": 1024,
                "n_embd": 768,
                "n_head": 12,
                "n_layer": 12
            }
            """
            config = GPTConfig(
                vocab_size=hparams["n_vocab"],
                hidden_size=hparams["n_embd"],
                intermediate_size=hparams["n_embd"] * 4,
                num_hidden_layers=hparams["n_layer"],
                num_attention_heads=hparams["n_head"],
                max_position_embeddings=hparams["n_ctx"],
                qkv_bias=True,
            )

            # 2.3 加载checkpoint权重
            from model.gpt2.gpt_download import load_gpt2_params_from_tf_ckpt
            params = load_gpt2_params_from_tf_ckpt(model_path, hparams)

        # ── 通用: 转换并加载 ────────────────────────────────
        state_dict = convert_openai_to_custom(params, config.num_hidden_layers)

        # 如果 map_location 不是 cpu，需要将 state_dict 中的 tensor 移动到目标设备
        if map_location is not None and map_location != "cpu":
            device = torch.device(map_location) if isinstance(map_location, str) else map_location
            state_dict = {k: v.to(device) for k, v in state_dict.items()}

        model = cls(config)
        model.load_state_dict(state_dict)
        model.to(map_location)
        return model

    @classmethod
    def _from_pt(cls, model_name_or_path: str, map_location: Optional[Union[str, torch.device]] = "cpu"):
        """从自定义格式加载

        Args:
            model_name_or_path: 模型路径
            map_location: 设备映射位置，默认 "cpu"

        支持格式：
        - 优先加载 model.safetensors
        - 兼容 pytorch_model.bin / .pt
        """
        import os
        import json
        from safetensors.torch import load_file

        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, "config.json")

            # 优先safetensors，fallback到bin
            safetensors_file = os.path.join(model_name_or_path, "model.safetensors")
            bin_file = os.path.join(model_name_or_path, "pytorch_model.bin")

            if os.path.exists(safetensors_file):
                weight_file = safetensors_file
                use_safetensors = True
            elif os.path.exists(bin_file):
                weight_file = bin_file
                use_safetensors = False
            else:
                raise FileNotFoundError(f"找不到权重文件: {safetensors_file} 或 {bin_file}")
        else:
            # 文件格式
            config_file = model_name_or_path.replace(".pt", "_config.json").replace(".bin", "_config.json").replace(".safetensors", "_config.json")
            weight_file = model_name_or_path
            use_safetensors = weight_file.endswith(".safetensors")

        # 读取配置
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        with open(config_file) as f:
            config_dict = json.load(f)
        config = GPTConfig(**config_dict)

        # 加载权重
        if use_safetensors:
            state_dict = load_file(weight_file)
        else:
            state_dict = torch.load(weight_file, map_location=map_location, weights_only=False)

        model = cls(config)
        model.load_state_dict(state_dict)
        model.to(map_location)
        return model

# ─────────────────────────────────────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    # 模型初始化测试
    """
    from config import SOURCE_DIR, OUT_DIR
    from tokenizers import Tokenizer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_pretrained("gpt2")
    cfg = GPTConfig.gpt2_small()
    torch.manual_seed(123)
    model = GPTForCausalLM(cfg)
    model.to(device)
    
    
    """
    # 加载权重测试
    """
    load_pretrained_mode = 3 # 1: scratch, 2: load_pretrained, 3: load_continual_learning
    if load_pretrained_mode == 2:
        # model = GPTForCausalLM.from_pretrained("124M", source="openai")
        # 注意map_location和.to(device)的区别，前者一般是初始化模型权重的时候加载，后者一般是在运行的时候加载
        # model = GPTForCausalLM.from_pretrained(model_name_or_path="124M", source="openai", model_dir=f"{SOURCE_DIR}/trf/gpt2", map_location=device)
        # model = GPTForCausalLM.from_pretrained(f"{SOURCE_DIR}/trf/gpt2/124M", source="openai", map_location=device)
        model = GPTForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/gpt2/124M", source="hf", map_location=device)
    elif load_pretrained_mode == 3:
        model = GPTForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260722", source="pt", map_location=device)
    
    
    """
    # 前向测试
    """
    texts = ["Every effort moves you", "Every day holds a"]
    batch = torch.stack([torch.tensor(tokenizer.encode(t).ids) for t in texts]).to(device)
    # forward test
    logits = model(batch)
    print(f"Input shape:  {batch.shape}")
    print(f"Output shape: {logits.shape}")  # (2, 4, 50257)
    
    
    """
    # 参数量测试
    """
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")  # ~163M（含 out_head）

    
    """
    # 生成测试
    """
    model.eval()
    prompt = torch.tensor(tokenizer.encode("Hello, I am").ids).unsqueeze(0).to(device) # (3,) -> (1, 3)
    out = model.generate(prompt, max_new_tokens=50, top_k=50, temperature=0.8) # (1, 3) -> (1, 13)
    print(f"\nGenerated: {tokenizer.decode(out[0].tolist())}") # (1, 13) -> (13,) -> list -> str
    
    

