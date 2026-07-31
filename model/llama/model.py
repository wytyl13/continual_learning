#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/24 11:06
@Author  : weiyutao
@File    : model.py

llama model architecture.
小组件不进行测试，重要组件测试验证维度和数值，完成后测试训练和推理。
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
import math
import inspect
import json
import os
from accelerate import init_empty_weights

from model.llama.config import LLamaConfig
from model.gpt2.model import GELU
from model.llama.weight_convert import convert_safetensors_to_custom

from utils.model_loader import resolve_checkpoint_files, build_auto_device_map, fast_load_weights
from utils.logger import get_logger
from utils.enums import LoadMode, ModelType


logger = get_logger(__name__)


class RMSNorm(nn.Module):
    """
    RMSNorm
    μ = 0
    σ² = 1/(d - 1) ⋅ Σ(xᵢ - μ)² 
    y = (x - μ) / (√σ²+ϵ) @ γ + β = x / (√σ²+ϵ) @ γ
    """
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def _norm(self, x):
        # σ² = 1/(d - 1) ⋅ Σ(xᵢ - μ)² = 1/(d - 1) ⋅ Σ(xᵢ)² = x.pow(2).mean(-1), 
        # -1 means calculate the mean of last dimension what based on the hidden_size dim.
        # torch.rsqrt = 1 / torch.sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # x.float() means upcasting the x from float16 or bfloat16 to float32 to prevent numerical overflow.
        # type_as means downcasting the output from float32 to float16 or bfloat16
        output=  self._norm(x.float()).type_as(x)
        return output * self.weight
    

def precompute_freqs_cis(hidden_size: int, max_position_embeddings: int, theta: float = 10000.0):
    """
    cal the real part cosmθ, shape is (max_position_embeddings, hidden_size // 2)
    cal the imaginary part sinmθ, shape is (max_position_embeddings, hidden_size // 2)
    """
    
    # 1/θ^(2i/d), shape is (hidden_size // 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, hidden_size, 2)[: (hidden_size // 2)].float() / hidden_size))

    # shape(max_position_embeddings,) what is [0, 1, 2, ...] means the the absolutely position for each token.
    t = torch.arange(max_position_embeddings, device=freqs.device)

    # outer(t, freqs),shape is (max_position_embeddings, hidden_size // 2)
    # 在第t个位置的token，其对应的嵌入向量的第i个位置二维切片，需要旋转的总角度
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs) # real part: cosmθ
    freqs_sin = torch.sin(freqs) # imaginary part: sinmθ
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    reshape freqs_cis from (max_position_embeddings, hidden_size // 2) to (1, max_position_embedding, 1, hidden_size // 2)
    方便后续与shape为(batch, max_position_embeddings, num_heads, hidden_size // 2)的x进行广播
    """
    ndim = x.ndim # the number of dimensions
    assert 0 <= 1 <= ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] # [1, max_position_embedding, 1, hidden_size // 2]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    位置编码（详细可见docs/rope.txt）：
        自注意力机制需要额外编码每个词的位置信息。
    整数（1, 2, 3...）或者按比例缩放（把句子长度压缩到0-1之间）的位置编码：
        前者数值过大影响了词语本身的意思，后者在不同长度的句子中相同的步长代表不同位置。
    Transformer中的位置编码：
        亟需一种编码方式（数值要有界，对任何长度的序列都通用），于是找到了正弦波和余弦波。想象
        一下二进制，不同位的数值变化频率不同。最高位（最左侧）变化最慢，最低位（最右侧）变化极快。
        原版transformer的位置编码本质就是一个把0和1变成了-1到1之间平滑曲线的浮点数二进制计数器。

        PE(pos, 2i) = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        pos表示词绝对位置，i是维度的索引（0-63），d是总维度128。10000^(2i/d)就是来控制转速的。
        在低维度区（i=0），10000^(2i/d)=1，随着pos的微小变化，sin(pos)会剧烈震荡，像秒针，相当于二进制的低位。
        可以理解为嵌入128维度的前半部分，他随着pos位置的变化而产生幅度较大的震荡。
        在高维区（i=63），10000^(2i/d)接近10000，随着pos的变化，即便从1到100,sin(pos/10000)也不会产生剧烈震荡，
        类似于时针，相当于二进制的高位。可以理解为128维度的后半部分，它随着pos位置的变化产生较小的震荡。
        为什么要sin、cos成对出现（和差化积）？
        sin(A + B) = sin A * cos B + cos A * sin B
        cos(A + B) = cos A * cos B - sin A * sin B
        
        假设PE(pos) = [sin(pos, 0), cos(pos, 1), sin(pos, 2), cos(pos, 3), ...].T
            PE⁽ⁱ⁾(pos) = [sin(pos, wi), cos(pos, wi)].T, PE 向量在第i组的切片，是一对 i=0-63，是相邻的奇偶对
            比如PE⁽0⁾(pos) = [sin(pos, 0), cos(pos, 1)].T
            
            PE⁽ⁱ⁾(pos + k) = [sin(pos + k, wi), cos(pos + k, wi)].T，假设w是10000^(2i/d)，所以最终
            
            PE⁽ⁱ⁾(pos) = [sin(wi * pos), cos(wi * pos)].T
            PE⁽ⁱ⁾(pos + k) = [sin(wi * (pos + k)), cos(wi * (pos + k)] = [sin(wi*pos + wi*k), cos(wi*pos + wi*k)].T
            = [sin(wi*pos) * cos(wi*k) + cos(wi*pos) * sin(wi*k), cos(w*pos) * cos(wi*k) - sin(wi*pos) * sin(wi*k)].T
            = [[cos(wi*k), sin(wi*k)], [-sin(wi*k), cos(wi*k)]].T @ [sin(wi*pos), cos(wi*pos)].T
            PE⁽ⁱ⁾(pos + k) = [[cos(wi*k), sin(wi*k)], [-sin(wi*k), cos(wi*k)]].T @ PE⁽ⁱ⁾(pos)
            中间的那个就是转换矩阵M(k) = [[cos(wi*k), sin(wi*k)], [-sin(wi*k), cos(wi*k)]].T，和pos无关，所以可以使得模型可以学到相对位置。
            如果模型想表达：我想重点关注距离我k个位置的那个词，模型只要把自己的权重矩阵w学习调整成刚才那个只和k有关的转换矩阵就行了。
        Q = xq + pq, K = xk + pk, Q.T @ K =  (xq + pq).T @ (xk + pk) = (xq.T + pq.T) @ (xk + pk) 
        = xq.T @ xk + xq.T @ pk + pq.T @ xk + pq.T @ pk = 内容 @ 内容 + 内容 @ 位置 + 位置 @ 内容 + 位置 @ 位置
        在计算注意力分数的时候，因为位置编码的直接相加导致了严重的信息耦合，可以发现（内容 @ 位置 + 位置 @ 内容 + 位置 @ 位置）这三个里面有
        相对位置信息，但是和原始内容耦合在一起，导致模型学习起来吃力，且很难泛化到训练时没有见过的更长序列（因为训练数据的序列长度决定了
        位置编码pos的上限，在推理的时候遇到一个超长序列会导致没有见过该pos位置的信息）。
    RoPE破局思路：为每个位置m的向量xm添加位置信息，方法是将其旋转角度mθ，其中θ是固定参数
        欧拉公式：eⁱᶿ = cosθ + i̇⋅sinθ
        二维坐标系里面，(x, y)表示一个点。复平面里，横轴是实数轴，纵轴是虚数轴，复数z = a + b⋅i 对应的复平面点(a, b)
        在一个单位圆上（半径为1），如果从横轴正方向逆时针走θ角度，这个点的坐标为(cosθ, sinθ)，或者可以表述为一个长度为1、角度为θ的向量
        也就是说：复数cosθ + i⋅sinθ，对应的复平面点为(cosθ, sinθ)，其中i为复数轴，i² = -1
        e的本质：持续不断，不间断地作用。假设在银行存了 1 块钱，年利率是 100%。按年结息1年后会得到2块钱，按月结息利滚利一年后得到2.61块，按秒结息一年后得到2.718块，就是e的值
        eˣ表示顺着当前指的方向，以x的速度不断地增长。虚数i的本质，假设在实数轴1的位置，对应的复平面是1+0i，i对应的复平面是0+1i，相乘得到(1+0i)*(0+1i)=0+1i+0+0=0+1i
        所以得到1乘以i得到0+1i，其实就是i，意思就是旋转90度。然后在这个基础上再乘以i，i^2=-1，意味着继续旋转90度。所以在复数的世界里，乘上虚数i就等于发出一个向右侧转弯的
        指令，就是沿着单位圆旋逆时针转90度。
        eⁱᶿ表示以θ为时间/速度，持续不断地沿着单位圆逆时针旋转，持续下去会绘制出一个完美的单位圆。所以eⁱᶿ描述的是一个绘制单位圆的动作，而cosθ+i⋅sinθ是一个静态结果，描述的
        是逆时针旋转theta角度之后当前所处的单位圆上的坐标点为(cosθ, sinθ)。
        
        假设我们现在拿到了query和key，在进行注意力分数计算之前，我们要添加位置编码。给query两两分组得到 hidden_size//2 组坐标点。每个
        坐标点可以表述为复数形式：query: (a_q+i⋅b_q), key: (aₖ+i⋅bₖ) 
        分别给query添加位置m，key添加位置n，就是分别旋转m角度和n角度：
        z_q = (a_q+i*b_q)*eⁱᵐᶿ = (a_q+i*b_q)*(cosmθ + i⋅sinmθ) = a_q*cosmθ + a_q*i⋅sinmθ + i*b_q*cosmθ - b_q*sinmθ
                          = (a_q*cosmθ - b_q*sinmθ) + i*(a_q*sinmθ + b_q*cosmθ)
        z_k = (aₖ+i*bₖ) ⋅ eⁱⁿᶿ = (aₖ+i*bₖ)*(cosnθ+i*sinnθ) = aₖ*cosnθ + aₖ*i⋅sinnθ + i⋅bₖ*cosnθ - bₖ*sinnθ
                          = (aₖ*cosnθ - bₖ*sinnθ) + i*(aₖ*sinnθ + bₖ*cosnθ)
                          
        以上已经得到两个二维坐标的复数表示。现在计算两个复数的内积 = z_q*x_k*(*表示共轭)
        z_k = (aₖ+i*bₖ)*eⁱⁿᶿ
        z_k* = (aₖ-i*bₖ)*e⁻ⁱⁿᶿ(所有的i变成-i)
        z_q * z_k* = (a_q+i*b_q)*eⁱᵐᶿ * (aₖ-i*bₖ) ⋅ e⁻ⁱⁿᶿ = (a_q+i*b_q)*(aₖ-i*bₖ)*eⁱᵐᶿ*e⁻ⁱⁿᶿ
                = [(a_q*a_k-i*a_q*b_k+i*b_q*a_k+b_q*b_k)]*eⁱᵐᶿ*e⁻ⁱⁿᶿ = [(a_q*a_k+b_q*b_k)+i(b_q*a_k-a_q*b_k)]*eⁱᵐᶿ*e⁻ⁱⁿᶿ
                = [(a_q*a_k+b_q*b_k)+i(b_q*a_k-a_q*b_k)]*eⁱ⁽ᵐ⁻ⁿ⁾ᶿ
                = [(a_q*a_k+b_q*b_k)+i(b_q*a_k-a_q*b_k)]*(cos((m-n)θ+i*sin((m-n)θ))
                = [(a_q*a_k+b_q*b_k)*cos((m-n)θ] + [(a_q*a_k+b_q*b_k)*i*sin((m-n)θ)] + [i(b_q*a_k-a_q*b_k)*cos((m-n)θ] - [(b_q*a_k-a_q*b_k)*sin((m-n)θ)]
                = 实数                            + 虚数                              + 虚数                            - 实数
                = 取实部
                = (a_q*a_k+b_q*b_k)*cos((m-n)θ - (b_q*a_k-a_q*b_k)*sin((m-n)θ
        这种结合使得模型既能考虑内容相似性，又能考虑位置关系。且公式里只剩下了相对位置 (m-n)，绝对位置 m 和 n 被完美抵消了。
    """                                                                                                                                     
    # q.float() means casting the q from BF16 to float32.
    # reshape q from (batch, max_position_embeddings, num_attention_heads, head_dim) to 
    # (batch, max_position_embeddings, num_attention_heads, head_dim//2, 2), split rule is from [x0, x1, x2, x3] to [[x0, x1], [x2, x3]]
    # and unbind(-1) to split two shape(batch, max_position_embeddings, num_attention_heads, head_dim//2), split rule is from [[x0, x1], [x2, x3]] to [x0, x2], [x1, x3]
    # what is correspond to the transformer position embedding: PE(pos, 2i), PE(po, 2i+1)
    # q_r means real [0, 2, 4, ..., hidden_size // 2], q_i means imaginary [1, 3, 5, ..., hidden_size // 2 + 1].
    
    # shape(batch, max_position_embeddings, num_attention_heads, hidden_size // 2), shape(batch, max_position_embeddings, num_attention_heads, hidden_size // 2)
    q_r, q_i = q.float().reshape(q.shape[:-1] + (-1, 2)).unbind(-1) 

    # shape(batch, max_position_embeddings, num_attention_heads, hidden_size // 2), shape(batch, max_position_embeddings, num_attention_heads, hidden_size // 2)
    k_r, k_i = k.float().reshape(k.shape[:-1] + (-1, 2)).unbind(-1)
    
    # reshape freqs_cos and freqs_sin for broadcasting.
    freqs_cos = reshape_for_broadcast(freqs_cos, q_r) # shape(1, max_position_embeddings, 1, hidden_size // 2)
    freqs_sin = reshape_for_broadcast(freqs_sin, q_r) # shape(1, max_position_embeddings, 1, hidden_size // 2)

    # The final rotary reuslt for query and key.
    # (a_q+i*b_q)*eⁱᵐᶿ = (a_q*cosmθ - b_q*sinmθ) + i*(a_q*sinmθ + b_q*cosmθ), a_q is real part, b_q is imaginary.
    # (aₖ+i*bₖ) ⋅ eⁱⁿᶿ = (aₖ*cosnθ - bₖ*sinnθ) + i*(aₖ*sinnθ + bₖ*cosnθ), a_k is real part, b_k is imaginary.
    # apply rotation using real numbers for real part and imaginary part.
    # after rotation, the (a_q*cosmθ - b_q*sinmθ) is real part, (a_q*sinmθ + b_q*cosmθ) is imaginary.
    # (a_q+i*b_q)*eⁱᵐᶿ = (q_r*cosmθ - q_i*sinmθ) + i*(q_r*sinmθ + q_i*cosmθ), (q_r*cosmθ - q_i*sinmθ) is real part = q_out_r, (q_r*sinmθ + q_i*cosmθ) is imaginary part = q_out_i.
    # (aₖ+i*bₖ) ⋅ eⁱⁿᶿ = (k_r*cosnθ - k_i*sinnθ) + i*(k_r*sinmθ + k_i*cosmθ), (k_r*cosnθ - k_i*sinnθ) is real part = k_out_r, (k_r*sinmθ + k_i*cosmθ) is imaginary part = k_ouy_i.
    q_out_r = q_r * freqs_cos - q_i * freqs_sin # shape(batch, max_position_embeddings, num_attention_heads, hidden_size // 2)
    q_out_i = q_r * freqs_sin + q_i * freqs_cos
    k_out_r = k_r * freqs_cos - k_i * freqs_sin
    k_out_i = k_r * freqs_sin + k_i * freqs_cos
    
    # stack real part and imaginary part to shape(batch_size, max_position_embeddings, num_attention_heads, hidden_size // 2, 2) 
    # and flatten from the third dimension hidden_size//2, the final shape is (batch_size, max_position_embeddings, num_attention_heads, hidden_size)
    q_out = torch.stack([q_out_r, q_out_i], dim=-1).flatten(3)
    k_out = torch.stack([k_out_r, k_out_i], dim=-1).flatten(3)

    return q_out.type_as(q), k_out.type_as(k) # shape(batch_size, max_position_embeddings, num_attention_heads, hidden_size)
    

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    input x: key and value tensor.
    reshap x from shape(batch, max_position_embeddings, num_key_value_heads, hidden_size)
    to shape(batch, max_position_embeddings, num_key_value_heads * n_rep, hidden_size)
    """
    batch, max_position_embeddings, num_key_value_heads, hidden_size = x.shape
    if n_rep == 1:
        return x
    
    return (
        x[:, :, :, None, :]
        .expand(batch, max_position_embeddings, num_key_value_heads, n_rep, hidden_size)
        .reshape(batch, max_position_embeddings, num_key_value_heads * n_rep, hidden_size)
    )


class Attention(nn.Module):
    """
    支持 MHA (Multi-Head Attention) 和 GQA (Grouped Query Attention)。
    集成了 RoPE (Rotary Position Embedding) 和 Flash Attention。
    """
    
    def __init__(self, cfg: LLamaConfig):
        super().__init__()
        assert cfg.hidden_size % cfg.num_attention_heads == 0, \
            "hidden_size 必须能被 num_attention_heads 整除"
        
        self.hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        
        # llama1 7B, 13B, 33B, 65B 均使用MHA
        # LLama2 34B, 70B 首次引入了GQA
        # LLama3 全面使用 GQA
        self.num_key_value_heads = self.num_attention_heads if cfg.num_key_value_heads is None else cfg.num_key_value_heads # MHA or GQA
        
        # the numbers of key, value repeat.
        self.n_rep = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        # Q, K, V 和输出投影矩阵
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=cfg.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=cfg.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=cfg.qkv_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=cfg.qkv_bias)

        # Dropout 设置
        self.dropout = nn.Dropout(cfg.attn_pdrop)
        
        self.flash = hasattr(nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            logger.warning(f"using slow attention. Flash Attention requires Pytorch >= 2.0")
            # (1, 1, max_position_embeddings, max_position_embeddings), fill with -inf
            mask = torch.full((1, 1, cfg.max_position_embeddings, cfg.max_position_embeddings), float("-inf"))
            # 把掩码矩阵的左下角及对角线（diagnoal=1）全部置为0
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask, persistent=False)
            
    def forward(
        self, 
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
    ) -> torch.Tensor:
        batch, max_position_embeddings, _ = x.shape

        # 线性投影
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(batch, max_position_embeddings, self.num_attention_heads, self.head_dim)
        k = k.view(batch, max_position_embeddings, self.num_key_value_heads, self.head_dim)
        v = v.view(batch, max_position_embeddings, self.num_key_value_heads, self.head_dim)

        # RoPE（仅对Q和K，因为位置编码仅仅是为了计算注意力权重）
        q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values.
        k = repeat_kv(k, self.n_rep) # (batch, max_position_embeddings, num_key_value_heads * n_rep, head_dim)
        v = repeat_kv(v, self.n_rep) # (batch, max_position_embeddings, num_key_value_heads * n_rep, head_dim)

        # transpose(1, 2)
        q = q.transpose(1, 2) # shape(batch, num_key_value_heads * n_rep, max_position_embeddings, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # flash implementation
        if self.flash:
            output = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=True)
        else:
            # manual implementation.
            # q shape(batch, num_key_value_heads * n_rep, max_position_embeddings, head_dim) @
            # k.transpose(2, 3) shape(batch, num_key_value_heads * n_rep, head_dim, max_position_embeddings)
            # output shape(batch, num_key_value_heads * n_rep, max_position_embeddings, max_position_embeddings)
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            # shape(batch, num_key_value_heads, max_position_embeddings, max_position_embeddings)
            scores = scores + self.mask[:, :, :max_position_embeddings, :max_position_embeddings]
            scores = nn.functional.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.dropout(scores)
            
            # scores shape(batch, num_key_value_heads * n_rep, max_position_embeddings, max_position_embeddings) @ 
            # v shape(batch, num_key_value_heads * n_rep, max_position_embeddings, head_dim)
            # out shape = (batch, num_key_value_heads * n_rep, max_position_embeddings, head_dim)
            output = torch.matmul(scores, v)
        # restore time as batch dimension and concat heads
        # transpose(1, 2): shape(batch, max_position_embeddings, num_key_value_heads * n_rep, head_dim)
        # view: shape(batch, max_position_embeddings, num_key_value_heads * n_rep * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch, max_position_embeddings, -1)
        
        # Apply final projection.
        output = self.o_proj(output)
        return output
        

class SwiGLU(nn.Module):
    """
    SwiGLU
    input: shape(batch, max_position_embeddings, hidden_size) 
    bias=false
    SILU(x) = x * sigmoid(x) = x / 1 + e⁻ˣ
    和GELU相比，都是在x>0时近似y=x，在x<0时有一个微笑的复制，然后在极端负值时趋近于0.
    他俩的非线性能力几乎一样，但是SILU计算更简单，且在配置门控结构（GLU）时，SILU的平滑效果更好。
    门控？都是非线性过滤，但是GELU是被动过滤，SwiGLU中的门控是主动过滤。经过SILU激活后的门控
    生成了一个掩码，再结合up_proj路的无情特征提取，保留可信特征，丢弃不可信的。
    总之，GELU是我提取了什么特征就输出什么特征。SwiGLU是我提取了一堆特征，然后我再自己决定哪些特征值得输出。
    LLaMA 改用 SwiGLU：Linear(d → 8/3*d) * sigmoid gate → Linear(8/3*d → d)
    三个矩阵而非两个，但参数量通过调整维度（详见config配置）保持近似等效。
    是一个并行的双支路结构，将特征转换和特征过滤解耦，在降维前使用逐元素相乘进行融合。
    """
    def __init__(self, cfg: LLamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False) 
        
        if cfg.activation_function == "gelu" or cfg.activation_function == "gelu_new":
            self.act_fn = GELU(cfg.activation_function) 
        elif cfg.activation_function == "silu" or cfg.activation_function == "swish":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {cfg.activation_function}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate_proj and active function, up_proj
        # down_proj
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    

class DecoderLayer(nn.Module):
    def __init__(self, cfg: LLamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.layer_norm_epsilon)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.layer_norm_epsilon)
        self.mlp = SwiGLU(cfg)
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(
        self, 
        x: torch.Tensor, 
        freqs_cos: torch.Tensor, 
        freqs_sin: torch.Tensor
    ) -> torch.Tensor:
        # attention layer. layer_norm -> self_attn -> dropout -> short connect.
        x = x + self.dropout(self.self_attn(self.input_layernorm(x), freqs_cos, freqs_sin))
        
        # feed forward layer. post_attn_layernorm -> SwiGLU -> dropout -> short connect.
        return x + self.dropout(self.mlp(self.post_attention_layernorm(x)))


class LLamaModel(nn.Module):
    """
    LLama Model.
    """
    def __init__(self, cfg: LLamaConfig):
        super().__init__()
        self.cfg = cfg
        
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.drop_embeddings = nn.Dropout(cfg.embd_pdrop)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.layer_norm_epsilon)

        # RoPE，注册为buffer，不参与梯度
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, cfg.max_position_embeddings)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


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
                    
        tok_emb = self.embed_tokens(idx)
        x = self.drop_embeddings(tok_emb)
        freqs_cos = self.freqs_cos[:max_position_embeddings]
        freqs_sin = self.freqs_sin[:max_position_embeddings]
        for layer in self.layers:
            # x = layer(x, freqs_cos, freqs_sin)
            layer_device = next(layer.parameters()).device
            x = layer(x.to(layer_device), freqs_cos.to(layer_device), freqs_sin.to(layer_device))
        x = x.to(next(self.norm.parameters()).device)
        x = self.norm(x)
        return x


class LLamaForCausalLM(nn.Module):
    def __init__(self, cfg: LLamaConfig):
        super().__init__()
        self.cfg = cfg
        self.model = LLamaModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def tie_weights(self):
        """Re-tie lm_head ↔ embed_tokens after load_state_dict(assign=True).

        load_state_dict with assign=True replaces embed_tokens.weight with a
        new Parameter object, breaking the alias set in __init__.  This method
        restores the tie so that fast_load_weights / accelerate dispatch_model
        never see a stale meta tensor on lm_head.
        """
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # shape(batch, max_position_embeddings)
        hidden_state = self.model(idx)
        # batch, max_position_embedding, vocab_size
        # NOTE: do NOT manually .to() here — when device_map="auto" is used,
        # accelerate's AlignDevicesHook.pre_forward already moves inputs to the
        # correct execution device. Calling .to(lm_head.param.device) reads the
        # offloaded (meta) device and corrupts hidden_state into a meta tensor.
        return self.lm_head(hidden_state)


    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        source: str = "local",
        map_location: Optional[Union[str, torch.device]] = "cpu",
        device_map: Optional[Union[str, dict]] = None
    ) -> "LLamaForCausalLM":
        """加载预训练权重。

        Args:
            model_name_or_path: 模型名称或路径
            source: "local" | "hf"
                - "local": 本地训练的模型
                - "hf": HuggingFace官方下载的预训练权重
            map_location: 设备映射位置，默认 "cpu"
            device_map: "auto" 或自定义字典，多GPU推理时自动分配层
        """
        if source == "local":
            return cls._from_local(model_name_or_path, map_location, device_map)
        elif source == "hf":
            return cls._from_hf(model_name_or_path, map_location, device_map)
        else:
            raise ValueError(f"Unknown source: {source}. Must be 'local' or 'hf'")

    @classmethod
    def _from_hf(cls, model_name_or_path: str, map_location: Optional[Union[str, torch.device]] = "cpu", device_map: Optional[Union[str, dict]] = None):
        """从 HuggingFace 官方格式加载，按照权重数据格式加载，不设置自定义格式"""
        # 1. config
        with open(os.path.join(model_name_or_path, "config.json")) as f:
            config = LLamaConfig.from_dict(json.load(f))

        # 2. 解析权重文件
        shard_files, use_safetensors = resolve_checkpoint_files(model_name_or_path)

        with init_empty_weights():
            model = cls(config)
            
        if device_map == "auto":
            # 4. 推断多卡分配
            device_map = build_auto_device_map(model, no_split_classes=["DecoderLayer"])# 5. 逐shard加载，模型专属的convert逻辑作为 convert_fn 传入
        else:
            # 单设备：所有参数都去同一个设备，复用同一套加载逻辑
            device_map = {"": map_location}
        def convert_fn(shard):
            return convert_safetensors_to_custom(
                shard, config.num_hidden_layers,
                tie_word_embeddings=config.tie_word_embeddings,
                n_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
            )
        fast_load_weights(model, shard_files, use_safetensors, device_map, convert_fn)
        return model

    @classmethod
    def _from_local(cls, model_name_or_path: str, map_location: Optional[Union[str, torch.device]] = "cpu", device_map: Optional[Union[str, dict]] = None):
        """从本地训练保存的模型加载，按照权重数据格式加载，不设置自定义格式"""
        from utils.model_loader import resolve_checkpoint_files

        # 1. config
        config_file = os.path.join(model_name_or_path, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        with open(config_file) as f:
            config = LLamaConfig.from_dict(json.load(f))

        # 2. 解析权重文件（支持单文件和分片格式）
        shard_files, use_safetensors = resolve_checkpoint_files(model_name_or_path)

        # 3. convert_fn：本地格式通常已是自定义格式，但兼容 HF 格式保存的情况
        def convert_fn(shard):
            if any("rotary_emb.inv_freq" in k for k in shard):
                logger.info("检测到 HF 格式权重，执行转换...")
                return convert_safetensors_to_custom(
                    shard, config.num_hidden_layers,
                    tie_word_embeddings=config.tie_word_embeddings,
                    n_heads=config.num_attention_heads,
                    n_kv_heads=config.num_key_value_heads,
                )
            return shard  # 已是自定义格式，直接透传

        # 4. meta tensor 初始化
        with init_empty_weights():
            model = cls(config)

        # 5. device_map
        if device_map == "auto":
            device_map = build_auto_device_map(model, no_split_classes=["DecoderLayer"])
        else:
            device_map = {"": map_location}

        # 6. 加载（支持单文件和分片）
        fast_load_weights(model, shard_files, use_safetensors, device_map, convert_fn)
        return model

    @classmethod
    def from_empty(
        cls,
        cfg: "LLamaConfig",
        map_location: Optional[Union[str, torch.device]] = "cpu",
        device_map: Optional[Union[str, dict]] = None,
    ) -> "LLamaForCausalLM":
        """仅用于架构测试（不加载任何权重），支持 device_map='auto'。
        没有 checkpoint 可以参考 dtype
        Args:
            cfg: 模型配置
            map_location: 单设备模式下的目标设备，默认 "cpu"
            device_map: "auto" 表示多卡自动分配，否则走单设备路径
        """
        torch.set_default_dtype(torch.bfloat16)
        try:
            if device_map == "auto":
                with init_empty_weights():
                    model = cls(cfg)
                resolved_map = build_auto_device_map(model, no_split_classes=["DecoderLayer"])
                # 将 meta tensor 就地实例化到目标设备（值随机，仅验证架构维度）
                for mod_name, mod in model.named_modules():
                    tgt = resolved_map.get(mod_name, resolved_map.get("", "cpu"))
                    for param_name, p in mod.named_parameters(recurse=False):
                        if p.is_meta:
                            mod._parameters[param_name] = nn.Parameter(
                                torch.empty(p.shape, dtype=p.dtype, device=tgt),
                                requires_grad=p.requires_grad,
                            )
                    for buf_name, b in mod.named_buffers(recurse=False):
                        if b.is_meta:
                            mod._buffers[buf_name] = torch.empty(b.shape, dtype=b.dtype, device=tgt)
            else:
                with torch.device(map_location or "cpu"):
                    model = cls(cfg)
        finally:
            torch.set_default_dtype(torch.float32)
        return model

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        pad_token_id = pad_token_id or self.cfg.pad_token_id
        eos_token_id = eos_token_id or self.cfg.eos_token_id

        idx = input_ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_position_embeddings:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            # Upcast and sanitize before any masking — bfloat16 overflow can
            # produce NaN in the raw logits, which poisons topk and softmax.
            logits = logits.float()
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

            if top_k is not None and do_sample:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if temperature == 0.0 or not do_sample:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                probs = nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next.to(idx.device)), dim=1)

            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return idx



if __name__ == "__main__":
    """
    print("llama model architecture")
    # 模型初始化测试
    """
    from model.llama.config import LLamaConfig
    from config import SOURCE_DIR, OUT_DIR
    from tokenizers import Tokenizer
    load_pretrained_mode = LoadMode.CONTINUAL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/tiny_llama/tokenizer.json")
    tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/llama-2-7b/tokenizer.json")
    # tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/llama/llama-3-8b/tokenizer.json")
    # cfg = LLamaConfig.tiny_llama()
    # cfg = LLamaConfig.llama2_7b()
    cfg = LLamaConfig.llama3_8b()
    torch.manual_seed(123)

    """
    # 加载权重测试
    """
    if load_pretrained_mode == LoadMode.PRETRAINED:
        # model = LLamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/tiny_llama", source="hf", map_location=device)
        # model = LLamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/llama-2-7b", source="hf", map_location=device)
        model = LLamaForCausalLM.from_pretrained(f"{SOURCE_DIR}/hf/llama/llama-3-8b", source="hf", device_map="auto")
    elif load_pretrained_mode == LoadMode.CONTINUAL:
        # model = LLamaForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260730_tiny_llama", source="local", map_location=device)
        # model = LLamaForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260730_llama2_7b", source="local", map_location=device)
        model = LLamaForCausalLM.from_pretrained(f"{OUT_DIR}/train_simple_20260730_llama2_7b", source="local", device_map="auto")
    else:
        # 仅测试模型架构（不加载权重），与 PRETRAINED/CONTINUAL 同步支持 device_map="auto"
        model = LLamaForCausalLM.from_empty(cfg, device_map="auto")
    
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
    out = model.generate(prompt, max_new_tokens=250, top_k=50, temperature=0.8) # (1, 3) -> (1, 13)
    print(f"\nGenerated: {tokenizer.decode(out[0].tolist())}") # (1, 13) -> (13,) -> list -> str