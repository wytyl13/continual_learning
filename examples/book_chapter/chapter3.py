#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/05/17 19:32:46
@Author : 
@File : chapter3.py
In self-attention, our goal is to calculate context vectors z(i) for each element x(i) in the input sequence. 
A context vector can be interpreted as an enriched embedding vector.
    Context vectors play a critial role in self-attention. Their purpose is to create enriched representations of 
each element in an input sequence(like a sentence) by incorporating information form
all other elements in the sequence. This is essential in LLMs, which need to understand the relationship
and relevance of words in a sentence to construct these context vectors so that they are relevant
for the LLM to generate the next token.
    The dot product is a measure of similarity because it quantifies how closely two vectors are
aligned: a higher dot product indicates a greater degree of alignment or similarity between the \
vectors. In the context of self-attention mechanisms, the dot product determines the extent to which
each element in a sequence focuses on, or "attends to," any other element: the higher the dot product, the
higher the similarity and attention score between two elements.
    We have got the attension score by calculating the dot product of query and key. Then we normalize each of
the attention scores we computed previously. The main goal behind the normalization is to obtain attention
weights that sum up to 1. This normalization is a convention that is useful for interpretation
and maintaining training stability in an LLM. Just like attention_weights = attention_scores / attention_scores.sum()
In practice, it's more common and advisable to use the softmax function for normalization.This approach is
better at managing extreme values and offers more favorable gradient properties during training. 
attention_weights = torch.exp(attention_score) / torch.exp(attention_score).sum(dim=0). the softmax
function also meets the objective and normalizes the attention weights such that they sum to 1. In addition, the softmax function
ensures that the attention weights are always positive. This makes the output interpretable as probabilities or
relative importance, where higher weights indicate greater importance. Note that this native softmax implementation may encounter
numerical instability problems, such as overflow and underflow, when dealing with large or small input values. It is
advisable to use the pytorch implementation of softmax, which has been extensively optimized for performance.
Just like torch.softmax(attension_scores, dim=0).
    We have computed the normalized attention weights, we are ready for the final step: calculating the context
vector z_(2) is weighted sum of all input vectors, obtained by multiplying each input vector by its corresponding
attention weight. Z_(2) = a21 * x_(1) + a22 * x_(2) + a_2T * x_(T), attention_score(2) = torch.dot(x, query), query = x_(2).
attention_weights(2) = torch.softmax(attention_score(2), dim=0).
    Sor far, we have computed attention weights and the context vector for input2, now let's extend this computation
to calculate attention weights and context vectors for all inputs. When computing the preceding attention score tensor, we 
used for loops in Python. However, for loops are generally slow, and we can acheive the same results using matrix multiplication:
attention_scores = inputs @ inputs.T, then we normalize each row so that the values in each row sum to 1: attention_weights = torch.softmax(attention_scores, dim=-1).
Then we use these attention weights to compute all context vectors via matrix multiplication: all_context_vectors = attention_weights @ inputs.
"""

import torch
inputs = torch.tensor([
    [0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]
])

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value


# we can obtain all keys and values via matrix multiplication:
keys = inputs @ W_key
values = inputs @ W_value
# then we will compute the attention scores.
keys_2 = keys[1]
attention_score_2 = query_2.dot(keys_2) # dimension is (1, 1), is (1. 3) @ (3, 1) = (1, 1),what means the w22(attention scores about the second token with itself)
attention_scores_2 = query_2 @ keys.T # dimension is (1, 6), is (1, 3) @ (3, 6) = (1, 6), what means the w2t(attrntion scores about the second token with all tokens)

d_k = keys.shape[-1]
attention_weights_2 = torch.softmax(attention_scores_2 / d_k**0.5, dim=-1)



"""
The retionale behind scaled-dot product attention
The reason for the normalization by the embedding dimension size is to improve the training
peformance by avoiding small gradients. For instance, when scaling up the embedding dimension, 
which is typically greater than 1,000 for GPT-like LLMs, large dot products can result in very small
gradients during backpropagation due to the softmax function applied to them. As dot products
increase, the softmax function behaves more like a step function, resulting in gradients nearing zero. These
small gradients can drastically slow down learning or cause training to stagnate.

The scaling by the square root of the embedding dimension is the reason why this self-attention mechanism is also
called scaled-dot product attention.
为什么一定要分别训练QKV三个权重？
1、当然可以直接使用原始向量X互相做点积，向量和它自己的点击永远是最大的（因为点积衡量的是相似性），这意味着每个单词都会把绝大部分注意力放在自己身上，而无法
有效地关注句子里的其他单词。引入W_Q和W_K将原始向量投影到不同的空间，使得我想找的特征Q和我本身的标签K分离开来，这样
词语就去寻找与其互补或相关的其他词，而不是只盯着自己。
2、保证注意力的非对称性。在语言中注意力是有方向性的，比如i love you，love对you的关注度和you对love的关注度是不一样的。
比如使用原始向量X_A@X_B=X_B@X_A。但是引入Q和K之后，A的查询匹配B的标签Q_A@K_B，和B的查询匹配A的标签Q_B@K_A，结果是完全不同的。
注意理解语法天生就是非对称的重要性：人类语言中，词与词之间的关系几乎都是有明确方向的，如果模型只能表达对称的关系，它就无法真正理解复杂的语法。
比如：招聘会上，一个HR是一个token，一个程序员是一个token，在招聘会环境下，HR的query也就是需求是我需要一个资深程序员，他的key也就是他的标签是20K月薪，
而程序员token它的query也就是需求是找一个月薪50万的工作，它的标签也就是key是资深程序员。那么很容易理解，Q_A@K_B也就是HR对程序员的注意力是极高的，因为它满足了自己的需求
而Q_B@K_A也就是程序员对HR的注意力是极低的，因为它没有满足程序员的薪资要求。
比如：the dog bit the man. 动词bit对man的注意力是很强的，而man对bit的注意力很弱。
在数学上：ScoreA->B = X_A @ X_B，这在数学计算上满足点积交换律，是对称的，也就是说算出来的注意力矩阵一定是一个对称矩阵。
而通过引入两个独立训练的参数矩阵W_W, W_K，点积变成了
Q_A @ K_B=(X_A @ W_Q) @ (X_B @ W_K) = X_A @ W_Q @ (W_K).T @ X_B.T，这会有极小的概率成为一个对称矩阵。
3、提取特征的过滤器。原始的词向量X非常庞大，里面包含了这个词的词性、感情色彩、指代关系等所有信息。但是在某一次特定的注意力计算中，
模型可能只需要这个词的某一部分特征。矩阵W_V就像是一个特征过滤器，从X中提炼出当前上下文真正需要的信息V，如果直接使用X作为V去加权求和，
就会把大量不需要的噪音也混入最终的结果中。
4、为多头注意力提供空间。头1可能负责寻找语法关系，头2可能负责寻找代词指代。每一个头都需要自己独立的W_Q, W_K, W_V矩阵，从而把
同一个原始向量X映射到不同的语义子空间中。
"""
import torch.nn as nn
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vector = attention_weights @ values
        return context_vector
        
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))



class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vectors = attention_weights @ values
        return context_vectors
torch.manual_seed(789)
context_vectors = SelfAttention_v2(d_in, d_out)
print(context_vectors(inputs))


"""
Hiding future words with causal attention.
Causal attention, also known as masked attention, is a specialized form of self-attention. It restricts a model to only
consider previous and current inputs in a sequence when processing any given token when computing attention scores. This 
is in contrast to the standard self-attention mechanism, which allows access to the entire input sequence at once. Just like
BERT, which is standard self-attention processes the entire input sequence for deep contextual understanding, whereas causal
attention masks future tokens to enable autoregressive text generation.
We can use the SelfAttention_v2 to implement the causal attention.
"""
self_attention_v2 = SelfAttention_v2(d_in, d_out)
queries = self_attention_v2.W_query(inputs)
keys = self_attention_v2.W_key(inputs)
attention_scores = queries @ keys.T
attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)
print(attention_weights)


context_length = attention_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length)) # tril can generate one triangle lower matrix.
print(mask_simple)


masked_simple = attention_weights * mask_simple
print(masked_simple)


row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)


"""
attention(Q, K, V) = softmax(Q @ K.T / (d_k **0.5)) @ V
为什么要除以d_k**0.5？目的是缩放点积注意力，d_k是query和key向量的维度，目的是为了防止点积方差过大，
假设两个向量中的每个元素都是相互独立的随机变量，且均值为0，方差为1，当对两个维度为d_k的向量进行
点积相乘时，得到的结果的均值依然是0，但是方差会放大到d_k。还有一个是避免softmax梯度消失，如果输入给哦softmax的
数值非常大，softmax的输出分布会变得极其陡峭，最大值对应的概率会无限趋近于1，而其他所有值趋近于0.
这里注意softmax执行位置，如果先进性softmax计算，然后应用掩码矩阵，那么还需要进行归一化操作，目的是使得每行注意力权重之和为1。
如果是在softmax之前应用的掩码矩阵，比如后面的负无穷大的掩码，那么不需要再进行归一化操作。

为什么要使用隐藏层维度进行缩放注意力分数？
softmax的马太效应，注意力分数会经过softmax层输出为注意力权重
softmax是一个基于指数的函数，如果输入数字都很小，softmax的结果比较温和，如果输入
数字很大，比如极端情况下，softmax的输入结果会成为一个one-hot向量，完全无视了句子里其他的上下文。
为什么注意力分数的数值会变大？如果Q和K里的每一个数字都是独立且均值为0，方差为1的随机变量，那么这d_k
个乘积相加厚，虽然结果的均值还是0，但是他的方差会随着维度d_k的增加而线性增长，变成d_k。根据概率论，如果我们将
整个结果除以d_k**0.5，它的方差就会被拉回1，即d_k/d_k**0.5=1。注意方差的物理意义是“数据波动幅度的平方”


information leakage
When we apply a mask and renormalize the attention weights, it might initially appear that information 
from future tokens could still influence the current token because their values are part of the softmax calculation.
However, the key insight is that when we renormalize the attention weights after masking, what we are essentially doing
is recalculating the softmax over a smaller subset(since masked position don't contribute to the softmax value).
The mathematical elegance of softmax is that despite initially including all positions in the denominator, after
masking and renormalizing, the effect of the masked positions is nullfilled——they don't contirbute to the softmax
score in any meaningful way.

In simpler terms, after masking and renormalization, the distribution of attention weights is as if it was
calculated only among the unmasked positions to begain with. This ensures there's no information leakage from future
tokens as we intend.

The softmax function converts its inputs into a probability distribution. When negative infinity values are present
in a row, the softmax function treats them as zero probability. (Mathematically, this is because e^-∞ approaches 0).
"""

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attention_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
attention_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attention_weights)

"""
Dropout in deep learning is a technique where randomly selected hidden layer units are ignored during training, effectively
"dropping" them out. This method helps prevent overfitting by ensuring that a model does not become overly
reliant on any specific set of hidden layer units. It's important to emphasize that dropout is only used during training and is disabled afterward.

In the transformer architecture, including models like GPT, dropout in the attention mechanism is typically applied at
two specific times: after calculating the attention weights or after applying the attention weights to
the value vectors. Here we will apply the dropout mask after computing the attention weights.
"""

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(f"dropout: {dropout(example)}")

"""
When applying dropout to an attention weight matrix with a rate of 50%, half of the elements
in the matrix are randomly set to zero. To compensate fro the reduction in active elements, the 
values of the remaining elements in the matrix are scaled up by a factor of 1/0.5=2. This scaling
is crucial to maintain the overall balance of the attention weights, ensuring that the average influence of the attention mechainsm 
remains consistant during both the training and inference phases.
"""

torch.manual_seed(123)
print(dropout(attention_weights))




"""
Implementing a compact causal attention class.
"""
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, 
                 dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        # because the mask will be used in gpu, so we can use the register_buffer 
        # to tell torch that you should move it to GPU together with the trainable weights.
        # not let the mask stop at cpu.
        # diagonal=0保留主对角线为1
        # diagonal=1不保留主对角线，即主对角线为0
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attention_scores = queries @ keys.transpose(1, 2)
        
        # 原地更改变量
        # self.mask是支持最大长度的上下文掩码，当前输入不需要完整的mask，所以要切片
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1
        )

        attention_weights = self.dropout(attention_weights)
        
        context_vectors = attention_weights @ values
        return context_vectors
    
torch.manual_seed(123)
context_length = batch.shape[1]
causal_attention = CausalAttention(d_in, d_out, context_length, 0.0)
context_vectors = causal_attention(batch)
print("context_vectors.shape: ", context_vectors.shape)


"""
Extending single-head attention to multi-head attention.
"""
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        
    def forward(self, x):
        # (2, 6, 2) cat (2, 6, 2), dim=-1  ==> (2, 6, 4)
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
    

torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)

context_vectors = mha(batch)
print(context_vectors)
print("context_vectors.shape: ", context_vectors.shape)






"""
吕梁
煤矸石中提取高岭土，高岭土成分：
al2o3（高）> 42%、
sio2（低）< 55%、
fe2o3（低）< 1%、
tio2（低）
"""














