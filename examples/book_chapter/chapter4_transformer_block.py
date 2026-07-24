#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/07/05 15:08:27
@Author : weiyutao
@File : chapter4_transformer_block.py
Combines several concepts we have previously covered: multi-head attention, layer normalization, dropout, feed forward layers, and
GELU activations. Later we will connect this transformer block to the remaining parts of GPT architecture.
Transformer的核心思路：用attention横向跨词融合全局上下文，用FFN纵向闭门深挖单词内在逻辑，两者在维度守恒的积石上交替前行，从而实现复杂推理能力的无限堆叠。
为什么必须交替堆叠而不是先全部注意力，再全部FFN？
1、防止过度平滑(over smoothing)。Attention的本质是加权求和。如果连续走多层注意力，所有的词向量都会变成一模一样灰色（所有token特征同化了，就像把不同颜色的墨水混合到水里一样）
加入FFN就是为了每次混合之后，使用非线性把它重新拉开。
2、构造螺旋上升的推理能力。
第一层：Attention发现苹果旁边有科技，FFN把苹果+科技升华为苹果公司
第二层：Attention发现苹果公司和远处的创始人，FFN把苹果公司+创始人升华为乔布斯

真正的记忆权重在哪里？这些特征权重是以什么形式存储的？
FFN是真正的大模型训练数据的知识库，FFN一般有两层全连接，第一个全连接W1是升维一般是4*input，第二个全连接层是降维。
假设第一层的1024号神经元，它的权重专门用来匹配苹果+科技，当带有科技特征的苹果想向量输入进来的时候，它和1024号神经元权重做点积，匹配极高算出来的数极大
而其他神经元算出来的数值为负，GELU将会放行数值较大的，其它杂音全被屏蔽。W2里面存储的是答案，被点亮的1024号神经元对应着W2里面的一行向量，这行向量存储着
乔布斯、iphone、macbook等高阶知识，比如W1层1024号神经元输出是5.2， W2中对应W1中的1024号神经元的神经元比如600号神经元立即激活并乘以激活值5.2，这两个
神经元的配对激活在训练过程中已经被激活了。以上特征都是一个个向量对应的就是W1、W2这些神经元。每个权重是当前层的权重，对应多个神经元，每个神经元的维度
在每一层不一样，如W2层，W2维度是3072,768其每个神经元的维度是768，也就是特征向量的维度是768，也就是每个特征比如乔布斯、iphone、macbook使用这个大小的维度
去表示，然后总计3072个神经元。比如第600个神经元对应W1中的1024号神经元其意思是乔布斯、iphone、macbook，第601号神经元对应的是比尔盖茨等。残差网络增量学习
就是将以上经过W2层权重输出的激活值加上输入x，去让梯度学习该增量。

因为特征提取是交替堆叠的，如果第一层的特征提取已经升华到了顶点，后续层会产生歧义吗？
token的768维向量不是一个杯子，而是一条拥有768条车道的高速公路，第一层的FNN在10-50维写入了输入序列苹果为科技公司的特征。
第二层的FNN并不是去覆盖第10-50维，而是读取这部分特征然后再第200-250维写入更高级的史蒂夫乔布斯特征。就好比一块巨大的黑板上左上角和右下角写字
它们通过在每层进行加法叠加到同一个向量里面，不存在覆写也不会产生歧义。

如果第一层就升华到了顶点，那么后续层的残差网络会全部发挥功效，在反向传播的时候，最外层的损失直接通过高速通道返回给第一层，而除了第一层所有的激活输出梯度都为0，
除了第一层所有层的权重将不会被更新。这也就是在深度网络中残差的重要性，它是神经网络调整每一层的重要工具。


为什么特征提取一定是逐步的？也就是浅层是低阶特征，深层是高阶特征？
因为高阶逻辑在数学上是由低阶逻辑组合而成的。
第一层attention看到苹果和logo，FFN升华出苹果品牌
第二层attention看到苹果和财报，FFN升华出公司财报利润
第三层attention看到苹果品牌和上文的财报和利润表，FNN升华出商业巨头和股价预测这些高阶逻辑。
如果只有一层FFN，根本看不到苹果和公司财报利润的关联信息，因为attention还没有来得及把那么远这么复杂的关系搬运过来，必须是attention搬运一点，FFN总结一点，带着
新的总结，Attention再去搬运更远的，FFN在做更深的总结，这就是逐步升华。而且因为残差流的关系，后续写入的特征不会覆盖掉上一层甚至第一层写入的特征。
假设第一层 第二层 第三层 第四层
经过第一层之后768维度的向量大概是这样的：基础词义：苹果 ......
经过第二层之后：基础词义：苹果 特征：水果植物 
经过第三层之后：基础词义：苹果 特征：水果植物 特征：科技公司
经过第四层之后：基础词义：苹果 特征：水果植物 特征：科技公司 特征：乔布斯 

也就是说第一层输出特征为A
第二层基于第一层的输出B
第三层基于第一层+第二层的输出C  A+B->C
第四层基于第一层第二层第三层的输出D A+B+C->D
注意以上：每进行一次前向传播，所有层可以看到的token数是一致的。比如我是谁三个字
输入我推理预测是
输入我是推理预测谁
第一个所有层都会看到我并计算注意力
第二个所有层都会看到我是并计算注意力

大模型的深度网络不是一台粉碎机，而是一台高纬度的3D打印机，残差连接保证了打印出来的底座永远不会被抹掉，高维空间保证了每一层都有足够的地方去喷涂新的材料，
LOSS梯度就是哪张极其严苛的图纸，强迫这太打印机只能按照从微观到宏观，从字面到逻辑的顺序，一层层把智能堆叠出来。



直通估计器STE(Straight-Through Estimator)
假设二值化符号函数为：sign(x) 
    = +1, x >= 0
    = -1, x < 0
这就是一个台阶，解约函数的梯度在台阶平坦的地方为0，在台阶跳变的地方为无穷大。所以该阶跃函数内部的参数权重一定不会被更新因为梯度为0。
STE破局法则
前向传播：sign(0.3)=+1，抹除了高精度的量化细节0.3，网络向下传递的是极其粗粒度的+1，这就构建了坚固的拓扑骨架。
反向传播：当最终的误差传递回来的时候，STE强行规定此时阶跃函数的梯度为1.

使用直通估计其STE可以在残差网络的基础上引入非量化规则。
何为非量化规则？
想象一个弹珠游戏，扔的弹珠越多，经过随机翻倍之后（翻倍的数值有2 4 6 8 10），
然后如果赢了按照投的弹珠数量乘以翻倍输出赢得的弹珠数量，并且每输出30个弹珠会输出一张卡。
大人可以看到自己投入弹珠的具体数额，也可以看到翻倍的值，但是小孩不会看这些量化指标，
他们只知道一个规则就是投的弹珠数量越多，那么在赢了的时候出的卡片就越多。
从现实来看小孩这种非量化规则优于大人的量化规则。是否可以在现有网络的基础上新增一个非量化规则的路由呢？

为什么非量化规则更优？
1、极低的认知负载。量化规则需要进行大量的乘法和除法，这在神经科学中意味着高度依赖前额叶皮层的显性工作记忆。能耗极高。
非量化规则是一种低精度的序数关系，只需要极少的神经元突触激活即可完成判断。
2、对环境突变的绝对鲁棒性。遇到不同分布的数据，量化规则将彻底失效，非量化规则的单调性规则在参数变化之后依然成立。
3、抓住了流形空间的拓扑特征，而非离散坐标。

具体如何实现？依靠残差块QH-Block，由双规特征提取和自动调节阀门组成
Q(x): 量化输出
N(x): 非量化输出，可以是一个sign函数，输出是几何拓扑或者单调规律。注意这里N(X) = sign(Wₙ · X)，Wₙ是可学习权重。
α = σ(Wgₐₜₑ · X) σ可以是一个sigmoid函数，该函数输出0-1之间的权重，网络会根据输入X自动计算出分配给量化规则和非量化规则的比例。
Y = X + [α · N(X) + ((1 - α) · Q(X))]

优点很多，比如在前向传播的时候
1、每个路由的特征都会自动根据可学习参数α(因为α是根据可学习的权重Wgₐₜₑ计算得到的)去调节对应的输出比例。
2、如果α很大，那么相对应的分配给量化指标Q(X)的比例将会减少。同时在反向传播的时候对应的更新Q(X)里面权重的梯度将会很小（因为对应的梯度
计算由(1 - α)决定）。意味着较小幅度或者不更新量化权重，较大幅度更新非量化权重Wₙ。
3、在全局损失传递回来的时候，α · (1 - α)决定回传的整体梯度大小，该乘法结果在α = 0.5的时候最大，在α接近最大值1和最小值0的时候乘法结果接近0。
$$\frac{\partial L}{\partial W_{\text{gate}}} = \frac{\partial L}{\partial Y} \cdot \big[ N(X) - Q(X) \big] \cdot \color{red}{\alpha (1 - \alpha)} \cdot X$$
这意味着一旦确信走哪条路线，物理锁死梯度，此时Wgₐₜₑ的更新将被阻止，否则一旦犹豫不决（也就是α接近0.5的时候），巨大的梯度会引导Wgₐₜₑ迅速更新。

缺点也很多，sigmoid激活函数去做概率映射会导致死神经元的问题，也就是说会导致Wgₐₜₑ这个可学习参数不更新的情况。
详细见chapter4_hq_block模块
"""

import torch
import torch.nn as nn
print(f"CUDA 是否可用：{torch.cuda.is_available()}")
print(f"当前使用的显卡：{torch.cuda.get_device_name(0)}")



from examples.book_chapter.chapter3_multi_head_attention import MultiHeadAttention
from examples.book_chapter.chapter4_layer_norm import GPT_CONFIG_124M, LayerNorm
from examples.book_chapter.chapter4_feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in = cfg["embedding_dim"],
            d_out = cfg["embedding_dim"],
            context_length=cfg["context_length"],
            num_heads = cfg["num_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.feed_forward = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dim"])
        self.norm2 = LayerNorm(cfg["embedding_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        
    
    def forward(self, x):
        """
        Layer normalization is applied before each of these two components, and dropout is applied after them to regularize the model
        and prevent overfitting. This is also known as Pre-LayerNorm. Older architectures, such as the original transformer mdoel, applied
        layer normalization after the self-attention and feed forward networks instead, known as Post-LaterNorm, which oftenl leads to worse
        training dynamics. The class also implement the forward pass, where each component is followed by a shortcut connection that adds the
        input of the block to its input. This critical feature helps gradients flow through the network during training and improves the learning of deep models.
        注意在进行注意力机制计算的时候，在计算完注意权重之后直接对权重进行了dropout操作，然后紧接着在注意力机制计算中使用权重和进行了矩阵操作，然后进行了out_proj映射
        这步是必要的因为注意力机制计算得到的结果需要进行线性映射之后重新洗牌才具有意义。
        """
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x




if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.randn(2, 4, 768)
    block = TransformerBlock(cfg=GPT_CONFIG_124M)
    output = block(x)

    print("Input shape: ", x.shape)
    print("Output shape: ", output.shape)


