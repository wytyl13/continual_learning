#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/06/21 12:03
@Author  : weiyutao
@File    : chapter4_feed_forward.py

激活函数的本质
激活函数是为了增加模型的非线性能力。

sigmoid(x) = 1/(1+e**-x)。S型曲线，sigmoid(0) = 0.5，取值范围为(0, 1)
优势：输出为0,1刚好为概率值，依然是二分类问题输出层的绝对霸主。
缺陷：最大斜率只有0.25所以会造成梯度消失，还有就是输出不是零中心化的，输出永远是正数，
这会导致反向传播时，权重的更新方向出现捆绑（只能一起变大或一起变小），导致网络收敛的路径像个
锯齿，极其缓慢。

tanh(x) = (e**x-e**-x)/(e**x+e**-x)。S型曲线，看起来比sigmoid更陡峭一点，取值范围为(-1, 1)
中心点事0，tanh(0)=0，tanh(x) = 2 * sigmoid(2x) - 1。tanh本质上就是把sigmoid放大两倍然后向下
平移一个单位。
优势：零中心化的，且输出有正有负，平均值是0。完美解决了sigmoid导致权重更新锯齿状的问题。
在绝大多数隐藏层中tanh总是毫无悬念吊打sigmoid。另外，在原点处的最大斜率是1，比sigmoid的0.25好一些。
但是最终还是难逃两端梯度消失的命运。

为什么sigmoid的激活函数输出取值范围为0,1，也就是非零中心化这么严重？因为如果某一个激活函数的输出
恒为正，如果没有其它层没有为负输出的激活函数，那么整个网络的梯度方向将会一致为正。

假设两层简单的全连接神经网络。
前向传播
x@w1=a1
z1=sigmoid(a1)
a2=z1@w2
z2=sigmoid(z2)
其中z1, z2恒为正(0, 1)

反向传播：根据链式法则
dsigmoid/dx=z2(1-z2)，恒为正（因为z2取值范围为(0, 1)）
dL/dw2=dL/dz2 * dz2/da2 * da2/dw2 = [dL/dz2*z2(1-z2)] * z1
以上，因为z1和z2这两层的激活函数都是sigmoid所以以上dL/dw2的正负取决于dL/dz2的正负，也就是说权重的更新和
最外层的损失强行捆绑，引发锯齿状震荡。

同理，
dL/dw1=dL/dw2 * dz1/da1 * da1/dw1 = [dL/dz2*z2(1-z2)] * z1 * z1(1-z1) * x
w1的更新方向直接被初始化数据x的正负号控制，如果输入数据是图像(0, 255)都是正数，那么w1权重也会遭受
只能同增同减的锯齿命运。同增同减指的是w1权重内部所有权重其梯度值的正负完全取决于自己权重对应的输入x，比如
x@w1 = x11*w11+x12*w12=a1
那么在反向传播的时候，da1/dw11=x11, da1/dw12=x12，在x11和x12同为正的情况下（比如图像），w1权重的更新就是锯齿状



注意这里的锯齿命运针对的是同一层不同权重参数之间，而不是不同层之间。首先：第二层的权重更新方向取决于全局损失（也就是最外层的损失值）
因为第二层的输入是第一层的激活输出恒为正。而第一层的权重更新方向完全取决于第一层权重的输入也就是x（原始输入），除了第二层链式传播过来的正负。
不管是哪一层，都会有多个权重比如w1是：w11、w12等等，这些权重更新的梯度都取决于自己的输入（第一层的输入就是x11、x12原始输入，第二层的输入就是
z11、z12是第一层的激活输出），当然这是在sigmoid函数激活值恒为正的情况下的，如果是其他激活函数不恒为正或者负，那么会有更多的变通性。
如果是在sigmoid激活函数下，假设第一层的激活函数是sigmoid，并且第一层的输入都恒为正（假设是图像），那么第一层所有的权重更新就是锯齿状，
因为恒为正。

以上就是sigmoid激活函数的劣势，它使得网络训练过程不存在不确定性，使得网络不能承担复杂的训练功能。
这就是为什么激活函数如此重要，同样的，tanh虽然取值范围为(-1, 1)，但是它仅仅只是解决了锯齿状问题（当前层权重梯度更新方向不同增同减），但是
它并没有解决梯度消失的问题。为什么会梯度消失？
这取决于tanh和sigmoid的曲线，tanh的最大斜率（也就是梯度）为0.25，sigmoid的最大斜率是0.25。也就是说即时是只有1层激活函数是sigmoid，
其误差信号都会被扣掉75%，如果是10层权重并且10个sigmoid激活，那么其第一层的误差信号为最外层损失的0.25**10，几乎为0.
同样的，tanh只有在输入为0的时候其梯度才为1，只要输入大于3或者小于-3，其梯度值几乎为0.意味着tanh和sigmoid在输入激活的数值较大的时候
都会发生梯度消失，梯度消失意味着：当前神经元的权重将不会被更新，因为当前神经元对应的激活函数的梯度为0，所以他的参数不会被更新。

让我们理解一下神经元这个概念
我们要区分层这个概念，而神经元是在每个层级下面的概念
假设我们之前的案例，两个全连接层
每个全连接层下面对应的有多个神经元，每个神经元由输入、权重和激活函数等组成。权重是神经元的突触，输出值是轴突信号，而神经元
就是一个完整的计算单元，包含了接收所有输入，乘以对应权重，求和加偏置，通过激活函数整个过程。
所以现在可以理解每个神经元对应的激活函数的梯度都不一样，那么对应的在反向传播过程中，假设第一层的a11输出为3，并且该层的
激活函数是tanh，那么它大概率的梯度就是0，那么对应的在反向传播过程中不论其输入x11是多少，最终计算的w11更新的梯度就是0，
这就是因为激活函数导致的梯度消失，从而导致w11权重变为死神经元。一旦陷入了死循环，比如是比较死板的sigmoid和tanh激活函数
w11的更新梯度将会持续为0，导致w11一直不做修正，那么在前向传播的过程中其输出a11就恒等于3，其反向传播梯度就恒等于0.
但是a12等于1，那么其梯度将不为9，这样不会造成梯度消失的问题，使得w12不会变为死神经元，但是sigmoid和tanh这种性质导致其
梯度为0的情况概率较大，所以被relu和gelu、swish-gelu替代了，RELU的特性就是在输入激活的值为正的时候恒为自己，也就是斜率为1
这就使得不论网络的纵深有多大，链式求导法则计算出来的梯度不会被缩放，恒为1，使得最外层的损失无损传播到第一层，更适合深层网络
的激活，同时，其对于负值恒为0的方式并没有解决死神经元的问题，因为如果一个神经元其输出为负，那就会造成该神经元对应的突触
w11的权重永远不会被更新。反之，gelu和swish-gelu因为在负值附近给予了较小的梯度，这就导致某个神经元的梯度不可能为0，梯度不为0
意味着其权重有更新的激活，随着权重的更新该神经元的输出有可能为正，使得神经元向损失较小的方向优化。

为什么relu和gelu都是对输出正直其恒等于自己，而输出是负值的时候赋值为0或者一个接近0的值？因为在神经科学中，正直一般表示有
正信号，而负值表示没有信号输出表示该神经元没有被激活。所以负值对激活与否没有意义。但是直接将负值置为0会导致死神经元，
所以才给与一个较小的变换值，使得负值在后续的更新中有可能向正值靠近。这具有伟大的意义。


Implementing a feed forward network with GELU activations.
Historically, the ReLU activation function has been commonly used in deep learning due to its simplicity
and effectiveness across various neural network architectures. However, in LLMs, several other activation
functions are employed beyond the traditional ReLU. Two notable examples are GELU (Gaussian error linear unit)
and SwiGLU (Swish-gated linear unit).

前向传播
a = GELU(z) = z · Φ(z)
Φ(z)是标准正泰分布的累积分布函数(CDF)
Φ(z) = P(Z ≤ z) =  1 / √2π · ∫₋∞²e(⁻ᵗ^²/²)dt
因为以上累积分布函数在GPU理算起来太慢，所以使用以下两种近似公式之一：
a ≈ 0.5 · z · [1 + tanh(√1/π · (z + 0.044715 · z³))]
a ≈ z · σ(1.702 · z) = z / 1 + e⁻(¹.⁷⁰²·z)

反向传播
∂L / ∂z = ∂L / ∂a * ∂a / ∂z
∂a / ∂z = 1 · Φ(z) + z · Φ'(z) = 1 / √2π · ∫₋∞²e(⁻ᵗ^²/²)dt + z · (1 / √2π · e⁽⁻z²/²⁾)

完整的反向传播链条
假设输入x维度是(2, 3)，2个词3维嵌入。
权重w维度是(3, 2)
假设当前层是全连接层，z = x · w = (2, 2)
∂L / ∂z = ∂L / ∂a · [Φ(z) + z · Φ'(z)] = (2, 2) 初步损失
∂L / ∂w = xᵀ · ∂L / ∂z  = xᵀ · ∂L / ∂a · [Φ(z) + z · Φ'(z)]  = (3, 2) · (2, 2) = (3, 2)
根据以上梯度计算公式可以发现：某一个神经元的梯度方向由 ∂L / ∂z也就是激活函数的梯度和输入x决定，如果sigmoid激活函数会导致 ∂L / ∂w 梯度方向和最外层的损失捆绑，因为∂L / ∂z在激活函数为sigmoid的时候恒为正
并且∂L / ∂w的梯度方向严格取决于输入x的正负。

在激活函数是sigmoid的时候：z -> +∞，∂L / ∂z -> 1；z -> -∞，∂L / ∂z -> 0；梯度消失，不会再次更新权重w，导致死神经元
在激活函数是tanh的时候：z -> +∞，∂L / ∂z -> 1；z -> -∞，∂L / ∂z -> -1；梯度消失，不会再次更新权重w，导致死神经元
在激活函数是GELU的时候，z -> +∞，∂L / ∂z -> 1；z -> -∞，∂L / ∂z -> 0；z -> -0，∂L / ∂z -> -0.084，不会导致梯度消失，会更新w权重，逐步更新w权重有概率使得z值由负转正，从而梯度趋向于1
sigmoid函数在z值为0.5的时候梯度最大为0.5
tanh在z值为0的时候梯度最大为1
relu在z值为0的时候梯度为0，在z值大于0的时候梯度恒为1，在z值小于0的时候值恒为0且梯度恒为0
gelu在z值大于3的时候梯度恒为1，在z值大于0小于3的时候梯度大于0小于1，在z值趋向于-0的时候，梯度值不为0为-0.084，在梯度值趋向于-∞的时候，梯度趋向于0。以此可见gelu适合深层次网络。
为什么在z值为正的时候梯度必须为1？因为随着网络的加深，如果梯度介于0和1之间，那么反向传播到前面层的梯度将被缩放至无限接近0，会导致梯度消失。
只有梯度为1才能将靠近输出层的全局损失值无损回传给前面层。

当然比gelu更优的激活函数比如swish-gelu
"""



"""
An implement of the GELU activation function.
GELU(x) ≈ 0.5 · x · (1 + tanh[√2/π · (x + 0.044715 · x³)])
"""

import torch
import torch.nn as nn
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        x 输入x是tensor
        torch.pi 不是tensor，因为底层是math.pi是一个float标量
        torch.function(a) a在经过运算后必须是tensor
        1 + x 是tensor
        torch.pow(x, 3) 是tensor
        x + 0.044715 是tensor
        2.0 / torch.pi 是float不是tensor，在使用torch.sqrt运算之前必须将其转换为tensor
        下面代码更高效的做法：直接将torch.sqrt(torch.tensor(2.0 / torch.pi)) 使用math.sqrt(2.0 / math.pi)
        代替，以为内每次都进行tensor的转换会耗费GPU计算时间。在init的时候直接初始化
        self.const_val = math.sqrt(2.0 / math.pi)
        然后再forward中直接
        return 0.5 * x * (1 + torch.tanh(self.const_val * (1 + 0.044715 * torch.pow(x, 3))))
        因为self.const_val * (1 + 0.044715 * torch.pow(x, 3))一定是一个tensor，所以可行
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))
        
        

"""
COMPARE RELU AND GELU
The smoothness of GELU can lead to better optimization properties during training, 
as it allows for more nuanced adjustments to the model's parameters. In contrast,
RELU has a sharp corner at zero, which can sometimes make optimization harder, especially
in networks that are very deep or have complex architectures. Moreover, unlike RELU, which 
outputs zero for any negative input, GELU allows for a small, non-zero input for negative values. 
This characteristic means that during the training process, neurons that receive negative input can
still contribute to the learning process, albeit to a lesser extent than positive inputs.

"""
def compare_relu_gelu():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gelu, relu = GELU(), nn.ReLU()

    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "RELU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label} (x)")
        plt.grid(True)
    plt.tight_layout()
    plt.savefig("test.png")
    plt.close()
    print("运行完毕，图像已保存为 test.png")



class FeedForward(nn.Module):
    """
    Let's use the GELU function to implement the small nueral network module FeedForward.
    The FeedForward module plays a crucial role in enhancing the model's ability to learn
    from and generalize the data. Although the input and output dimensions of this module
    are the same, it internally expands the embedding dimension into a higher-dimensional
    space through the first layer. This expansion is followed by a nonlinear GELY activation and then a
    contraction back to the orginal dimension with the second linear transformation. Such a design
    allows for the exploration of a richer representation space. Moreover, the uniformity in input
    and output dimensions simplifies the architecture by enabling the stacking of multiple laters, 
    as we will do later, without the need to adjust dimensions between them, thus making the model more scalable.
    """
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["embedding_dim"], 4 * cfg["embedding_dim"]),
            GELU(),
            nn.Linear(4 * cfg["embedding_dim"], cfg["embedding_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class ExampleDeepNeuralNetwork(nn.Module):
    """
    shortcut, residual neural network.
    本质：
    1、标准网络里的连乘诅咒，这里可以说有点像激活函数GELU的功能，随着网络越深，其梯度值会累乘，如果梯度值介于0-1之间
    则会出现梯度消失，但是残差的数学公式是y = F(x) + x, y' = F(x)' + 1，这里无论F(x)'是否为0-1之间，都会加上1使其累乘值大于1
    2、从重塑到雕刻。残差的本质是学习差值，假设在绘制一幅画，y=F(x)时候每层网络都在尽力绘制整张图，有了y = F(x) + x之后，
    F(x)的目标不再负责绘制整张图，而是只负责修细节。这意味着即便F(x)=0,y = 0 + x = x.假设我们在当前层绘制的理想输出应该是给蒙娜丽莎画上双眼皮，画上双眼皮之后蒙娜丽莎
    的完美画像输出应该是H(x)，但是传统方法到这里y = F(x) = H(x)，F(x)这个权重矩阵必须从零开始记住蒙娜丽莎的所有像素（脸型、鼻子、嘴巴、双眼皮），如果他的初始化权重很随机，
    它要经过极其漫长的训练才能把所有东西都学对，一旦学歪了画就毁了。但是残差为了达到完美，网络要逼迫自己做到F(x) + x = H(x), F(x) = H(x) - x，F(x)的权重的学习目标变为了
    拟合残差也就是 H(x) - x，就是当前层完美的蒙娜丽莎减去半成品的蒙娜丽莎，就是双眼皮。这里残差就是给蒙娜丽莎绘制双眼皮。
    3、假设总领导在安排一个任务，普通层 y = F(x) 像是一层层按照部门向下传播指令，然后会层层衰减，每个部门的领导之间没有任何沟通，上一个部门的最后一个工人和
    下一个部门的第一个工人直接沟通。shortcut类似于总领导下发指令之后，每个部门的领导会把自己部门收到的指令直接传递给下一个部门的领导，以此类推。
    4、让我们理清一下概念
    神经元，由突触（权重）、层归一化、激活函数等构成，每一个神经元有一个权重，比如wii
    层：权重、归一化、激活、残差
    激活输出F(x)，层输出y=F(x) + x
    残差的本质：在反向传播过程中，全局损失可以跳跃多层（甚至所有层）权重直接回到起点。但是注意前向传播每一层的计算都要做。假设有一个三层残差网络
    前向传播: 输入 -> layer_1 -> layer_2 -> layer_3 -> output
    反向传播: loss -> 输入 (F1(x) = 0, F2(x) = 0, F3(x) = 0)注意是因为当前层激活输出为0导致激活函数更新梯度为0导致当前层的权重不做更新。
    注意激活函数为0不一定当前层的权重为0而且肯定不为0，只是说激活函数为0的话当前层的权重将不被更新。GELU可以防止激活函数为0导致的梯度消失。
    当然3层残差网络可以组合多种反向传播通路，比如从输出直接到输出这条通络是最捷径，而随着网络的深度增加，残差会根据任务的难易程度自动调整网络权重更新
    而选择对应的激活通路，这是一项重要的功能。同时残差还能防止梯度消失，解决从重塑到雕刻的问题。
    5、激活函数和残差的异同
    激活函数（RELU、GELU、Swish-GELU）和残差缺一不可，一般残差shortcut在激活函数之后，搭配起来。
    激活函数解决的是“局部死神经元”问题，即便输入是比较大的负数，它也会给一个微小的负梯度，保证反向传播到这层的梯度不为0，那么该层的权重将会被更新，同时在下次前向传播的时候激活函数输出会逐步转为正数。
    残差永远是在F(x)'接近0的时候才体现出来的，因为残差是在激活函数的基础上加上原始输入x，求导之后也就是激活函数的导数+1，如果激活函数的导数为1，那么也就是导数为2，这个没有意义，但是如果激活函数
    导数接近0，那么最终在经过残差之后梯度就为1。
    
    如果当前层的激活函数梯度为0，那么在反向传播到当前层的时候，当前层的权重将不会做更新，也就是越过当前层，直接将梯度传递给下一层，因为梯度为0+1。如果没有残差呢？首先当前层的梯度为0那么当前层的权重不会
    被更新，其次梯度传递到当前层即为0了，F(x)'=0。继续往前回传，因为没有残差回传给每一层的梯度都为0，那么会造成在梯度为0的当前层往前所有层都为死神经元。这个是局部激活函数GELU无法解决的，即便前面某一层的
    GELU激活函数不为0，因为回传过去的损失梯度为0，所以早层前面层的梯度都为0。在这可以明显体会到GELU和shortcut的区别。如果加上残差呢？即便当前层的激活函数输出为0，+1之后梯度为1，然后回传给当前层的损失值
    会原封不动的直接回传到上一层，上一层即便梯度为0，损失梯度也会原封不动回传到上上层，如果上上层的激活函数输出梯度不为0，如果用了GELU，那么大概率其梯度是一个大于0的值（因为GELU会让其输出不为0），所以
    上上层的权重将会被更新。但是这里有一个注意：如果之前所有层的激活函数输出都为0，那么损失梯度会一路回传到输入层，如果输入层之前没有可更新的权重了，那么本次回传将不会有任何权重被更新。但是如果输入
    层之前还有嵌入层等可更新权重，那么权重将会被更新。权重的更新量=上游传来的误差*当前层激活函数输出*输入特征x
    """
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_out = layer(x)
            if self.use_shortcut and x.shape == layer_out.shape:
                x = x + layer_out
            else:
                x = layer_out
        return x


def print_gradients(model: nn.Module = None, x: torch.tensor = None):
    output = model(x)
    target = torch.tensor([[0.]])
    
    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
        


if __name__ == "__main__":
    """
    # compare_relu_gelu()
    from chapter4_layer_norm import GPT_CONFIG_124M
    ffn = FeedForward(GPT_CONFIG_124M)
    x = torch.randn(2, 3, 768)
    out = ffn(x)
    print(out.shape)
    """

    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )
    print_gradients(model_without_shortcut, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes=layer_sizes, use_shortcut=True)
    print_gradients(model=model_with_shortcut, x=sample_input)
    
    



