## 预训练数据集
数据清洗流水线：语言过滤language filtering、质量过滤quality filtering、去重deduplication、隐私脱敏PII removal
Dolma OLMo
emergent behavior紧急行为
The ability to perform tasks that the model wasn't explicitly trained to perform is called an emergent behavior. The capability isn't explicitly taught durinig training but emerges as a nature consequence of the model's exposure to vast quantities of multilingual data in diverse contexts. The fact that GPT models can "learn" the translation patterns between languages and perform translation tasks even though they weren't specifically trained for it demonstrates the benefits and capabilities of these large-scale, generative language models. we can perform diverse tasks without using diverse models for each.

At its core, an embedding is a mapping from discrete objects, such as words, images, or even entire documents, to points in a continuous vector space -- the primary purpose of embeddings is to convert nonnumeraic data into a format that neural networks can process.

While word embeddings are the most common form of text embedding, there are also embeddings for sentences, paragraph, or whole documents. Senrence or paragraph embeddings are popular choices for retrieval-augmented generation. Retrieval-augmented generation combines generation with retrieval to pull relevant information when genearting text, which is a technical that is beyond the scope of this book. Since our goal is to train GPT-like LLMs, which learn to generate text one word at a time, we will focus on word embeddings.

Word embeddings can have varing dimensions, from one to thousands. A higher dimensionality might capture more nuanced relationships but at the cost of computational efficiency.

While we can use pretrained models such as Word2Vec to generate embeddings for machine learning models, LLMs commonly produce their own embeddings that are part of the input layer and are updated during training. The advantage of optimizing the embeddings as part of the LLM training instead of using Word2Vec is that the embeddings are optimized to the specific task and data at hand.

export http_proxy="http://127.0.0.1:7890" https_proxy="http://127.0.0.1:7890" ftp_proxy="http://127.0.0.1:7890" all_proxy="socks5://127.0.0.1:7890" HTTP_PROXY="http://127.0.0.1:7890" HTTPS_PROXY="http://127.0.0.1:7890" FTP_PROXY="http://127.0.0.1:7890" ALL_PROXY="socks5://127.0.0.1:7890"

unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY 

pip install -U huggingface_hub
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download openai-community/gpt2 \
  --local-dir /mnt/wsl/fast_disk/continual_learning/source/hf/gpt2/124M \
  --include "*.safetensors" "*.json" "*.txt" "*.model"


HF_ENDPOINT=https://hf-mirror.com huggingface-cli download openai-community/gpt2-medium \
  --local-dir /mnt/wsl/fast_disk/continual_learning/source/hf/gpt2/355M \
  --include "*.safetensors" "*.json" "*.txt" "*.model"


HF_ENDPOINT=https://hf-mirror.com huggingface-cli download openai-community/gpt2 \
  --local-dir /root/autodl-tmp/continual_learning_data/source/hf \
  --include "*.safetensors" "*.json" "*.txt" "*.model"

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --local-dir /mnt/wsl/fast_disk/continual_learning/source/hf/llama/tiny_llama \
  --include "*.safetensors" "*.json" "*.txt" "*.model"

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --local-dir /root/autodl-tmp/continual_learning_data/source/hf/llama/tiny_llama \
  --include "*.safetensors" "*.json" "*.txt" "*.model"

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download meta-llama/Llama-2-7b-hf \
  --local-dir /root/autodl-tmp/continual_learning_data/source/hf/llama/llama-2-7b \
  --include "*.safetensors" "*.json" "*.txt" "*.model" --exclude "*.bin"

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download NousResearch/Llama-2-7b-hf \
  --local-dir /root/autodl-tmp/continual_learning_data/source/hf/llama/llama-2-7b \
  --include "*.safetensors" "*.json" "*.txt" "*.model" --exclude "*.bin"


deepspeed --num_gpus=1 --module training.train_llama
python -m training.train_llama


~/.claude/projects




| 符号分类 (Category) | 符号名称 / 含义 (Meaning) | LaTeX 代码 (Code) | 渲染效果 (Render) |
| --- | --- | --- | --- |
| **希腊字母 (Greek)** | 小写基础字母 1 | `\alpha, \beta, \gamma, \delta` | $\alpha, \beta, \gamma, \delta$ |
|  | 小写基础字母 2 | `\epsilon, \theta, \lambda, \mu` | $\epsilon, \theta, \lambda, \mu$ |
|  | 小写基础字母 3 | `\pi, \sigma, \phi, \omega` | $\pi, \sigma, \phi, \omega$ |
|  | 常用大写字母 | `\Gamma, \Delta, \Theta, \Sigma, \Omega` | $\Gamma, \Delta, \Theta, \Sigma, \Omega$ |
| **代数与运算 (Math)** | 乘, 除, 加减, 减加 | `\times, \div, \pm, \mp` | $\times, \div, \pm, \mp$ |
|  | 点乘, 星乘, 圆点 | `\cdot, \ast, \circ` | $\cdot, \ast, \circ$ |
|  | 直和, 张量积 (Kronecker乘积) | `\oplus, \otimes` | $\oplus, \otimes$ |
|  | 分数, 平方根, $n$次根号 | `\frac{a}{b}, \sqrt{x}, \sqrt[n]{x}` | $\frac{a}{b}, \sqrt{x}, \sqrt[n]{x}$ |
| **关系与逻辑 (Logic)** | 小于等于, 大于等于, 不等于 | `\leq, \geq, \neq` | $\leq, \geq, \neq$ |
|  | 约等于, 等价于, 正比于 | `\approx, \equiv, \propto` | $\approx, \equiv, \propto$ |
|  | 远小于, 远大于 | `\ll, \gg` | $\ll, \gg$ |
|  | 因为, 所以 | `\because, \therefore` | $\because, \therefore$ |
|  | 任意, 存在 | `\forall, \exists` | $\forall, \exists$ |
| **集合与区间 (Sets)** | 属于, 不属于 | `\in, \notin` | $\in, \notin$ |
|  | 子集, 包含于, 并集, 交集 | `\subset, \subseteq, \cup, \cap` | $\subset, \subseteq, \cup, \cap$ |
|  | 空集, 无穷大 | `\emptyset, \infty` | $\emptyset, \infty$ |
|  | 实数集, 整数集 (黑板粗体) | `\mathbb{R}, \mathbb{Z}, \mathbb{N}, \mathbb{C}` | $\mathbb{R}, \mathbb{Z}, \mathbb{N}, \mathbb{C}$ |
|  | 算法复杂度 / 损失函数 (花体) | `\mathcal{O}, \mathcal{L}, \mathcal{N}` | $\mathcal{O}, \mathcal{L}, \mathcal{N}$ |
| **微积分 (Calculus)** | 偏导, 梯度 (Nabla) | `\partial, \nabla` | $\partial, \nabla$ |
|  | 积分, 双重积分, 闭合积分 | `\int, \iint, \oint` | $\int, \iint, \oint$ |
|  | 连加, 连乘 | `\sum_{i=1}^n, \prod_{i=1}^n` | $\sum_{i=1}^n, \prod_{i=1}^n$ |
|  | 极限 | `\lim_{x \to \infty}` | $\lim_{x \to \infty}$ |
| **线性代数 (LinAlg)** | 向量, 粗体向量 (矩阵) | `\vec{x}, \mathbf{X}` | $\vec{x}, \mathbf{X}$ |
|  | 矩阵范数 (双竖线) | `\Vert x \Vert` | $\Vert x \Vert$ |
|  | 预测值, 宽帽 (估计值) | `\hat{y}, \widehat{y}` | $\hat{y}, \widehat{y}$ |
|  | 向量内积 (尖括号) | `\langle x, y \rangle` | $\langle x, y \rangle$ |
| **修饰与箭头 (Arrows)** | 左箭头, 右箭头, 双向箭头 | `\leftarrow, \rightarrow, \leftrightarrow` | $\leftarrow, \rightarrow, \leftrightarrow$ |
|  | 推出, 等价 (长箭头) | `\Rightarrow, \Longleftrightarrow` | $\Rightarrow, \Longleftrightarrow$ |
|  | 上划线, 下划线 | `\overline{AB}, \underline{AB}` | $\overline{AB}, \underline{AB}$ |
|  | 顶部波浪线, 点导数 (物理/时间) | `\tilde{x}, \dot{x}, \ddot{x}` | $\tilde{x}, \dot{x}, \ddot{x}$ |



# 现在有以下问题
```
项目中很多函数需要进行config的转换：openai模型权重读取的config转换为自定义的gpt2_model config，自定义的config转换为transformers config
transformers config转换为 自定义config。

还有就是在训练的时候：我可以使用transformers类去加载模型训练，还可以使用自定义model加载模型训练，对应的这两个模型保存的时候config和model配置不一样
我现在可以加载任何训练模型（包括自定义model训练保存的模型，还有transformers类加载训练的模型），但是我现在无法兼容加载这两个模型，因为这两个模型
保存的权重命名规则不一样啊？还有config也不一样

还有transformer的config中需要配置eos_token_id pad_token_id这两个参数，但是我的自定义模型架构config配置中没有
token_ids = model.generate(
                input_ids=encoded,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=50256,
                eos_token_id=50256,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                top_k=top_k if top_k and temperature > 0 else 50
            )
因为model.generate和我的自定义gptmodel中的generate不一样

以上问题帮我完整的分析一下重构思路，我希望大道至简，不希望我的框架做的像transformers一样有很多的判断条件
```

# BF16 与 FP16 的核心区别

`bf16` (Bfloat16) 和 `float16` (FP16) 都是占用 **16个比特（2个字节）**的半精度浮点数。它们的核心区别在于**对这16个比特的分配策略不同**，这导致了它们在“表达范围”和“表达精度”上的巨大差异。

简单来说：**FP16 选择了更高的精度，而 BF16 选择了更大的范围。**

## 1. 比特位分配对比
3.14159 * 10^5
  3.14159 就是“尾数”
  5 就是“指数”
  10 是基数（在计算机里基数是 2）

十进制 125 -> 科学计数法 1.25 * 10^2

十进制 6.5 -> 计算机的科学技术法
  二进制110 -> 十进制6=2^2+2^1
  二进制0.1 -> 十进制0.5=2^-1
  二进制110.1 -> 十进制6.5

十进制6.5的二进制存储规则：
  二进制110.1=1.101*2^2，小数点左移2位，2^2中底数2就是基数，指数2就是指数，1.101中的101就是尾数
  因此十进制的6.5换算成计算机的科学技术法是1.101*2^2
  符号是正数对应的符号位0，指数为2，尾数是101，为了同时表示正指数和负指数，在存储
  指数时，会给真实指数加上一个固定的偏移量，然后再转成二进制存进去。FP32和BF16的偏移量是127，FP16的偏移量是15

单精度float32：符号位 1bit，指数位 8bit，尾数位 23bit。
  符号位正数是0，负数是1.
  指数为8bit，真实指数2+127=129，129二进制是10000001，129=2^7+2^0，8bit的第0位置和第7位置为1，其余地方为0。
  因此二进制是10000001
  尾数位是23bit，放进去真实二进制101即可（小数是从左往右补）
  1.101中的整数1不用存储，因为一定是1，在十进制里面这个数字可能是0-9，但是在二进制表示中一定是1

  0 | 10000001 | 10100000000000000000000
  符号位0表示正数，指数位10000001转换为十进制为129，减去偏移量127等于2，表示小数点向右偏移2位，
  10100000000000000000000表示1.101，向右偏移2位最终二进制表示为110.1，转换为十进制为2^2+2^1+2^-1=6.5

半精度FP16：符号位 1bit，指数位 5bit，指数偏移为15，尾数位 10bit
  十进制6.5半精度FP16的存储为：指数为十进制=15+2=17，十进制17二进制表示为=2^4+2^0
  0 | 10001 | 1010000000
  10001转换为十进制为17，减去偏移量15等于2，1.101向右偏移2位最终二进制表示为110.1

BF16：符号位 1bit，指数位 8bit，指数偏移量为127，尾数位=16-1-8=7bit
  十进制6.5 BF16存储为：指数为十进制127+2=29，十进制29二进制表示为=2^7+2^0=128+1=129
  0 | 10000001 | 1010000
  将单精度float32从中间截断，只保留前16位就是BF16

比如 1.0 * 2^20，FP32和BF16因为指数上限是127（因为指数位均为8bit，考虑偏移量127，
8bit的存储的十进制最大值为255，最小值是0，去掉两个特殊情况8bit全为0或者全为1，全为0代表数字0和极其微小的非规格化数字
全1保留给无穷大和非数字Nan使用，可以使用的指数只有1到254，这是254个指数情况，254是最大的指数状态，再减去127漂移量，
最大的正指数状态为127，也就是这两个数据类型可以保存的最大数值为尾数全为1的情况下，指数向右偏移127位，可以表示的最小
状态为1-127=-126）

什么时候指数为是负？比如一个十进制小数0.125=2^-3，二进制形式为0.001，现在要存储二进制0.001，小数点左边必须是1，
所以0.001=1*2^-3，指数为是-3，尾数位0，考虑指数偏移量，键入时BF16最终偏移后的指数为是-3+127=124=64+32+16+8+4=2^6+2^5+2^4+2^3+2^2
0 | 01111100 | 0000000
以上如果这个小数位很大，比如2^-126，其指数位为-126，加上偏移量127等于1，也就是说BF16可以表示的最接近0的正小数为
0 | 00000001 | 1111111
该数值的十进制表示为 1.9921875 * 2^-126



| 数据类型 | 符号位 (Sign) | 指数位 (Exponent) | 尾数位 (Fraction) | 特点 |
| :--- | :--- | :--- | :--- | :--- |
| **单精度 FP32** | 1 bit | 8 bit | 23 bit | 标准基准，范围大，精度高。 |
| **半精度 FP16** | 1 bit | **5 bit** | **10 bit** | **精度较高**，但最大只能表示到 ~65504。 |
| **大脑浮点 BF16** | 1 bit | **8 bit** | **7 bit** | **动态范围极大**（同 FP32），但精度较低。 |

> **什么是 BF16？**
> BF16 全称是 Brain Floating Point，最初由 Google Brain 团队为机器学习专门设计。它的设计极其巧妙：直接**生硬地截断** FP32 的后 16 位尾数，保留了和 FP32 完全相同的 8 位指数。这意味着 BF16 能表示的数字范围和 FP32 一模一样（最大到 ~3.4 x 10^38）。

---

## 2. 工程部署中的核心差异

这两种数据类型在深度学习的训练和推理架构中扮演着截然不同的角色：

### 2.1 神经网络训练：为什么大模型偏爱 BF16？
在训练深度神经网络时，梯度的变化范围极广。如果使用 FP16，由于其最大值只有 65504，非常容易发生**梯度溢出**（Overflow）或下溢出（Underflow），导致训练崩溃（出现 NaN）。
*   **BF16 的优势：** 拥有和 FP32 一样的动态范围，几乎永远不会溢出。在混合精度训练中，直接使用 BF16 可以省去复杂的“损失缩放”（Loss Scaling）操作，训练极度稳定。目前绝大多数百亿级 LLM（大语言模型）的预训练都采用 BF16。

### 2.2 端侧与实时推理：为什么 TensorRT 常用 FP16？
在部署边缘侧实时计算机视觉（CV）系统时，通常面对的是推理（Inference）阶段。此时模型的权重已经固定，数值范围是已知且可控的。
*   **FP16 的优势：** 相比 BF16，FP16 多了 3 个比特的尾数位，**精度更高**。在处理图像数据时，这微小的精度优势有助于更好地保留空间特征和细节。
*   **落地实践：** 部署高性能 C++ 推理引擎（如 TensorRT）时，通常可以通过校准（Calibration）或量化感知训练，将模型权重安全地映射到 FP16 的范围内，从而在不损失精度的前提下，最大化利用 GPU 的 Tensor Core，轻松达成 200ms 以内的实时延迟要求。