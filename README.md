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
