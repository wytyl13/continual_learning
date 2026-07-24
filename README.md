## 预训练数据集
数据清洗流水线：语言过滤language filtering、质量过滤quality filtering、去重deduplication、隐私脱敏PII removal
Dolma OLMo
emergent behavior紧急行为
The ability to perform tasks that the model wasn't explicitly trained to perform is called an emergent behavior. The capability isn't explicitly taught durinig training but emerges as a nature consequence of the model's exposure to vast quantities of multilingual data in diverse contexts. The fact that GPT models can "learn" the translation patterns between languages and perform translation tasks even though they weren't specifically trained for it demonstrates the benefits and capabilities of these large-scale, generative language models. we can perform diverse tasks without using diverse models for each.

At its core, an embedding is a mapping from discrete objects, such as words, images, or even entire documents, to points in a continuous vector space -- the primary purpose of embeddings is to convert nonnumeraic data into a format that neural networks can process.

While word embeddings are the most common form of text embedding, there are also embeddings for sentences, paragraph, or whole documents. Senrence or paragraph embeddings are popular choices for retrieval-augmented generation. Retrieval-augmented generation combines generation with retrieval to pull relevant information when genearting text, which is a technical that is beyond the scope of this book. Since our goal is to train GPT-like LLMs, which learn to generate text one word at a time, we will focus on word embeddings.

Word embeddings can have varing dimensions, from one to thousands. A higher dimensionality might capture more nuanced relationships but at the cost of computational efficiency.

While we can use pretrained models such as Word2Vec to generate embeddings for machine learning models, LLMs commonly produce their own embeddings that are part of the input layer and are updated during training. The advantage of optimizing the embeddings as part of the LLM training instead of using Word2Vec is that the embeddings are optimized to the specific task and data at hand.


unset http_proxy https_proxy ftp_proxy all_proxy HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY 

pip install -U huggingface_hub
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download openai-community/gpt2 \
  --local-dir /mnt/wsl/fast_disk/continual_learning/source/hf \
  --include "*.safetensors" "*.json" "*.txt" "*.model"

~/.claude/projects