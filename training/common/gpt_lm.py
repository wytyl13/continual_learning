#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/21 11:23
@Author  : weiyutao
@File    : gpt_lm.py
"""

from tokenizers import Tokenizer

from lm_eval.api.model import LM
import torch
import torch.nn.functional as F


class GPT2LM(LM):
    def __init__(
        self,
        model,
        device,
        tokenizer=None,                 # 新增：可选的 tokenizer 参数（支持 transformers.GPT2Tokenizer）
        model_name: str = "gpt2",
        context_length: int = 1024,
        vocab_size: int = 50257,
        eot_token_id: int = None,
        max_gen_toks: int = 200,
        batch_size: int = 2
    ):
        """
        LM 包装器，支持自定义模型和 transformers 模型

        Args:
            model: 模型实例（可以是自定义的 GPTModel 或 transformers 的 GPT2LMHeadModel）
            device: 设备
            tokenizer: tokenizer 实例（必须提供）
            model_name: 模型名称（保留参数，未使用）
            context_length: 上下文长度
            vocab_size: 词表大小
            max_gen_toks: 最大生成 token 数
            batch_size: 批次大小

        使用示例：
            from tokenizers import Tokenizer
            model = GPTForCausalLM.from_pretrained(...)
            tokenizer = Tokenizer.from_pretrained("gpt2")
            lm = GPT2LM(model=model, device=device, tokenizer=tokenizer)
        """
        super().__init__()
        # 如果模型已经通过 accelerate dispatch_model / device_map="auto" 分配到多卡，
        # 不能再调用 .to(device)，否则会破坏 hook 的设备映射，导致 device 混乱。
        _has_accelerate_hooks = any(hasattr(m, '_hf_hook') for m in model.modules())
        if not _has_accelerate_hooks:
            model = model.to(device)
        self.model = model.eval()
        self._device = device
        self.tokenizer = Tokenizer.from_pretrained(model_name) if tokenizer is None else tokenizer
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")

        self.context_length = context_length
        self.vocab_size = vocab_size
        self._eot_token_id = eot_token_id if eot_token_id is not None else vocab_size - 1
        self._max_gen_toks = max_gen_toks
        self._batch_size = batch_size
    
    @property
    def eot_token_id(self):
        return self._eot_token_id
    
    @property
    def max_length(self):
        return self.context_length

    @property
    def max_gen_toks(self):
        return self._max_gen_toks
    
    @property
    def batch_size(self):
        return self._batch_size
    
    def tok_encode(self, string: str):
        encoded = self.tokenizer.encode(string).ids
        return encoded
    
    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)
    
    def loglikelihood(self, requests):
        """
        给定一个上下文和一个续写，
        1、模型给这段续写打分（所有续写token的概率值之和）
        2、模型是否greedy地预测出了这段续写（谈心采样去二值化每一个续写token，然后取并集all()得到一个布尔值）
        例如：
        上下文："The cat sat on the"
        候选A:  "mat"
        候选B:  "car"
        候选C:  "moon"
        对每个候选调用一次loglikelihood，看模型觉得哪个续写最合理，概率最大（所有续写token的概率值之和）的就是模型的答案。
        """
        
        results = []
        for req in requests:
            # 给定上下文和续写
            # "The cat sat on the" and "mat"
            context, continuation = req.args
            # ctx_ids = self.tok_encode(context) # [464, 3797, 3332, 319, 262]
            # cont_ids = self.tok_encode(continuation) # [2603]
            # cont_len = len(cont_ids) # 1

            # 1. 整体拼接后分词，保证内部不会被错误插入特殊 token
            full_text = context + continuation
            full_ids = self.tok_encode(full_text)
            
            # 2. 单独对 context 分词，用来确定分割点
            ctx_ids = self.tok_encode(context)
            
            # 3. 通过切片获得干净的 continuation ids
            cont_ids = full_ids[len(ctx_ids):]
            cont_len = len(cont_ids)

            
            # 拼接，如果总长度超过max_length就左截断，保证续写永远在末尾
            # [464, 3797, 3332, 319, 262, 2603]
            input_ids = (ctx_ids + cont_ids)[-self.max_length:]
            
            # shape(1, 6)
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self._device)

            with torch.no_grad():
                # shape(1, 6, vocab_size)
                logits = self.model(input_tensor)
                if hasattr(logits, 'logits'):  # transformers 模型输出
                    logits = logits.logits

            # 注意logits的前两个维度等于输入的维度，输入维度是(1, 6)，logits维度是(1, 6, vocab_size)
            # 同时logits预测的6个token是输入维度的next token。即：
            # logits[0, 0]也就是第一个批次第1个token("The")的预测下一个token的概率分布（概率分布的维度就是vocab_size）
            # logits[0, 1]也就是第一个批次第1和第2个token ("The cat")的预测下一个token的概率分布
            # logits[0, 2]也就是第一个批次第1和第2个和第3个token ("The cat sat")的预测下一个token的概率分布
            # 在该案例中，我们关注的是2603也就是mat对应的预测位置，它的预测位置是logits[0, 4:5, :]也就是logits[0, -2:-1, :]位置
            # shape(1, vocab_size)
            cont_logits = logits[0, -cont_len - 1:-1, :]
            
            cont_ids_t = torch.tensor(cont_ids, dtype=torch.long).to(self._device) # 真实内容是[2603]

            # shape: (1, vocab_size)
            # 注意这里的log_softmax就是先对输出分布求softmax得到概率分布，然后对每一个概率分布求对数
            # 但是这里只需要计算得到log_softmax概率分布即可，然后对targets索引对应的概率分布进行求和，
            # 得到的结果越大就说明性能越好（因为这个log_softmax输出的是负数，负数越大说明求log前概率分布越大）。所以不需要加符号，但是如果加了符号就
            # 换个方向，负对数似然越小说明求log前的概率分布越大，模型预测效果越好。但是困惑度不同，困惑度是e^负对数似然的平均值
            # 负对数似然越小越好，但是对数似然是越大越好，因为似然属于0-1（也就是softmax的取值范围），对0-1范围的数取对数，值一定是负数。
            # 越接近1的似然概率，取对数之后其值越接近0，因为e^0=1，所以对数似然越接近0，也就是越小，其对应的似然值越大。
            # 而负对数似然越小，其对应的似然值越大。所以这里不要混淆。
            # 这里的log_softmax也是前向传播中需要做的，前向传播的步骤：
            # 求概率分布，log_probs = F.log_softmax(logits, dim=-1)
            # 求负对数似然：loss = F.nll_loss(log_probs, targets)
            # 以上两步骤可以使用以下函数替代：loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            # 负对数似然本质是拿到所有targets索引对应的概率值，然后求负对数并做平均（因为在当前batch_size无意义），然后最小大该值就是最大化对应目标索引的概率分布
            # 而这里为了做评估，第一步照做log_softmax拿到概率分布，然后直接对目标target索引对应概率进行求和。
            # 后续为了求困惑度PPL，只需要在这儿概率求和的基础上做平均也就是除以len(tokens)，然后在求e的对数即可。
            # loss越低其对应的目标token的概率分布越大，模型性能越好，PPL越低模型越高，因为PPL就是损失和的指数。
            # 注意前向传播中求当前批次损失值的是基于当前批次所有token的负对数似然均值，比如模型输入(batch, seq_len)，输出(batch, seq_len, vocab_size)
            # 然后在计算负对数似然的时候需要合并前两个维度，也就是logits.flatten(0, 1)合并前两个维度，
            # target_batch.flatten()，target_batch本身shape(batch, seq_len)，flatten之后就是shape(batch*seq_len,)
            # 然后在这个基础上去求负对数似然的平均，因为输出(batch, seq_len, vocab_size)中这个seq_len就是基于因果掩码矩阵
            # 对每一个输入索引及之前所有token的next_token预测，注意是错位的，向后位移一个索引。
            # 参考/home/weiyutao/ai/continual_learning/training/trainer_base.py中的calc_loss_batch.
            log_probs = F.log_softmax(cont_logits, dim=-1)
            
            # 从log_probs概率分布中取出位置索引为cont_ids_t对应的概率值
            token_log_probs = log_probs[range(cont_len), cont_ids_t]

            # 对概率值求和，因为续写的token数不一定，有可能是1也可能大于1
            total_log_prob = token_log_probs.sum().item()
            
            # 模型是否贪心地选了正确token？True/False
            is_greedy = (cont_logits.argmax(dim=-1) == cont_ids_t).all().item()
            results.append((total_log_prob, bool(is_greedy)))
        return results
            
    def loglikelihood_rolling(self, requests):
        """
        评估模型对整篇文章的建模能力（困惑度PPL）
        PPL回答的问题：模型预测每个词时，平均面临多少种有效选择？
        PPL=2 模型每次基本能在2个选项里确定答案，很自信
        PPL=100 模型每次面对100个等可能得选项，很困惑
        PPL=50000 完全随机猜（等于词表大小）
        """
        
        results = []
        for req in requests:
            # 整篇文章
            string = req.args[0]
            tokens = self.tok_encode(string)
            stride = self.max_length // 2
            total_log_prob = 0.0
            prev_end = 0 # 上一个窗口结束的位置
            start = 0

            while True:
                end = min(start + self.max_length, len(tokens))
                # shape(max_length,)
                chunk = tokens[start:end]
                
                # shape(1, max_length)
                input_tensor = torch.tensor(chunk).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    # shape(1, max_length, vocab_size)
                    logits = self.model(input_tensor)
                    if hasattr(logits, 'logits'):  # transformers 模型输出
                        logits = logits.logits

                # 只对新的部分计算log pro，因为有重叠，去掉本窗口和上个窗口重叠的部分
                count_from = max(prev_end - start, 1)
                target_tokens = chunk[count_from:]
                if target_tokens:
                    n = len(target_tokens)
                    # 找出target_tokens对应的logits
                    # shape(n, vocab_size)
                    pred_logits = logits[0, count_from-1:count_from-1+n, :]

                    log_probs = F.log_softmax(pred_logits, dim=-1)
                    target_t = torch.tensor(target_tokens).to(self._device)
                    # shape(n,)
                    token_lp = log_probs[range(n), target_t]
                    total_log_prob += token_lp.sum().item()
                prev_end = end
                
                if end >= len(tokens):
                    break
                start += stride
            # rolling不计算is_greedy，只计算困惑度
            # -total_log_prob是负对数似然
            # -total_log_prob / len(tokens) 是负对数似然平均值
            # PPL = exp(-total_log_prob / len(tokens))
            # PPL越低，说明模型对这段文字越熟悉，也就是说明性能越好。
            # 注意：loglikelihood_rolling每个请求只返回一个float（总对数似然），
            # 不像loglikelihood返回(log_prob, is_greedy)元组。
            # 下游任务(如wikitext)用 (loglikelihood,) = results 解包，
            # 如果返回元组会导致 weighted_mean 里 sum() 报 int+tuple 错误。
            results.append(total_log_prob)
        return results

    def generate_until(self, requests):
        """
        用于开放式生成任务，模型需要自己生成答案，直到遇到停止符。
        """
        results = []
        for req in requests:
            # "Q: What is the capital of France?\nA:"
            # gen_kwargs = {"until": ["\n", "Q:"], "max_gen_toks": 50}
            context, gen_kwargs = req.args
            
            # stop_seqs = ["\n", "Q:"]
            # 遇到这些字符串就停止
            stop_seqs = gen_kwargs.get("until", [])

            # shape(1, len(context))
            input_ids = torch.tensor(self.tok_encode(context), dtype=torch.long).unsqueeze(0).to(self._device)

            with torch.no_grad():
                # shape(1, len(context)+self._max_gen_toks)
                out = self.model.generate(
                    input_ids, 
                    max_new_tokens = self._max_gen_toks,
                    top_k=1, # greedy解码（top_k=1等价于argmax）
                    temperature=1.0,
                )
                
            # 只去新生成的部分
            new_tokens = out[0][input_ids.shape[1]:].tolist()

            # 解码字符串
            generated = self.tok_decode(new_tokens)

            # 截断
            for stop in stop_seqs:
                if stop in generated:
                    # 比如生成了"Paris\nQ: What is..."
                    # 遇到了 "\n" -> 截断 -> "Paris"
                    generated = generated[:generated.index(stop)]
            results.append(generated)
        return results
                    

if __name__ == "__main__":
    import math
    from model.gpt2.model import GPTModel, GPTConfig
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = GPTConfig.gpt2_small()
    model = GPTModel(cfg)
    lm = GPT2LM(model=model, device=device)

    class FakeReq:
        def __init__(self, *args):
            self.args = args
    
    requests = [
        FakeReq("The cat sat on the", " mat"),
        FakeReq("The cat sat on the", " car"),
        FakeReq("The cat sat on the", " moon")
    ]

    results = lm.loglikelihood(requests)
    print("=========== loglikelihood ===========")
    labels = [" mat", " car", " moon"]
    for label, (log_prob, is_greedy) in zip(labels, results):
        print(f"{label!r:10s} log_prob={log_prob:8.4f} is_greedy={is_greedy}")
    best = max(range(len(results)), key=lambda i: results[i][0])
    print(f"模型选择： {labels[best]!r}")
    
    
    class FakeRollingReq:
        def __init__(self, string):
            self.args = (string,)
    text = "The quick brown for jumps over the lazy dog. " * 10
    rolling_requests = [FakeRollingReq(text)]
    rolling_results = lm.loglikelihood_rolling(rolling_requests)
    total_lp = rolling_results[0]
    n_tokens = len(lm.tok_encode(text))
    ppl = math.exp(-total_lp / n_tokens)
    print(f"\n=== loglikelihood_rolling ===")
    print(f"total_log_prob={total_lp:.2f} n_tokens={n_tokens} PPL={ppl:.2f}")