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
from utils.logger import get_logger

logger = get_logger(__name__)


class GPT2LM(LM):
    def __init__(
        self,
        model,
        device,
        tokenizer=None,
        model_name: str = "gpt2",
        context_length: int = 1024,
        vocab_size: int = 50257,
        eot_token_id: int = None,
        max_gen_toks: int = 200,
        batch_size: int = 2,
    ):
        """
        LM 包装器，支持自定义模型和 transformers 模型，自动适配 data/pipeline parallelism。

        Args:
            model: 单个模型或模型列表
                   - 单个模型 → pipeline parallelism（模型可能跨多卡，串行推理）
                   - 模型列表 → data parallelism（每个模型占一张卡，并行推理）
            device: 主设备（单模型时使用；多模型时忽略，各模型用自己的设备）
            tokenizer: tokenizer 实例（必须提供）
            model_name: 模型名称（保留参数，未使用）
            context_length: 上下文长度
            vocab_size: 词表大小
            max_gen_toks: 最大生成 token 数
            batch_size: 批次大小（data parallel 时，每个模型的实际 batch = batch_size // len(models)）

        使用示例：
            # Pipeline parallelism (单个模型，可能跨多卡)
            model = LlamaForCausalLM.from_pretrained(..., device_map="auto")
            lm = GPT2LM(model=model, device=device, tokenizer=tokenizer)

            # Data parallelism (多个模型，各占一卡)
            model_0 = LlamaForCausalLM.from_pretrained(..., device_map="cuda:0")
            model_1 = LlamaForCausalLM.from_pretrained(..., device_map="cuda:1")
            lm = GPT2LM(model=[model_0, model_1], device=None, tokenizer=tokenizer)
        """
        super().__init__()

        # 支持单个模型或模型列表
        if isinstance(model, list):
            self.models = [m.eval() for m in model]
            self.use_data_parallel = len(self.models) > 1
            self._device = device if device else next(self.models[0].parameters()).device
        else:
            # 如果模型已经通过 accelerate dispatch_model / device_map="auto" 分配到多卡，
            # 不能再调用 .to(device)，否则会破坏 hook 的设备映射，导致 device 混乱。
            _has_accelerate_hooks = any(hasattr(m, '_hf_hook') for m in model.modules())
            if not _has_accelerate_hooks:
                model = model.to(device)
            self.models = [model.eval()]
            self.use_data_parallel = False
            self._device = device

        self.tokenizer = Tokenizer.from_pretrained(model_name) if tokenizer is None else tokenizer
        if self.tokenizer is None:
            raise ValueError("tokenizer must be provided")

        self.context_length = context_length
        self.vocab_size = vocab_size
        self._eot_token_id = eot_token_id if eot_token_id is not None else vocab_size - 1
        self._max_gen_toks = max_gen_toks
        # Data parallel 时每个模型的实际 batch_size 应该更小（因为同时跑多个模型）
        # 为了保持总吞吐量不变，每个模型用 batch_size // n_models
        self._batch_size = batch_size // len(self.models) if self.use_data_parallel else batch_size

        if self.use_data_parallel:
            logger.info(f"[GPT2LM] Data Parallelism: {len(self.models)} 模型, "
                       f"每模型 batch_size={self._batch_size}")
        else:
            logger.info(f"[GPT2LM] Pipeline Parallelism: 1 模型, batch_size={self._batch_size}")

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
        """loglikelihood 入口：统一走真正批量前向传播"""
        return self._loglikelihood_batched(requests)
    
    def loglikelihood_rolling(self, requests):
        """loglikelihood_rolling 入口：统一走真正批量前向传播"""
        return self._loglikelihood_rolling_batched(requests)

    def _loglikelihood_batched(self, requests):
        """
        真正的批量推理：将多条 requests 左padding对齐后合并成 (B, L) 一次 forward pass。
        支持 data parallelism：若有多个模型，并行处理各自的请求子集。

        为什么用左padding（而不是右padding）：
          - 续写 tokens 永远在序列末尾
          - 左padding让所有样本的末尾对齐，logits[-cont_len-1:-1] 索引方式可以统一使用
          - 右padding会把续写token推到中间，位置索引变复杂
        注意：需要传入 attention_mask 来让模型忽略 padding 位置。
        """
        if not self.use_data_parallel:
            # Pipeline parallelism 或单GPU：串行处理
            return self._process_requests(requests, self.models[0], self._device)

        # Data parallelism：拆分请求到多个模型，并行处理
        import threading
        n_models = len(self.models)
        chunk_size = (len(requests) + n_models - 1) // n_models  # 向上取整
        chunks = [requests[i:i+chunk_size] for i in range(0, len(requests), chunk_size)]

        results_list = [None] * n_models
        threads = []

        for i, (chunk, model) in enumerate(zip(chunks, self.models)):
            device = next(model.parameters()).device
            t = threading.Thread(
                target=lambda idx, reqs, m, d: results_list.__setitem__(idx, self._process_requests(reqs, m, d)),
                args=(i, chunk, model, device)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 合并结果
        results = []
        for r in results_list:
            if r:
                results.extend(r)
        return results

    def _process_requests(self, requests, model, device):
        """处理一批请求的核心逻辑，供串行和并行两种模式复用"""
        results = []
        pad_id = 0

        for batch_start in range(0, len(requests), self._batch_size):
            batch_reqs = requests[batch_start : batch_start + self._batch_size]

            # 1. 对每条请求做分词
            batch_input_ids = []
            batch_cont_ids  = []
            for req in batch_reqs:
                context, continuation = req.args
                full_ids = self.tok_encode(context + continuation)
                ctx_ids  = self.tok_encode(context)
                cont_ids = full_ids[len(ctx_ids):]
                input_ids = (ctx_ids + cont_ids)[-self.max_length:]
                batch_input_ids.append(input_ids)
                batch_cont_ids.append(cont_ids)

            # 2. 左padding对齐到同一长度
            max_len = max(len(x) for x in batch_input_ids)
            padded       = []
            attn_masks   = []
            for ids in batch_input_ids:
                pad_len = max_len - len(ids)
                padded.append([pad_id] * pad_len + ids)
                attn_masks.append([0] * pad_len + [1] * len(ids))

            input_tensor = torch.tensor(padded,     dtype=torch.long).to(device)
            attn_tensor  = torch.tensor(attn_masks, dtype=torch.long).to(device)

            # 3. 单次 forward pass，batch_size = B
            with torch.no_grad():
                out = model(input_tensor, attention_mask=attn_tensor)
                logits = out.logits if hasattr(out, 'logits') else out   # (B, L, V)

            # 4. 逐样本取续写位置的 logits 并计算分数
            for j, cont_ids in enumerate(batch_cont_ids):
                cont_len     = len(cont_ids)
                # 由于左padding，续写 token 对应的 logit 永远在末尾相同位置
                cont_logits  = logits[j, -cont_len - 1 : -1, :]          # (cont_len, V)
                cont_ids_t   = torch.tensor(cont_ids, dtype=torch.long).to(device)
                log_probs    = F.log_softmax(cont_logits, dim=-1)
                token_lp     = log_probs[range(cont_len), cont_ids_t]
                total_lp     = token_lp.sum().item()
                is_greedy    = (cont_logits.argmax(dim=-1) == cont_ids_t).all().item()
                results.append((total_lp, bool(is_greedy)))

        return results

    def _loglikelihood_serial(self, requests):
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
       
    def _loglikelihood_parallel(self, requests):
        """并行推理（使用多个 CUDA streams）"""
        results = [None] * len(requests)
        
        # 分配每个请求到不同的 stream
        for i, req in enumerate(requests):
            stream_idx = i % self.num_streams
            stream = self.streams[stream_idx]
            
            with torch.cuda.stream(stream):
                context, continuation = req.args
                full_text = context + continuation
                full_ids = self.tok_encode(full_text)
                ctx_ids = self.tok_encode(context)
                cont_ids = full_ids[len(ctx_ids):]
                cont_len = len(cont_ids)
                
                input_ids = (ctx_ids + cont_ids)[-self.max_length:]
                input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self._device)
                
                with torch.no_grad():
                    logits = self.model(input_tensor)
                    if hasattr(logits, 'logits'):
                        logits = logits.logits
                
                cont_logits = logits[0, -cont_len - 1:-1, :]
                cont_ids_t = torch.tensor(cont_ids, dtype=torch.long).to(self._device)
                log_probs = F.log_softmax(cont_logits, dim=-1)
                token_log_probs = log_probs[range(cont_len), cont_ids_t]
                total_log_prob = token_log_probs.sum().item()
                is_greedy = (cont_logits.argmax(dim=-1) == cont_ids_t).all().item()
                
                results[i] = (total_log_prob, bool(is_greedy))
        
        # 同步所有 streams
        for stream in self.streams:
            stream.synchronize()
        
        return results
            
    def _loglikelihood_rolling_batched(self, requests):
        """
        真正的批量 rolling 推理：把所有文档的所有滑动窗口摊平，按 batch_size 批量做 forward pass。
        支持 data parallelism：若有多个模型，并行处理各自的文档子集。

        思路：
          - 每篇文档的滑动窗口是顺序依赖的（total_log_prob 累加），但窗口之间彼此独立。
          - 不同文档的窗口也彼此独立。
          - 因此把所有（文档, 窗口）二元组摊平成一个 list，统一批量推理，最后按 doc_idx 归并。

        左padding说明：
          - 窗口内的 token 占据 padded tensor 的末尾，padding 在前。
          - count_from（需要跳过的重叠前缀）是相对于原始 chunk 的偏移量，
            换算到 padded tensor 的偏移 = pad_len + count_from。
        """
        if not self.use_data_parallel:
            # Pipeline parallelism 或单GPU：串行处理
            return self._process_rolling_requests(requests, self.models[0], self._device)

        # Data parallelism：拆分文档到多个模型，并行处理
        import threading
        n_models = len(self.models)
        chunk_size = (len(requests) + n_models - 1) // n_models
        chunks = [requests[i:i+chunk_size] for i in range(0, len(requests), chunk_size)]

        results_list = [None] * n_models
        threads = []

        for i, (chunk, model) in enumerate(zip(chunks, self.models)):
            device = next(model.parameters()).device
            t = threading.Thread(
                target=lambda idx, reqs, m, d: results_list.__setitem__(idx, self._process_rolling_requests(reqs, m, d)),
                args=(i, chunk, model, device)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 合并结果
        results = []
        for r in results_list:
            if r:
                results.extend(r)
        return results

    def _process_rolling_requests(self, requests, model, device):
        """处理 rolling 请求的核心逻辑，供串行和并行两种模式复用"""
        # 1. 为每篇文档预计算所有滑动窗口
        #    all_windows: list of (doc_idx, chunk_tokens, count_from)
        all_windows = []
        for doc_idx, req in enumerate(requests):
            tokens    = self.tok_encode(req.args[0])
            stride    = self.max_length // 2
            start     = 0
            prev_end  = 0
            while True:
                end        = min(start + self.max_length, len(tokens))
                chunk      = tokens[start:end]
                count_from = max(prev_end - start, 1)
                all_windows.append((doc_idx, chunk, count_from))
                prev_end = end
                if end >= len(tokens):
                    break
                start += stride

        # 2. 按 batch_size 批量推理所有窗口
        doc_log_probs = [0.0] * len(requests)
        pad_id = 0

        for batch_start in range(0, len(all_windows), self._batch_size):
            batch = all_windows[batch_start : batch_start + self._batch_size]

            # 左padding对齐
            max_len    = max(len(w[1]) for w in batch)
            padded     = []
            attn_masks = []
            for _, chunk, _ in batch:
                pad_len = max_len - len(chunk)
                padded.append([pad_id] * pad_len + list(chunk))
                attn_masks.append([0] * pad_len + [1] * len(chunk))

            input_tensor = torch.tensor(padded,     dtype=torch.long).to(device)
            attn_tensor  = torch.tensor(attn_masks, dtype=torch.long).to(device)

            with torch.no_grad():
                out    = model(input_tensor, attention_mask=attn_tensor)
                logits = out.logits if hasattr(out, 'logits') else out   # (B, L, V)

            for j, (doc_idx, chunk, count_from) in enumerate(batch):
                target_tokens = chunk[count_from:]
                if not target_tokens:
                    continue
                n       = len(target_tokens)
                pad_len = max_len - len(chunk)
                # count_from 是在原始 chunk 里的偏移，换算到 padded tensor 的偏移
                actual_from = pad_len + count_from
                pred_logits = logits[j, actual_from - 1 : actual_from - 1 + n, :]  # (n, V)
                log_probs   = F.log_softmax(pred_logits, dim=-1)
                target_t    = torch.tensor(target_tokens, dtype=torch.long).to(device)
                token_lp    = log_probs[range(n), target_t]
                doc_log_probs[doc_idx] += token_lp.sum().item()

        return doc_log_probs

    def _loglikelihood_rolling_serial(self, requests):
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

    def _loglikelihood_rolling_parallel(self, requests):
        """并行 rolling 推理"""
        results = [None] * len(requests)
        
        for i, req in enumerate(requests):
            stream_idx = i % self.num_streams
            stream = self.streams[stream_idx]
            
            with torch.cuda.stream(stream):
                string = req.args[0]
                tokens = self.tok_encode(string)
                stride = self.max_length // 2
                total_log_prob = 0.0
                prev_end = 0
                start = 0
                
                while True:
                    end = min(start + self.max_length, len(tokens))
                    chunk = tokens[start:end]
                    input_tensor = torch.tensor(chunk).unsqueeze(0).to(self._device)
                    
                    with torch.no_grad():
                        logits = self.model(input_tensor)
                        if hasattr(logits, 'logits'):
                            logits = logits.logits
                    
                    count_from = max(prev_end - start, 1)
                    target_tokens = chunk[count_from:]
                    if target_tokens:
                        n = len(target_tokens)
                        pred_logits = logits[0, count_from-1:count_from-1+n, :]
                        log_probs = F.log_softmax(pred_logits, dim=-1)
                        target_t = torch.tensor(target_tokens).to(self._device)
                        token_lp = log_probs[range(n), target_t]
                        total_log_prob += token_lp.sum().item()
                    
                    prev_end = end
                    if end >= len(tokens):
                        break
                    start += stride
                
                results[i] = total_log_prob
        
        # 同步所有 streams
        for stream in self.streams:
            stream.synchronize()
        
        return results

    def generate_until(self, requests):
        """
        用于开放式生成任务，模型需要自己生成答案，直到遇到停止符。
        支持批量推理和 data parallelism。
        """
        if not self.use_data_parallel:
            # Pipeline parallelism 或单GPU：批量处理
            return self._generate_batch(requests, self.models[0])

        # Data parallelism：拆分请求到多个模型，并行处理
        import threading
        n_models = len(self.models)
        chunk_size = (len(requests) + n_models - 1) // n_models
        chunks = [requests[i:i+chunk_size] for i in range(0, len(requests), chunk_size)]

        results_list = [None] * n_models
        threads = []

        for i, (chunk, model) in enumerate(zip(chunks, self.models)):
            t = threading.Thread(
                target=lambda idx, reqs, m: results_list.__setitem__(idx, self._generate_batch(reqs, m)),
                args=(i, chunk, model)
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 合并结果
        results = []
        for r in results_list:
            if r:
                results.extend(r)
        return results

    def _generate_batch(self, requests, model):
        """
        批量生成的核心逻辑。

        难点：不同 request 的 stop_seqs 可能不同，需要逐个检查停止条件。
        策略：先统一生成到 max_new_tokens，再逐个截断。
        """
        results = []
        device = next(model.parameters()).device

        for batch_start in range(0, len(requests), self._batch_size):
            batch_reqs = requests[batch_start : batch_start + self._batch_size]

            # 1. 分词每条 context
            batch_contexts = []
            batch_stop_seqs = []
            for req in batch_reqs:
                context, gen_kwargs = req.args
                batch_contexts.append(self.tok_encode(context))
                batch_stop_seqs.append(gen_kwargs.get("until", []))

            # 2. 左padding对齐（生成时需要从右边开始）
            max_len = max(len(c) for c in batch_contexts)
            pad_id = self.tokenizer.token_to_id("<pad>") if hasattr(self.tokenizer, 'token_to_id') else 0
            if pad_id is None:
                pad_id = 0  # fallback

            padded = []
            attn_masks = []
            for ctx in batch_contexts:
                pad_len = max_len - len(ctx)
                padded.append([pad_id] * pad_len + ctx)
                attn_masks.append([0] * pad_len + [1] * len(ctx))

            input_ids = torch.tensor(padded, dtype=torch.long).to(device)       # (B, L)
            attn_mask = torch.tensor(attn_masks, dtype=torch.long).to(device)   # (B, L)

            # 3. 批量生成
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=self._max_gen_toks,
                    do_sample=False,  # greedy decoding
                    pad_token_id=pad_id,
                    use_cache=True,   # 确保启用 KV cache
                )

            # 4. 逐样本解码并应用 stop_seqs
            for i, (ctx_len, stop_seqs) in enumerate(zip([len(c) for c in batch_contexts], batch_stop_seqs)):
                # 跳过原始 context（包括左padding），只取新生成的部分
                # outputs[i] 的前 max_len 是 padded context，之后是生成的 token
                generated_ids = outputs[i][max_len:].tolist()
                generated_text = self.tok_decode(generated_ids)

                # 截断到第一个 stop_seq
                for stop in stop_seqs:
                    if stop in generated_text:
                        generated_text = generated_text[:generated_text.index(stop)]
                        break

                results.append(generated_text)

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