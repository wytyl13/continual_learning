#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/19 11:15
@Author  : weiyutao
@File    : trainer_base.py
"""
import torch
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from safetensors.torch import save_file
from huggingface_hub import split_torch_state_dict_into_shards
from dataclasses import asdict

from model.gpt2.model import ModelType
from utils.logger import get_logger

logger = get_logger(__name__)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text).ids
    # 增加批次维度用于模型推理
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.tensor, tokenizer):
    # 去掉批次维度用于解码
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    calculate the batch loss.
    """
    # input_batch(batch, seq_length)
    # accelerate接管之后不再需要to指定设别
    # input_batch = input_batch.to(device)

    # target_batch(batch, seq_length)
    # target_batch = target_batch.to(device)

    # iuput(batch, seq_length) -> out(batch, seq_length, vocab_size)
    logits = model(input_batch)
    
    if hasattr(logits, 'logits'):
        logits = logits.logits
    else:
        logits = logits
    
    # logits.flatten(0, 1): input(batch, seq_length, vocab_size) -> output(batch*seq_length, vocab_size)
    # target_batch.flatten(): input(batch, seq_length) -> output(batch*seq_length,)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def generate_text_simple(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=None):
    """
    通用的文本生成函数（默认实现）。
    支持温度采样和 top-k 采样。

    Args:
        model: 语言模型
        idx: 输入 token ids，形状 (batch, seq_len)
        max_new_tokens: 生成的最大 token 数量
        context_size: 模型的上下文窗口大小
        temperature: 温度参数，控制随机性
                    - temperature = 1.0: 标准采样
                    - temperature < 1.0: 更确定性（更保守）
                    - temperature > 1.0: 更随机（更有创造性）
                    - temperature → 0: 接近贪婪解码
        top_k: 只从概率最高的 top_k 个 token 中采样
              - None: 不限制（从全部 vocab 采样）
              - int: 只考虑前 k 个最高概率的 token

    Returns:
        生成的 token ids，形状 (batch, seq_len + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        # 截取最后 context_size 个 token 作为输入
        idx_cond = idx[:, -context_size:]

        # 前向传播获取 logits
        logits = model(idx_cond)  # (batch, seq_len, vocab_size)

        # 只取最后一个位置的 logits
        logits = logits[:, -1, :]  # (batch, vocab_size)

        # 应用 top-k 过滤
        if top_k is not None:
            # 获取 top_k 个最大值
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            # 将小于第 k 大的值设为 -inf（softmax 后概率接近 0）
            logits[logits < v[:, [-1]]] = float('-inf')

        # 应用温度缩放
        logits = logits / temperature

        # 转换为概率分布
        probas = torch.softmax(logits, dim=-1)

        # 从概率分布中采样
        idx_next = torch.multinomial(probas, num_samples=1)  # (batch, 1)

        # 拼接到序列末尾
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_and_print_sample(model, tokenizer, device, start_context, context_size,
                              max_new_tokens=50, temperature=1.0, top_k=None, model_type=ModelType.CUSTOM):
    """
    生成并打印文本样本。

    优先使用模型自己的 generate 方法（如果存在），
    否则使用通用的 generate_text_simple 函数。

    Args:
        model: 语言模型
        tokenizer: 分词器
        device: 设备（CPU/GPU）
        start_context: 起始文本
        context_size: 模型的上下文窗口大小
        max_new_tokens: 生成的最大 token 数量
        temperature: 温度参数，控制随机性（默认 1.0）
        top_k: 只从概率最高的 top_k 个 token 中采样（默认 None，不限制）
    """
    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        if model_type == ModelType.TRANSFORMERS:
            # transformers 模型使用 GenerationConfig
            attention_mask = torch.ones_like(encoded)
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
        # 检查模型是否有自己的 generate 方法
        elif hasattr(model, 'generate') and callable(getattr(model, 'generate')):
            # 使用模型自己的 generate 方法
            token_ids = model.generate(
                encoded,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k
            )
        else:
            # 使用通用的生成函数
            token_ids = generate_text_simple(
                model=model,
                idx=encoded,
                max_new_tokens=max_new_tokens,
                context_size=context_size,
                temperature=temperature,
                top_k=top_k
            )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    logger.info(decoded_text.replace("\n", " "))
    model.train()
    

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    # 可视化训练过程中保存的每个epoch下的损失
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Trainig loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig("loss_plot.png", dpi=300, bbox_inches="tight")


def save_model(model, save_dir, tokenizer=None, state_dict=None):
    os.makedirs(save_dir, exist_ok=True)
    if state_dict is None:
        # 保存权重，处理共享权重
        state_dict = model.state_dict()
    # 如果有weight tying，复制一份lm_head.weight避免共享内存
    config = getattr(model, 'cfg', None) or getattr(model, 'config', None)
    if config and getattr(config, 'tie_word_embeddings', False):
        if 'lm_head.weight' in state_dict:
            state_dict['lm_head.weight'] = state_dict['lm_head.weight'].clone()
    split = split_torch_state_dict_into_shards(state_dict, max_shard_size="5GB")
    for filename, tensor_keys in split.filename_to_tensors.items():
        shard = {k: state_dict[k] for k in tensor_keys}
        save_file(shard, os.path.join(save_dir, filename))
    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        with open(os.path.join(save_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)

    # 保存config
    if config:
        # 兼容dataclass和transformers config
        if hasattr(config, 'to_dict'):  # transformers config
            config_dict = config.to_dict()
        else:  # dataclass
            config_dict = asdict(config)
        with open(f"{save_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    # 保存tokenizer
    if tokenizer is not None:
        if hasattr(tokenizer, 'save'):  # tokenizers库
            tokenizer.save(f"{save_dir}/tokenizer.json")
            logger.info(f"Tokenizer saved: {save_dir}/tokenizer.json")
        elif hasattr(tokenizer, 'save_pretrained'):  # transformers库
            tokenizer.save_pretrained(save_dir)
            logger.info(f"Tokenizer saved: {save_dir}")
        else:
            logger.warning(f"tokenizer类型 {type(tokenizer)} 不支持保存")

    logger.info(f"Model saved: {save_dir}")


def train_model(model, train_loader, val_loader,
                optimizer, device, num_epochs,
                eval_freq, eval_iter, start_context, context_size, tokenizer,
                generate_sample_temperature=1.0, generate_sample_top_k=None,
                patience=5, save_dir="best_model.pt", model_type="gpt2", accelerator=None):
    """
    通用的模型训练函数。

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 设备（CPU/GPU）
        num_epochs: 训练轮数
        eval_freq: 每多少步评估一次
        eval_iter: 评估时使用多少批次
        start_context: 用于生成样本的起始文本
        context_size: 模型的上下文窗口大小
        tokenizer: 分词器
        generate_sample_temperature: 生成样本时的温度参数（默认 1.0）
        generate_sample_top_k: 生成样本时的 top_k 参数（默认 None）
    """
    train_losses, val_losses, track_token_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )

            if accelerator is not None:
                # accelerator 模式
                accelerator.backward(loss)
            else:
                # 原生 PyTorch 模式
                loss.backward()
            optimizer.step()

            # 计算累积token数
            tokens_seen += input_batch.numel()

            global_step += 1

            # 定期评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter=eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_token_seen.append(tokens_seen)
                logger.info(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
                if  val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if accelerator is not None:
                        accelerator.wait_for_everyone() # 同步所有进程
                        unwrapped_model = accelerator.unwrap_model(model)  # 去掉 DDP/ZeRO 包装
                        # ZeRO-3 需要用 get_state_dict 触发 all-gather，
                        # 直接调 state_dict() 只得到本卡分片（shape=[0]）
                        state_dict = accelerator.get_state_dict(model)
                        if accelerator.is_main_process:  # 只在主进程保存
                            alloc = torch.cuda.memory_allocated() / 1024**3
                            reserved = torch.cuda.memory_reserved() / 1024**3
                            print(f"GPU memory — allocated: {alloc:.2f} GiB | reserved: {reserved:.2f} GiB")
                            save_model(unwrapped_model, save_dir, tokenizer, state_dict)
                        # all-gather 会临时占满整卡显存，保存完立即释放
                        del state_dict
                        torch.cuda.empty_cache()
                        accelerator.wait_for_everyone()
                    else:
                        save_model(model, save_dir, tokenizer)
                    
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping triggered (patience={patience}).")
                        return train_losses, val_losses, track_token_seen
                    

        # 每个 epoch 结束后生成样本
        generate_and_print_sample(
            model, tokenizer, device, start_context, 
            context_size,
            temperature=generate_sample_temperature,
            top_k=generate_sample_top_k,
            model_type=model_type
        )

    return train_losses, val_losses, track_token_seen