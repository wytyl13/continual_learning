#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/08/11
@Author : weiyutao
@File : model.py

Qwen2/2.5 model architecture.
完全复用 LLaMA 组件，仅配置不同。
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional, Union, Generator
import json
import os
from accelerate import init_empty_weights
from tokenizers import Tokenizer

from model.qwen.config import QwenConfig, MODEL_CONFIGS
from model.llama.model import (
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    repeat_kv,
    Attention,
    SwiGLU,
    DecoderLayer
)

from config import SOURCE_DIR, OUT_DIR

from utils.model_loader import (
    resolve_checkpoint_files,
    build_auto_device_map,
    fast_load_weights,
    materialize_meta_model,
    auto_device_map
)

from utils.logger import get_logger, log_stage
from utils.enums import LoadMode, ModelName


logger = get_logger(__name__)


class QwenModel(nn.Module):
    """
    Qwen2/2.5 Model.
    架构与 LLaMA 完全相同，直接复用 LLaMA 组件。
    """
    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.drop_embeddings = nn.Dropout(cfg.embd_pdrop)

        # 直接使用 LLaMA 的 DecoderLayer，传入 QwenConfig
        # DecoderLayer 内部会使用 cfg 的所有参数（qkv_bias, layer_norm_epsilon 等）
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.layer_norm_epsilon)

        # RoPE，注册为buffer，不参与梯度
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.gradient_checkpointing = False


    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            idx: (batch, seq_len) token id 序列
            attention_mask: (batch, seq_len) left padding mask applied for the inference.
        Returns:
            hidden_states: (batch, seq_len, hidden_size)
        """
        batch, max_position_embeddings = idx.shape
        assert max_position_embeddings <= self.cfg.max_position_embeddings, \
                    f"序列长度 {max_position_embeddings} 超过 context_length {self.cfg.max_position_embeddings}"

        tok_emb = self.embed_tokens(idx)
        x = self.drop_embeddings(tok_emb)
        freqs_cos = self.freqs_cos[:max_position_embeddings]
        freqs_sin = self.freqs_sin[:max_position_embeddings]
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, freqs_cos, freqs_sin, use_reentrant=False)
            else:
                layer_device = next(layer.parameters()).device
                attn_mask_on_device = attention_mask.to(layer_device) if attention_mask is not None else None
                x = layer(
                    x.to(layer_device),
                    freqs_cos.to(layer_device),
                    freqs_sin.to(layer_device),
                    attn_mask_on_device
                )
        x = x.to(next(self.norm.parameters()).device)
        x = self.norm(x)
        return x


class QwenForCausalLM(nn.Module):
    _no_split_modules = ["DecoderLayer"]

    def __init__(self, cfg: QwenConfig):
        super().__init__()
        self.cfg = cfg
        self.model = QwenModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.tie_weights()

    def tie_weights(self):
        """Re-tie lm_head ↔ embed_tokens after load_state_dict(assign=True)."""
        if self.cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点：重计算激活值换显存，与 ZeRO-3 兼容。"""
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点。"""
        self.model.gradient_checkpointing = False

    def forward(
        self,
        idx: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_state = self.model(idx, attention_mask)
        return self.lm_head(hidden_state)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        source: str = "local",
        map_location: Optional[Union[str, torch.device]] = "cpu",
        device_map: Optional[Union[str, dict]] = None
    ) -> "QwenForCausalLM":
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
        """从 HuggingFace 官方格式加载 Qwen2/2.5 权重"""
        from model.qwen.weight_convert import convert_safetensors_to_custom

        # 1. 加载配置
        with open(os.path.join(model_name_or_path, "config.json")) as f:
            config = QwenConfig.from_dict(json.load(f))

        # 2. 解析权重文件
        shard_files, use_safetensors = resolve_checkpoint_files(model_name_or_path)

        # 3. 创建 meta tensor 模型
        with init_empty_weights():
            model = cls(config)

        # 4. 设备映射
        device_map = auto_device_map(model, device_map, map_location)

        # 5. 权重转换函数
        def convert_fn(shard):
            return convert_safetensors_to_custom(
                shard, config.num_hidden_layers,
                tie_word_embeddings=config.tie_word_embeddings,
                n_heads=config.num_attention_heads,
                n_kv_heads=config.num_key_value_heads,
            )

        # 6. 加载权重
        fast_load_weights(model, shard_files, use_safetensors, device_map, convert_fn)
        return model

    @classmethod
    def _from_local(cls, model_name_or_path: str, map_location: Optional[Union[str, torch.device]] = "cpu", device_map: Optional[Union[str, dict]] = None):
        """从本地训练保存的模型加载"""
        from model.qwen.weight_convert import convert_safetensors_to_custom

        # 1. 加载配置
        config_file = os.path.join(model_name_or_path, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        with open(config_file) as f:
            config = QwenConfig.from_dict(json.load(f))

        # 2. 解析权重文件
        shard_files, use_safetensors = resolve_checkpoint_files(model_name_or_path)

        # 3. 权重转换函数（兼容 HF 格式的本地保存）
        def convert_fn(shard):
            if any("rotary_emb.inv_freq" in k for k in shard):
                logger.info("检测到 HF 格式权重，执行转换...")
                return convert_safetensors_to_custom(
                    shard, config.num_hidden_layers,
                    tie_word_embeddings=config.tie_word_embeddings,
                    n_heads=config.num_attention_heads,
                    n_kv_heads=config.num_key_value_heads,
                )
            return shard

        # 4. 创建 meta tensor 模型
        with init_empty_weights():
            model = cls(config)

        # 5. 设备映射
        device_map = auto_device_map(model, device_map, map_location)

        # 6. 加载权重
        fast_load_weights(model, shard_files, use_safetensors, device_map, convert_fn)
        return model

    @classmethod
    def from_empty(
        cls,
        cfg: "QwenConfig",
        map_location: Optional[Union[str, torch.device]] = "cpu",
        device_map: Optional[Union[str, dict]] = None,
    ) -> "QwenForCausalLM":
        """仅用于架构测试（不加载任何权重），支持 device_map='auto'。"""
        torch.set_default_dtype(torch.bfloat16)
        try:
            if device_map == "auto":
                with init_empty_weights():
                    model = cls(cfg)
                resolved_map = build_auto_device_map(model)
                materialize_meta_model(model, resolved_map)
            else:
                with torch.device(map_location or "cpu"):
                    model = cls(cfg)
        finally:
            torch.set_default_dtype(torch.float32)
        return model

    @torch.no_grad()
    def _stream_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        do_sample: bool = True,
    ) -> Generator[int, None, None]:
        pad_token_id = pad_token_id or self.cfg.pad_token_id
        eos_token_id = eos_token_id or self.cfg.eos_token_id

        idx = input_ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_position_embeddings:]
            logits = self(idx_cond)
            logits = logits[:, -1, :].float()
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
            yield idx_next[0].item()

            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

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
        stream: bool = False,
    ) -> Union[torch.Tensor, Generator[int, None, None]]:

        if stream:
            return self._stream_generate(
                input_ids, max_new_tokens, temperature, top_k,
                pad_token_id, eos_token_id, do_sample
            )

        pad_token_id = pad_token_id or self.cfg.pad_token_id
        eos_token_id = eos_token_id or self.cfg.eos_token_id

        idx = input_ids
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_position_embeddings:]
            attn_cond = attention_mask[:, -self.cfg.max_position_embeddings:] if attention_mask is not None else None

            logits = self(idx_cond, attention_mask=attn_cond)
            logits = logits[:, -1, :].float()
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
    load_pretrained_mode = LoadMode.SCRATCH
    model_name = ModelName.QWEN2_0_5B.value
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{SOURCE_DIR}/hf/qwen/{model_name}/tokenizer.json")
    tokenizer = Tokenizer.from_file(f"{SOURCE_DIR}/hf/qwen/{model_name}/tokenizer.json")
    cfg = MODEL_CONFIGS[model_name]
    torch.manual_seed(123)

    """
    # 加载权重测试
    """
    if load_pretrained_mode == LoadMode.PRETRAINED:
        # 磁盘读取权重 → CPU内存 → GPU↑ 这一步无法绕过，文件必须先经过CPU内存
        with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}"):
            model = QwenForCausalLM.from_pretrained(
                f"{SOURCE_DIR}/hf/llama/{model_name}",
                source="hf", 
                device_map="auto"
            )
    elif load_pretrained_mode == LoadMode.CONTINUAL:
        # 磁盘读取权重 → CPU内存 → GPU↑ 这一步无法绕过，文件必须先经过CPU内存
        with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}"):
            model = QwenForCausalLM.from_pretrained(
                f"{OUT_DIR}/train_simple_20260730_{model_name}",
                source="local", 
                device_map="auto"
            ) 
    else:
        # 仅测试模型架构（不加载权重），与 PRETRAINED/CONTINUAL 同步支持 device_map="auto"
        with log_stage(logger, f"加载 {model_name} 权重! {load_pretrained_mode}"):
            model = QwenForCausalLM.from_empty(cfg, device_map="auto")
    
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
    # 生成测试-非流式
    """
    model.eval()
    prompt = torch.tensor(tokenizer.encode("Hello, I am").ids).unsqueeze(0).to(device) # (3,) -> (1, 3)
    out = model.generate(prompt, max_new_tokens=250, top_k=50, temperature=0.8) # (1, 3) -> (1, 13)
    print(f"\nGenerated: {tokenizer.decode(out[0].tolist())}") # (1, 13) -> (13,) -> list -> str


    """
    # 生成测试-流式
    prompt = torch.tensor(tokenizer.encode("Hello, I am").ids).unsqueeze(0).to(device) # (3,) -> (1, 3)
    model.eval()
    for token_id in model.generate(prompt, max_new_tokens=250, top_k=50, temperature=0.8, stream=True):
        token_str = tokenizer.decode([token_id])
        print(token_str, end="", flush=True)
    print()  # 最后换行
    """