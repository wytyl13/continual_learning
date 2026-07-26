#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/07/26 17:26:40
@Author : weiyutao
@File : config.py


GPT-2 配置类及格式转换工具

"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class GPTConfig:
    """GPT 模型配置
    
    设计参考 transformers 库的 PretrainedConfig，但简化为 dataclass：
    - 所有超参数都有默认值，对应 GPT-2 Small（124M）
    - 通过类方法快速创建各尺寸预设

    使用方式：
        cfg = GPTConfig()                    # GPT-2 Small（默认）
        cfg = GPTConfig.gpt2_medium()        # GPT-2 Medium
        cfg = GPTConfig(num_layers=6)        # 自定义
    """
    
    # 模型架构参数
    vocab_size: int = 50257 # GPT-2 tokenizer 词表大小
    hidden_size: int = 768 # 词向量维度，同时也是所有层的宽度
    intermediate_size: Optional[int] = None # 默认None，表示自动计算
    max_position_embeddings: int = 1024 # 最大序列长度（位置编码的上限）
    num_attention_heads: int = 12 # 注意力头数，head_dim = hidden_size / num_attention_heads = 64
    num_hidden_layers: int = 12 # Transformer 块堆叠层数
    layer_norm_epsilon: float = 1e-5
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    activation_function: str = "gelu_new"
    tie_word_embeddings: bool = True
    qkv_bias: bool = True # Q/K/V 投影是否带 bias（原始 GPT-2 为 True，现代模型倾向 False）
    
    # 生成相关参数
    pad_token_id: int = 50256
    eos_token_id: int = 50256
    bos_token_id: int = 50256

    def __post_init__(self):
        """实例化后自动计算intermediate_size"""
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * 4

    # ═══════════════════════════════════════════════════════════════
    # 预设配置
    # ═══════════════════════════════════════════════════════════════
    
    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        """124M 参数"""
        return cls()

    @classmethod
    def gpt2_medium(cls) -> "GPTConfig":
        """355M 参数"""
        return cls(hidden_size=1024, num_attention_heads=16, num_hidden_layers=24)

    @classmethod
    def gpt2_large(cls) -> "GPTConfig":
        """774M 参数"""
        return cls(hidden_size=1280, num_attention_heads=20, num_hidden_layers=36)

    @classmethod
    def gpt2_xl(cls) -> "GPTConfig":
        """1.5B 参数"""
        return cls(hidden_size=1600, num_attention_heads=25, num_hidden_layers=48)
    
    # ═══════════════════════════════════════════════════════════════
    # Config格式转换（核心方法）
    # ═══════════════════════════════════════════════════════════════
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GPTConfig":
        """从字典创建配置（自动检测格式）
        
        支持三种格式：
        1. 自定义格式（hidden_size, num_hidden_layers）
        2. Transformers格式（n_embd, n_layer）
        3. OpenAI格式（n_vocab, n_ctx）
        """
        # 检测格式
        if "n_embd" in config_dict:
            # Transformers格式
            return cls._from_transformers_dict(config_dict)
        elif "hidden_size" in config_dict:
            # 自定义格式
            return cls._from_custom_dict(config_dict)
        elif "n_vocab" in config_dict:
            # OpenAI格式
            return cls._from_openai_dict(config_dict)
        else:
            raise ValueError(
                f"无法识别config格式。期望包含以下键之一："
                f"'n_embd'(Transformers) / 'hidden_size'(自定义) / 'n_vocab'(OpenAI)，"
                f"实际键: {list(config_dict.keys())}"
            )
            
    
    @classmethod
    def _from_custom_dict(cls, config_dict: Dict[str, Any]) -> "GPTConfig":
        """从自定义格式字典创建"""
        return cls(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            intermediate_size=config_dict.get("intermediate_size"),
            max_position_embeddings=config_dict["max_position_embeddings"],
            num_attention_heads=config_dict["num_attention_heads"],
            num_hidden_layers=config_dict["num_hidden_layers"],
            layer_norm_epsilon=config_dict["layer_norm_epsilon"],
            embd_pdrop=config_dict["embd_pdrop"],
            resid_pdrop=config_dict["resid_pdrop"],
            attn_pdrop=config_dict["attn_pdrop"],
            activation_function=config_dict["activation_function"],
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
            qkv_bias=config_dict.get("qkv_bias", True),
            pad_token_id=config_dict.get("pad_token_id", 50256),
            eos_token_id=config_dict.get("eos_token_id", 50256),
            bos_token_id=config_dict.get("bos_token_id", 50256),
        )
        
    @classmethod
    def _from_transformers_dict(cls, config_dict: Dict[str, Any]) -> "GPTConfig":
        """从Transformers格式字典创建"""
        return cls(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["n_embd"],
            intermediate_size=config_dict.get("n_inner", config_dict["n_embd"] * 4),
            max_position_embeddings=config_dict["n_positions"],
            num_attention_heads=config_dict["n_head"],
            num_hidden_layers=config_dict["n_layer"],
            layer_norm_epsilon=config_dict.get("layer_norm_epsilon", 1e-5),
            embd_pdrop=config_dict.get("embd_pdrop", 0.1),
            resid_pdrop=config_dict.get("resid_pdrop", 0.1),
            attn_pdrop=config_dict.get("attn_pdrop", 0.1),
            activation_function=config_dict.get("activation_function", "gelu_new"),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", True),
            qkv_bias=True,
            pad_token_id=config_dict.get("pad_token_id", 50256),
            eos_token_id=config_dict.get("eos_token_id", 50256),
            bos_token_id=config_dict.get("bos_token_id", 50256),
        )
        
    @classmethod
    def _from_openai_dict(cls, config_dict: Dict[str, Any]) -> "GPTConfig":
        """从OpenAI hparams.json创建
        # hparams config json
        {
            "n_vocab": 50257,
            "n_ctx": 1024,
            "n_embd": 768,
            "n_head": 12,
            "n_layer": 12
        }
        """
        return cls(
            vocab_size=config_dict["n_vocab"],
            hidden_size=config_dict["n_embd"],
            intermediate_size=config_dict["n_embd"] * 4,
            max_position_embeddings=config_dict["n_ctx"],
            num_attention_heads=config_dict["n_head"],
            num_hidden_layers=config_dict["n_layer"],
            qkv_bias=True,
        )
        
    def to_transformers_dict(self) -> Dict[str, Any]:
        """转换为Transformers格式的字典"""
        return {
            "vocab_size": self.vocab_size,
            "n_positions": self.max_position_embeddings,
            "n_embd": self.hidden_size,
            "n_layer": self.num_hidden_layers,
            "n_head": self.num_attention_heads,
            "n_inner": self.intermediate_size,
            "activation_function": self.activation_function,
            "resid_pdrop": self.resid_pdrop,
            "embd_pdrop": self.embd_pdrop,
            "attn_pdrop": self.attn_pdrop,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
        }