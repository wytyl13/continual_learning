#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/07/26 23:57:27
@Author : weiyutao
@File : config.py

llama重新定义config，不继承gpt2_config，因为很多参数不共有。
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LLamaConfig:
    """
    LLama 模型配置
    """
    # 模型架构参数
    vocab_size: int = 32000 # LLAMA1/2: 32000, LLAMA3: 128256
    hidden_size: int = 4096
    intermediate_size: Optional[int] = None
    max_position_embeddings: int = 4096 # LLAMA1: 2048, LLAMA2/3: 4096+
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    num_hidden_layers: int = 32
    layer_norm_epsilon: float = 1e-5
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    activation_function: str = "silu"
    tie_word_embeddings: bool = True
    qkv_bias: bool = False
    
    # multiple_of 属性，默认为 LLaMA 官方使用的 256
    multiple_of: int = 256
    
    rope_theta: float = 10000.0
    
    # 生成相关参数
    pad_token_id: int = 0
    eos_token_id: int = 2
    bos_token_id: int = 1

    
    def __post_init__(self):
        """实例化后自动计算intermediate_size"""
        if self.intermediate_size is None:
            hidden_dim = self.hidden_size * 4
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)
            self.intermediate_size = hidden_dim

    
    #═══════════════════════════════════════════════════════════════
    # 预设配置（参考 GPTConfig 风格）
    # ═══════════════════════════════════════════════════════════════
    @classmethod
    def baby_llama(cls) -> "LLamaConfig":
        """
        Baby LLaMA: ~40M 参数
        专用于本地单卡（如 8G 显存）的快速架构验证、前向/反向传播测试。
        """
        return cls(
            hidden_size=512,
            num_attention_heads=8,
            num_key_value_heads=8,
            num_hidden_layers=4,
            max_position_embeddings=512
        )

    @classmethod
    def tiny_llama(cls) -> "LLamaConfig":
        """
        TinyLlama: ~1.1B 参数
        基于官方 config 提取
        """
        return cls(
            vocab_size=32000,
            hidden_size=2048,
            intermediate_size=5632,
            max_position_embeddings=2048,
            num_attention_heads=32,
            num_key_value_heads=4,
            num_hidden_layers=22,
            layer_norm_epsilon=1e-5,
            rope_theta=10000.0,
            tie_word_embeddings=False,
            qkv_bias=False
        )

    @classmethod
    def llama2_7b(cls) -> "LLamaConfig":
        """LLaMA-2 7B:6.7B 参数"""
        return cls()
    
    @classmethod
    def llama2_13b(cls) -> "LLamaConfig":
        """LLaMA-2 13B: 13B 参数"""
        return cls(
            hidden_size=5120,
            num_attention_heads=40,
            num_key_value_heads=40,
            num_hidden_layers=40
        )
    
    @classmethod
    def llama2_70b(cls) -> "LLamaConfig":
        """LLaMA-2 70B: 70B 参数，GQA"""
        return cls(
            hidden_size=8192,
            num_attention_heads=64,
            num_key_value_heads=8,
            num_hidden_layers=80
        )
        
    @classmethod
    def llama3_8b(cls) -> "LLamaConfig":
        """LLaMA-3 8B"""
        return cls(
            vocab_size=128256,
            hidden_size=4096,
            num_attention_heads=32,
            num_key_value_heads=8,
            num_hidden_layers=32,
            max_position_embeddings=8192,
            rope_theta=500000.0
        )
        
    # ═══════════════════════════════════════════════════════════════
    # Config格式转换（核心方法）
    # ═══════════════════════════════════════════════════════════════
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLamaConfig":
        """从字典创建配置（自动检测格式）"""
        # 官方的 LLaMA config 通常包含 'model_type' 或特有字段 'rms_norm_eps'
        if config_dict.get("model_type") == "llama" or "rms_norm_eps" in config_dict:
            return cls._from_transformers_dict(config_dict)
        else:
            return cls._from_custom_dict(config_dict)

    @classmethod
    def _from_custom_dict(cls, config_dict: Dict[str, Any]) -> "LLamaConfig":
        """从自定义格式字典创建"""
        # 提取字典中在dataclass中有的字段
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    @classmethod
    def _from_transformers_dict(cls, config_dict: Dict[str, Any]) -> "LLamaConfig":
        """从 Hugging Face 官方格式的字典创建 LLaMA 配置"""
        return cls(
            vocab_size=config_dict.get("vocab_size", 32000),
            hidden_size=config_dict.get("hidden_size", 4096),
            intermediate_size=config_dict.get("intermediate_size", None),
            max_position_embeddings=config_dict.get("max_position_embeddings", 4096),
            num_attention_heads=config_dict.get("num_attention_heads", 32),
            num_key_value_heads=config_dict.get("num_key_value_heads", 32),
            num_hidden_layers=config_dict.get("num_hidden_layers", 32),
            # 参数映射
            layer_norm_epsilon=config_dict.get("rms_norm_eps", 1e-5),
            attn_pdrop=config_dict.get("attention_dropout", 0.0),
            activation_function=config_dict.get("hidden_act", "silu"),
            qkv_bias=config_dict.get("attention_bias", False),
            
            tie_word_embeddings=config_dict.get("tie_word_embeddings", False),
            rope_theta=config_dict.get("rope_theta", 10000.0),
            
            pad_token_id=config_dict.get("pad_token_id", 0),
            eos_token_id=config_dict.get("eos_token_id", 2),
            bos_token_id=config_dict.get("bos_token_id", 1),
        )
        
    def to_transformers_dict(self) -> Dict[str, Any]:
        """将当前自定义配置转换为 Hugging Face 官方 LLaMA 格式的字典"""
        return {
            "model_type": "llama",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            # 参数映射
            "rms_norm_eps": self.layer_norm_epsilon,
            "attention_dropout": self.attn_pdrop,
            "hidden_act": self.activation_function,
            "attention_bias": self.qkv_bias,
            
            "tie_word_embeddings": self.tie_word_embeddings,
            "rope_theta": self.rope_theta,
            
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            
            # 以下为 LLaMA 配置中常见但未在 dataclass 中特别维护的默认项
            "mlp_bias": False,
            "pretraining_tp": 1,
            "rope_scaling": None,
            "use_cache": True
        }
    
if __name__ == "__main__":
    from tokenizers import Tokenizer
    from config import SOURCE_DIR, OUT_DIR
    """
    # check llama config.
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(f"{SOURCE_DIR}/hf/llama/tiny_llama")
    hf_config_dict = config.to_dict()
    print(hf_config_dict)
    
    """
    # Test transform from HF config to auto defining config.
    """
    print("-----------------transform1-------------------")
    my_config = LLamaConfig.from_dict(hf_config_dict)
    print(my_config)
    print("-----------------transform1-------------------")
    """
    # Test transform fom auto defining config to HF config.
    """
    print("-----------------transform2-------------------")
    back_to_hf_dict = LLamaConfig.tiny_llama().to_transformers_dict()
    print(back_to_hf_dict)
    print("-----------------transform2-------------------")

    print("-----------------transform3-------------------")
    back_to_hf_dict = my_config.to_transformers_dict()
    print(back_to_hf_dict)
    print("-----------------transform3-------------------")
    
    
