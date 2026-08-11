#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/08/07
@Author : weiyutao
@File : config.py

Qwen2/2.5 系列模型配置

架构说明：
    Qwen2 / Qwen2.5 / Qwen2.5-Coder / Qwen2.5-Math / Qwen3 (dense) 架构完全相同
    与 LLaMA 架构基本一致，主要区别：
        - 词表更大：151936 / 152064（LLaMA 是 32000 / 128256）
        - rope_theta 更大：1000000.0（LLaMA2 是 10000.0，LLaMA3 是 500000.0）
        - qkv_bias = True（LLaMA 通常是 False）
        - rms_norm_eps = 1e-6（LLaMA 通常是 1e-5）
        
        
  ┌─────────────────────┬────────────────────────────┬───────────────────────────────────────┬
  │        参数         │           Qwen2            │                Qwen2.5                │
  ├─────────────────────┼────────────────────────────┼───────────────────────────────────────┼
  │ 上下文长度           │ 0.5B/1.5B: 32K7B/72B: 128K │ 0.5B/1.5B/3B: 32K7B/14B/32B/72B: 128K │
  ├─────────────────────┼────────────────────────────┼───────────────────────────────────────┼
  │ tie_word_embeddings │ 0.5B/1.5B: True7B+: False  │ 0.5B/1.5B/3B: True7B+: False          │
  ├─────────────────────┼────────────────────────────┼───────────────────────────────────────┼
  │ 模型尺寸             │ 0.5B/1.5B/7B/72B           │ 0.5B/1.5B/3B/7B/14B/32B/72B           │
  ├─────────────────────┼────────────────────────────┼───────────────────────────────────────┼
  │ hidden/intermediate │ 相同                        │ 相同                                  │
  └─────────────────────┴────────────────────────────┴───────────────────────────────────────┴
  
  ┌───────┬─────────────┬───────────────────┬────────┬────────────┬────────────┬────────────────┬───────────────┐
  │ Model │ Hidden Size │ Intermediate Size │ Layers │ Q/KV Heads │ Vocab Size │ Context Length │ Tie Embedding │
  ├───────┼─────────────┼───────────────────┼────────┼────────────┼────────────┼────────────────┼───────────────┤
  │ 0.5B  │ 896         │ 4,864             │ 24     │ 14 / 2     │ 151,936    │ 32K            │ True          │
  ├───────┼─────────────┼───────────────────┼────────┼────────────┼────────────┼────────────────┼───────────────┤
  │ 1.5B  │ 1,536       │ 8,960             │ 28     │ 12 / 2     │ 151,936    │ 32K            │ True          │
  ├───────┼─────────────┼───────────────────┼────────┼────────────┼────────────┼────────────────┼───────────────┤
  │ 3B    │ 2,048       │ 11,008            │ 36     │ 16 / 2     │ 151,936    │ 32K            │ True          │
  ├───────┼─────────────┼───────────────────┼────────┼────────────┼────────────┼────────────────┼───────────────┤
  │ 7B    │ 3,584       │ 18,944            │ 28     │ 28 / 4     │ 152,064    │ 128K           │ False         │
  ├───────┼─────────────┼───────────────────┼────────┼────────────┼────────────┼────────────────┼───────────────┤
  │ 14B   │ 5,120       │ 13,824            │ 48     │ 40 / 8     │ 152,064    │ 128K           │ False         │
  ├───────┼─────────────┼───────────────────┼────────┼────────────┼────────────┼────────────────┼───────────────┤
  │ 32B   │ 5,120       │ 27,648            │ 64     │ 40 / 8     │ 152,064    │ 128K           │ False         │
  ├───────┼─────────────┼───────────────────┼────────┼────────────┼────────────┼────────────────┼───────────────┤
  │ 72B   │ 8,192       │ 29,568            │ 80     │ 64 / 8     │ 152,064    │ 128K           │ False         │
  └───────┴─────────────┴───────────────────┴────────┴────────────┴────────────┴────────────────┴───────────────┘
  
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class QwenConfig:
    """
    Qwen 模型配置
    适用于 Qwen2 / Qwen2.5 / Qwen2.5-Coder / Qwen2.5-Math / Qwen3 (dense)
    """
    # 模型架构参数
    vocab_size: int = 151936  # Qwen2.5: 151936 或 152064
    hidden_size: int = 2048
    intermediate_size: int = 11008
    max_position_embeddings: int = 32768  # Qwen2.5 支持 32K 上下文
    num_attention_heads: int = 16
    num_key_value_heads: int = 2  # GQA
    num_hidden_layers: int = 36

    # 归一化和正则化
    layer_norm_epsilon: float = 1e-6  # Qwen 使用更小的 epsilon
    embd_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    attn_pdrop: float = 0.0

    # 激活函数和注意力
    activation_function: str = "silu"
    qkv_bias: bool = True  # Qwen 的注意力层有 bias

    # 位置编码
    rope_theta: float = 1000000.0  # Qwen 特有的大 theta

    # 其他
    tie_word_embeddings: bool = False  # 3B+ 模型不绑定
    multiple_of: int = 256

    # 特殊 token
    pad_token_id: int = 151643
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    def __post_init__(self):
        """实例化后的自动计算（如果需要）"""
        # Qwen 的 intermediate_size 通常是手动指定的，不需要自动计算
        pass

    # ═══════════════════════════════════════════════════════════════
    # 预设配置 - Qwen2.5 系列
    # ═══════════════════════════════════════════════════════════════

    @classmethod
    def qwen2_5_0_5b(cls) -> "QwenConfig":
        """Qwen2.5-0.5B: ~0.5B 参数"""
        return cls(
            vocab_size=151936,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2,
            max_position_embeddings=32768, # 32K
            rope_theta=1000000.0,
            tie_word_embeddings=True  # 小模型绑定
        )
    
    @classmethod
    def qwen2_0_5b(cls) -> "QwenConfig":
        """Qwen2-0.5B: ~0.5B 参数"""
        return cls.qwen2_5_0_5b()
    
    @classmethod
    def qwen2_5_1_5b(cls) -> "QwenConfig":
        """Qwen2.5-1.5B: ~1.5B 参数"""
        return cls(
            vocab_size=151936,
            hidden_size=1536,
            intermediate_size=8960,
            num_hidden_layers=28,
            num_attention_heads=12,
            num_key_value_heads=2,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            tie_word_embeddings=True
        )
        
    @classmethod
    def qwen2_1_5b(cls) -> "QwenConfig":
        """Qwen2-1.5B: ~1.5B 参数"""
        return cls.qwen2_5_1_5b()
    
    @classmethod
    def qwen2_5_3b(cls) -> "QwenConfig":
        """Qwen2.5-3B: ~3B 参数"""
        return cls(
            vocab_size=151936,
            hidden_size=2048,
            intermediate_size=11008,
            num_hidden_layers=36,
            num_attention_heads=16,
            num_key_value_heads=2,
            max_position_embeddings=32768,
            rope_theta=1000000.0,
            tie_word_embeddings=True  # 7B+ 不绑定
        )

    @classmethod
    def qwen2_5_7b(cls) -> "QwenConfig":
        """Qwen2.5-7B: ~7B 参数"""
        return cls(
            vocab_size=152064,
            hidden_size=3584,
            intermediate_size=18944,
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            max_position_embeddings=131072,
            rope_theta=1000000.0,
            tie_word_embeddings=False
        )
    
    @classmethod
    def qwen2_7b(cls) -> "QwenConfig":
        """Qwen2-7B: ~7B 参数"""
        return cls.qwen2_5_7b()

    @classmethod
    def qwen2_5_14b(cls) -> "QwenConfig":
        """Qwen2.5-14B: ~14B 参数"""
        return cls(
            vocab_size=152064,
            hidden_size=5120,
            intermediate_size=13824,
            num_hidden_layers=48,
            num_attention_heads=40,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            rope_theta=1000000.0,
            tie_word_embeddings=False
        )

    @classmethod
    def qwen2_5_32b(cls) -> "QwenConfig":
        """Qwen2.5-32B: ~32B 参数"""
        return cls(
            vocab_size=152064,
            hidden_size=5120,
            intermediate_size=27648,
            num_hidden_layers=64,
            num_attention_heads=40,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            rope_theta=1000000.0,
            tie_word_embeddings=False
        )

    @classmethod
    def qwen2_5_72b(cls) -> "QwenConfig":
        """Qwen2.5-72B: ~72B 参数"""
        return cls(
            vocab_size=152064,
            hidden_size=8192,
            intermediate_size=29568,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            max_position_embeddings=131072,
            rope_theta=1000000.0,
            tie_word_embeddings=False
        )
        
    @classmethod
    def qwen2_72b(cls) -> "QwenConfig":
        """Qwen2-72B: ~72B 参数"""
        return cls.qwen2_5_72b()

    # ═══════════════════════════════════════════════════════════════
    # Qwen2.5 变体（架构相同，只是训练数据不同）
    # ═══════════════════════════════════════════════════════════════

    @classmethod
    def qwen2_5_coder_1_5b(cls) -> "QwenConfig":
        """Qwen2.5-Coder-1.5B: 代码专用模型"""
        return cls.qwen2_5_1_5b()

    @classmethod
    def qwen2_5_coder_7b(cls) -> "QwenConfig":
        """Qwen2.5-Coder-7B: 代码专用模型"""
        return cls.qwen2_5_7b()

    @classmethod
    def qwen2_5_coder_32b(cls) -> "QwenConfig":
        """Qwen2.5-Coder-32B: 代码专用模型"""
        return cls.qwen2_5_32b()

    @classmethod
    def qwen2_5_math_1_5b(cls) -> "QwenConfig":
        """Qwen2.5-Math-1.5B: 数学专用模型"""
        return cls.qwen2_5_1_5b()

    @classmethod
    def qwen2_5_math_7b(cls) -> "QwenConfig":
        """Qwen2.5-Math-7B: 数学专用模型"""
        return cls.qwen2_5_7b()

    @classmethod
    def qwen2_5_math_72b(cls) -> "QwenConfig":
        """Qwen2.5-Math-72B: 数学专用模型"""
        return cls.qwen2_5_72b()

    # ═══════════════════════════════════════════════════════════════
    # Config 格式转换（核心方法）
    # ═══════════════════════════════════════════════════════════════

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "QwenConfig":
        """从字典创建配置（自动检测格式）"""
        # 检测是否为 Hugging Face 官方格式
        if config_dict.get("model_type") == "qwen2" or "rms_norm_eps" in config_dict:
            return cls._from_transformers_dict(config_dict)
        else:
            return cls._from_custom_dict(config_dict)

    @classmethod
    def _from_custom_dict(cls, config_dict: Dict[str, Any]) -> "QwenConfig":
        """从自定义格式字典创建"""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    @classmethod
    def _from_transformers_dict(cls, config_dict: Dict[str, Any]) -> "QwenConfig":
        """从 Hugging Face 官方格式的字典创建 Qwen 配置"""
        return cls(
            vocab_size=config_dict.get("vocab_size", 151936),
            hidden_size=config_dict.get("hidden_size", 2048),
            intermediate_size=config_dict.get("intermediate_size", 11008),
            max_position_embeddings=config_dict.get("max_position_embeddings", 32768),
            num_attention_heads=config_dict.get("num_attention_heads", 16),
            num_key_value_heads=config_dict.get("num_key_value_heads", 2),
            num_hidden_layers=config_dict.get("num_hidden_layers", 36),

            # 参数映射
            layer_norm_epsilon=config_dict.get("rms_norm_eps", 1e-6),
            attn_pdrop=config_dict.get("attention_dropout", 0.0),
            activation_function=config_dict.get("hidden_act", "silu"),
            qkv_bias=config_dict.get("attention_bias", True),

            rope_theta=config_dict.get("rope_theta", 1000000.0),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", False),

            pad_token_id=config_dict.get("pad_token_id", 151643),
            bos_token_id=config_dict.get("bos_token_id", 151643),
            eos_token_id=config_dict.get("eos_token_id", 151645),
        )

    def to_transformers_dict(self) -> Dict[str, Any]:
        """将当前自定义配置转换为 Hugging Face 官方 Qwen 格式的字典"""
        return {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],

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

            "rope_theta": self.rope_theta,
            "tie_word_embeddings": self.tie_word_embeddings,

            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,

            # Qwen 特有的其他字段
            "hidden_act": self.activation_function,
            "initializer_range": 0.02,
            "max_window_layers": self.num_hidden_layers,
            "sliding_window": None,
            "use_sliding_window": False,
            "use_cache": True,
        }


# 预设配置字典
MODEL_CONFIGS = {
    "Qwen2-0.5B": QwenConfig.qwen2_0_5b(),
    "Qwen2-1.5B": QwenConfig.qwen2_1_5b(),
    "Qwen2-7B": QwenConfig.qwen2_7b(),
    "Qwen2-72B": QwenConfig.qwen2_72b(),
    "Qwen2.5-0.5B": QwenConfig.qwen2_5_0_5b(),
    "Qwen2.5-1.5B": QwenConfig.qwen2_5_1_5b(),
    "Qwen2.5-3B": QwenConfig.qwen2_5_3b(),
    "Qwen2.5-7B": QwenConfig.qwen2_5_7b(),
    "Qwen2.5-14B": QwenConfig.qwen2_5_14b(),
    "Qwen2.5-32B": QwenConfig.qwen2_5_32b(),
    "Qwen2.5-72B": QwenConfig.qwen2_5_72b(),
}


if __name__ == "__main__":
    """测试配置转换"""
    print("=" * 80)
    print("Qwen2.5-3B 配置测试")
    print("=" * 80)

    # 1. 创建配置
    config = QwenConfig.qwen2_5_3b()
    print("\n1. 自定义配置:")
    print(config)

    # 2. 转换为 Hugging Face 格式
    print("\n2. 转换为 Hugging Face 格式:")
    hf_dict = config.to_transformers_dict()
    for k, v in hf_dict.items():
        print(f"  {k}: {v}")

    # 3. 从 Hugging Face 格式恢复
    print("\n3. 从 Hugging Face 格式恢复:")
    recovered_config = QwenConfig.from_dict(hf_dict)
    print(recovered_config)

    print("\n" + "=" * 80)
    print("配置测试完成")
    print("=" * 80)
