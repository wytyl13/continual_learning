#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/24 10:03
@Author  : weiyutao
@File    : weight_convert.py

GPT 架构命名规则（用于后续开源模型实现）

    ⚠️ 重要说明：
        本文件使用的是自定义命名（适合学习理解）
        后续实现新的开源模型架构时，应遵循 Transformers 库的统一命名规则
        这样可以最大化减少权重转换的复杂度，实现与 HuggingFace 生态的无缝兼容

    ══════════════════════════════════════════════════════════════════════════
    一、核心命名规则（基于 LLaMA/Qwen/Mistral 等现代模型，2023+ 标准）
    ══════════════════════════════════════════════════════════════════════════

    1. Attention 层（自注意力模块）
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       self.q_proj        Query 投影层      (hidden_size → hidden_size)
       self.k_proj        Key 投影层        (hidden_size → kv_hidden_size)
       self.v_proj        Value 投影层      (hidden_size → kv_hidden_size)
       self.o_proj        Output 投影层     (hidden_size → hidden_size)

       self.head_dim              每个注意力头的维度
       self.num_heads             Query 头数量
       self.num_key_value_heads   KV 头数量（GQA 时 < num_heads）

       ❌ 避免使用：wq/wk/wv/wo (LLaMA原始)、c_attn/c_proj (GPT-2历史)

    2. MLP/FFN 层（前馈网络，SwiGLU 架构）
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       self.gate_proj     Gate 投影层        (hidden_size → intermediate_size)
       self.up_proj       Up 投影层          (hidden_size → intermediate_size)
       self.down_proj     Down 投影层        (intermediate_size → hidden_size)
       self.act_fn        激活函数           nn.SiLU() 或 nn.GELU()

       ❌ 避免使用：w1/w2/w3 (数学符号)、c_fc/c_proj (GPT-2历史)、fc1/fc2

    3. DecoderLayer（Transformer 块）
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       self.self_attn                  自注意力模块
       self.mlp                        前馈网络模块
       self.input_layernorm            注意力前的归一化层
       self.post_attention_layernorm   MLP 前的归一化层
       self.hidden_size                隐藏层维度

       ❌ 避免使用：attn (应该用 self_attn)、ffn (应该用 mlp)、ln_1/ln_2

    4. 主模型（基础模型，不含 LM head）
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       self.embed_tokens   Token 嵌入层      (vocab_size, hidden_size)
       self.layers         Transformer 层列表  nn.ModuleList([DecoderLayer...])
       self.norm           最终归一化层       RMSNorm 或 LayerNorm
       self.rotary_emb     旋转位置编码       (仅 RoPE 模型)

       ❌ 避免使用：wte (GPT-2)、h/blocks (不够描述性)、ln_f

    5. ForCausalLM（带 LM head 的完整模型）⚠️ 关键结构
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
       self.model          基础模型实例       MyGPTModel(config)
       self.lm_head        语言模型输出头     (hidden_size, vocab_size)
       self.vocab_size     词表大小

       ⚠️ 两层结构的重要性：
          - 创建 'model.' 前缀，与 HuggingFace 结构完全一致
          - state_dict keys:
              model.embed_tokens.weight
              model.layers.0.self_attn.q_proj.weight
              lm_head.weight  (顶层，无 model. 前缀)

    ══════════════════════════════════════════════════════════════════════════
    二、配置类命名规则（Config）
    ══════════════════════════════════════════════════════════════════════════

    vocab_size                  词表大小
    hidden_size                 隐藏层维度 (不用 embedding_dim, d_model)
    intermediate_size           MLP 中间层维度 (不用 ffn_dim)
    num_hidden_layers           Transformer 层数 (不用 n_layers, num_layers)
    num_attention_heads         注意力头数量 (不用 n_heads)
    num_key_value_heads         KV 头数量 (GQA，默认 = num_attention_heads)
    max_position_embeddings     最大序列长度 (不用 max_seq_len, context_length)
    rms_norm_eps                RMSNorm epsilon (不用 layer_norm_eps)
    rope_theta                  RoPE 的 theta 参数
    hidden_act                  激活函数类型 ("silu", "gelu", "swiglu")

    pad_token_id, bos_token_id, eos_token_id    特殊 token ID
    use_cache                   是否使用 KV cache

    ══════════════════════════════════════════════════════════════════════════
    三、完整示例（符合 Transformers 标准）
    ══════════════════════════════════════════════════════════════════════════

    from dataclasses import dataclass
    import torch.nn as nn

    @dataclass
    class MyGPTConfig:
        vocab_size: int = 50257
        hidden_size: int = 768
        intermediate_size: int = 3072
        num_hidden_layers: int = 12
        num_attention_heads: int = 12
        num_key_value_heads: int = 12
        max_position_embeddings: int = 2048
        rms_norm_eps: float = 1e-6
        rope_theta: float = 10000.0
        hidden_act: str = "silu"

    class Attention(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.num_key_value_heads = config.num_key_value_heads

            # ✅ 使用 Transformers 命名规则
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    class MLP(nn.Module):
        def __init__(self, config):
            super().__init__()
            # ✅ 使用 Transformers 命名规则（SwiGLU 架构）
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            self.act_fn = nn.SiLU()

    class DecoderLayer(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
            # ✅ 使用 Transformers 命名规则
            self.self_attn = Attention(config)
            self.mlp = MLP(config)
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    class MyGPTModel(nn.Module):
        # 基础模型（不含 LM head）
        def __init__(self, config):
            super().__init__()
            # ✅ 使用 Transformers 命名规则
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
            self.layers = nn.ModuleList([
                DecoderLayer(config, i) for i in range(config.num_hidden_layers)
            ])
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = RotaryEmbedding(config)

    class MyGPTForCausalLM(nn.Module):
        # 完整模型（含 LM head）⚠️ 两层结构是关键
        def __init__(self, config):
            super().__init__()
            self.config = config

            # ⚠️ 关键：使用 self.model 包装，创建 'model.' 前缀
            self.model = MyGPTModel(config)

            # LM head 在顶层
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        def forward(self, input_ids, attention_mask=None, labels=None):
            hidden_states = self.model(input_ids, attention_mask)
            logits = self.lm_head(hidden_states)
            # ... 计算 loss
            return {"loss": loss, "logits": logits}

    ══════════════════════════════════════════════════════════════════════════
    四、权重加载最佳实践
    ══════════════════════════════════════════════════════════════════════════

    1. 使用两层结构（Model + ForCausalLM）
       - 确保 state_dict key 路径与 HuggingFace 一致
       - 最小化权重转换逻辑

    2. 从 HuggingFace 加载到自定义模型
       from safetensors.torch import load_file
       state_dict = load_file("model.safetensors")  # 或 torch.load("pytorch_model.bin")
       model.load_state_dict(state_dict, strict=False)

    3. 保存自定义模型权重
       # 推荐：同时保存两种格式
       torch.save(model.state_dict(), "my_model.bin")
       from safetensors.torch import save_file
       save_file(model.state_dict(), "my_model.safetensors")

    4. 让 Transformers 加载你的权重
       from transformers import LlamaForCausalLM, LlamaConfig
       hf_model = LlamaForCausalLM.from_config(hf_config)
       hf_model.load_state_dict(load_file("my_model.safetensors"), strict=False)
       

================================================================================
自定义GPT格式：
================================================================================
GPTForCausalLM(
(model): GPTModel(
    (embed_tokens): Embedding(50257, 768)
    (embed_pos_tokens): Embedding(1024, 768)
    (drop_embedding): Dropout(p=0.1, inplace=False)
    (layers): ModuleList(
    (0-11): 12 x DecoderLayer(
        (input_layernorm): LayerNorm()
        (self_attn): Attention(
        (q_proj): Linear(in_features=768, out_features=768, bias=True)
        (k_proj): Linear(in_features=768, out_features=768, bias=True)
        (v_proj): Linear(in_features=768, out_features=768, bias=True)
        (o_proj): Linear(in_features=768, out_features=768, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        )
        (post_attention_layernorm): LayerNorm()
        (mlp): MLP(
        (up_proj): Linear(in_features=768, out_features=3072, bias=True)
        (act_fn): GELU()
        (down_proj): Linear(in_features=3072, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.1, inplace=False)
    )
    )
    (norm): LayerNorm()
)
(lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

"""

import torch
import numpy as np

def show(d, prefix=""):
    """递归打印 params 的结构：每个叶子节点的名字和 shape"""
    if isinstance(d, dict):
        for k, v in d.items():
            show(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(d, list):
        # blocks 是个 list（每层一个 dict），只看第 0 层就够了，结构都一样
        print(f"{prefix}  <list, len={len(d)}>  (只展开第 0 个)")
        show(d[0], f"{prefix}[0]")
    elif isinstance(d, np.ndarray):
        print(f"{prefix:45s} shape={d.shape}")
    else:
        print(f"{prefix:45s} = {d}")


def print_state_dict_keys(state_dict, title="State Dict Keys"):
      """美观输出state_dict的键名结构"""
      from collections import defaultdict

      print(f"\n{'='*80}")
      print(f"{title}")
      print(f"{'='*80}")

      keys = sorted(state_dict.keys())

      # 按模块分组
      groups = defaultdict(list)
      for key in keys:
          if key.startswith('h.'):
              # 提取层级和模块
              parts = key.split('.')
              layer_id = int(parts[1])
              module = '.'.join(parts[2:])
              groups[f'h.{layer_id}'].append(module)
          else:
              groups['root'].append(key)

      # 输出根级别的键
      if 'root' in groups:
          print("\n根级别:")
          for k in sorted(groups['root']):
              print(f"  {k}")

      # 只显示第0层的详细结构
      if 'h.0' in groups:
          print("\nTransformer层 (h.0 示例):")
          modules = defaultdict(list)
          for k in sorted(groups['h.0']):
              parts = k.rsplit('.', 1)
              if len(parts) == 2:
                  module, param = parts
                  modules[module].append(param)
              else:
                  modules[k].append('')

          for module in sorted(modules.keys()):
              params = modules[module]
              if params and params[0]:
                  print(f"  h.0.{module}: {params}")
              else:
                  print(f"  h.0.{module}")

      # 统计信息
      layer_count = len([k for k in groups.keys() if k.startswith('h.')])
      print(f"\n统计:")
      print(f"  总键数: {len(keys)}")
      print(f"  Transformer层数: {layer_count}")
      print(f"  根级别键数: {len(groups.get('root', []))}")


def convert_openai_to_custom(openai_params: dict, n_layers: int) -> dict:
    """将 OpenAI GPT-2 的 params dict 转换为 GPTModel 的 state_dict 格式。
            
    ================================================================================
    OpenAI 格式：
    ================================================================================
    blocks  <list, len=12>  (只展开第 0 个)
    blocks[0].attn.c_attn.b                       shape=(2304,)
    blocks[0].attn.c_attn.w                       shape=(768, 2304)
    blocks[0].attn.c_proj.b                       shape=(768,)
    blocks[0].attn.c_proj.w                       shape=(768, 768)
    blocks[0].ln_1.b                              shape=(768,)
    blocks[0].ln_1.g                              shape=(768,)
    blocks[0].ln_2.b                              shape=(768,)
    blocks[0].ln_2.g                              shape=(768,)
    blocks[0].mlp.c_fc.b                          shape=(3072,)
    blocks[0].mlp.c_fc.w                          shape=(768, 3072)
    blocks[0].mlp.c_proj.b                        shape=(768,)
    blocks[0].mlp.c_proj.w                        shape=(3072, 768)
    b                                             shape=(768,)
    g                                             shape=(768,)
    wpe                                           shape=(1024, 768)
    wte                                           shape=(50257, 768)


    OpenAI 的存储格式与本模型命名不一致，需要做两件事：
        1. 键名映射：params["wte"] → "token_embedding.weight"
        2. 矩阵转置：OpenAI 的线性层权重是 (in, out) 存储，
        PyTorch nn.Linear 期望 (out, in)，所以需要 .T

    Q/K/V 合并存储：OpenAI 把三个投影矩阵合并为 c_attn，
    shape 为 (d, 3d)，需要用 np.split 沿最后一维拆成三份。
    """
    sd = {}

    # 嵌入层
    sd["model.embed_tokens.weight"]    = torch.tensor(openai_params["wte"])
    sd["model.embed_pos_tokens.weight"] = torch.tensor(openai_params["wpe"])

    # 最终归一化层
    sd["model.norm.scale"] = torch.tensor(openai_params["g"])
    sd["model.norm.shift"] = torch.tensor(openai_params["b"])

    # 输出头：OpenAI 使用 weight tying，out_head 复用 token_embedding 权重
    sd["lm_head.weight"] = torch.tensor(openai_params["wte"])

    for b in range(n_layers):
        blk = openai_params["blocks"][b]
        p = f"model.layers.{b}"

        # Q/K/V 权重：shape (d, 3d) → 拆成三个 (d, d)，再转置为 (d, d)
        # 先拆分后转置，转置是因为tensorflow和pytorch存储权重的方式不同
        # Linear(768, 2304)，其对应的weight权重是(2304, 768)，是因为pytorch优化矩阵运算。
        # tensorflow(768, 2304)，其对应的weight权重是(768, 2304)，所以先拆分，拆分完后要进行转置。
        q_w, k_w, v_w = np.split(blk["attn"]["c_attn"]["w"], 3, axis=-1)
        sd[f"{p}.self_attn.q_proj.weight"]  = torch.tensor(q_w.T)
        sd[f"{p}.self_attn.k_proj.weight"]    = torch.tensor(k_w.T)
        sd[f"{p}.self_attn.v_proj.weight"]  = torch.tensor(v_w.T)

        q_b, k_b, v_b = np.split(blk["attn"]["c_attn"]["b"], 3)
        sd[f"{p}.self_attn.q_proj.bias"]    = torch.tensor(q_b)
        sd[f"{p}.self_attn.k_proj.bias"]      = torch.tensor(k_b)
        sd[f"{p}.self_attn.v_proj.bias"]    = torch.tensor(v_b)

        # 输出投影
        sd[f"{p}.self_attn.o_proj.weight"] = torch.tensor(blk["attn"]["c_proj"]["w"].T)
        sd[f"{p}.self_attn.o_proj.bias"]   = torch.tensor(blk["attn"]["c_proj"]["b"])

        # MLP
        sd[f"{p}.mlp.up_proj.weight"] = torch.tensor(blk["mlp"]["c_fc"]["w"].T)
        sd[f"{p}.mlp.up_proj.bias"]   = torch.tensor(blk["mlp"]["c_fc"]["b"])
        sd[f"{p}.mlp.down_proj.weight"] = torch.tensor(blk["mlp"]["c_proj"]["w"].T)
        sd[f"{p}.mlp.down_proj.bias"]   = torch.tensor(blk["mlp"]["c_proj"]["b"])

        # LayerNorm（OpenAI 用 g/b 命名，对应 scale/shift）
        sd[f"{p}.input_layernorm.scale"] = torch.tensor(blk["ln_1"]["g"])
        sd[f"{p}.input_layernorm.shift"] = torch.tensor(blk["ln_1"]["b"])
        sd[f"{p}.post_attention_layernorm.scale"] = torch.tensor(blk["ln_2"]["g"])
        sd[f"{p}.post_attention_layernorm.shift"] = torch.tensor(blk["ln_2"]["b"])

    return sd


def convert_safetensors_to_custom(hf_state_dict: dict, n_layers: int, tie_word_embeddings: bool) -> dict:
    """
    ================================================================================
    State Dict Keys
    safetensors.load_file加载输出的结构
    logger.info(print_state_dict_keys(hf_state_dict))
    注意实际转换过程中这个是标准的加载方式，比transoformers封装的少了最外层的transformer这个键，同时也少了lm_head，
    因为tie_word_embeddings=True
    ================================================================================

    根级别:
    ln_f.bias
    ln_f.weight
    wpe.weight
    wte.weight

    Transformer层 (h.0 示例):
    h.0.attn: ['bias']
    h.0.attn.c_attn: ['bias', 'weight']
    h.0.attn.c_proj: ['bias', 'weight']
    h.0.ln_1: ['bias', 'weight']
    h.0.ln_2: ['bias', 'weight']
    h.0.mlp.c_fc: ['bias', 'weight']
    h.0.mlp.c_proj: ['bias', 'weight']
    """
    
    sd = {}
    # 嵌入层
    wte_weight = hf_state_dict["wte.weight"]
    sd["model.embed_tokens.weight"] = wte_weight
    sd["model.embed_pos_tokens.weight"] = hf_state_dict["wpe.weight"]

    # 最终归一化层
    sd["model.norm.scale"] = hf_state_dict["ln_f.weight"]
    sd["model.norm.shift"] = hf_state_dict["ln_f.bias"]

    # 输出头
    sd["lm_head.weight"] = hf_state_dict["lm_head.weight"] if not tie_word_embeddings else wte_weight
    
    for i in range(n_layers):
        hf_prefix = f"h.{i}"
        custom_prefix = f"model.layers.{i}"

        # 注意力层LayerNorm
        sd[f"{custom_prefix}.input_layernorm.scale"] = hf_state_dict[f"{hf_prefix}.ln_1.weight"]
        sd[f"{custom_prefix}.input_layernorm.shift"] = hf_state_dict[f"{hf_prefix}.ln_1.bias"]

        # Q/K/V权重：自定义Conv1D(2304, 768) 对应的weight shape (768, 2304)-> 转置后 (2304, 768) -> split 3份 (768, 768)
        c_attn_weight = hf_state_dict[f"{hf_prefix}.attn.c_attn.weight"] # (768, 2304)
        c_attn_weight_t = c_attn_weight.T # (2304, 768)
        q_w, k_w, v_w = torch.chunk(c_attn_weight_t, 3, dim=0) # (768, 768)
        sd[f"{custom_prefix}.self_attn.q_proj.weight"] = q_w
        sd[f"{custom_prefix}.self_attn.k_proj.weight"] = k_w
        sd[f"{custom_prefix}.self_attn.v_proj.weight"] = v_w
        
        # Q/K/V bias
        c_attn_bias = hf_state_dict[f"{hf_prefix}.attn.c_attn.bias"] # (2304,)
        q_b, k_b, v_b = torch.chunk(c_attn_bias, 3, dim=0) # (768,) * 3 
        sd[f"{custom_prefix}.self_attn.q_proj.bias"] = q_b
        sd[f"{custom_prefix}.self_attn.k_proj.bias"] = k_b
        sd[f"{custom_prefix}.self_attn.v_proj.bias"] = v_b

        # 输出投影：Conv1D（768, 768) -> 转置
        c_proj_weight = hf_state_dict[f"{hf_prefix}.attn.c_proj.weight"] # (768, 768)
        sd[f"{custom_prefix}.self_attn.o_proj.weight"] = c_proj_weight.T # (768, 768)
        sd[f"{custom_prefix}.self_attn.o_proj.bias"] = hf_state_dict[f"{hf_prefix}.attn.c_proj.bias"]
        
        # MLP LayerNorm
        sd[f"{custom_prefix}.post_attention_layernorm.scale"] = hf_state_dict[f"{hf_prefix}.ln_2.weight"]
        sd[f"{custom_prefix}.post_attention_layernorm.shift"] = hf_state_dict[f"{hf_prefix}.ln_2.bias"]

        # MLP层：Conv1D -> 转置 注意Conv1D这个OpenAI自定义的网络数据维度的流转方式不同于nn.Conv1D
        c_fc_weight = hf_state_dict[f"{hf_prefix}.mlp.c_fc.weight"] # (768, 3072)
        sd[f"{custom_prefix}.mlp.up_proj.weight"] = c_fc_weight.T
        sd[f"{custom_prefix}.mlp.up_proj.bias"] = hf_state_dict[f"{hf_prefix}.mlp.c_fc.bias"]

        c_proj_weight = hf_state_dict[f"{hf_prefix}.mlp.c_proj.weight"] # (3072, 768)
        sd[f"{custom_prefix}.mlp.down_proj.weight"] = c_proj_weight.T
        sd[f"{custom_prefix}.mlp.down_proj.bias"] = hf_state_dict[f"{hf_prefix}.mlp.c_proj.bias"]
    
    return sd


def convert_custom_to_hf(custom_state_dict: dict, num_layers: int) -> dict:
    """
    将自定义 GPTModel 的 state_dict 转换为 transformers GPT2LMHeadModel 格式
    
    ================================================================================
    huggingface gpt2模型架构
    格式为以下代码输出
    from transformers import GPT2LMHeadModel
    hf_model = GPT2LMHeadModel.from_pretrained(f"{SOURCE_DIR}/hf/gpt2/124M")
    print(hf_model)
    ================================================================================
    GPT2LMHeadModel(
        (transformer): GPT2Model(
            (wte): Embedding(50257, 768)
            (wpe): Embedding(1024, 768)
            (drop): Dropout(p=0.1, inplace=False)
            (h): ModuleList(
            (0-11): 12 x GPT2Block(
                (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (attn): GPT2Attention(
                    (c_attn): Conv1D(nf=2304, nx=768)
                    (c_proj): Conv1D(nf=768, nx=768)
                    (attn_dropout): Dropout(p=0.1, inplace=False)
                    (resid_dropout): Dropout(p=0.1, inplace=False)
                )
                (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): GPT2MLP(
                    (c_fc): Conv1D(nf=3072, nx=768)
                    (c_proj): Conv1D(nf=768, nx=3072)
                    (act): NewGELUActivation()
                    (dropout): Dropout(p=0.1, inplace=False)
                )
            )
            )
            (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
        (lm_head): Linear(in_features=768, out_features=50257, bias=False)
        )


    关键差异：
    1. 命名空间：自定义模型没有 'transformer.' 前缀
    2. LayerNorm：自定义用 scale/shift，transformers 用 weight/bias
    3. QKV 合并：transformers 将 Q/K/V 合并到 c_attn 中
    4. 权重转置：transformers 的线性层权重是转置存储的

    Args:
        custom_state_dict: 自定义模型的 state_dict
        num_layers: transformer 层数

    Returns:
        transformers 格式的 state_dict
    """
    hf_state_dict = {}

    # 嵌入层
    hf_state_dict["transformer.wte.weight"] = custom_state_dict["model.embed_tokens.weight"]
    hf_state_dict["transformer.wpe.weight"] = custom_state_dict["model.embed_pos_tokens.weight"]

    # Final LayerNorm
    hf_state_dict["transformer.ln_f.weight"] = custom_state_dict["model.norm.scale"]
    hf_state_dict["transformer.ln_f.bias"] = custom_state_dict["model.norm.shift"]

    # lm_head
    hf_state_dict["lm_head.weight"] = custom_state_dict["lm_head.weight"]

    # 注意力层
    for i in range(num_layers):
        prefix_custom = f"model.layers.{i}"
        prefix_hf = f"transformer.h.{i}"

        # --- LayerNorm 1 ---
        hf_state_dict[f"{prefix_hf}.ln_1.weight"] = custom_state_dict[f"{prefix_custom}.input_layernorm.scale"]
        hf_state_dict[f"{prefix_hf}.ln_1.bias"] = custom_state_dict[f"{prefix_custom}.input_layernorm.shift"]

        # --- Attention: Q/K/V 合并 + 转置 ---
        # 自定义模型：W_query/W_key/W_value 分开，shape = (out, in) = (768, 768)
        # transformers：c_attn 合并，shape = (in, 3*out) = (768, 2304)，但存储为转置
        q_w = custom_state_dict[f"{prefix_custom}.self_attn.q_proj.weight"].T  # (768, 768)
        k_w = custom_state_dict[f"{prefix_custom}.self_attn.k_proj.weight"].T    # (768, 768)
        v_w = custom_state_dict[f"{prefix_custom}.self_attn.v_proj.weight"].T  # (768, 768)
        c_attn_weight = torch.cat([q_w, k_w, v_w], dim=1)  # (768, 2304)
        hf_state_dict[f"{prefix_hf}.attn.c_attn.weight"] = c_attn_weight

        q_b = custom_state_dict[f"{prefix_custom}.self_attn.q_proj.bias"]  # (768,)
        k_b = custom_state_dict[f"{prefix_custom}.self_attn.k_proj.bias"]    # (768,)
        v_b = custom_state_dict[f"{prefix_custom}.self_attn.v_proj.bias"]  # (768,)
        c_attn_bias = torch.cat([q_b, k_b, v_b], dim=0)  # (2304,)
        hf_state_dict[f"{prefix_hf}.attn.c_attn.bias"] = c_attn_bias

        # --- Attention: output projection + 转置 ---
        hf_state_dict[f"{prefix_hf}.attn.c_proj.weight"] = custom_state_dict[f"{prefix_custom}.self_attn.o_proj.weight"].T
        hf_state_dict[f"{prefix_hf}.attn.c_proj.bias"] = custom_state_dict[f"{prefix_custom}.self_attn.o_proj.bias"]

        # --- LayerNorm 2 ---
        hf_state_dict[f"{prefix_hf}.ln_2.weight"] = custom_state_dict[f"{prefix_custom}.post_attention_layernorm.scale"]
        hf_state_dict[f"{prefix_hf}.ln_2.bias"] = custom_state_dict[f"{prefix_custom}.post_attention_layernorm.shift"]

        # --- MLP + 转置 ---
        hf_state_dict[f"{prefix_hf}.mlp.c_fc.weight"] = custom_state_dict[f"{prefix_custom}.mlp.up_proj.weight"].T
        hf_state_dict[f"{prefix_hf}.mlp.c_fc.bias"] = custom_state_dict[f"{prefix_custom}.mlp.up_proj.bias"]
        hf_state_dict[f"{prefix_hf}.mlp.c_proj.weight"] = custom_state_dict[f"{prefix_custom}.mlp.down_proj.weight"].T
        hf_state_dict[f"{prefix_hf}.mlp.c_proj.bias"] = custom_state_dict[f"{prefix_custom}.mlp.down_proj.bias"]

    return hf_state_dict



if __name__ == "__main__":
    import tiktoken
    from model.gpt2.model import GPTConfig, GPTForCausalLM
    from config import SOURCE_DIR, OUT_DIR
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    cfg = GPTConfig.gpt2_small()
    torch.manual_seed(123)
    model = GPTForCausalLM(cfg)

    """
    输出格式对比
    """

    # OpenAI格式
    from model.gpt2.gpt_download import download_and_load_gpt2
    _, openai_params = download_and_load_gpt2("124M", models_dir=f"{SOURCE_DIR}/trf/gpt2")
    print("=" * 80)
    print("OpenAI 格式：")
    print("=" * 80)
    show(openai_params)

    # 自定义模型架构
    print("\n" + "=" * 80)
    print("自定义GPT格式：")
    print("=" * 80)
    print(model)
    
    # 输出huggingface模型架构
    from transformers import GPT2LMHeadModel
    hf_model = GPT2LMHeadModel.from_pretrained(f"{SOURCE_DIR}/hf/gpt2/124M")
    print("\n" + "=" * 80)
    print(hf_model)
    print("=" * 80)