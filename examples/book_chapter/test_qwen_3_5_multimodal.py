#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/07/13
@Author : weiyutao
@File : test_qwen_multimodal.py
@Desc : 纯 transformers 测试 Qwen3.5-0.8B-Base 的多模态（图文）能力，不依赖 qwen_vl_utils。
"""

import os
import warnings
import torch
from PIL import Image

for proxy in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']:
    os.environ.pop(proxy, None)

# 跨设备 offload 时 model.device 会报 'meta'，generate 会误判 input_ids 设备不匹配而告警。
# 实际上 embed_tokens 在 CPU、input_ids 也在 CPU，是对的，属误报，静音之。
warnings.filterwarnings("ignore", message=".*input_ids.*device.*")

# WSL2 下 transformers 5.x 的 caching_allocator_warmup 会预分配大块连续显存，
# WSL2 虚拟 GPU 不支持，替换为空操作绕过。
import transformers.modeling_utils as _mu
_mu.caching_allocator_warmup = lambda *a, **k: None

from modelscope import AutoProcessor, AutoConfig, AutoModelForImageTextToText
from accelerate import init_empty_weights

# model_name = "Qwen/Qwen3.5-0.8B-Base"
model_name = "Qwen/Qwen3-0.6B-Base"
image_path = "/home/weiyutao/ai/continual_learning/00071.jpg"
question = "请仔细观察这张图片。这是什么作物的叶片？叶片上出现了什么症状？可能是什么病害？"

# ---------- 1. 动态构建跨设备 device_map ----------
# embed_tokens / lm_head（与 embed 权重绑定）单张量 ~508MB，超 WSL2 连续显存上限，留 CPU；
# visual 视觉编码器也留 CPU，使图像特征与文本 embedding 在同一设备融合，避免跨设备报错；
# 其余（语言层、norm）放 GPU。
print("正在推导 device_map ...")
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
with init_empty_weights():
    skeleton = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)

device_map = {name: "cuda:0" for name, _ in skeleton.named_children()}
for name, module in skeleton.named_modules():
    leaf = name.split(".")[-1]
    if name.endswith("embed_tokens") or leaf == "lm_head" or leaf == "visual":
        device_map[name] = "cpu"
print("device_map =", device_map)

# ---------- 2. 加载 processor 与模型 ----------
print(f"\n正在加载 {model_name} ...")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=True,
)
model.eval()
print("模型加载成功。")

# ---------- 3. 预取视觉占位 token（无 chat template 时手动拼接用） ----------
_tok = processor.tokenizer
_VS = _tok.convert_ids_to_tokens(config.vision_start_token_id)
_VE = _tok.convert_ids_to_tokens(config.vision_end_token_id)
_IP = _tok.convert_ids_to_tokens(config.image_token_id)


def ask(question, image=None, max_new_tokens=512):
    """构造图文输入并生成。image 为 None 时是纯文本推理。"""
    if image is not None:
        content = [{"type": "image"}, {"type": "text", "text": question}]
    else:
        content = [{"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Base 无 chat template：有图则手动拼视觉占位符，纯文本直接用问题
        text = f"{_VS}{_IP}{_VE}{question}" if image is not None else question

    proc_kwargs = {"text": [text], "return_tensors": "pt"}
    if image is not None:
        proc_kwargs["images"] = [image]
    inputs = processor(**proc_kwargs)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


leaf_image = Image.open(image_path).convert("RGB")

# ---------- 题 A：纯文本 · 逻辑推理 ----------
q_a = (
    "某小麦田 5 月初发现条锈病，病叶率 3%。已知条锈病在 15-20℃、有结露条件下潜育期约 7 天，"
    "每个病斑可产孢导致约 5 片新叶发病。若未来 21 天持续适温高湿且不施药，按此扩繁速度，"
    "理论上病叶率将达到多少？这个推算的现实局限是什么？"
)
print("\n" + "=" * 60)
print("【题 A · 纯文本逻辑推理】")
print("=" * 60)
print(ask(q_a, image=None))

# ---------- 题 B：图文结合 · 鉴别诊断 ----------
q_b = (
    "请仔细观察这张小麦叶片图片。叶锈病、条锈病、秆锈病三者在孢子堆颜色、排列方式、着生部位上"
    "如何区分？依据本图特征，最可能是哪一种？为什么排除另外两种？"
)
print("\n" + "=" * 60)
print("【题 B · 图文结合鉴别诊断】")
print("=" * 60)
print(ask(q_b, image=leaf_image))
