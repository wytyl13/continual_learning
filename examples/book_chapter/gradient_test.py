#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/02/15 08:51
@Author  : weiyutao
@File    : gradient_test.py
"""


import torch
import torch.nn as nn
import torch.optim as optim

# --- 定义模型 ---
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.linear.weight, 0.0) # 初始权重为 0

    def forward(self, x):
        return self.linear(x)

# --- 准备数据 ---
input_feature = torch.tensor([[1.0]]) 
target_A = torch.tensor([[10.0]]) # 旧知识 
target_B = torch.tensor([[12.0]]) # 新知识

print(f"{'='*15} 修正后的测试 {'='*15}\n")

# ==========================================
# 场景 1：同一个梯度累积 (Gradient Accumulation) - 循环训练
# 模拟：每次更新前，都完整地看了一遍 A 和 B
# ==========================================
print("【场景 1】同一个梯度累积 (真正的融合)")
model_1 = SimpleModel()
optimizer = optim.SGD(model_1.parameters(), lr=0.05)

# 我们让这个过程循环 50 次，模拟模型收敛的过程
for epoch in range(150):
    optimizer.zero_grad() # 清空梯度
    
    # 1. 积累 A 的梯度
    loss_a = nn.MSELoss()(model_1(input_feature), target_A)
    loss_a.backward() 
    
    # 2. 积累 B 的梯度 (不清空，直接叠加)
    loss_b = nn.MSELoss()(model_1(input_feature), target_B)
    loss_b.backward() 
    
    # 3. 根据 A+B 的总意见迈出一步
    optimizer.step()

print(f"  > 最终权重 w: {model_1.linear.weight.item():.2f}")
print(f"  > 结论：权重完美停在了 10 和 12 的中间 (11.00)。\n")


# ==========================================
# 场景 2：同一个版本，但不同批次 (Interleaved Batch)
# 模拟：数据 A 和 B 交替出现
# ==========================================
print("【场景 2】同一个版本，数据交替 (动态平衡)")
model_2 = SimpleModel()
optimizer = optim.SGD(model_2.parameters(), lr=0.05)

for epoch in range(150):
    # Step A
    optimizer.zero_grad()
    loss_a = nn.MSELoss()(model_2(input_feature), target_A)
    loss_a.backward()
    optimizer.step()
    
    # Step B
    optimizer.zero_grad()
    loss_b = nn.MSELoss()(model_2(input_feature), target_B)
    loss_b.backward()
    optimizer.step()

print(f"  > 最终权重 w: {model_2.linear.weight.item():.2f}")
print(f"  > 结论：权重在 11.0 附近微小震荡，宏观上实现了融合。\n")


# ==========================================
# 场景 3：不同的训练版本 (Sequential / Continual Learning)
# 模拟：先完全学会 A，再转而去学 B (不再看 A)
# ==========================================
print("【场景 3】不同版本，顺序训练 (灾难性遗忘)")
model_3 = SimpleModel()
optimizer = optim.SGD(model_3.parameters(), lr=0.05)

# --- 阶段 1：只训练 A ---
print("  [阶段 1] 训练旧数据 A...")
for i in range(150):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model_3(input_feature), target_A)
    loss.backward()
    optimizer.step()
print(f"    -> 阶段 1 结束，权重 w: {model_3.linear.weight.item():.2f}")

# --- 阶段 2：只训练 B ---
print("  [阶段 2] 训练新数据 B (A 缺席)...")
for i in range(50):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model_3(input_feature), target_B)
    loss.backward()
    optimizer.step()

print(f"  > 最终权重 w: {model_3.linear.weight.item():.2f}")
print(f"  > 结论：权重完全变成了 12.00，彻底忘记了 A。")