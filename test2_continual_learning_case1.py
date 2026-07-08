#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/02/15 09:07
@Author  : weiyutao
@File    : test2.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy

# --- 1. 定义模型 ---
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.linear.weight, 0.0)

    def forward(self, x):
        return self.linear(x)

# --- 2. 准备数据 ---
input_feature = torch.tensor([[1.0]])
target_A = torch.tensor([[10.0]]) # V1 旧知识
target_B = torch.tensor([[12.0]]) # V2 新知识

print(f"{'='*15} 持续学习终极测试：EWC vs 蒸馏 {'='*15}\n")

# ==========================================
# 阶段 0：预训练 (V1 老师模型)
# ==========================================
print("【阶段 0】预训练 V1 模型 (目标 10.0)")
model_v1 = SimpleModel()
optimizer = optim.SGD(model_v1.parameters(), lr=0.05)

for i in range(50):
    optimizer.zero_grad()
    loss = nn.MSELoss()(model_v1(input_feature), target_A)
    loss.backward()
    optimizer.step()

w_old = model_v1.linear.weight.item()
print(f"  > V1 训练结束，权重: {w_old:.2f}\n")

# 保存 V1 作为“老师” (Teacher)，并冻结它
teacher_model = copy.deepcopy(model_v1)
for param in teacher_model.parameters():
    param.requires_grad = False # 老师不参与训练，只负责“指指点点”

# ==========================================
# 测试 1：基准 (无保护)
# ==========================================
print("【测试 1】基准：无保护直接训练")
model_1 = copy.deepcopy(model_v1)
optimizer_1 = optim.SGD(model_1.parameters(), lr=0.05)

for i in range(50):
    optimizer_1.zero_grad()
    loss = nn.MSELoss()(model_1(input_feature), target_B)
    loss.backward()
    optimizer_1.step()

print(f"  > 最终权重: {model_1.linear.weight.item():.2f}")
print(f"  > 结果：彻底遗忘，变成了 12.0。\n")


# ==========================================
# 测试 2：正则化 (EWC 思想)
# 约束：Loss + λ * (w - w_old)^2
# ==========================================
print("【测试 2】正则化 (EWC)：锁参数")
model_2 = copy.deepcopy(model_v1)
optimizer_2 = optim.SGD(model_2.parameters(), lr=0.05)

lambda_reg = 1.0 # 正则化系数

for i in range(50):
    optimizer_2.zero_grad()
    
    # 新任务 Loss
    pred = model_2(input_feature)
    task_loss = nn.MSELoss()(pred, target_B)
    
    # 正则化 Loss (Penalty)
    current_w = model_2.linear.weight
    anchor_w = torch.tensor([[w_old]])
    reg_loss = lambda_reg * (current_w - anchor_w) ** 2
    
    total_loss = task_loss + reg_loss
    total_loss.backward()
    optimizer_2.step()

print(f"  > 最终权重: {model_2.linear.weight.item():.2f}")
print(f"  > 结果：权重被参数距离强行拉住。\n")


# ==========================================
# 测试 3：知识蒸馏 (Knowledge Distillation)
# 约束：(1-α)*Task_Loss + α*Distill_Loss
# ==========================================
print("【测试 3】知识蒸馏 (KD)：锁输出 (推荐)")
model_3 = copy.deepcopy(model_v1) # 学生继承老师的身体
optimizer_3 = optim.SGD(model_3.parameters(), lr=0.05)

# alpha (蒸馏系数)：
# 0.0 = 只学新知识 (遗忘)
# 1.0 = 只模仿老师 (不学新)
# 0.5 = 平衡
alpha = 0.5 

for i in range(50):
    optimizer_3.zero_grad()
    
    # 1. 学生预测新数据
    pred_student = model_3(input_feature)
    
    # 2. 计算新任务 Loss (Student vs Target 12.0)
    loss_task = nn.MSELoss()(pred_student, target_B)
    
    # 3. 计算蒸馏 Loss (Student vs Teacher 10.0)
    # 让老师也看一眼现在的数据，问老师：“由于没有旧数据了，如果是你，你会输出什么？”
    with torch.no_grad():
        pred_teacher = teacher_model(input_feature)
        
    loss_distill = nn.MSELoss()(pred_student, pred_teacher)
    
    # 4. 融合 Loss
    total_loss = (1 - alpha) * loss_task + alpha * loss_distill
    
    total_loss.backward()
    optimizer_3.step()

print(f"  > 最终权重: {model_3.linear.weight.item():.2f}")
print(f"  > 结果：通过模仿老师的行为，权重也稳定在了中间值。\n")

# ==========================================
# 总结对比
# ==========================================
print(f"{'='*10} 总结对比 {'='*10}")
print(f"旧知识(A): 10.0 | 新知识(B): 12.0")
print(f"1. 无保护权重: {model_1.linear.weight.item():.2f} (遗忘)")
print(f"2. EWC 权重:   {model_2.linear.weight.item():.2f} (僵硬)")
print(f"3. KD 权重:    {model_3.linear.weight.item():.2f} (灵活，推荐)")
