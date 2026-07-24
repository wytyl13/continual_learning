#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/07 10:55
@Author  : weiyutao
@File    : chapter4_hq_block.py
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch.optim as optim
import copy
import torch.nn as nn
import torch.nn.functional as F


def create_task_dataloader(dataset, target_classes, batch_size=64, is_train=True):
    """
    从完整数据集中提取指定类别的样本，构成当前任务的 DataLoader
    :param dataset: 原始数据集 (如 full_train_dataset)
    :param target_classes: 当前任务需要学习的类别列表，例如 [0, 1]
    """
    # 提取所有标签
    targets = dataset.targets.numpy()
    
    # 找到属于目标类别的所有样本的索引 (Indices)
    # np.isin 会返回一个布尔数组，np.where 将其转换为具体的位置索引
    indices = np.where(np.isin(targets, target_classes))[0]
    
    # 使用 Subset 取出这些特定索引的数据子集
    task_subset = Subset(dataset, indices)
    
    # 包装成 DataLoader
    loader = DataLoader(
        task_subset, 
        batch_size=batch_size, 
        shuffle=is_train  # 训练集打乱，测试集不打乱
    )
    
    print(f"提取类别 {target_classes} 完成，共包含样本数: {len(indices)}")
    return loader


class SignSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        # -1 when x < 0 else 1.
        return torch.sign(x + 1e-8)


    @staticmethod
    def backward(ctx, grad_output):
        # 强制把阶跃函数的倒数置为1.
        return grad_output.clone()


class QH_Block(nn.Module):
    def __init__(self, in_dim, out_dim, k=8):
        super().__init__()
        
        # 门控权重
        self.w_gate = nn.Linear(in_dim, 1)

        # 非量化分支
        self.w_n = nn.Linear(in_dim, k)
        self.w_n_proj = nn.Linear(k, out_dim)

        # 量化分支
        self.q_branch = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )


    def forward(self, x):
        # 门控权重
        alpha = torch.sigmoid(self.w_gate(x))
        
        # 非量化分支
        n_x = SignSTE.apply(self.w_n(x)) 
        n_out = self.w_n_proj(n_x)
        
        # 量化分支
        q_out = self.q_branch(x)
        
        # 混合叠加
        return x + alpha * n_out + (1 - alpha) * q_out



class QHNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.qh_block = QH_Block(in_dim=256, out_dim=256, k=8)
        self.fc2 = nn.Linear(256, 2)
    
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.qh_block(x)
        x = self.fc2(x)
        return x



def train_continual_task(
    train_loader, 
    task_name, 
    target_classes, 
    epochs=2
):
    model = QHNet()
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("\n================开始训练：{task_name}====================")

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
    
            mapped_target = torch.where(target == target_classes[0], 0, 1).to(data.device)
            output = model(data)
            loss = criterion(output, mapped_target)

            loss.backward()

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    features = F.relu(model.fc1(model.flatten(data)))
                    alpha_tensor = torch.sigmoid(model.qh_block.w_gate(features))
                    alpha_mean = alpha_tensor.mean().item()

                    q_grad = model.qh_block.q_branch[0].weight.grad
                    q_grad_norm = q_grad.norm().item() if q_grad is not None else 0.0
                    
                    print(f"Epoch {epoch+1} | Batch {batch_idx:03d} | Loss: {loss.item():.4f} | "
                          f"Alpha: {alpha_mean:.4f} | Q分支梯度Norm: {q_grad_norm:.6f}")
            optimizer.step()
            total_loss += loss.item()
            

if __name__ == "__main__":
    # ==========================================
    # 1. 下载原始完整数据集 (PyTorch 会自动下载到 ./data 文件夹)
    # ==========================================
    print("正在下载/加载原始 MNIST 数据集...")
    # 定义数据预处理：转为 Tensor 并进行标准化 (MNIST的标准均值和方差)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # train=True 下载训练集，download=True 表示如果本地没有就自动联网下载
    full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    

    # ==========================================
    # 2. 构建持续学习流水线 (Task A 和 Task B)
    # ==========================================
    print("\n开始构建持续学习任务流...")

    # 任务 A: 只学习识别数字 0 和 1
    task_A_classes = [0, 1]
    task_A_train_loader = create_task_dataloader(full_train_dataset, task_A_classes, is_train=True)
    task_A_test_loader  = create_task_dataloader(full_test_dataset, task_A_classes, is_train=False)

    # 任务 B: 只学习识别数字 2 和 3
    task_B_classes = [2, 3]
    task_B_train_loader = create_task_dataloader(full_train_dataset, task_B_classes, is_train=True)
    task_B_test_loader  = create_task_dataloader(full_test_dataset, task_B_classes, is_train=False)

    print("\n数据流准备完毕！可以开始喂给 QH-Block 网络进行测试了。")
    
    train_continual_task(
        train_loader=task_A_train_loader,
        task_name="Task A (数字 0-1)",
        target_classes=task_A_classes,
        epochs=2
    )
    
    
    train_continual_task(
        train_loader=task_B_train_loader,
        task_name="Task B (数字 2-3)",
        target_classes=task_B_classes,
        epochs=2
    )

    print("\n持续学习训练流测试完毕！")







