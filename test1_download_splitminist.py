#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time : 2026/07/06 20:28:26
@Author : weiyutao
@File : test1.py
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

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
# 2. 定义核心的“任务过滤”函数
# ==========================================
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


# ==========================================
# 3. 构建你的持续学习流水线 (Task A 和 Task B)
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