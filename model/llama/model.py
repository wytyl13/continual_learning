#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/24 11:06
@Author  : weiyutao
@File    : model.py

llama model architecture.

"""
import torch
import torch.nn as nn



class RMSNorm(nn.Module):
    """
    RMSNorm
    μ = 0
    σ² = 1/(d - 1) ⋅ Σ(xᵢ - μ)² 
    y = (x - μ) / (√σ²+ϵ) @ γ + β = x / (√σ²+ϵ) @ γ
    """
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def _norm(self, x):
        # σ² = 1/(d - 1) ⋅ Σ(xᵢ - μ)² = 1/(d - 1) ⋅ Σ(xᵢ)² = x.pow(2).mean(-1), 
        # -1 means calculate the mean of last dimension what based on the hidden_size dim.
        # torch.rsqrt = 1 / torch.sqrt
        return x * torch.rsqrt(x.pow(2).mean(-1, keep_dim=True) + self.eps)

    def forward(self, x):
        # x.float() means upcasting the x from float16 or bfloat16 to float32 to prevent numerical overflow.
        # type_as means downcasting the output from float32 to float16 or bfloat16
        output=  self._norm(x.float()).type_as(x)
        return output * self.weight
        
    


if __name__ == "__main__":
    print("llama model architecture")
