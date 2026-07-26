#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training package for GPT models.
"""

from .common.trainer_base import (
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    calc_loss_loader,
    evaluate_model,
    generate_text_simple,
    generate_and_print_sample,
    train_model
)
from .common.dataset_base import (
    PretrainDataset,
    create_dataloader
)
from .common.gpt_lm import GPT2LM


__all__ = [
    'text_to_token_ids',
    'token_ids_to_text',
    'calc_loss_batch',
    'calc_loss_loader',
    'evaluate_model',
    'generate_text_simple',
    'generate_and_print_sample',
    'train_model',
    'PretrainDataset',
    'create_dataloader',
    'GPT2LM'
]
