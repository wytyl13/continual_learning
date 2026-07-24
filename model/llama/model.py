#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/24 11:06
@Author  : weiyutao
@File    : model.py

llama model architecture.
"""


if __name__ == "__main__":
    import urllib.request
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"

    urllib.request.urlretrieve(url, file_path)
