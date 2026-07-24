#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/08 08:53
@Author  : weiyutao
@File    : chapter5_train.py
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import torch
import tiktoken
from examples.book_chapter.chapter4_gpt_model import GPTModel, generate_text_simple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "embedding_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids: torch.tensor, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


"""
Next, we implement a utility function to calculate the cross entropy loss of a given batch returned via the training and validation loader.
"""
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


"""
We can implement this training flow via the train_model_simple function in code.
"""

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
    
    

def train_model_simple(model, train_loader, val_loader, 
                       optimizer, device, num_epochs, 
                       eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            
            # 反向传播
            loss.backward()

            # 优化器执行梯度更新权重
            optimizer.step()
            
            # 计算累积token数
            tokens_seen += input_batch.numel()


            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter=eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )
        
        # 在每个epoch结束之后进行随堂测验
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, token_seen, train_losses, val_losses):
    # 可视化训练过程中保存的每个epoch下的损失
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Trainig loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.savefig("loss_plot.png", dpi=300, bbox_inches="tight")



if __name__ == "__main__":
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10, 
        context_size=GPT_CONFIG_124M["context_length"]
    )
    
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    """
    So far, we have generate the next token before training the GPTModel. In order to continue our training work, we should calculate
    a loss metric for the generated outputs. This loss serves as a progress and success indicator of the training progress. Furthermore, 
    in later chapters, when we fine-tune our LLM, we will review additional methodologies for assessing model quality.
    
    The model training aims to increase the softmax probability in the index positions corresponding to the correct target token IDs.
    this softmax probability is also used in the evaluation metric we will implement next to numerically assess the models's generated outputs: 
    the higher the probability in the correctg positions, the better. The vocanbulary we are using for our GPT-2 model has 50257 tokens, so most
    of the initial probabilities will hover around 0.00002 (1 / 50257)
    """
    inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                           [40,    1107, 588]]) # "I really like"]
    
    targets = torch.tensor([[3626, 6100, 345 ], # ["effort moves you",
                            [1107, 588, 11311]]) # "really like chocolate"]
    
    with torch.no_grad():
        logits = model(inputs)
    
    probas = torch.softmax(logits, dim=-1) # batch, num_tokens, vocab_size
    print(probas.shape)
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
    
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)
    
    """
    The goal of training an LLM is to maximize the likelihood of the correct token, which involves increasing its probability
    relative to other tokens. This way, we ensure the LLM consistently picks the target token——essentially the next word in 
    the sentence——as the next token it generates. we poroceed it by applying the logarithm to the probability scores.
    """
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)
    
    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)
    
    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)
    
    print("Logits shape:", logits.shape) # 2, 3, 50257
    print("Targets shape:", targets.shape) # 2, 3

    """
    At its core, the cross entropy loss is a popular measure in machine learning and deep learning
    that measures the difference between two probability distributions——typically, the true distribution of labels and
    the predicted distribution from a model. In the context of machine learning and specifically in frameworks like PyTorch, the
    cross_entropy funciton computes this measure for discrete outcomes, which is similar to the negative average log probability of the 
    target tokens given the model's generated token probabilitie, making the terms "cross entropy" and "negative average log probability"
    related and often used interchangeably in practice.
    """
    logits_flat = logits.flatten(0, 1) # 6, 50257
    targets_flat = targets.flatten() # 6
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)

    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)

    """
    Perplexity is a measure often used alongside cross entropy loss to evaluate the performance
    of models in tasks like language modeling. It can provide a more interpretable way to understand the uncertainty of a model
    in predicting the next token in a sequence.
    
    Perplexity measures how well the probability distribution predicted by the model matches the actual distribution of the words
    in the dataset. Similar to the loss, a lower perplexity indicates that the model predictions are closer to the actual distribution.
    
    Perplexity can be calculated as perplexity = torch.exp(loss), which returns tensor(48725.8203) when applied the previously calculated loss.
    
    Perplexity is often considered more interpretable than the raw loss value because it signifies the effective vocabulary size about which
    the model is uncertain at each step. In the given example, this would translate to the model being unsure about which among 48725
    tokens in the vocabulary to generate as the next token.
    也就是说eˡᴼˢˢ的值如果越小，比如是1，说明下一个token预测输出基本可以确定为某一个索引，但是如果该困惑度值接近词表大小，比如48725已经非常接近词表大小50257
    说明模型基本在瞎猜。
    we have now calculated the loss for two small text inputs for illustration purpose. Next, we will apply the loss computation
    to the entire training and validation sets.
    
    To compute the loss on the training and validation datasets, we use a very small text dataset, the "The Verdict" short story by Edith Wharton.
    """
    
    file_path = "The_Verdict.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Characters:", total_characters)
    print("Tokens:", total_tokens)

    
    
    train_ratio = 0.9
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    from examples.book_chapter.chapter2_dataloader import create_dataloader_v1
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("Val loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)
    
    
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
        print("Training loss:", train_loss)
        print("Validation loss:", val_loss)
        
        
    
    """
    Adam optimizers are a popular choice for training deep neural networks. However, in our training loop, we 
    opt for the AdamW optimizer. AdamW is a variant of Adam that improves the weight decay approach, which aims 
    to minimize model complexity and prevent overfitting by penalizing larger weights. This adjustment allows AdamW
    to acheive more effective regularization and better generalization; thus, AdamW is frequently used in the training of LLMs.
    """
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1
    )
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, 
        start_context="Every effort moves you", tokenizer=tokenizer
    )
    
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    
    
    

    
    