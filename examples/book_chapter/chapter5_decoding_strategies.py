#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2026/07/08 16:45
@Author  : weiyutao
@File    : chapter5_decoding_strategies.py
Let's look at text generation strategies to generate more original text. First, we will briefly revisit the
generate_text_simple function that we used inside generate_and_print_sample earlier. Then we will cover two
techniques, temperature scaling and top-k sampling, to improve this function.

Previously, inside the generate_text_simple function, we always sampled the token with the highest probablity as the next
token using torch.argmax, also known as greedy decoding. To generate text with more variety, we can replace argmax with a function
that samples from a probability distribution.
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
import tiktoken
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from examples.book_chapter.chapter4_gpt_model import GPTModel
from examples.book_chapter.chapter5_train import GPT_CONFIG_124M, train_model_simple, text_to_token_ids, token_ids_to_text
from examples.book_chapter.chapter2_dataloader import create_dataloader_v1
from examples.book_chapter.chapter4_gpt_model import generate_text_simple


def generate(model, idx, max_new_tokens, context_size, 
                temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # batch, context_size
        with torch.no_grad():
            logits = model(idx_cond) # batch, tokens, vocab_size
        logits = logits[:, -1, :] # batch, tokens-1, vocab_size

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k) # batch, tokens-1, top_k
            min_val = top_logits[:, -1] # batch, 1
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probas, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

if __name__ == "__main__":
    """
    # load data, model and train it.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    print(len(train_loader))
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
    
    model.to("cpu")
    model.eval()
    """
    # generate text used the trained model.
    
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
    """
    
    """
    test the decoding strategies.
    """
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8
    }
    inverse_vocab = {v: k for k, v in vocab.items()}
    print(f"inverse_vocab: {inverse_vocab}")
    
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print(inverse_vocab[next_token_id])
    """
    Since the largest logit value and, correspondingly, the largest softmax probability score are
    int the fourth position, the generate word is "forward". To implement a probability sampling process,
    we can now replace argmax with the multinomial function in PyTorch. As we can see, the word forward is sampled
    most of time, but  other tokens such as closer, inches, and toward will also be sampled some of the time.
    This means that if we replaced the argmax function with the multinomial function inside the generate_and_print_sample
    function, the LLM would sometimes generate texts such as every effort moves you toward, every effort moves you inches,
    and every effort moves you closer instead of every effort moves you forward. We can further control the distribution
    and selection process via a concept called temperature scaling. Temperature scaling is just a fancy description for dividing
    the logits by a number greater than 0. Temperatures greater than 1 result in more uniformly distributed token probabilities, 
    and temperatures smaller than 1 will result in more confident distributions.
    """ 
    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])
    
    def print_sampled_tokens(probas):
        torch.manual_seed(123)
        sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
        sampled_ids = torch.bincount(torch.tensor(sample)) # [freq_index_1, freq_index_2, freq_index_3]
        print(sampled_ids)
        for i, freq in enumerate(sampled_ids):
            print(f"{freq} x {inverse_vocab[i]}")

    print_sampled_tokens(probas)
    # 71 x closer
    # 2 x every
    # 0 x effort
    # 544 x forward
    # 2 x inches
    # 1 x moves
    # 0 x pizza
    # 376 x toward
    # 4 x you
    
    def softmax_with_temperature(logits, temperature):
        """
        A temperature of 1 divides the logits by 1 before passing them to the softmax function to compute the probabilty scores.
        In other words, using a temperature of 1 is the same as not using any temperature scaling. In this case, the tokens are selected
        with a probability equal to the original softmax probability scores via the multinomial sampling function in PyTorch. For example,
        for the temperature setting 1, the token corresponding to "forward" would be selected about 60% of the time. Also, applying very small
        temperatures, such as 0.1, will result in sharper distributions such that the behavior of the multinomial functiohn selects the most
        like token almost 100% of the time, approaching the behavior of the argmax function. Likewise, a temperature of 5 results in a more uniform
        distribution where other tokens are selected more often. This can add more variety to the generated texts but also more often results in
        nonsensical text.
        除以温度系数，如果温度系数小于1越近近0，所有z值会被无线放大，导致较大的z值经过softmax后计算得到的概率值越大。而温度系数如果大于1，会导致
        z值被无限缩小，导致各个z值经过softmax后计算得到的概率值被均匀化，概率值之间会非常接近。温度系数为1和原始z值一模一样。
        """
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=0)
    
    
    temperatures = [1, 0.1, 5]
    scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
    x = torch.arange(len(vocab))
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(x + i * bar_width, scaled_probas[i], bar_width, label=f'Temperature = {T}')
        ax.set_ylabel('Probability')
        ax.set_xticks(x)
        ax.set_xticklabels(vocab.keys(), rotation=90)
        ax.legend()
        plt.tight_layout()
        plt.savefig("temperatures_fig.png", dpi=300, bbox_inches="tight")
    
    
    """
    Top-K sampling.
    We've now implemented a probabilistic sampling approach coupled with temperature scaling to increase
    the diversity of the outputs. We saw that higher temperature values result in more uniformly distributed
    next-token probabilities, which result in more diverse outputs as it recduces the likelihood of the model
    repeatedly selecting the most probable token. This method allows for the exploring of less likely but potentially
    more interesting and creative paths in the generation process. However, one downside of this approach is that it 
    sometimes leads to grammatically incorrect or completely nonsensical outputs such as every effort moves you pizza.
    Top-K sampling, when combined with probabilistic sampling and temperature scaling, can improve the text
    generation results. In top-k sampling, we can restrict the sampled tokens to the top-k most likely tokens
    and exclude all other tokens from the selection process by masking their probability scores.
    The top-k approach replaces all nonselected logits with negative infinity value (-inf), such that when
    computing the softmax values, the probability scores of the non-top-k tokens are 0, and the remaining probabilities
    sum up to 1.
    """
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("Top logits:", top_logits)
    print("Top positions:", top_pos)
    
    new_logits = torch.where(condition=next_token_logits < top_logits[-1], input=torch.tensor(float('-inf')), other=next_token_logits)
    print(f"new_logits: {new_logits}")

    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)
    """
    Then we can now apply the temperature scaling and multinomial function for probabilistic sampling to select
    the next token among these three non-zero probability scores to generate the next token.

    Now, let's combine tremperature sampling and top-k sampling to modify the generate_text_simple function
    we used to generate text via the LLM earlier, creating a new generate function.
    """
    
    
    torch.manual_seed(123)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Evert effort moves you", tokenizer),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    # torch.save(model.state_dict(), "model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_and_optimizer.pth"     
    )
    
    