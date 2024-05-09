import os
import gc
import json
import torch
import shutil
import argparse
from tqdm import tqdm
from os.path import join, exists
from collections import defaultdict
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

import tracemalloc

# constants for Llama-1
LORA_PARA = {
    '7b':  {'dim': 4096, 'ffn': 11008, 'n_layer': 32},
    '13b': {'dim': 5120, 'ffn': 13824, 'n_layer': 40},
    '30b': {'dim': 6656, 'ffn': 17920, 'n_layer': 60},
    '65b': {'dim': 8192, 'ffn': 22016, 'n_layer': 80},
}

SELF_ATTENTION_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
]

FFN_MODULES = [
    'gate_proj', 'down_proj', 'up_proj'
]

def create_directory(path):
    """Creates a directory if it does not exist."""
    if os.path.exists(path):
        raise FileExistsError(f"The directory {path} already exists.")
    else:
        os.makedirs(path)

def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Computes a summed-area table
# This table allows us to quickly calculate the sum of any sub-matrix
def compute_summed_area_table(matrix):
    return torch.cumsum(torch.cumsum(matrix, dim=0), dim=1)

# Computes the sum of a submatrix given a summed-area table
# Uses the top left and bottom right coordinates of the desired submatrix
def sum_submatrix(sum_area_table, top_left, bottom_right):
    total = sum_area_table[bottom_right[0], bottom_right[1]]
    left = sum_area_table[top_left[0] - 1, bottom_right[1]] if top_left[0] > 0 else 0
    above = sum_area_table[bottom_right[0], top_left[1] - 1] if top_left[1] > 0 else 0
    above_left = sum_area_table[top_left[0] - 1, top_left[1] - 1] if top_left[0] > 0 and top_left[1] > 0 else 0
    return total - left - above + above_left

# Finds the submatrix of size tgt_x x tgt_y within the sensitivity matrix that has the maximum sum of elements
def max_sens_submatrix(sens, tgt_x, tgt_y):
    x_len, y_len = sens.shape
    max_sum = -float("inf")
    top_left_max = (0, 0)

    sum_area_table = compute_summed_area_table(sens)

    # Iterate over all possible submatrices of the desired size within the sensitivity matrix
    for i in range(tgt_x - 1, x_len):
        for j in range(tgt_y - 1, y_len):
            # Compute the top left coordinate of the current submatrix
            top_left = (i - tgt_x + 1, j - tgt_y + 1)
            # Compute the sum of the current submatrix
            current_sum = sum_submatrix(sum_area_table, top_left, (i, j))
            # Update max_sum and top_left_max if the current sum is larger than the maximum found so far
            if current_sum > max_sum:
                max_sum = current_sum
                top_left_max = top_left

    # Compute the bottom right coordinate of the submatrix with maximum sum
    bottom_right_max = (top_left_max[0] + tgt_x - 1, top_left_max[1] + tgt_y - 1)
    return top_left_max, bottom_right_max

def tokenize(text, tokenizer, device="cpu"):
    result = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding=False,
        return_tensors='pt'
    ).to(device)

    result["labels"] = result["input_ids"].clone()

    return result

def get_content(messages, tokenizer):
    message_text = ''
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token
    return message_text

def calculate_sensitivity(model, tokenizer, data, device="cpu", loss_on_output=True):
    sample_sensitivity = {}
        
    full_prompt = get_content(data['messages'], tokenizer)
    tokenized_full_prompt = tokenize(full_prompt, tokenizer)

    if loss_on_output:
        user_prompt = get_content(data['messages'][:-1], tokenizer)
        user_prompt += "<|assistant|>\n"
        tokenized_user_prompt = tokenize(user_prompt, tokenizer)
        user_prompt_len = tokenized_user_prompt["input_ids"].size(1)
            
        tokenized_full_prompt["labels"] = torch.cat([
            torch.full((1, user_prompt_len), -100, dtype=torch.long, device=device),
            tokenized_full_prompt["labels"][:, user_prompt_len:]
        ], dim=-1)

    # forward pass with gradient tracking
    model.zero_grad()
    outputs = model(**tokenized_full_prompt)
    predictions = outputs.logits

    # calculate and backward loss
    # loss_fct = torch.nn.CrossEntropyLoss()
    # loss = loss_fct(predictions.view(-1, model.config.vocab_size)[:-1], 
    #                 inputs['input_ids'].view(-1)[1:])
    loss = outputs.loss
    loss.backward()

    # calculate sensitivity for each parameter
    for name, param in model.named_parameters():
        if param.grad is not None:
            cur_sensitivity = torch.abs(param * param.grad)
            cur_sensitivity = cur_sensitivity.detach()
            sample_sensitivity[name] = cur_sensitivity

	# clear memory after computing the sensitivity for one sample
    del full_prompt, tokenized_full_prompt, outputs, predictions, loss
    model.zero_grad()
    gc.collect()

    return sample_sensitivity

def select_top_k_layers(sensitivity, args):
    k = LORA_PARA[args.lora_size]['n_layer']
    layer_sensitivity = defaultdict(float)
    for name in sensitivity:
        if "layers" in name:
            layer_index = name.split(".")[2]
            cur_sensiticity = sensitivity[name].sum().item()
            layer_sensitivity[layer_index] += cur_sensiticity
    
    # sort layers by sensitivity
    sorted_layers = sorted(layer_sensitivity.items(), key=lambda item: item[1], reverse=True)
    print(sorted_layers)

    # keep the layers with the highest sensitivity, and again by layer index
    sorted_layers = sorted_layers[:k]
    sorted_layers = sorted(sorted_layers, key=lambda item: int(item[0]))

    return [int(layer) for layer, sensitivity in sorted_layers]

def resize_params(params, sens, tgt_x, tgt_y):
    if tgt_x > params.size(0) or tgt_y > params.size(1):
        raise ValueError('Only dimensionality reduction of the parameter matrix is currently supported!')
    if params.shape != sens.shape:
        raise ValueError('Inconsistent dimensions of the parameter and sensitivity matrices.')
    
    '''
    # Compute the average sensitivity for each row and each column
    row_sens_avg = torch.mean(sens, dim=1)
    col_sens_avg = torch.mean(sens, dim=0)

    # Get the indices of the rows and columns with the highest average sensitivity
    high_sens_row_indices = torch.topk(row_sens_avg, tgt_x)[1]
    high_sens_col_indices = torch.topk(col_sens_avg, tgt_y)[1]

    # Sort the indices to maintain the original order
    high_sens_row_indices = torch.sort(high_sens_row_indices)[0]
    high_sens_col_indices = torch.sort(high_sens_col_indices)[0]

    # Get the parameters with high sensitivity
    resized_params = params[high_sens_row_indices, :][:, high_sens_col_indices]
    '''
    
    top_left, bottom_right = max_sens_submatrix(sens, tgt_x, tgt_y)
    resized_params = params[top_left[0] : bottom_right[0] + 1, top_left[1] : bottom_right[1] + 1]
    return resized_params

# Perform SVD and keep only top r singular values/vectors
def svd(mat, r):

    # torch.linalg.svd does not support fp16
    U, S, Vh = torch.linalg.svd(mat) # , driver='gesvd')

    # truncate U, S, Vh
    U = U[:, :r]
    S = S[:r]
    S_diag = torch.diag(S)
    U = U @ S_diag
    Vh = Vh[:r, :]

    return U.half(), Vh.half()

def main(args):
    # set up
    create_directory(write_path := join('extracted_lora',
                     f'{args.model_size}-to-{args.lora_size}-{args.task}'))

    # load models
    device = "cpu"
    model = LlamaForCausalLM.from_pretrained(
        # f'elinas/llama-{args.model_size}-hf-transformers-4.29', # Llama-1
        f'meta-llama/Llama-2-{args.model_size}-hf',
        # torch_dtype=torch.float16, # doesn't work for cpu
        low_cpu_mem_usage=True,
    ).to(device)
    # tokenizer = LlamaTokenizer.from_pretrained(f'elinas/llama-{args.model_size}-hf-transformers-4.29')
    tokenizer = AutoTokenizer.from_pretrained(f'meta-llama/Llama-2-{args.model_size}-hf')

    # load seed data
    data = load_data(path=f'data/{args.task}/sample_{args.task}.jsonl')
    
    # calculate sensitivity for all the parameters
    print('Start calculating sensitivity!')
    sensitivity = {}
    for sample in tqdm(data):
        sample_sensitivity = calculate_sensitivity(model, tokenizer, sample, device)
        for name, cur_sensitivity in sample_sensitivity.items():
            if name in sensitivity:
                sensitivity[name] += cur_sensitivity
            else:
                sensitivity[name] = cur_sensitivity
        del sample_sensitivity
        gc.collect()

    # select top k layers
    selected_layers = select_top_k_layers(sensitivity, args)
    print(f'The following layers are selected by using sensitivity:\n{selected_layers}')

    # extract task-sepcific knowledge
    lora = {}
    model_params = model.state_dict()

    d_small = LORA_PARA[args.lora_size]['dim']
    d_large = LORA_PARA[args.model_size]['dim']
    d_ffn_small = LORA_PARA[args.lora_size]['ffn']
    d_ffn_large = LORA_PARA[args.model_size]['ffn']
    vocab_dim = model.config.vocab_size

    # extract the embedding layer for lora
    if args.include_emb:
        key = 'model.embed_tokens.weight'
        # [d_large, vocab_dim] -> [d_small, vocab_dim]
        resized_params = resize_params(model_params[key].t(), sensitivity[key].t(), 
                                       d_small, vocab_dim)
        # [d_small, vocab_dim] -> [d_small, r], [r, vocab_dim]
        B, A = svd(resized_params, args.r)
        key_A = 'base_model.model.model.embed_tokens.lora_embedding_A'
        key_B = 'base_model.model.model.embed_tokens.lora_embedding_B'
        lora[key_A] = A
        lora[key_B] = B

    print('Start extracting parameters from each layer!')
    # extract the parameters in different layers
    new_layer_index = 0
    for layer in tqdm(selected_layers):
        # extract the self attention module for lora
        if args.include_self_attention:
            for module in SELF_ATTENTION_MODULES:
                key = f'model.layers.{layer}.self_attn.{module}.weight'
                # [d_large, d_large] -> [d_small, d_small]
                resized_params = resize_params(model_params[key], sensitivity[key], 
                                               d_small, d_small)
                # [d_small, d_small] -> [d_small, r], [r, d_small]
                key = key.replace(f'layers.{layer}', f'layers.{new_layer_index}')
                B, A = svd(resized_params, args.r)
                key_A = 'base_model.model.' + key.replace('.weight', '.lora_A.weight')
                key_B = 'base_model.model.' + key.replace('.weight', '.lora_B.weight')
                lora[key_A] = A
                lora[key_B] = B
        # extract the self attention module for lora
        if args.include_ffn:
            for module in FFN_MODULES:
                key = f'model.layers.{layer}.mlp.{module}.weight'
                if module == 'down_proj':
                    # [d_large, d_ffn_large] -> [d_small, d_ffn_small]
                    resized_params = resize_params(model_params[key], sensitivity[key], 
                                                   d_small, d_ffn_small)
                else:
                    # [d_ffn_large, d_large] -> [d_ffn_small, d_small]
                    resized_params = resize_params(model_params[key], sensitivity[key], 
                                                   d_ffn_small, d_small)
                key = key.replace(f'layers.{layer}', f'layers.{new_layer_index}')
                # [d_small, d_ffn_small] -> [d_small, r], [r, d_ffn_small] or
                # [d_ffn_small, d_small] -> [d_ffn_small, r], [r, d_small]
                B, A = svd(resized_params, args.r)
                key_A = 'base_model.model.' + key.replace('.weight', '.lora_A.weight')
                key_B = 'base_model.model.' + key.replace('.weight', '.lora_B.weight')
                lora[key_A] = A
                lora[key_B] = B
        new_layer_index += 1

    # extract the last projection layer for lora
    if args.include_emb:
        key = 'lm_head.weight'
        # [vocab_dim, d_large] -> [vocab_dim, d_small]
        resized_params = resize_params(model_params[key], sensitivity[key], 
                                       vocab_dim, d_small)
        # [vocab_dim, d_small] -> [vocab_dim, r], [r, d_small]
        B, A = svd(resized_params, args.r)
        key_A = 'base_model.model.lm_head.lora_A.weight'
        key_B = 'base_model.model.lm_head.lora_B.weight'
        lora[key_A] = A
        lora[key_B] = B

    # save new weights and config
    torch.save(lora, join(write_path, 'adapter_model.bin'))
    shutil.copy(join('configs/Llama-2', f'adapter_{args.lora_size}.json'), 
                join(write_path, 'adapter_config.json'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract task-specific parametric knowledge from large model into small LoRA.'
    )
    parser.add_argument('--model_size', default='13b',
                        choices=['7b', '13b', '30b', '65b'],
                        help='size of large model', type=str)
    parser.add_argument('--lora_size', default='7b',
                        choices=['7b', '13b', '30b', '65b'],
                        help='size of extracted LoRA', type=str)
    parser.add_argument('--task', default='gsm',
                        help='task corresponding to seed data', type=str)
    parser.add_argument('--r', default=16,
                        help='rank of LoRA model', type=int)
    parser.add_argument('--include_emb',
                        default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='include embedding layer in extracted LoRA')
    parser.add_argument('--include_ffn',
                        default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='include FFN module in extracted LoRA')
    parser.add_argument('--include_self_attention',
                        default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='include self attention module in extracted LoRA')
    args = parser.parse_args() 
    main(args)
