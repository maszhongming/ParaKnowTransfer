import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def apply_lora(base_model_path, target_model_path, lora_path, delta_path=None):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    
    # adding delta
    print(f"Loading the LoRA adapter from {delta_path}")
    delta_init_model = PeftModel.from_pretrained(
        base,
        delta_path,
        # torch_dtype=torch.float16,
    )
    
    print("Adding the delta init")
    model = delta_init_model.merge_and_unload()
    
    # merging lora
    print(f"Loading the LoRA adapter from {lora_path}")
    lora_model = PeftModel.from_pretrained(
        model,
        lora_path,
    )
    
    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(base_tokenizer) > embedding_size:
        print(f"The vocabulary size of the tokenizer in the lora model folder contains {len(base_tokenizer)-embedding_size} more tokens than the base model.")
        print("Resizing the token embeddings of the merged model...")
        model.resize_token_embeddings(len(base_tokenizer))

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True)

    args = parser.parse_args()

    apply_lora(args.base_model_path, args.target_model_path, args.lora_path, args.delta_path)
