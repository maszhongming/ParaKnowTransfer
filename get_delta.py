import os
import torch
import shutil
import argparse
from os.path import join, dirname, basename


def get_delta_lora(args):
    model = torch.load(join(args.path, 'adapter_model.bin'))
    for key in model:
        if 'lora_A' in key or 'lora_embedding_A' in key:
            model[key] *= -1
    return model

def main(args):
    write_name = basename(args.path) + '-delta'
    parent_path = dirname(args.path)
    write_path = join(parent_path, write_name)
    os.mkdir(write_path)
    
    delta_lora = get_delta_lora(args)
    torch.save(delta_lora, join(write_path, 'adapter_model.bin'))
    shutil.copy(join(args.path,  'adapter_config.json'),
                join(write_path, 'adapter_config.json'))
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Save the weights of LoRA delta'
    )
    parser.add_argument('--path', type=str, required=True,
                        help='Path to the original LoRA')
    args = parser.parse_args()
    main(args)
