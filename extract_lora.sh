python extract_lora_with_sensitivity.py \
    --model_size 13b \
    --lora_size 7b \
    --task gsm

python get_delta.py \
  --path extracted_lora/13b-to-7b-gsm