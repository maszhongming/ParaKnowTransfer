python merge_lora.py \
    --base-model-path "meta-llama/Llama-2-7b-hf" \
    --delta-path "extracted_lora/13b-to-7b-gsm-delta" \
    --lora-path "trained_lora/7b-gsm-with-13b" \
    --target-model-path "merged_lora/7b-gsm-with-13b"
