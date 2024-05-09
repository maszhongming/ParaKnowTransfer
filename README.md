# Seeking Neural Nuggets: Knowledge Transfer in LLMs from a Parametric Perspective (ICLR 2024)
<p align="center">
  <a href="https://arxiv.org/abs/2310.11451"><img src="https://img.shields.io/badge/📝-Paper-blue" height="25"></a>
  <a href="https://maszhongming.github.io/ParaKnowTransfer/"><img src="https://img.shields.io/badge/🌐-Website-red" height="25"></a>
</p>

🖋 **Authors:** [Ming Zhong](https://maszhongming.github.io/), [Chenxin An](https://scholar.google.com.hk/citations?user=fY69CxIAAAAJ&hl=en), [Weizhu Chen](https://www.microsoft.com/en-us/research/people/wzchen/), [Jiawei Han](https://hanj.cs.illinois.edu/), [Pengcheng He](https://scholar.google.com/citations?user=TS1RoxAAAAAJ&hl=en)

## 📜 Overview

Large Language Models (LLMs) inherently encode extensive knowledge within their parameters. Previous studies have demonstrated that this parametric knowledge can be **detected** (e.g., via cloze tests) or **modified** (e.g., through knowledge editing).

Taking this further, *can task-specific parametric knowledge be **transferred** across LLMs of different scales?*

Absolutely! Our paper provides empirical evidence supporting the transferability of parametric knowledge.

## 🚀 Setting Up the Environment
To begin, set up your environment with the necessary packages:
```bash
conda create --name paratransfer python=3.10
conda activate paratransfer
pip install -r requirements.txt
```

## 🔄 Parametric Knowledge Transfer

### Knowledge Extraction
We start by extracting task-specific parametric knowledge from the larger teacher model into the LoRA module for the smaller student model. Using Llama-2 13B as the teacher and Llama-2 7B as the student for the GSM task:

```bash
python extract_lora_with_sensitivity.py \
    --model_size 13b \
    --lora_size 7b \
    --task gsm

python get_delta.py \
  --path extracted_lora/13b-to-7b-gsm
```

Modify the settings in `extracted_lora.sh` as needed.

### Knowledge Injection

Next, we use the extracted parameters to initialize the LoRA module in the student model and fine-tune it:

```bash
./train.sh
```

The models will be saved in the `trained_lora` folder.

### Evaluation

Merge the LoRA module with the base model for evaluation:

```bash
./merge.sh
```

Subsequently, employ [Open-Instruct](https://github.com/allenai/open-instruct) to evaluate the model across various benchmarks.

## 📚 Citation
If you find this work useful, please consider citing our paper:
```
@inproceedings{zhong2023seeking,
  title={Seeking Neural Nuggets: Knowledge Transfer in Large Language Models from a Parametric Perspective},
  author={Zhong, Ming and An, Chenxin and Chen, Weizhu and Han, Jiawei and He, Pengcheng},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
