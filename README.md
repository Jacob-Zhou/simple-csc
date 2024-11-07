# Simple CSC

This repository provides a implementation of paper [A Simple yet Effective Training-free Prompt-free Approach to Chinese Spelling Correction Based on Large Language Models](xxx).

## Requirements
- torch>=2.0.1
- transformers>=4.27.0
- xformers==0.0.21
- accelerate
- bitsandbytes
- sentencepiece
- pypinyin
- pypinyin-dict
- opencc-python-reimplemented
- modelscope (optional)
- streamlit (optional)

You can simply set up the environment by running:
```bash
bash scripts/set_enviroment.sh
```
It will automatically create a virtual environment and install the required packages.

You can install flash-attn for better performance by running:
```
pip install flash-attn --no-build-isolation
```

## Usage

### Command Line
```bash
python -u run.py  \
    --input-file <input-file>  \
    --path <path>  \
    --model-name <model-name>  \
    --n-observed-chars <n-observed-chars>  \
    --n-beam <n-beam>  \
    --batch-size <batch-size>  \
    --alpha <alpha>  \
    --use-faithfulness-reward
```

### Data Preparation
Before running, you are required to preprocess each sentence pair into the format of
```txt
[src]	[tgt]
[src]	[tgt]
[src]	[tgt]
```
Where `[src]` and `[tgt]` are the source and target sentences, respectively.
A `\t` is used to separate them.

The process of the data preparation can be found in the `scripts/download_datasets.sh`.
It will automatically excute during the first run of the `scripts/set_enviroment.sh`.

The data are downloaded from the original sources, which are hosted on `raw.githubusercontent.com` and `Google Drive`.

### Model Preparation
The code will automatically download the model from the Huggingface model hub, if the model is not found in the local cache.

### Demo App

We also provide a demo for our approach. You can run the demo by
```bash
streamlit run demo.py
```

before running the demo, make sure you have installed the `streamlit` package.

By default, the demo uses the `Qwen/Qwen1.5-0.5B`, a small model can be run on a V100 GPU with 32GB memory.
You can change to other models by selecting the model name in the sidebar of the demo, during the running of the demo.
Directly modify the `default_model` in the `demo_config.py` to change the default model is a better way, if you don't want to use `Qwen/Qwen1.5-0.5B` at all.

You can also change the `n_beam`, `alpha`, and `use_faithfulness_reward` in the sidebar.

Couple of examples can also be found in the sidebar, including a long sentence example with 1866 characters.

__Here is a screenshot of the demo:__

https://github.com/Jacob-Zhou/llm-csc/assets/13799122/a0f7545d-424e-4b73-b147-f7930b0a7a4a

## Supported Models

- `GPT2`
- `Baichuan2`
- `Qwen1.5`
- `Qwen2`
- `InternLM2`
