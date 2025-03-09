# Simple CSC

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

[English](README.md) | [中文](README.zh.md)

This repository provides an implementation of following papers:
- (**[v2.0.0](https://github.com/Jacob-Zhou/simple-csc/tree/v2.0.0)**) [A Training-free LLM-based Approach to General Chinese Character Error Correction](https://arxiv.org/abs/2502.15266).
- (**[v1.0.0](https://github.com/Jacob-Zhou/simple-csc/tree/v1.0.0)**) [A Simple yet Effective Training-free Prompt-free Approach to Chinese Spelling Correction Based on Large Language Models](https://arxiv.org/abs/2410.04027).

__News__
- 2025/02/21: New version (v2.0.0) released.
  - Now we support insert and delete character operations.
  - Add prompted model, which can be used to improve the performance of the corrector.
- 2024/12/09: We **won 1st place** in the [Kingsoft Office 2024 Algorithm Challenge: Chinese Text Correction Competition (金山办公2024算法挑战赛-中文文本智能校对大赛)](https://datastudio.wps.cn/matchcenter/competition/1/introduction), with this codebase serving as a key module of our solution. Notably, our solution achieved an $F_{0.5}$ score that was **2.02** points higher than the second-place team.
- 2024/09/20: Our paper is accepted by **EMNLP 2024 main** conference.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Model Preparation](#model-preparation)
  - [Python API](#python-api)
  - [RESTful API Server & API Call](#restful-api-server--api-call)
  - [Demo App](#demo-app)
  - [Run Experiments](#run-experiments)
    - [Data Preparation](#data-preparation)
- [Supported Models](#supported-models)
- [Contributing](#contributing)
- [License](#license)

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
- modelscope *(optional, for model download from modelscope)*
- streamlit *(optional, for demo app)*
- uvicorn *(optional, for RESTful API server)*
- fastapi *(optional, for RESTful API server)*
- loguru *(optional, for RESTful API server)*
- sse_starlette *(optional, for RESTful API server)*

## Installation

You can set up the environment by running:

```bash
bash scripts/set_environment.sh
```

This will automatically create a virtual environment and install the required packages.

For better performance, you can install flash-attn:

```bash
pip install flash-attn --no-build-isolation
```

<!-- Add a warning about Qwen2.5 -->

> [!WARNING] 
> Reported by a user, using Qwen2 or Qwen2.5 family models without flash-attn will lead unexpected errors. **Specifically, the corrector will stuck in the beam search process.**
> 
> Please install flash-attn to avoid this issue. Or you can set `torch_dtype=torch.bfloat16` in the `LMCorrector` class to avoid this issue.
>
> Though we strongly recommend using flash-attn, which will significantly reduce the memory usage and speed up the inference process.

## Usage

### Using Specific Version

We have tagged two versions of the code:
- v1.0.0: The first version of the code, which is the repository for the paper [A Simple yet Effective Training-free Prompt-free Approach to Chinese Spelling Correction Based on Large Language Models](https://arxiv.org/abs/2410.04027).
- v2.0.0: The second version of the code, which is the repository for the paper [A Training-free LLM-based Approach to General Chinese Character Error Correction](https://arxiv.org/abs/2502.15266).

The repository will be under continuous development.
If you want to use a specific version, you can use the following command to switch to the specific version:

```bash
git checkout v1.0.0
```

or

```bash
git checkout v2.0.0
```

### Model Preparation
The code will automatically download the model from the Huggingface model hub, if the model is not found in the local cache.

> [!TIP]
> In our experiments, we found that the `Base` versions of LLMs perform better than their `Instruct` or `Chat` counterparts for this task.
>
> **Recommended models:**
> - For `v1.0.0`: `baichuan-inc/Baichuan2-13B-Base` provides the best performance.
> - For `v2.0.0`: `Qwen/Qwen2.5-[7/14]B` is recommended for good balance between performance and speed.


### Python API
We provide a simple Python API for the corrector:

```python
from lmcsc import LMCorrector
import torch

corrector = LMCorrector(
    model="Qwen/Qwen2.5-1.5B",
    prompted_model="Qwen/Qwen2.5-1.5B", # Suggested to use the same model for model and prompted_model. In this way, we only need to load the model once.
    config_path="configs/c2ec_config.yaml", # You can always use the default config file to disable the insert and delete operations.
    torch_dtype=torch.bfloat16, # the default torch_dtype is torch.float16, but it will lead unexpected errors when using Qwen2 or Qwen2.5 family models without flash-attn.
)

outputs = corrector("完善农产品上行发展机智。")
print(outputs)
# [('完善农产品上行发展机制。',)]
```

Stream mode is also available:

```python
outputs = corrector("完善农产品上行发展机智。", stream=True)
for output in outputs:
    print(output[0][0], end="\r", flush=True)
print()
```

### RESTful API Server & API Call
We also provide the RESTful API server for the corrector.

```bash
# Suggested to use the same model for model and prompted_model. In this way, we only need to load the model once.
# You can always use the default config file to disable the insert and delete operations.
# use bf16 to avoid unexpected errors when using Qwen2 or Qwen2.5 family models without flash-attn.
python api_server.py  \
    --model "Qwen/Qwen2.5-1.5B"  \
    --prompted_model "Qwen/Qwen2.5-1.5B"  \
    --config_path "configs/c2ec_config.yaml"  \
    --host 127.0.0.1  \
    --port 8000  \
    --workers 1  \
    --bf16
```

You can use `curl` to test the RESTful API server.

```bash
# Default
curl -X POST 'http://127.0.0.1:8000/correction' -H 'Content-Type: application/json' -d '{"input": "完善农产品上行发展机智。"}'
# > {"id":"","object":"correction","choices":[{"index":0,"message":{"content":"完善农产品上行发展机制。"}}],"created":1727058762}

# Stream
curl -X POST 'http://127.0.0.1:8000/correction' -H 'Content-Type: application/json' -d '{"input": "完善农产品上行发展机智。", "stream": "True"}'
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善农产品"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善农产品上"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善农产品上行"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善农产品上行发展"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善农产品上行发展机制"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善农产品上行发展模式。"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善农产品上行发展机制。"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善农产品上行发展机制。"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善农产品上行发展机制。"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"完善农产品上行发展机制。"},"index":0}],"created":1727058762}
# > data: [DONE]

# Correction with contexts
curl -X POST 'http://127.0.0.1:8000/correction' -H 'Content-Type: application/json' -d '{"input": "未挨前兆", "contexts": "患者提问：", "stream": "True"}'
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"未"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"未挨"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"胃癌前"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"胃癌前兆"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"胃癌前兆"},"index":0}],"created":1727058762}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"胃癌前兆"},"index":0}],"created":1727058763}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"胃癌前兆"},"index":0}],"created":1727058763}
# > data: {"id":"","object":"correction.chunk","choices":[{"delta":{"content":"胃癌前兆"},"index":0}],"created":1727058763}
# > data: [DONE]
```

### Demo App
We provide a demo application for our approach. To run the demo:

1. Ensure you have installed the `streamlit` package.
2. Run the following command:

```bash
streamlit run demo.py
```

By default, the demo uses `Qwen/Qwen2.5-0.5B`, which can run on a V100 GPU with 32GB memory. You can change to other models in the demo's sidebar or by modifying the `default_model` in `configs/demo_app_config.yaml`.

The corrector config of the demo is `configs/demo_config.yaml`, which is the same as the `c2ec_config.yaml` in `v2.0.0`.
You can also modify `config_path` in `configs/demo_app_config.yaml` to use other configs.

The sidebar also allows you to adjust `n_beam`, `alpha`, and `use_faithfulness_reward` parameters.

Several examples are provided in the sidebar, including a long sentence with 1866 characters.

![Demo Screenshot](https://github.com/Jacob-Zhou/llm-csc/assets/13799122/a0f7545d-424e-4b73-b147-f7930b0a7a4a)

### Run Experiments
The experiments on the datasets mentioned in the paper can be run by the following command:

```bash
python -u run.py  \
    --input-file <input-file>  \
    --path <path>  \
    --model-name <model-name>  \
    --prompted-model-name <prompted-model-name>  \
    --config-path <config-path>  \
    --n-observed-chars <n-observed-chars>  \
    --n-beam <n-beam>  \
    --batch-size <batch-size>  \
    --alpha <alpha>  \
    --use-faithfulness-reward
```

#### Data Preparation
Before running, you are required to preprocess each sentence pair into the format of
```
[src]	[tgt]
[src]	[tgt]
[src]	[tgt]
```
Where `[src]` and `[tgt]` are the source and target sentences, respectively.
A `\t` is used to separate them.

The process of the data preparation can be found in the `scripts/download_datasets.sh`.
This script will download the datasets from the original sources, which are hosted on `raw.githubusercontent.com` and `Google Drive`, and preprocess them into the required format.

##### Instruction of building the C2EC dataset
We build the C2EC dataset to test the performance our approach in handling missing and redundant characters.
You can build the C2EC dataset by following the steps below.

- Download the CCTC v1.1 dataset from [CTCResource](https://github.com/destwang/CTCResources?tab=readme-ov-file#datasets).
  - Place the `cctc_v1.1.zip` (MD5: `ecd94ad85c33d7c0ace11b6da316f81e`) file in the `data/c2ec` directory. You don't need to unzip it.
  - After placing the zip file, the directory structure should be like this:
    ```
    datasets/
      c2ec/
        metadata/
          dev.index
          test.index
        cctc_v1.1.zip
    ```
- Run the following command to build the C2EC dataset. The Lemon dataset will automatically be downloaded from [Lemon](https://github.com/gingasan/lemon).
  ```bash
  bash scripts/build_c2ec_dataset.sh
  ```

- Voilà, you have the C2EC dataset!


###### Citation for Original Datasets
> [!NOTE]
> The C2EC dataset building by combining the CCTC dataset and the Lemon dataset.
> If you use the C2EC dataset, please also cite the original papers 😀.

- CCTC Dataset
  ```bibtex
  @inproceedings{wang-etal-2022-cctc,
      title = "{CCTC}: A Cross-Sentence {C}hinese Text Correction Dataset for Native Speakers",
      author = "Wang, Baoxin  and
        Duan, Xingyi  and
        Wu, Dayong  and
        Che, Wanxiang  and
        Chen, Zhigang  and
        Hu, Guoping",
      editor = "Calzolari, Nicoletta  and
        Huang, Chu-Ren  and
        Kim, Hansaem  and
        Pustejovsky, James  and
        Wanner, Leo  and
        Choi, Key-Sun  and
        Ryu, Pum-Mo  and
        Chen, Hsin-Hsi  and
        Donatelli, Lucia  and
        Ji, Heng  and
        Kurohashi, Sadao  and
        Paggio, Patrizia  and
        Xue, Nianwen  and
        Kim, Seokhwan  and
        Hahm, Younggyun  and
        He, Zhong  and
        Lee, Tony Kyungil  and
        Santus, Enrico  and
        Bond, Francis  and
        Na, Seung-Hoon",
      booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
      month = oct,
      year = "2022",
      address = "Gyeongju, Republic of Korea",
      publisher = "International Committee on Computational Linguistics",
      url = "https://aclanthology.org/2022.coling-1.294/",
      pages = "3331--3341"
  }
  ```

- Lemon Dataset
  ```bibtex
  @inproceedings{wu-etal-2023-rethinking,
      title = "Rethinking Masked Language Modeling for {C}hinese Spelling Correction",
      author = "Wu, Hongqiu  and
        Zhang, Shaohua  and
        Zhang, Yuchen  and
        Zhao, Hai",
      editor = "Rogers, Anna  and
        Boyd-Graber, Jordan  and
        Okazaki, Naoaki",
      booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
      month = jul,
      year = "2023",
      address = "Toronto, Canada",
      publisher = "Association for Computational Linguistics",
      url = "https://aclanthology.org/2023.acl-long.600/",
      doi = "10.18653/v1/2023.acl-long.600",
      pages = "10743--10756"
  }
  ```

## Supported Models
- GPT2
- Baichuan2
- Qwen1.5
- Qwen2
- Qwen2.5
- InternLM2

## Future Plans
- [x] Enable insert and delete operations.
- [ ] Speed up the inference process.
- [ ] Package the code into a library.
- [ ] Enable continuous batching.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Citation
If you find this work useful, please consider citing:

```bibtex
@inproceedings{zhou-etal-2024-simple,
    title = "A Simple yet Effective Training-free Prompt-free Approach to {C}hinese Spelling Correction Based on Large Language Models",
    author = "Zhou, Houquan  and
      Li, Zhenghua  and
      Zhang, Bo  and
      Li, Chen  and
      Lai, Shaopeng  and
      Zhang, Ji  and
      Huang, Fei  and
      Zhang, Min",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.966",
    pages = "17446--17467",
    abstract = "This work proposes a simple training-free prompt-free approach to leverage large language models (LLMs) for the Chinese spelling correction (CSC) task, which is totally different from all previous CSC approaches. The key idea is to use an LLM as a pure language model in a conventional manner. The LLM goes through the input sentence from the beginning, and at each inference step, produces a distribution over its vocabulary for deciding the next token, given a partial sentence. To ensure that the output sentence remains faithful to the input sentence, we design a minimal distortion model that utilizes pronunciation or shape similarities between the original and replaced characters. Furthermore, we propose two useful reward strategies to address practical challenges specific to the CSC task. Experiments on five public datasets demonstrate that our approach significantly improves LLM performance, enabling them to compete with state-of-the-art domain-general CSC models.",
}
```
