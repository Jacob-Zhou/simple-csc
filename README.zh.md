# Simple CSC

[![License: Apache](https://img.shields.io/badge/License-Apache-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

[English](README.md) | [中文](README.zh.md)

一键让中文大模型化身中文拼写纠错模型!!!

本仓库提供了论文 [A Simple yet Effective Training-free Prompt-free Approach to Chinese Spelling Correction Based on Large Language Models](https://arxiv.org/abs/2410.04027) 的实现。

__新闻__
- 2024/12/09: 我们荣获[金山办公2024算法挑战赛-中文文本智能校对大赛](https://datastudio.wps.cn/matchcenter/competition/1/introduction)冠军。在比赛中，我们以本代码库作为核心模块，取得了 83.37 的 $F_{0.5}$ 分数，比第二名高 2.02 分。
- 2024/09/20: 我们的论文被 EMNLP 2024 主会接收。

## 目录

- [环境要求](#环境要求)
- [安装](#安装)
- [使用方法](#使用方法)
  - [模型准备](#模型准备)
  - [Python API](#python-api)
  - [RESTful API 服务器和调用](#restful-api-服务器和调用)
  - [演示应用](#演示应用)
  - [运行实验](#运行实验)
    - [数据准备](#数据准备)
- [支持的模型](#支持的模型)
- [贡献](#贡献)
- [许可证](#许可证)

## 环境要求
- torch>=2.0.1
- transformers>=4.27.0
- xformers==0.0.21
- accelerate
- bitsandbytes
- sentencepiece
- pypinyin
- pypinyin-dict
- opencc-python-reimplemented
- modelscope *(可选，用于从 modelscope 下载模型)*
- streamlit *(可选，用于演示应用)*
- uvicorn *(可选，用于 RESTful API 服务器)*
- fastapi *(可选，用于 RESTful API 服务器)*
- loguru *(可选，用于 RESTful API 服务器)*
- sse_starlette *(可选，用于 RESTful API 服务器)*

## 安装

您可以通过运行以下命令来配置环境：

```bash
bash scripts/set_enviroment.sh
```

这将自动创建虚拟环境并安装所需的包。

为获得更好的性能，您可以安装 flash-attn：

```bash
pip install flash-attn --no-build-isolation
```

<!-- Add a warning about Qwen2.5 -->

> [!WARNING] 
> 一位用户报告说，使用 Qwen2.5 系列模型时，如果没有安装 flash-attn，代码会产生意料之外的行为。
> 
> 因此如在本代码库中使用 Qwen2.5 系列模型，请务必安装 flash-attn。
> 
> 同时我们正在努力解决这个问题，使未安装 flash-attn 的用户也能正常使用 Qwen2.5 系列模型。

## 使用方法

### 模型准备
如果在本地缓存中未找到模型，代码将自动从 Huggingface 模型仓库下载模型。

### Python API
我们提供了一个简单的 Python API 用于纠错：

```python
from lmcsc import LMCorrector

corrector = LMCorrector(
    model="Qwen/Qwen2.5-0.5B",
    config_path="configs/default_config.yaml",
)

outputs = corrector("完善农产品上行发展机智。")
print(outputs)
# [('完善农产品上行发展机制。',)]
```

也支持流式模式：

```python
outputs = corrector("完善农产品上行发展机智。", stream=True)
for output in outputs:
    print(output[0][0], end="\r", flush=True)
print()
```

### RESTful API 服务器和调用
我们还提供了纠错器的 RESTful API 服务器。

```bash
python api_server.py  \
    --model "Qwen/Qwen2.5-0.5B"  \
    --host 127.0.0.1  \
    --port 8000  \
    --workers 1
```

您可以使用 `curl` 来测试 RESTful API 服务器。

```bash
# 默认模式
curl -X POST 'http://127.0.0.1:8000/correction' -H 'Content-Type: application/json' -d '{"input": "完善农产品上行发展机智。"}'
# > {"id":"","object":"correction","choices":[{"index":0,"message":{"content":"完善农产品上行发展机制。"}}],"created":1727058762}

# 流式模式
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

# 带上下文的纠错
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

### 演示应用
我们为我们的方法提供了一个演示应用。要运行演示：

1. 确保您已安装 `streamlit` 包。
2. 运行以下命令：

```bash
streamlit run demo.py
```

默认情况下，演示使用 `Qwen/Qwen2.5-0.5B`，可以在具有 32GB 内存的 V100 GPU 上运行。您可以在演示的侧边栏中或通过修改 `configs/demo_app_config.yaml` 中的 `default_model` 来更换其他模型。

侧边栏还允许您调整 `n_beam`、`alpha` 和 `use_faithfulness_reward` 参数。

侧边栏中提供了几个示例，包括一个包含 1866 个字的长句。

![演示截图](https://github.com/Jacob-Zhou/llm-csc/assets/13799122/a0f7545d-424e-4b73-b147-f7930b0a7a4a)

### 运行实验
可以通过以下命令运行论文中提到的数据集的实验：

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

#### 数据准备
运行之前，您需要将每个句子对预处理成以下格式：
```
[src]	[tgt]
[src]	[tgt]
[src]	[tgt]
```
其中 `[src]` 和 `[tgt]` 分别是源句子和目标句子。
使用 `\t` 分隔它们。

数据准备过程可以在 `scripts/download_datasets.sh` 中找到。
该脚本将从原始来源（托管在 `raw.githubusercontent.com` 和 `Google Drive` 上）下载数据集，并将它们预处理成所需格式。

## 支持的模型
- GPT2
- Baichuan2
- Qwen1.5
- Qwen2
- Qwen2.5
- InternLM2

## 未来计划
- [ ] 允许字的插入和删除（几乎完成）。
- [ ] Top-k 投票以获得更好的性能。
- [ ] 将代码打包成库。
- [ ] 加快推理过程。
- [ ] 重构代码以兼容 vLLM（长期计划）。

## 贡献
欢迎贡献！请随时提交 Pull Request。

## 许可证
该项目采用 Apache 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 引用
如果您觉得这个工作有用，请引用：

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
