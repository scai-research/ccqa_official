# 🔄 CCQA: Generating Question from Solution Can Improve Inference-Time Reasoning in SLMs

[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue)](https://2025.emnlp.org/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ACL Anthology](https://img.shields.io/badge/ACL%20Anthology-EMNLP%202025-blue)](https://aclanthology.org/2025.emnlp-main.704/)

<img width="2152" height="1090" alt="image" src="https://github.com/user-attachments/assets/b510410f-36fb-47d3-9252-cbf60a670a7d"  />

<div align="center">

  
**Official implementation of "CCQA: Generating Question from Solution Can Improve Inference-Time Reasoning in SLMs" 
[EMNLP 2025 Main]**

</div>


## Overview

CCQA is a novel inference-time reasoning method designed specifically for **Small Language Models (SLMs)**. Inspired by cycle consistency, CCQA generates questions from candidate solutions and selects the most reliable answer by measuring similarity to the original question.

### Key Features

- **Designed for SLMs**: Addresses limitations of existing reasoning methods for smaller models (135M-3B parameters)
- **Cycle Consistency**: Evaluates reasoning quality by regenerating questions from solutions
- **Efficient**: Uses lightweight Flan-T5 model for backward question generation

## 🛠️ Supported Models & Benchmarks

### Base Models
- **Llama3.2**: 1B, 3B
- **Qwen2.5**: 0.5B, 1.5B, 3B
- **SmolLM2**: 135M, 360M
- **Falcon**: 1B

### Question Generator
- **Flan-T5-base** (258M)

### Benchmarks
| Type | Benchmarks |
|------|-----------|
| **Arithmetic** | GSM8K, SVAMP, MultiArith |
| **Commonsense** | CommonSenseQA, StrategyQA, ARC-Challenge |

### Citation
If you find this work useful, please cite:

```bibtex
@inproceedings{kim-yoon-2025-ccqa,
  title = "{CCQA}: Generating Question from Solution Can Improve Inference-Time Reasoning in {SLM}s",
  author = "Kim, Jinyoung  and Yoon, Ji Won",
  booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
  month = nov,
  year = "2025",
  address = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2025.emnlp-main.704/",
  doi = "10.18653/v1/2025.emnlp-main.704",
  pages = "13944--13956"
}

