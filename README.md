# ccqa_official

### CCQA: Generating Question from Solution Can Improve Inference-Time Reasoning in SLMs (EMNLP 2025)
<img width="2152" height="1090" alt="image" src="https://github.com/user-attachments/assets/d0dca0b6-9165-476c-b2ba-45cb30c8bfab" />

<div align="center">

# 🔄 CCQA: Cycle-Consistency in Question Answering

[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue)](https://2025.emnlp.org/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2501.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2501.xxxxx)

**Official implementation of "CCQA: Generating Question from Solution Can Improve Inference-Time Reasoning in SLMs"**

[Paper](https://arxiv.org/abs/2509.18536) | [Code](https://github.com/scai-research/ccqa_official) | [Models](upload soon)

</div>

---

## Overview

CCQA is a novel inference-time reasoning method designed specifically for **Small Language Models (SLMs)**. Inspired by cycle consistency, CCQA generates questions from candidate solutions and selects the most reliable answer by measuring similarity to the original question.

### Key Features

- **Designed for SLMs**: Addresses limitations of existing reasoning methods for smaller models (135M-3B parameters)
- **Cycle Consistency**: Evaluates reasoning quality by regenerating questions from solutions
- **Efficient**: Uses lightweight Flan-T5 model for backward question generation

