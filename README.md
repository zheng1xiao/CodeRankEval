# CodeRankEval: Benchmarking and Analyzing LLM Performance for Code Ranking

This project provides the benchmark dataset introduced in the paper "CodeRankEval: Benchmarking and Analyzing LLM Performance for Code Ranking". It aims to evaluate and analyze the performance of large language models (LLMs) on code ranking tasks.

## Directory Structure

- `CodeRankEval/`: The original (base) version of the benchmark dataset, including:
  - `pointwise_data.json`
  - `pairwise_data.json`
  - `listwise_data.json`
- `CodeRankEval-Perturbed/`: The perturbed version of the benchmark dataset, including:
  - `pointwise_data_perturb.json`
  - `pairwise_data_perturb.json`
  - `listwise_data_perturb.json`

## Dataset Description

- **Base Version**: The original code ranking evaluation data, used to assess model performance under standard conditions.
- **Perturbed Version**: The base dataset with specific perturbations introduced, used to test model robustness against data distribution changes or noise.

## Citation

If you use this benchmark, please cite the following paper:

> CodeRankEval: Benchmarking and Analyzing LLM Performance for Code Ranking

For questions or suggestions, please contact the authors.
