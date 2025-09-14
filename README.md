Please refer to the script files for detailed configuration and parameter options.
# CodeRankEval: Benchmarking and Analyzing LLM Performance for Code Ranking

This project provides the benchmark dataset introduced in the paper "CodeRankEval: Benchmarking and Analyzing LLM Performance for Code Ranking". It aims to evaluate and analyze the performance of large language models (LLMs) on code ranking tasks.


## Directory Structure

- `CodeRankEval/`: The original (base) version of the benchmark dataset, including:
- `CodeRankEval-Perturbed/`: The perturbed version of the benchmark dataset, including:
- `eva/`: Evaluation framework and scripts
  - `evaluate.py`, `evaluate_pointwise.py`, `evaluate_pairwise.py`, `evaluate_listwise.py`: Main evaluation scripts

## Dataset Description

- **Base Version**: The original code ranking evaluation data, used to assess model performance under standard conditions.
- **Perturbed Version**: The base dataset with specific perturbations introduced, used to test model robustness against data distribution changes or noise.


## Usage

```bash
# For API-based evaluation
sh eva/scripts/api_evaluate.sh

# For local model evaluation
sh eva/scripts/local_evaluate.sh
```

### Parameter Replacement

You may need to modify certain parameters in the scripts according to your environment and requirements, such as:

- Model name or API key
- Method(pointwise/pairwise/listwise)
- Dataset path

Open the corresponding `.sh` script and update the variables or arguments as needed before running. Please refer to the comments in each script for details.

## Citation

If you use this benchmark, please cite the following paper:

> CodeRankEval: Benchmarking and Analyzing LLM Performance for Code Ranking

For questions or suggestions, please contact the authors.
