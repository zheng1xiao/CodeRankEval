# CodeRankEval: Benchmarking and Analyzing LLM Performance for Code Ranking

本项目为论文《CodeRankEval: Benchmarking and Analyzing LLM Performance for Code Ranking》所提出的基准测试集，旨在评估和分析大语言模型（LLM）在代码排序任务中的表现。

## 目录结构

- `CodeRankEval/`：基础版本的benchmark数据集，包含：
  - `pointwise_data.json`
  - `pairwise_data.json`
  - `listwise_data.json`
- `CodeRankEval-Perturbed/`：引入扰动后的benchmark数据集，包含：
  - `pointwise_data_perturb.json`
  - `pairwise_data_perturb.json`
  - `listwise_data_perturb.json`

## 数据说明

- **基础版本**：原始的代码排序评测数据，用于评估模型在标准场景下的性能。
- **扰动版本**：在基础数据集上引入了特定扰动，用于测试模型在面对数据分布变化或噪声时的鲁棒性。

## 参考文献

如需引用本基准集，请参考原论文：

> CodeRankEval: Benchmarking and Analyzing LLM Performance for Code Ranking

如有问题或建议，欢迎联系作者。
