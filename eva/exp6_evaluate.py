import os
from tqdm import tqdm
from utils import *
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import Levenshtein
from sklearn import metrics
import numpy as np
from scipy.stats import pearsonr
from prompt import *
import argparse

def get_metrics(pred, ground_truth, solutions):
    gt_str = ''.join(str(x) for x in ground_truth)
    pred_str = ''.join(str(x) for x in pred)
    distance = Levenshtein.distance(gt_str, pred_str)

    eval_scores = np.array([sol['eval_score'] for sol in solutions])
    pred_scores = np.array([len(pred) - pred.index(i) for i in range(1, len(pred) + 1)])
    ndcg = metrics.ndcg_score([eval_scores], [pred_scores], k=len(pred))

    pearson_corr, _ = pearsonr(ground_truth, pred)
    return distance, ndcg, pearson_corr


def evaluate_predictions(data):
    valid_samples = 0
    invalid_samples = 0
    total_distance = 0
    total_ndcg = 0
    total_pearson = 0
    true_samples = 0
    
    for item in data:
        ground_truth = item['rank_list']
        pred = item.get('eval_list')
        
        if pred is None or len(pred) != len(ground_truth):
            invalid_samples += 1
            continue

        n = len(pred)
        if not set(pred) == set(range(1, n + 1)):
            invalid_samples += 1
            continue
        
        if pred == ground_truth:
            true_samples += 1
        
        distance, ndcg, pearson_corr = get_metrics(pred, ground_truth, item['solutions'])
        total_distance += distance
        total_ndcg += ndcg
        total_pearson += pearson_corr if not np.isnan(pearson_corr) else 0
        
        valid_samples += 1
    
    print_metrics(total_distance, total_ndcg, total_pearson, valid_samples, invalid_samples, true_samples)

def print_metrics(total_distance, total_ndcg, total_pearson, valid_samples, invalid_samples, true_samples):
    total_samples = valid_samples + invalid_samples
    valid_percentage = (valid_samples / total_samples * 100) if total_samples > 0 else 0
    invalid_percentage = (invalid_samples / total_samples * 100) if total_samples > 0 else 0
    true_percentage = (true_samples / total_samples * 100) if total_samples > 0 else 0

    print(f"总样本数: {total_samples}")
    print(f"有效样本比例: {valid_percentage:.2f}%")
    print(f"无效样本比例: {invalid_percentage:.2f}%")
    print(f"正确样本比例: {true_percentage:.2f}%")
    
    if valid_samples > 0:
        avg_distance = total_distance / valid_samples
        avg_ndcg = total_ndcg / valid_samples
        avg_pearson = total_pearson / valid_samples
        print(f"有效样本平均编辑距离: {avg_distance:.4f}")
        print(f"有效样本平均NDCG: {avg_ndcg:.4f}")
        print(f"有效样本平均Pearson相关系数: {avg_pearson:.4f}")
    else:
        print("全部样本都无效")


def evaluate_list_code(listwise_data, api, params):

    problem = listwise_data['question']
    generated_code_list = [item['pure_code'] for item in listwise_data['solutions']]

    prompt = (
        f"{IN_CONTEXT_EXAMPLE_LISTWISE}\n\n"
        f"Problem Statement:\n{problem}\n\n"
        f"Here are multiple generated solutions for the above problem:\n"
        + "\n".join(
            [f"Solution {i+1}:\n```python\n{code}```\n" for i, code in enumerate(generated_code_list)]
        )
        + "\nCarefully evaluate all the solutions based on correctness, efficiency, readability, and compliance with best practices. "
        "Provide a **ranked list** from best to worst, ensuring that your ranking strictly follows these criteria. "
        "Conclude with your final ranked list in the exact format: **Ranked Order: [index1, index2, ..., indexN]**, "
        "where indexX represents the original position (1-based index) of the solution in the provided list."
    )
    
    return api.generate(LISTWISE_SYSTEM_PROMPT, prompt, params["temperature"], params["max_tokens"])

def evaluate_code(filename, api, params, max_workers):
    data = read_json(filename)

    evaluate_func = partial(evaluate_list_code, api=api, params=params)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        evaluations = list(tqdm(
            executor.map(evaluate_func, data),
            total=len(data),
            desc="Evaluating code"
        ))

    for item, evaluation in zip(data, evaluations):
        item["evalution"] = evaluation
        item["eval_list"] = extract_rank_list(evaluation) if evaluation is not None else None

    evaluate_predictions(data)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some inputs for judgement.")

    # Model Configuration
    parser.add_argument("--api-model", type=str, help="Specify the language model to use.")
    parser.add_argument("--local-model", type=str, help="Specify the local model to use.")
    parser.add_argument("--temperature", type=float, default=0, help="Set the sampling temperature for the language model.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Define the maximum number of tokens to generate.")

    # File Paths
    parser.add_argument("--data-path", type=str, help="Data to be evaluated")

    # API Configuration
    parser.add_argument("--api-base", type=str, default="https://api2.aigcbest.top/v1", help="Set the base URL for the API.")
    parser.add_argument("--api-key", type=str, default="sk-M3bKoucx7o9TSEkK0eC5249f951644E69d880eC269Df061d", help="Provide the API key for authentication.")

    # Other
    parser.add_argument("--max-workers", type=int, default=32, help="Set the number of workers for multiple thread.")

    return parser.parse_args()
    

def main():

    args = parse_arguments()

    api = get_api(args)

    model_path = args.local_model if args.local_model else args.api_model

    params = {}
    params["model"] = args.local_model if args.local_model else args.api_model
    params["temperature"] = args.temperature
    params["max_tokens"] = args.max_tokens

    print(f"---------------开始对{os.path.basename(model_path)}评估listwise---------------")
    evaluate_code(args.data_path, api, params, args.max_workers)

if __name__ == "__main__":
    main()