import os
from tqdm import tqdm
from utils import *
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from prompt import *


def match_score_rate(data):
    count = len(data)
    invalid = 0
    matchScore = 0
    for item in data:
        if item['eval_score'] is None:
            invalid += 1
        elif item['eval_score'] == item['score']:
            matchScore += 1
    return matchScore/count*100, invalid/count*100


def evaluate_pointwise_code(problem, generated_code, api, params):
    if generated_code is None:
        return None

    prompt = (
        f"{IN_CONTEXT_EXAMPLE}\n\n"
        f"Problem Statement:\n{problem}\n\n"
        f"Generated Code:\n```python\n{generated_code}```\n\n"
        "Thoroughly evaluate the above code by examining its correctness, efficiency, and compliance with the problem requirements, ensuring that your analysis and final score are strictly based on the provided Scoring Criteria."
        "Provide a detailed analysis and conclude with the final score in the exact format: **Score: x points** (where x is an integer from 1 to 5)."
    )
    
    return api.generate(POINTWISE_SYSTEM_PROMPT, prompt, params["temperature"], params["max_tokens"])


def evaluate_code(filename, api, params, output_file, max_workers):
    data = read_json(filename)
    problems = [item['question'] for item in data]
    generated_codes = [item['pure_code'] for item in data]

    evaluate_func = partial(evaluate_pointwise_code, api=api, params=params)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        evaluations = list(tqdm(
            executor.map(evaluate_func, problems, generated_codes),
            total=len(problems),
            desc="Evaluating code"
        ))

    for item, evaluation in zip(data, evaluations):
        item["evaluation"] = evaluation
        if evaluation is not None: 
            item["eval_score"] = extract_score(evaluation)
        else:
            item["eval_score"] = None
    
    matchScore, invalid = match_score_rate(data)
    print(f"准确率: {matchScore:.2f}% 无效率: {invalid:.2f}%")

    write_json(data, output_file)
    print(f"评估结果保存到 {output_file}")


def pointwise_main(args):

    api = get_api(args)
    
    model_path = args.local_model if args.local_model else args.api_model

    output_file = args.output_file_dir+os.path.basename(model_path)+"_poinwise.json"

    params = {}
    params["model"] = args.local_model if args.local_model else args.api_model
    params["temperature"] = args.temperature
    params["max_tokens"] = args.max_tokens

    print(f"---------------开始对{os.path.basename(model_path)}评估pointwise---------------")
    evaluate_code(args.data_path, api, params, output_file, args.max_workers)
