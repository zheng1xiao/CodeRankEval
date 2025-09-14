import re
import json
from prompt import *
from vllmAPI import vllmAPI

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data


def write_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def extract_score(evaluation):
    score_match = re.search(r"Score: (\d+) points?", evaluation)
    return int(score_match.group(1)) if score_match else None


def extract_rank_list(evaluation):
    rank_match = re.search(r"Ranked Order:\s*\[(.*?)\]", evaluation)
    if rank_match:
        cleaned_str = ','.join(filter(None, rank_match.group(1).replace(' ', '').split(',')))
        if cleaned_str and all(part.isdigit() for part in cleaned_str.split(',')):
            return list(map(int, cleaned_str.split(',')))
    return None


def extract_code(generated_code):
    if not isinstance(generated_code, (str, bytes)):
        generated_code = str(generated_code)

    patterns = [
        r'```python\n(.*?)```',
        r'```python\r\n(.*?)```',
        r'```\n(.*?)```',
        r'\[PYTHON\]\n(.*?)\[/PYTHON\]'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, generated_code, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    return None


def get_api(args):
    if args.local_model is not None:
        api = vllmAPI(
            model=args.local_model,
            api_base="http://localhost:5000/v1"
        )
    else:
        api = vllmAPI(
            model=args.api_model,
            api_base=args.api_base,
            api_key=args.api_key
        )
    return api


def evaluate_list_code(listwise_data, api):

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
    
    return api.generate(LISTWISE_SYSTEM_PROMPT, prompt)


def evaluate_pointwise_code(problem, generated_code, api):
    if generated_code is None:
        return None

    prompt = (
        f"{IN_CONTEXT_EXAMPLE}\n\n"
        f"Problem Statement:\n{problem}\n\n"
        f"Generated Code:\n```python\n{generated_code}```\n\n"
        "Thoroughly evaluate the above code by examining its correctness, efficiency, and compliance with the problem requirements, ensuring that your analysis and final score are strictly based on the provided Scoring Criteria."
        "Provide a detailed analysis and conclude with the final score in the exact format: **Score: x points** (where x is an integer from 1 to 5)."
    )
    
    return api.generate(POINTWISE_SYSTEM_PROMPT, prompt)
