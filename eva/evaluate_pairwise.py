import os
import re
from tqdm import tqdm
from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed


def write_states(output_file, win, tie, lose, total):
    final_stats = {
        "win": win,
        "tie": tie,
        "lose": lose,
        "total": total,
        "win_rate": win / total if total > 0 else 0,
        "tie_rate": tie / total if total > 0 else 0,
        "lose_rate": lose / total if total > 0 else 0,
    }

    states_path = output_file.replace(".json", "_states.json")
    write_json(final_stats, states_path)

def prepare_conversations(template_path, data_path):
    template = {
        t['name']: {
            'system_prompt': t.get('system_prompt', ''),
            'prompt_template': t.get('prompt_template', '')
            } for t in read_jsonl(template_path)
        }
    
    data = read_json(data_path)

    for item in data: 
        item["messages"] = [
            {"role": "system", "content": template["pairwise"]["system_prompt"]},
            {"role": "user", "content": template["pairwise"]["prompt_template"].format(question=item["question"], answer_a=item["high_score_pure_code"], answer_b=item["low_score_pure_code"])}
        ]
        item["messages_reverse"] = [
            {"role": "system", "content": template["pairwise"]["system_prompt"]},
            {"role": "user", "content": template["pairwise"]["prompt_template"].format(question=item["question"], answer_a=item["low_score_pure_code"], answer_b=item["high_score_pure_code"])}
        ]

    return data
    

def win_or_lose(pairwise_result, pairwise_reverse_result):
    win_table = {
        ("A", "A"): "tie",
        ("A", "B"): "win",
        ("A", "C"): "tie",
        ("B", "A"): "lose",
        ("B", "B"): "tie",
        ("B", "C"): "tie",
        ("C", "A"): "tie",
        ("C", "B"): "tie",
        ("C", "C"): "tie",
    }

    match = re.search(r'\[\[(.*?)\]\](?!.*\[\[)', pairwise_result)
    result = match.group(1).strip() if match else None

    match_reverse = re.search(r'\[\[(.*?)\]\](?!.*\[\[)', pairwise_reverse_result)
    result_reverse = match_reverse.group(1).strip() if match_reverse else None

    return win_table.get((result, result_reverse), None)

def process_conversation(api, conv, params):
    API_MAX_RETRY = 3
    for _ in range(API_MAX_RETRY):
        try:
            response = api.generateWithMessage(
                messages=conv["messages"],
                temperature=params["temperature"],
                max_tokens=params["max_tokens"]
            )

            response_reverse = api.generateWithMessage(
                messages=conv["messages_reverse"],
                temperature=params["temperature"],
                max_tokens=params["max_tokens"]
            )

            win_result = win_or_lose(response, response_reverse)
            
            return {
                "question_id": conv["question_id"],
                "question": conv["question"],
                "low_score_pure_code": conv["low_score_pure_code"],
                "high_score_pure_code": conv["high_score_pure_code"],
                "low_score": conv["low_score"],
                "high_score": conv["high_score"],
                "pairwise_result": response,
                "pairwise_reverse_result": response_reverse,
                "win_result": win_result,
            }
        except Exception as e:
            print(f"Error processing conversation with question_id {conv['question_id']}: {e}")
            return None

class States:
    def __init__(self):
        self.win = 0
        self.tie = 0
        self.lose = 0
        self.total = 0
    
    def update(self, result):
        if result == "win":
            self.win += 1
        elif result == "lose":
            self.lose += 1
        elif result == "tie":
            self.tie += 1
        self.total += 1
    
    def get_rates(self):
        if self.total == 0:
            return 0.0, 0.0, 0.0
        return (
            self.win / self.total,
            self.tie / self.total,
            self.lose / self.total
        )
    
    def get_desc(self):
        win_rate, tie_rate, lose_rate = self.get_rates()
        return f"Win: {win_rate:.2%}, Tie: {tie_rate:.2%}, Lose: {lose_rate:.2%}"

def evaluate_code(data_and_convs, api, params, output_file, max_workers):
    final_results = []
    states = States()
    
    progress_bar = tqdm(total=len(data_and_convs), desc=states.get_desc())
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_conversation, api, conv, params): conv["question_id"]
            for conv in data_and_convs
        }

        for future in as_completed(futures):
            result = future.result()
            if result:
                states.update(result["win_result"])
                final_results.append(result)
                progress_bar.set_description(states.get_desc())
                progress_bar.update(1)

    progress_bar.close()
    final_results.sort(key=lambda x: x["question_id"])
    write_json(final_results, output_file)
    write_states(output_file, states.win, states.tie, states.lose, states.total)
    print(f"评估结果保存到 {output_file}")

    return final_results


def pairwise_main(args):

    api = get_api(args)

    model_path = args.local_model if args.local_model else args.api_model

    output_file = args.output_file_dir+os.path.basename(model_path)+"_pairwise.json"

    params = {}
    params["model"] = args.local_model if args.local_model else args.api_model
    params["temperature"] = args.temperature
    params["max_tokens"] = args.max_tokens

    data_and_convs = prepare_conversations(args.pairwise_template_path, args.data_path)

    print(f"---------------开始对{os.path.basename(model_path)}评估pairwise---------------")
    evaluate_code(data_and_convs, api, params, output_file, args.max_workers)
