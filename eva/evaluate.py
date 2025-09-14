import argparse
from evaluate_listwise import listwise_main
from evaluate_pointwise import pointwise_main
from evaluate_pairwise import pairwise_main

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process some inputs for judgement.")

    # Model Configuration
    parser.add_argument("--api-model", type=str, help="Specify the language model to use.")
    parser.add_argument("--local-model", type=str, help="Specify the local model to use.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Set the sampling temperature for the language model.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Define the maximum number of tokens to generate.")

    # File Paths
    parser.add_argument("--data-path", type=str, help="Data to be evaluated")
    parser.add_argument('--pairwise-template-path', type=str, default="/shd/zzr/xz/EvalDatasetsForCodeJudge/baseline/templates/template_MT-bench.jsonl", help='Path to the template file for pairwise judgement.')
    parser.add_argument('--output-file-dir', type=str, default="/shd/zzr/xz/EvalDatasetsForCodeJudge/eva_results/", help='Path to store the judgement.')

    # API Configuration
    parser.add_argument("--api-base", type=str, default="https://api2.aigcbest.top/v1", help="Set the base URL for the API.")
    parser.add_argument("--api-key", type=str, default="", help="Provide the API key for authentication.")

    # Other
    parser.add_argument("--max-workers", type=int, default=32, help="Set the number of workers for multiple thread.")
    parser.add_argument("--method", type=str, choices=["listwise", "pointwise", "pairwise"], help="Set the method for judgement (listwise/pointwise/pairwise).")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.method == "listwise":
        listwise_main(args)
    elif args.method == "pointwise":
        pointwise_main(args)
    elif args.method == "pairwise":
        pairwise_main(args)

if __name__ == "__main__":
    main()




