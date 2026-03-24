import argparse
import json

import tiktoken
import tqdm


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_log_file", type=str, required=True)
    return parser


# Mapping of prompt names to model names
# TODO: should load from the config file
template_names = {
    "generate_schema_qa.prompt": "gpt4.1",
    "extract_schema.prompt": "gpt4.1-mini",
    "schema_merging_sql.prompt": "gpt4.1",
    "answer_from_schema_no_table.prompt": "gpt4.1-mini",
    "answer_from_schema_tool_use.prompt": "gpt4.1",
    "answer_with_tool_use_output.prompt": "gpt4.1",
    "direct_tool_use_answer.prompt": "gpt4.1",
    "direct_tool_use.prompt": "gpt4.1",
}

# Tokenizer for gpt-4.1
tokenizer = tiktoken.get_encoding("cl100k_base")

# Model costs in USD per million tokens
model_costs = {
    "gpt4.1": {
        "input": 2.00,
        "output": 8.0,
    },
    "gpt4.1-mini": {
        "input": 0.4,
        "output": 1.6,
    },
}


def calculate_cost(prompt_file: str):
    with open(prompt_file, "r") as f:
        data = [json.loads(line) for line in f]

    total_cost = 0
    # cost calculation
    for item in tqdm.tqdm(data):
        prompt_name = item["template_name"]
        if prompt_name not in template_names:
            continue
        instruction = item["instruction"]
        input = item["input"]
        output = item["output"]

        instruction_tokens = tokenizer.encode(instruction)
        input_tokens = tokenizer.encode(input)
        output_tokens = tokenizer.encode(output)

        instruction_cost = len(instruction_tokens) * model_costs[template_names[prompt_name]]["input"] / 1000000
        input_cost = len(input_tokens) * model_costs[template_names[prompt_name]]["input"] / 1000000
        output_cost = len(output_tokens) * model_costs[template_names[prompt_name]]["output"] / 1000000

        total_cost += instruction_cost + input_cost + output_cost

    print(total_cost)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    calculate_cost(args.prompt_log_file)
