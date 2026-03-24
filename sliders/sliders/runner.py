"""Entry point for running SLIDERS experiments.

You can either run a specific experiment by passing the config file path to the --config argument,
or run the default experiment by not passing any arguments.

If you do not provide a config file, you can set which function to run. Default is to run the finance_bench_sliders function.
"""


from sliders.globals import SlidersGlobal
import argparse
import asyncio
import json
import os

import yaml
from pydantic import BaseModel
from tqdm import tqdm

from sliders.baselines import (
    LLMSequentialSystem,
    LLMWithoutToolUseSystem,
    LLMWithToolUseSystem,
    RLMSystem,
)
from sliders.experiment import print_result_summary
from sliders.experiments.finance_bench import FinanceBench
from sliders.experiments.loong import Loong
from sliders.experiments.babilong import BabiLong
from sliders.log_utils import logger
from sliders.system import SlidersAgent


SLIDERS_RESULTS = os.environ["SLIDERS_RESULTS"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--parallel", action="store_true")
    return parser.parse_args()


class Config(BaseModel):
    system: str
    experiment: str
    config_file: str
    system_config: dict
    experiment_config: dict
    output_file: str

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, "r") as f:
            config = yaml.safe_load(f)
            config["config_file"] = file_path
            return cls.model_validate(config)


EXPERIMENT_REGISTRY = {
    "finance_bench": FinanceBench,
    "loong": Loong,
    "babilong": BabiLong,
}

SYSTEM_REGISTRY = {
    "direct_tool_use": LLMWithToolUseSystem,
    "direct_no_tool_use": LLMWithoutToolUseSystem,
    "sequential": LLMSequentialSystem,
    "sliders": SlidersAgent,
    "rlm": RLMSystem,
}


async def run_experiment(config_file: str, parallel: bool = False):
    config = Config.from_file(config_file)
    experiment = EXPERIMENT_REGISTRY[config.experiment](config.experiment_config)
    system = SYSTEM_REGISTRY[config.system](config.system_config)
    results = await experiment.run(
        system,
        sample_size=config.experiment_config.get("num_questions"),
        random_state=config.experiment_config.get("random_state", 42),
        parallel=parallel,
    )
    try:
        results["config"] = config.model_dump(mode="json")
    except Exception as e:
        logger.error(f"Error dumping config: {e}")

    print_result_summary(results)

    output_file = os.path.join(
        SLIDERS_RESULTS, config.output_file.replace(".json", f"_{SlidersGlobal.experiment_id}.json")
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


async def run_finance_bench_direct_tool_use():
    experiment = FinanceBench(
        config={
            "soft_evaluator_engine": "gpt-4.1",
            "hard_evaluator_engine": "gpt-4.1",
        }
    )
    system = SlidersAgent(
        config={
            "tool_use": {
                "engine": "gpt-4.1",
                "template_file": "baselines/direct_tool_use.prompt",
                "max_tokens": 2048,
                "temperature": 0.0,
            },
            "answer": {
                "engine": "gpt-4.1",
                "template_file": "baselines/direct_tool_use_answer.prompt",
                "max_tokens": 2048,
                "temperature": 0.0,
            },
        }
    )

    print("Running Finance Bench Direct Tool Use experiment...")
    results = await experiment.run(
        system,
        sample_size=1,
        random_state=42,
    )

    with open(f"results/finance_bench_direct_tool_use_{SlidersGlobal.experiment_id}.json", "w") as f:
        json.dump(results, f, indent=2)


async def run_finance_bench_sliders():
    experiment = FinanceBench(
        config={
            "soft_evaluator_engine": "gpt-4.1",
            "hard_evaluator_engine": "gpt-4.1",
        }
    )
    system = SlidersAgent(config=Config.from_file("configs/finance_bench_sliders.yaml").system_config)

    print("Loading previous results to identify incorrect IDs...")
    with open(
        "results/finance_bench_sliders_objectives_based_20250829_003232.json", "r"
    ) as f:
        results = json.load(f)

    print("Processing incorrect IDs...")
    incorrect_ids = []
    for question in tqdm(results["results"], desc="Processing questions"):
        for key, value in question["evaluation_tools"].items():
            if "soft" in key:
                if not value["correct"]:
                    incorrect_ids.append(question["id"])
                    break

    print(f"Found {len(incorrect_ids)} incorrect IDs. Running experiment...")
    # results = await experiment.run(
    #     system,
    #     filter_func=lambda x: x["question"].startswith(
    #         "What is the FY2019 fixed asset turnover ratio for Activision Blizzard?"
    #     ),
    # )
    results = await experiment.run(
        system,
        # parallel=True,
        filter_func=lambda x: x["financebench_id"] in incorrect_ids,
    )

    with open(f"results/finance_bench_sliders_incorrect_ids_{SlidersGlobal.experiment_id}.json", "w") as f:
        json.dump(results, f, indent=2)


async def run_babilong_loong_sliders():
    system = SlidersAgent(config=Config.from_file("configs/babilong_loong_sliders_sample.yaml").system_config)

    qas = ["qa1", "qa2", "qa3", "qa4", "qa5", "qa6", "qa7", "qa8", "qa9", "qa10"]
    token_lens = ["128k", "256k", "512k", "1M"]

    # Create outer progress bar for QA iterations
    for qa in tqdm(qas, desc="QA Progress"):
        for token_len in tqdm(token_lens, desc="Token Length Progress"):
            experiment_config = Config.from_file("configs/babilong_loong_sliders_sample.yaml").experiment_config
            experiment_config["benchmark_path"] = f"babilong_data/babilong_{qa}_{token_len}.json"
            experiment = BabiLong(config=experiment_config)
            results = await experiment.run(system, sample_size=10)
            with open(
                f"results/babilong_loong_sliders_sample_{qa}_{token_len}_{SlidersGlobal.experiment_id}.json", "w"
            ) as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    try:
        if args.config:
            asyncio.run(run_experiment(args.config, args.parallel))
        else:
            asyncio.run(run_finance_bench_sliders())
    except Exception as e:
        logger.error(e)
        import traceback

        traceback.print_exc()
