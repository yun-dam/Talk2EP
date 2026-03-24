import os
from typing import Callable

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio

from sliders.document import contextualize_document_metadata
from sliders.baselines import System
from sliders.datasets import Dataset
from sliders.document import Document
from sliders.evaluation import Evaluator, LLMAsJudgeEvaluationTool
from sliders.globals import SlidersGlobal
from sliders.log_utils import logger
from sliders.experiments.base import Experiment


def log_finance_bench_results(result):
    logger.info(f"Gold Answer: {result['gold_answer']}")
    logger.info(f"Predicted Answer: {result['predicted_answer']}")
    for key, value in result["evaluation_tools"].items():
        # Handle different evaluation result formats
        if isinstance(value, dict):
            if "correct" in value:
                logger.info(f"{key}: {value['correct']}")
            else:
                logger.info(f"{key}: {value}")
        else:
            logger.info(f"{key}: {value}")


class FinanceBench(Experiment):
    def __init__(self, config: dict):
        self.config = config

        benchmark_path = self.config.get("benchmark_path")
        files_dir = self.config.get("files_dir")
        gpt_results_path = self.config.get("gpt_results_path")

        if benchmark_path is None:
            benchmark_path = "/data1/hypothesis_dataset/financebench/data/financebench_open_source.jsonl"
        if files_dir is None:
            files_dir = "/data1/hypothesis_dataset/financebench/markdown/pdfs/"
        if gpt_results_path is None:
            gpt_results_path = "/data1/hypothesis_dataset/financebench/results/gpt-4_oracle_reverse.jsonl"

        self.dataset = Dataset(benchmark_path)
        self.gpt_results = pd.read_json(gpt_results_path, lines=True)

        self.files_dir = files_dir
        self.evaluator = Evaluator()

        # Evaluation tools
        # Soft evaluator
        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/soft_evaluator.prompt",
                model=self.config["soft_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

        # Hard evaluator
        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/hard_evaluator.prompt",
                model=self.config["hard_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

    @property
    def description(self) -> str:
        return "Financial statement for a company (10K, 10Q, 8K, etc.)"

    async def _run_row(self, row: dict, system: System, all_metadata: list) -> dict:
        question = row["question"]
        file_path = os.path.join(self.files_dir, row["doc_name"] + ".md")
        document = await Document.from_markdown(
            file_path,
            description=self.description,
            document_name=row["doc_name"],
            **self.config.get("document_config", {}),
        )
        try:
            all_documents = await contextualize_document_metadata([document], question)
        except Exception:
            logger.error(f"Error contextualizing document metadata for question: {question}")
            all_documents = [document]

        try:
            answer, metadata = await system.run(question, all_documents, question_id=row["financebench_id"])
            metadata["gold_answer"] = row["answer"]
            metadata["predicted_answer"] = answer
        except Exception as e:
            logger.error(f"Error running system for question: {question}")
            import traceback

            logger.error(traceback.format_exc())
            logger.error(e)
            all_metadata.append(
                {
                    "question": question,
                    "error": str(e),
                    "answer": None,
                    "metadata": None,
                    "question_id": row["financebench_id"],
                }
            )
            return {"error": str(e), "question_id": row["financebench_id"]}

        metadata["id"] = row["financebench_id"]
        metadata["evidence"] = row["evidence"]

        result = await self.evaluator.evaluate(
            question_id=row["financebench_id"], question=question, gold_answer=row["answer"], predicted_answer=answer
        )

        all_metadata.append(metadata)
        return result

    async def run(
        self,
        system: System,
        filter_func: Callable[[dict], bool] | None = None,
        sample_size: int | None = None,
        random_state: int | None = None,
        parallel: bool = False,
    ) -> dict:
        results = []
        dataset = self.dataset
        if filter_func is not None:
            dataset = dataset.filter(filter_func)
        if sample_size is not None:
            dataset = dataset.sample(sample_size, random_state=random_state)
        all_metadata = []

        dataset_size = len(dataset)

        if parallel:
            tasks = [self._run_row(row, system, all_metadata) for row in dataset]
            results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
        else:
            for idx, row in enumerate(tqdm(dataset, desc="Running experiment")):
                logger.info(
                    f"===============================================\n{idx + 1} of {dataset_size} | Question {row.get('financebench_id', 'N/A')}\n==============================================="
                )

                result = await self._run_row(row, system, all_metadata)
                results.append(result)
                log_finance_bench_results(result)

                # Log current evaluation state and accuracies
                if len(results) > 0 and "evaluation_tools" in results[0]:
                    # Calculate current accuracies for each evaluation tool
                    current_accuracies = {}
                    for eval_tool in results[0]["evaluation_tools"].keys():
                        if isinstance(results[0]["evaluation_tools"][eval_tool]["correct"], bool):
                            correct_count = sum(
                                1
                                for r in results
                                if r.get("evaluation_tools", {}).get(eval_tool, {}).get("correct", False)
                            )
                            accuracy = correct_count / len(results)
                            current_accuracies[eval_tool] = accuracy
                        elif isinstance(results[0]["evaluation_tools"][eval_tool]["correct"], (int, float)):
                            correct_count = sum(
                                r.get("evaluation_tools", {}).get(eval_tool, {}).get("correct", 0) for r in results
                            )
                            accuracy = correct_count / len(results)
                            current_accuracies[eval_tool] = accuracy

                    logger.info(f"=== CURRENT EVALUATION STATE ({len(results)}/{dataset_size}) ===")
                    for eval_tool, accuracy in current_accuracies.items():
                        logger.info(f"{eval_tool} accuracy: {accuracy:.3f}")

                # Log progress and any errors
                if "error" in result:
                    logger.warning(f"Question {row.get('financebench_id', 'N/A')} had an error: {result['error']}")
                logger.info(f"Completed {len(results)}/{dataset_size} questions")

        for i, result in enumerate(results):
            result["id"] = all_metadata[i].get("id")
            result["evidence"] = all_metadata[i].get("evidence")

        # results summary
        results_summary = {}
        for result in results:
            for tool_name, tool_data in result.get("evaluation_tools", {}).items():
                # Each tool_result is a dict like {"ToolName": {"correct": bool, ...}} or {"ToolName": {"error": ...}}
                if tool_name not in results_summary:
                    results_summary[tool_name] = {"correct": 0, "total": 0}

                is_correct = False
                if isinstance(tool_data, dict):
                    is_correct = bool(tool_data.get("correct", False))

                results_summary[tool_name]["correct"] += int(is_correct)
                results_summary[tool_name]["total"] += 1

        # compute accuracy per tool
        for tool_name, agg in results_summary.items():
            total = agg.get("total", 0)
            correct = agg.get("correct", 0)
            agg["accuracy"] = (correct / total) if total else 0.0

        # Final summary
        successful_count = len([m for m in all_metadata if "error" not in m])
        error_count = len([m for m in all_metadata if "error" in m])
        logger.info("=== EXPERIMENT COMPLETE ===")
        logger.info(f"Total questions processed: {len(results)}")
        logger.info(f"Successful runs: {successful_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Expected sample size: {dataset_size}")

        return {
            "experiment_id": SlidersGlobal.experiment_id,
            "results": results,
            "all_metadata": all_metadata,
            "results_summary": results_summary,
        }
