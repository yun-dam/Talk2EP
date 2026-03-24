import glob
import json
import os
from typing import Callable

import numpy as np
from tqdm import tqdm

from sliders.baselines import System
from sliders.datasets import Dataset
from sliders.document import Document, contextualize_document_metadata
from sliders.evaluation import Evaluator, LLMAsJudgeEvaluationTool
from sliders.globals import SlidersGlobal
from sliders.llm_models import Evaluation, EvaluationScore
from sliders.log_utils import logger
from tqdm.asyncio import tqdm as tqdm_asyncio

file_handle_cache = {}


class Loong:
    def __init__(self, config: dict):
        self.config = config

        benchmark_path = self.config.get("benchmark_path")
        files_dir = self.config.get("files_dir")
        gpt_results_path = self.config.get("gpt_results_path")

        if benchmark_path is None:
            benchmark_path = "/data1/hypothesis_dataset/loong/loong.jsonl"
        if files_dir is None:
            files_dir = "/data1/hypothesis_dataset/loong/doc/"
        if gpt_results_path is None:
            gpt_results_path = None

        self.dataset = Dataset(benchmark_path)

        # Apply filters based on config
        self.dataset = self._apply_filters(self.dataset, config)

        self.files_dir = files_dir
        self.evaluator = Evaluator()

        # Evaluation tools
        # Soft evaluator
        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/soft_evaluator.prompt",
                eval_class=Evaluation,
                model=self.config["soft_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

        # Hard evaluator
        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/hard_evaluator.prompt",
                eval_class=Evaluation,
                model=self.config["hard_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/loong_evaluator.prompt",
                eval_class=EvaluationScore,
                model=self.config["hard_evaluator_model"],
                temperature=0.0,
                max_tokens=4096,
            )
        )

    def _apply_filters(self, dataset: Dataset, config: dict) -> Dataset:
        """Apply filters to the dataset based on configuration options."""
        # Priority: specific_ids_csv takes precedence over type/level filters
        if config.get("specific_ids_csv"):
            try:
                import pandas as pd

                id_sample_df = pd.read_csv(config["specific_ids_csv"])
                specific_ids = set(id_sample_df["id"].tolist())
                dataset = dataset.filter_by_specific_ids(specific_ids)
                logger.info(f"Filtered dataset by specific IDs: {len(specific_ids)} IDs")
            except Exception as e:
                logger.warning(f"Failed to load specific IDs CSV: {e}, falling back to type/level filters")

        # Apply type filters
        if config.get("filter_by_type"):
            filter_type = config["filter_by_type"]
            dataset = dataset.filter(lambda row: row.get("type") == filter_type)
            logger.info(f"Filtered dataset by type: {filter_type}")

        if config.get("filter_by_types"):
            filter_types = config["filter_by_types"]
            dataset = dataset.filter(lambda row: row.get("type") in filter_types)
            logger.info(f"Filtered dataset by types: {filter_types}")

        # Apply level filters
        if config.get("filter_by_level"):
            filter_level = config["filter_by_level"]
            dataset = dataset.filter(lambda row: row.get("level") == filter_level)
            logger.info(f"Filtered dataset by level: {filter_level}")

        if config.get("filter_by_levels"):
            filter_levels = config["filter_by_levels"]
            dataset = dataset.filter(lambda row: row.get("level") in filter_levels)
            logger.info(f"Filtered dataset by levels: {filter_levels}")

        logger.info(f"Filtered dataset size: {len(dataset)}")
        return dataset

    def description(self, doc_type: str) -> str:
        if doc_type == "paper":
            return "An academic paper from Arxiv."
        elif doc_type == "financial":
            return "A financial document primarily the quarterly and annual reports of a company."
        elif doc_type == "legal":
            return "Consists exclusively of cases adjudicated by the higher and intermediate courts "
        else:
            raise ValueError(f"Unknown document type: {doc_type}")

    async def run(
        self,
        system: System,
        filter_func: Callable[[dict], bool] | None = None,
        sample_size: int | None = None,
        random_state: int | None = None,
        parallel: bool = False,
    ):
        all_results = []
        dataset = self.dataset
        if filter_func is not None:
            dataset = dataset.filter(filter_func)
        if sample_size is not None:
            logger.info(f"Sampling {sample_size} questions")
            dataset = dataset.sample(sample_size, random_state=random_state)

        # Print dataset statistics
        num_document_counts = {}
        level_counts = {}
        type_counts = {}
        for row in dataset:
            level = row.get("level", "unknown")
            doc_type = row.get("type", "unknown")
            level_counts[level] = level_counts.get(level, 0) + 1
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
            num_document_counts[len(row["doc"])] = num_document_counts.get(len(row["doc"]), 0) + 1
        dataset_size = len(dataset)
        logger.info(
            f"Dataset level counts: {level_counts}\nDataset type counts: {type_counts}\nTotal dataset size: {dataset_size}"
        )
        logger.info(
            f"No. Documents per question:\navg: {np.mean(list(num_document_counts.keys())):.2f} ± {np.std(list(num_document_counts.keys())):.2f}\nmax: {max(list(num_document_counts.keys()))}\nmin: {min(list(num_document_counts.keys()))}"
        )

        all_results = []
        all_metadata = []
        for row in tqdm(dataset, desc="Running experiment"):
            doc_type = row["type"]
            doc_level = row["level"]

            logger.info(
                f"===============================================\n{len(all_results) + 1} of {dataset_size} | {row['type']}, L{row['level']} | Question {row['id']}\n==============================================="
            )

            logger.info("Loading documents...")
            all_documents = []
            load_document_tasks = []
            for doc_name in row["doc"]:
                file_path = os.path.join(self.files_dir, doc_type)
                if doc_type == "paper":
                    file_path = os.path.join(file_path, doc_name)

                    if not os.path.exists(file_path):
                        return None, {"error": f"Document {file_path} not found"}

                    if self.config.get("docprocesssing", True):
                        document_name = None
                    else:
                        document_name = doc_name
                    load_document_tasks.append(
                        Document.from_markdown(
                            file_path,
                            description=self.description(doc_type),
                            document_name=document_name,
                            **self.config.get("document_config", {}),
                        )
                    )
                elif doc_type == "financial":
                    if str(doc_level) == "4":
                        doc_path = glob.glob(os.path.join(file_path, f"*{doc_name}*.txt"))[0]
                    else:
                        doc_path = glob.glob(os.path.join(file_path, f"*2024-{doc_name}*.txt"))[0]

                    if not os.path.exists(doc_path):
                        return None, {"error": f"Document {doc_path} not found"}

                    json_table_path = os.path.join(
                        self.files_dir, "finance_processed_2", os.path.basename(doc_path) + ".new.tables.json"
                    )

                    if json_table_path and not os.path.exists(json_table_path):
                        logger.error(f"Document {doc_path} not found. JSON table path: {json_table_path}")

                    if self.config.get("docprocesssing", True):
                        document_name = None
                    else:
                        document_name = doc_name

                    load_document_tasks.append(
                        Document.from_file_path(
                            doc_path,
                            description=self.description(doc_type),
                            document_name=document_name,
                            tables_json_path=json_table_path,
                            **self.config.get("document_config", {}),
                        )
                    )
                elif doc_type == "legal":
                    doc_path = os.path.join(file_path, "legal.json")
                    if doc_path in file_handle_cache:
                        legal_js = file_handle_cache[doc_path]
                    else:
                        with open(doc_path, "r") as txt_file:
                            legal_js = json.load(txt_file)
                            file_handle_cache[doc_path] = legal_js

                    if doc_level == 4 and ("阅读以上判决文书，我将给你若干份判决结果：" in row["instruction"]):
                        legal_json_content = legal_js[doc_name]["content"]
                    else:
                        legal_json_content = legal_js[doc_name]["content"] + legal_js[doc_name]["result"]

                    load_document_tasks.append(
                        Document.from_plain_text(
                            legal_json_content,
                            description=self.description(doc_type),
                            document_name=doc_name,
                            file_path=doc_path,
                            **self.config.get("document_config", {}),
                        )
                    )
                else:
                    raise ValueError(f"Unknown document type: {doc_type}")

            all_documents = await tqdm_asyncio.gather(*load_document_tasks, desc="Loading documents")

            logger.info("All documents loaded.")

            logger.info("Contextualizing documents...")
            doc_names = [doc.document_name for doc in all_documents]
            question = row["prompt_template"].format(
                docs=doc_names, instruction=row.get("instruction", ""), question=row["question"]
            )
            try:
                # TODO: redo doc_names using LLM to make them differentiating from each other
                if self.config.get("docprocesssing", True):
                    all_documents = await contextualize_document_metadata(all_documents, question)
                else:
                    all_documents = all_documents
            except Exception as e:
                logger.error(
                    f"Error contextualizing document metadata: {e}, defaulting to original document descriptions"
                )

            logger.info("Running system...")
            try:
                answer, metadata = await system.run(question, all_documents, question_id=row["id"])
            except Exception as e:
                logger.error(f"Error running system: {e}")

                answer = f"Error running system: {e}"
                metadata = {"error": str(e), "question_id": row["id"]}

            # add question level and type to the metadata dict
            metadata["misc_question_metadata"] = {
                "question_type": row["type"],
                "question_level": row["level"],
                "question_id": row["id"],
            }

            # Evaluate immediately and await the result
            evaluation_result = await self.evaluator.evaluate(
                question_id=row["id"], question=question, gold_answer=row["answer"], predicted_answer=answer
            )

            all_results.append(evaluation_result)
            all_metadata.append(metadata)
            log_loong_results(evaluation_result)

            # Log current evaluation state and accuracies
            if len(all_results) > 0:
                # Calculate current accuracies for each evaluation tool
                current_accuracies = {}
                for eval_tool in all_results[0]["evaluation_tools"].keys():
                    if isinstance(all_results[0]["evaluation_tools"][eval_tool]["correct"], bool):
                        correct_count = sum(
                            1 for result in all_results if result["evaluation_tools"][eval_tool]["correct"]
                        )
                        accuracy = correct_count / len(all_results)
                        current_accuracies[eval_tool] = accuracy
                    elif isinstance(all_results[0]["evaluation_tools"][eval_tool]["correct"], (int, float)):
                        correct_count = sum(result["evaluation_tools"][eval_tool]["correct"] for result in all_results)
                        accuracy = correct_count / len(all_results)
                        current_accuracies[eval_tool] = accuracy
                    else:
                        raise ValueError(
                            f"Unknown evaluation tool type: {type(all_results[0]['evaluation_tools'][eval_tool]['correct'])}"
                        )
                logger.info(f"=== CURRENT EVALUATION STATE ({len(all_results)}/{dataset_size}) ===")
                for eval_tool, accuracy in current_accuracies.items():
                    logger.info(f"{eval_tool} accuracy: {accuracy:.3f}")

            # Log progress and any errors
            if "error" in metadata:
                logger.warning(f"Question {row['id']} had an error: {metadata['error']}")
            logger.info(f"Completed {len(all_results)}/{dataset_size} questions")

        # results summary
        results_summary = {}
        for result in all_results:
            for tool_name, tool_data in result.get("evaluation_tools", {}).items():
                # Each tool_result is a dict like {"ToolName": {"correct": bool, ...}} or {"ToolName": {"error": ...}}
                if tool_name not in results_summary:
                    results_summary[tool_name] = {"correct": 0, "total": 0}

                is_correct = False
                if isinstance(tool_data, dict):
                    is_correct = bool(tool_data.get("correct", False))

                results_summary[tool_name]["correct"] += int(is_correct)
                results_summary[tool_name]["total"] += 1

        # Final summary
        successful_count = len([m for m in all_metadata if "error" not in m])
        error_count = len([m for m in all_metadata if "error" in m])
        logger.info("=== EXPERIMENT COMPLETE ===")
        logger.info(f"Total questions processed: {len(all_results)}")
        logger.info(f"Successful runs: {successful_count}")
        logger.info(f"Errors: {error_count}")
        logger.info(f"Expected sample size: {dataset_size}")

        return {
            "experiment_id": SlidersGlobal.experiment_id,
            "results": all_results,
            "all_metadata": all_metadata,
            "results_summary": results_summary,
        }


def log_loong_results(result):
    logger.info(f"Gold Answer: {result['gold_answer']}")
    logger.info(f"Predicted Answer: {result['predicted_answer']}")
    for key, value in result["evaluation_tools"].items():
        logger.info(f"{key}: {value['correct']}")
