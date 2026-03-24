import json
import os
from typing import Callable, Any, Sequence

from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio

from sliders.baselines import System
from sliders.datasets import Dataset
from sliders.document import Document, contextualize_document_metadata
from sliders.evaluation import Evaluator, LLMAsJudgeEvaluationTool
from sliders.experiments.base import Experiment
from sliders.globals import SlidersGlobal
from sliders.log_utils import logger


def log_babilong_results(result: dict[str, Any]) -> None:
    """Standard logging helper so experiment output matches other benchmarks."""
    logger.info(f"Gold Answer: {result.get('gold_answer')}")
    logger.info(f"Predicted Answer: {result.get('predicted_answer')}")
    for tool_name, tool_data in result.get("evaluation_tools", {}).items():
        if isinstance(tool_data, dict):
            if "correct" in tool_data:
                logger.info(f"{tool_name}: {tool_data['correct']}")
            else:
                logger.info(f"{tool_name}: {tool_data}")
        else:
            logger.info(f"{tool_name}: {tool_data}")


class BabiLong(Experiment):
    """Experiment wrapper for the BabiLong benchmark."""

    def __init__(self, config: dict):
        super().__init__(config)
        benchmark_path = config.get("benchmark_path")
        files_dir = config.get("files_dir")

        if benchmark_path is None:
            benchmark_dir = config.get(
                "benchmark_dir",
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "babilong_data")),
            )
            dataset_variant = config.get("dataset_variant", "qa1")
            token_length = config.get("token_length", "128k")
            benchmark_filename = f"babilong_{dataset_variant}_{token_length}.json"
            benchmark_path = os.path.join(benchmark_dir, benchmark_filename)

        if files_dir is not None:
            files_dir = os.path.abspath(files_dir)

        self.dataset = self._load_dataset(benchmark_path)
        self.files_dir = files_dir
        self.default_description = config.get(
            "document_description", "Synthetic long-context stories used for multi-hop question answering."
        )

        self.question_field = config.get("question_field", "question")
        self.answer_field = config.get("answer_field", "target")
        self.id_field = config.get("id_field", "id")
        self.context_field = config.get("context_field", "input")
        self.documents_field = config.get("documents_field", "documents")
        self.document_text_field = config.get("document_text_field", "text")
        self.document_path_field = config.get("document_path_field", "path")

        self.evaluator = Evaluator()
        self._configure_evaluators(config)
        self._ensure_row_ids()

    class _InMemoryDataset:
        """Fallback dataset wrapper for pre-parsed records."""

        def __init__(self, records: list[dict[str, Any]]):
            self.data = records

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index: int):
            return self.data[index]

        def filter(self, condition_func: Callable[[dict[str, Any]], bool]):
            filtered = [item for item in self.data if condition_func(item)]
            return BabiLong._InMemoryDataset(filtered)

        def sample(self, n: int, random_state: int | None = None):
            import random

            if random_state is not None:
                random.seed(random_state)

            sampled = random.sample(self.data, min(n, len(self.data)))
            return BabiLong._InMemoryDataset(sampled)

        def filter_by_specific_ids(self, ids: list[str]):
            specific_ids = set(ids)
            filtered = [item for item in self.data if item.get("id") in specific_ids]
            return BabiLong._InMemoryDataset(filtered)

    def _load_dataset(self, benchmark_path: str):
        """Load the dataset, falling back to JSON-lines parsing when needed."""
        try:
            dataset = Dataset(benchmark_path)
            return dataset
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Falling back to JSON-lines loader for %s due to %s. "
                "Ensure the file contains one JSON object per line.",
                benchmark_path,
                exc,
            )
            records = self._load_json_lines(benchmark_path)
            return self._InMemoryDataset(records)

    def _load_json_lines(self, benchmark_path: str) -> list[dict[str, Any]]:
        """Manual loader for files that are JSON-lines formatted but saved with .json."""
        records: list[dict[str, Any]] = []
        with open(benchmark_path, "r", encoding="utf-8") as dataset_file:
            for line_number, raw_line in enumerate(dataset_file, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                    records.append(record)
                except json.JSONDecodeError as decode_error:
                    raise ValueError(
                        f"Failed to decode JSON on line {line_number} of {benchmark_path}: {decode_error}"
                    ) from decode_error
        return records

    def _ensure_row_ids(self) -> None:
        """Backfill deterministic IDs when the source data does not provide one."""
        dataset = self.dataset
        data = getattr(dataset, "data", None)
        if data is None:
            return

        missing_ids = [row for row in data if not row.get(self.id_field)]
        if not missing_ids:
            return

        for index, row in enumerate(data):
            if row.get(self.id_field):
                continue
            row[self.id_field] = str(index)

    def _configure_evaluators(self, config: dict) -> None:
        evaluator_kwargs = {
            "temperature": 0.0,
            "max_tokens": 4096,
        }
        soft_model = config.get("soft_evaluator_model")
        hard_model = config.get("hard_evaluator_model")

        if soft_model is None or hard_model is None:
            raise ValueError("Both `soft_evaluator_model` and `hard_evaluator_model` must be configured.")

        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/soft_evaluator.prompt",
                model=soft_model,
                **evaluator_kwargs,
            )
        )
        self.evaluator.add_evaluation_tool(
            LLMAsJudgeEvaluationTool(
                prompt_file="evaluators/hard_evaluator.prompt",
                model=hard_model,
                **evaluator_kwargs,
            )
        )

    async def _load_documents(self, row: dict) -> list[Document]:
        """Create Document objects from the row."""
        doc_config = self.config.get("document_config", {})
        documents: list[Document] = []

        # Prefer explicit documents list if available
        if self.documents_field in row and row[self.documents_field] is not None:
            documents_data = row[self.documents_field]
            if isinstance(documents_data, dict):
                iterable_docs = [documents_data]
            elif isinstance(documents_data, Sequence) and not isinstance(documents_data, (str, bytes)):
                iterable_docs = documents_data
            else:
                iterable_docs = [documents_data]

            for idx, doc_entry in enumerate(iterable_docs):
                document_name = None
                description = self.default_description
                try:
                    if isinstance(doc_entry, dict):
                        document_name = doc_entry.get("name") or doc_entry.get("document_name")
                        description = doc_entry.get("description", description)

                        if self.document_text_field in doc_entry:
                            documents.append(
                                await Document.from_plain_text(
                                    doc_entry[self.document_text_field],
                                    description=description,
                                    document_name=document_name,
                                    **doc_config,
                                )
                            )
                            continue

                        if self.document_path_field in doc_entry:
                            doc_path = doc_entry[self.document_path_field]
                            documents.append(
                                await self._load_document_from_path(doc_path, description, document_name, doc_config)
                            )
                            continue

                    if isinstance(doc_entry, str):
                        documents.append(
                            await self._load_document_from_path(
                                doc_entry, description, document_name or doc_entry, doc_config
                            )
                        )
                    else:
                        logger.warning(f"Unsupported document entry at index {idx}: {doc_entry}")
                except FileNotFoundError as e:
                    logger.error(f"Missing document for row {row.get(self.id_field)}: {e}")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to prepare document for row {row.get(self.id_field)}: {e}")

        # Fall back to context field treated as a single document
        if not documents and self.context_field in row:
            context_text = row[self.context_field]
            if isinstance(context_text, str) and context_text.strip():
                doc_name = row.get("doc_name") or row.get(self.id_field) or "context"
                documents.append(
                    await Document.from_plain_text(
                        context_text,
                        description=self.default_description,
                        document_name=str(doc_name),
                        **doc_config,
                    )
                )
            else:
                logger.warning(f"Context field for row {row.get(self.id_field)} is empty or not a string.")

        if not documents:
            raise ValueError(f"No documents available for row {row.get(self.id_field)}")

        try:
            return await contextualize_document_metadata(documents, row[self.question_field])
        except Exception:  # noqa: BLE001
            logger.exception(
                "Error contextualizing document metadata for row %s, returning raw documents.", row.get(self.id_field)
            )
            return documents

    async def _load_document_from_path(
        self,
        doc_path: str,
        description: str,
        document_name: str | None,
        doc_config: dict,
    ) -> Document:
        """Helper to load a document either from markdown or plain text path."""
        candidate_path = doc_path
        if not os.path.isabs(candidate_path):
            if self.files_dir:
                candidate_path = os.path.join(self.files_dir, candidate_path)
            else:
                candidate_path = os.path.abspath(candidate_path)

        if not os.path.exists(candidate_path):
            raise FileNotFoundError(candidate_path)

        loader = Document.from_markdown if candidate_path.endswith(".md") else Document.from_file_path
        return await loader(candidate_path, description=description, document_name=document_name, **doc_config)

    async def _run_row(self, row: dict, system: System, all_metadata: list) -> dict:
        question = row[self.question_field]
        answer = row.get(self.answer_field, "")
        question_id = row.get(self.id_field)

        try:
            all_documents = await self._load_documents(row)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error loading documents for row {question_id}: {e}")
            all_metadata.append(
                {
                    "question": question,
                    "error": str(e),
                    "answer": None,
                    "metadata": None,
                    "question_id": question_id,
                }
            )
            return {"error": str(e), "question_id": question_id}

        try:
            predicted_answer, metadata = await system.run(question, all_documents, question_id=question_id)
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error running system for row {question_id}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            all_metadata.append(
                {
                    "question": question,
                    "error": str(e),
                    "answer": None,
                    "metadata": None,
                    "question_id": question_id,
                }
            )
            return {"error": str(e), "question_id": question_id}

        metadata["gold_answer"] = answer
        metadata["predicted_answer"] = predicted_answer
        metadata["id"] = question_id
        if "evidence" not in metadata:
            metadata["evidence"] = row.get("evidence")

        result = await self.evaluator.evaluate(
            question_id=question_id,
            question=question,
            gold_answer=answer,
            predicted_answer=predicted_answer,
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
        dataset = self.dataset
        if filter_func is not None:
            dataset = dataset.filter(filter_func)
        if sample_size is not None:
            dataset = dataset.sample(sample_size, random_state=random_state)

        all_metadata: list[dict] = []
        results: list[dict] = []
        dataset_size = len(dataset)

        if parallel:
            tasks = [self._run_row(row, system, all_metadata) for row in dataset]
            results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
        else:
            for idx, row in enumerate(tqdm(dataset, desc="Running experiment")):
                question_id = row.get(self.id_field, "N/A")
                logger.info(
                    f"===============================================\n{idx + 1} of {dataset_size} | Question {question_id}\n==============================================="
                )
                result = await self._run_row(row, system, all_metadata)
                results.append(result)
                log_babilong_results(result)

                if "error" in result:
                    logger.warning(f"Question {question_id} had an error: {result['error']}")
                logger.info(f"Completed {len(results)}/{dataset_size} questions")

        for i, result in enumerate(results):
            if i < len(all_metadata):
                result["id"] = all_metadata[i].get("id")
                if "evidence" not in result:
                    result["evidence"] = all_metadata[i].get("evidence")

        results_summary: dict[str, dict[str, float | int]] = {}
        for result in results:
            for tool_name, tool_data in result.get("evaluation_tools", {}).items():
                if tool_name not in results_summary:
                    results_summary[tool_name] = {"correct": 0, "total": 0}

                is_correct = False
                if isinstance(tool_data, dict):
                    is_correct = bool(tool_data.get("correct", False))
                results_summary[tool_name]["correct"] += int(is_correct)
                results_summary[tool_name]["total"] += 1

        for tool_name, aggregate in results_summary.items():
            total = aggregate.get("total", 0)
            correct = aggregate.get("correct", 0)
            aggregate["accuracy"] = (correct / total) if total else 0.0

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
