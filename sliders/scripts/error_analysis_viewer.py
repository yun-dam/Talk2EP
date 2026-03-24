import asyncio
import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from rich.logging import RichHandler

from sliders.llm_models import ErrorAnalysisResponse


@dataclass
class ExperimentFiles:
    """Class to hold experiment file paths and metadata"""

    timestamp: str
    debug_log: Optional[str] = None
    results_file: Optional[str] = None

    @property
    def has_debug_log(self) -> bool:
        return self.debug_log is not None and os.path.exists(self.debug_log)

    @property
    def has_results(self) -> bool:
        return self.results_file is not None and os.path.exists(self.results_file)

    @property
    def display_name(self) -> str:
        """Get display name for UI with file availability indicators"""
        parts = []

        # Add timestamp
        parts.append(self.timestamp)

        # Add file indicators
        indicators = []
        if self.has_debug_log:
            indicators.append("📝")  # Debug logs
            parts.append(f"[{os.path.basename(self.debug_log)}]")
        if self.has_results:
            indicators.append("📊")  # Results
            parts.append(f"[{os.path.basename(self.results_file)}]")

        status = f"[{' '.join(indicators)}]"
        parts.append(status)

        return " ".join(parts)

    def load_debug_log(self) -> List[Dict]:
        """Load and parse debug log file"""
        if not os.path.exists(self.debug_log):
            return "❌ Debug log file not found"

        logs = []
        parse_errors = []
        current_object = []
        current_object_start_line = 0

        with open(self.debug_log, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                if not current_object:
                    # Start of new object
                    current_object_start_line = line_num

                current_object.append(line.rstrip("\n"))

                # Try to parse if we have a complete object
                current_text = "".join(current_object)
                if current_text[-2:] == '"}':
                    current_text = ErrorAnalysisViewer._sanitize_message(self, current_text)
                    try:
                        log_entry = json.loads(current_text)
                        logs.append(log_entry)
                        current_object = []
                    except json.JSONDecodeError as e:
                        error_context = {
                            "start_line": current_object_start_line,
                            "end_line": line_num,
                            "error": str(e),
                            "text_preview": current_text[:200] + "..." if len(current_text) > 200 else current_text,
                        }
                        parse_errors.append(error_context)
                        logger.warning(f"JSON parse error at lines {current_object_start_line}-{line_num}: {e}")
                        logger.warning(f"Problematic object: {current_text}")
                        current_object = []
                    except Exception as e:
                        error_context = {
                            "start_line": current_object_start_line,
                            "end_line": line_num,
                            "error": f"Unexpected error: {str(e)}",
                            "text_preview": current_text[:200] + "..." if len(current_text) > 200 else current_text,
                        }
                        parse_errors.append(error_context)
                        logger.error(f"Unexpected error at lines {current_object_start_line}-{line_num}: {e}")
                        current_object = []

        # Check if we have an incomplete object at the end
        if current_object:
            error_context = {
                "start_line": current_object_start_line,
                "end_line": "EOF",
                "error": "Incomplete JSON object at end of file",
                "text_preview": "".join(current_object)[:200] + "...",
            }
            parse_errors.append(error_context)
            logger.warning("Incomplete JSON object at end of file")

        if not logs:
            error_summary = "\n".join(
                [
                    f"Lines {err['start_line']}-{err['end_line']}: {err['error']}\n  {err['text_preview']}"
                    for err in parse_errors[:5]  # Show first 5 errors
                ]
            )
            return f"❌ No valid log entries found in file. First few errors:\n{error_summary}"
        return {"logs": logs, "parse_errors": parse_errors}

    def load_results(self) -> List[Dict]:
        """Load and parse results file"""
        if not self.has_results:
            return []

        try:
            with open(self.results_file, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    results, metadata = json_data
                elif isinstance(json_data, dict):
                    results = json_data["results"]
                    metadata = json_data["all_metadata"]
                else:
                    raise ValueError(f"Unexpected items in results file: {len(json_data)}, {type(json_data)}")
                # Results file contains a list of lists, we want to flatten it
                flattened_results = []
                for result_group, metadata_group in zip(results, metadata):
                    # turn result_group + metadata_group into a single dict
                    flattened_results.append({**result_group, **metadata_group})
                return flattened_results
        except Exception as e:
            logger.error(f"Error loading results file: {e}")
            return []


# Remove any existing handlers
logger.remove()

# Add only console handler with rich formatting
logger.add(
    RichHandler(markup=True, show_time=False),
    level="INFO",
    format="{message}",
    backtrace=True,
    diagnose=True,
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env file
load_dotenv()


class ErrorAnalysisViewer:
    def __init__(self):
        self.experiments = []  # List of loaded experiments
        self.current_experiment = None
        self.questions = []  # List of questions in current experiment
        self.statistics = {}  # Statistics for current experiment
        self.current_results = None  # Results data if available

        # Initialize LLM chain
        try:
            self.llm_error_analysis = llm_generation_chain(
                engine="azure/gpt-4.1",  # Use the same engine as extract_schema
                template_file="sliders/error_analysis.prompt",  # Just the relative path
                max_tokens=4096,
                temperature=0.0,
                pydantic_class=ErrorAnalysisResponse,
                progress_bar_desc="Generating error analysis",
                force_skip_cache=True,
            )
            logger.info("Successfully initialized LLM chain")
        except Exception as e:
            logger.error(f"Failed to initialize LLM chain: {e}")
            breakpoint()
            self.llm_error_analysis = None

    def _sanitize_message(self, text: str) -> str:
        """Sanitize message content to ensure valid JSON"""
        # Find the message field
        message_start = text.find('"message": "')
        if message_start == -1:
            return text

        # Start after the message field
        content_start = message_start + 11  # len('"message": "')

        # Find the end of the object
        end_marker = '"}'
        content_end = text.find(end_marker, content_start)
        if content_end == -1:
            return text

        # Get the parts
        prefix = text[: content_start + 1]
        content = text[content_start + 1 : content_end]
        suffix = text[content_end:]

        # Replace all double quotes in content with single quotes
        content = content.replace('"', "'")

        return prefix + content + suffix

    def load_experiment(self, experiment_files: ExperimentFiles) -> str:
        """Load experiment from debug logs and/or results file"""
        try:
            # Reset current state
            self.current_experiment = None
            self.current_results = None

            status_messages = []

            # First try to load results file if available
            if experiment_files.has_results:
                results = experiment_files.load_results()
                if not results:
                    status_messages.append("❌ No valid entries found in results file")
                else:
                    self.current_results = results
                    # Create experiment structure from results
                    self.current_experiment = {
                        "path": experiment_files.results_file,
                        "logs": [],
                        "questions": self._extract_questions_from_results(results),
                        "statistics": self._calculate_statistics_from_results(results),
                        "parse_errors": [],
                    }
                    status_messages.append(
                        f"✅ Loaded results with {len(self.current_experiment['questions'])} questions"
                    )
            else:
                status_messages.append("Results File not included.")

            # Load debug logs if available
            if experiment_files.has_debug_log:
                debug_log_data = experiment_files.load_debug_log()
                logs = debug_log_data.get("logs", None)
                if not logs:
                    status_messages.append("❌ No valid log entries found in debug log")
                else:
                    if not self.current_experiment:
                        # Only create from debug logs if we don't have results
                        self.current_experiment = {
                            "path": experiment_files.debug_log,
                            "logs": logs,
                            "questions": self._extract_questions(logs),
                            "statistics": self._calculate_statistics(logs),
                            "parse_errors": debug_log_data.get("parse_errors", []),
                        }
                        status_messages.append(
                            f"✅ Loaded debug log with {len(self.current_experiment['questions'])} questions"
                        )
                    else:
                        # Add debug logs to existing questions
                        self.current_experiment["logs"].extend(logs)
                        temp_questions = self._extract_questions(logs)
                        # update current experiments with question logs and document descriptions
                        for question, temp_question in zip(self.current_experiment["questions"], temp_questions):
                            question["logs"] = temp_question["logs"]
                            # Create mapping of doc names to descriptions from temp question
                            temp_doc_map = {doc["name"]: doc for doc in temp_question["documents"]}

                            # Update descriptions for matching documents
                            for doc in question["documents"]:
                                if doc["name"] in temp_doc_map:
                                    matching_doc = temp_doc_map[doc["name"]]
                                    doc["original_description"] = matching_doc["original_description"]
                                    doc["generated_description"] = matching_doc["generated_description"]
                                else:
                                    logger.warning(f"Document {doc['name']} not found in question logs")

                        status_messages.append("✅ Added debug logs to results data")
            else:
                status_messages.append("Debug log not included.")

            if not status_messages:
                return "❌ No valid files found"

            return "\n".join(status_messages)

        except Exception as e:
            logger.error(f"Error loading experiment: {e}")
            self.current_experiment = None
            self.current_results = None
            return f"❌ Error loading experiment: {str(e)}"

    def _extract_questions_from_results(self, results: List[Dict]) -> List[Dict]:
        """Extract questions from results file format"""
        questions = []

        for result in results:
            # Extract question text and documents
            question_text = result.get("question", "")
            documents = [
                {"name": doc, "original_description": "", "generated_description": ""}
                for doc in result.get("document_names", [])
            ]
            question = {
                "id": result.get("question_id", None),
                "metadata": result.get("misc_question_metadata", {}),
                "logs": [],
                "documents": documents,
                "gold_answer": str(result.get("gold_answer")),
                "predicted_answer": str(result.get("predicted_answer")),
                "evaluations": {},
                "question_text": question_text,  # Store full question text
                "timing": result.get("timing", {}),  # Store timing information
                "schema": result.get("schema", {}),  # Store schema information
                "extraction": result.get("extraction", {}),  # Store extraction stats
                "merging": result.get("merging", {}),  # Store merging stats
                "quality": result.get("quality", {}),  # Store quality metrics
            }

            # Extract evaluations
            evaluations = result.get("evaluation_tools", {})
            if isinstance(evaluations, list):
                # turn this into a dict
                new_eval_dict = {}
                for evaluation in evaluations:
                    new_eval_dict.update(evaluation)
                evaluations = new_eval_dict

            for eval_name, eval_data in evaluations.items():
                eval_type = eval_name.replace("LLMAsJudgeEvaluationTool", "")
                # Store both the score and explanation
                question["evaluations"][eval_type] = {
                    "score": eval_data.get("correct"),
                    "explanation": eval_data.get("explanation"),
                    "appeal_evaluation": eval_data.get("appeal_evaluation", {}),
                }

            # Extract metadata
            if "misc_question_metadata" in result:
                metadata = result["misc_question_metadata"]
                question["domain"] = metadata.get("question_type")
                question["level"] = str(metadata.get("question_level"))

            # Extract any additional metadata
            if "num_documents" in result:
                question["num_documents"] = result["num_documents"]
            if "document_sizes" in result:
                question["document_sizes"] = result["document_sizes"]
            if "total_chunks" in result:
                question["total_chunks"] = result["total_chunks"]

            questions.append(question)

        return questions

    def _calculate_statistics_from_results(self, results: List[Dict]) -> Dict:
        """Calculate statistics from results file format"""
        stats = {
            "total_questions": len(results),
            "correct": 0,
            "overall_accuracy": {},
            "metadata_breakdown": {},  # Will store breakdowns for each metadata field
        }

        # Calculate success rates
        for result in results:
            evaluations = result.get("evaluation_tools", {})
            all_correct = True

            if isinstance(evaluations, list):
                new_eval_dict = {}
                for evaluation in evaluations:
                    new_eval_dict.update(evaluation)
                evaluations = new_eval_dict

            for eval_name, eval_data in evaluations.items():
                eval_type = eval_name.replace("LLMAsJudgeEvaluationTool", "")
                correct = eval_data.get("correct")

                # Update overall accuracy
                if eval_type not in stats["overall_accuracy"]:
                    stats["overall_accuracy"][eval_type] = {"total": 0, "correct": 0, "explanations": []}
                stats["overall_accuracy"][eval_type]["total"] += 1

                # Handle both boolean and numeric scores
                if isinstance(correct, bool):
                    if correct:
                        stats["overall_accuracy"][eval_type]["correct"] += 1
                    else:
                        all_correct = False
                elif isinstance(correct, (int, float)):
                    if correct == 100:
                        stats["overall_accuracy"][eval_type]["correct"] += 1
                    else:
                        all_correct = False

                # Store explanation if available
                if eval_data.get("explanation"):
                    stats["overall_accuracy"][eval_type]["explanations"].append(
                        {"score": correct, "explanation": eval_data["explanation"]}
                    )

            if all_correct:
                stats["correct"] += 1

            # Update metadata breakdowns
            if "misc_question_metadata" in result:
                metadata = result["misc_question_metadata"]
                for metadata_key, metadata_value in metadata.items():
                    metadata_value = str(metadata_value)  # Convert to string for consistency

                    # Initialize breakdown for this metadata type if not exists
                    if metadata_key not in stats["metadata_breakdown"]:
                        stats["metadata_breakdown"][metadata_key] = {}

                    if metadata_value not in stats["metadata_breakdown"][metadata_key]:
                        stats["metadata_breakdown"][metadata_key][metadata_value] = {
                            "total": 0,
                            "correct": 0,
                            "evaluator_scores": {},
                            "examples": {"success": [], "failure": []},
                        }

                    breakdown = stats["metadata_breakdown"][metadata_key][metadata_value]
                    breakdown["total"] += 1
                    if all_correct:
                        breakdown["correct"] += 1

                    # Update evaluator scores
                    for eval_name, eval_data in evaluations.items():
                        eval_type = eval_name.replace("LLMAsJudgeEvaluationTool", "")
                        if eval_type not in breakdown["evaluator_scores"]:
                            breakdown["evaluator_scores"][eval_type] = {"total": 0, "correct": 0}

                        breakdown["evaluator_scores"][eval_type]["total"] += 1
                        if isinstance(eval_data.get("correct"), bool):
                            if eval_data["correct"]:
                                breakdown["evaluator_scores"][eval_type]["correct"] += 1
                        elif isinstance(eval_data.get("correct"), (int, float)):
                            if eval_data["correct"] == 100:
                                breakdown["evaluator_scores"][eval_type]["correct"] += 1

                    # Store example if it's a clear success or failure
                    if all_correct:
                        if len(breakdown["examples"]["success"]) < 3:
                            breakdown["examples"]["success"].append(
                                {
                                    "question": result.get("question", ""),
                                    "gold_answer": result.get("gold_answer", ""),
                                    "predicted_answer": result.get("predicted_answer", ""),
                                    "evaluations": evaluations,
                                }
                            )
                    else:
                        if len(breakdown["examples"]["failure"]) < 3:
                            breakdown["examples"]["failure"].append(
                                {
                                    "question": result.get("question", ""),
                                    "gold_answer": result.get("gold_answer", ""),
                                    "predicted_answer": result.get("predicted_answer", ""),
                                    "evaluations": evaluations,
                                }
                            )

        # Calculate percentages and averages
        # total = stats["total_questions"]

        # Overall accuracy percentages
        for eval_type in stats["overall_accuracy"]:
            eval_stats = stats["overall_accuracy"][eval_type]
            if eval_stats["total"] > 0:
                eval_stats["accuracy"] = (eval_stats["correct"] / eval_stats["total"]) * 100

                # Calculate average scores for numeric evaluations
                scores = []
                for exp in eval_stats["explanations"]:
                    if isinstance(exp["score"], (int, float)):
                        scores.append(exp["score"])
                if scores:
                    eval_stats["average_score"] = sum(scores) / len(scores)

        # Metadata breakdown percentages
        for metadata_key, metadata_values in stats["metadata_breakdown"].items():
            for value_stats in metadata_values.values():
                if value_stats["total"] > 0:
                    value_stats["accuracy"] = (value_stats["correct"] / value_stats["total"]) * 100
                    # Calculate evaluator percentages
                    for eval_scores in value_stats["evaluator_scores"].values():
                        if eval_scores["total"] > 0:
                            eval_scores["accuracy"] = (eval_scores["correct"] / eval_scores["total"]) * 100

        return stats

    def _merge_results_into_questions(self, results: List[Dict]) -> None:
        """Merge results data into existing questions"""
        if not self.current_experiment or not self.current_experiment["questions"]:
            return

        # Create mapping of questions by content
        questions_map = {}
        for question in self.current_experiment["questions"]:
            key = f"{question.get('gold_answer')}|{question.get('predicted_answer')}"
            questions_map[key] = question

        # Merge results
        for result in results:
            key = f"{result.get('gold_answer')}|{result.get('predicted_answer')}"
            if key in questions_map:
                question = questions_map[key]

                # Update evaluations
                evaluations = result.get("evaluation_tools", {})
                for eval_name, eval_data in evaluations.items():
                    eval_type = eval_name.replace("LLMAsJudgeEvaluationTool", "")
                    question["evaluations"][eval_type] = eval_data.get("correct")

                # Update metadata if not already set
                if not question["domain"] and "misc_question_metadata" in result:
                    metadata = result["misc_question_metadata"]
                    question["domain"] = metadata.get("question_type")
                    question["level"] = str(metadata.get("question_level"))

    def _extract_questions(self, logs: List[Dict]) -> List[Dict]:
        """Extract individual questions and their results from logs"""
        questions = []
        current_question = None

        for log in logs:
            message = log.get("message", "")

            # Stop processing if we hit the summary section
            if "SUMMARY" in message:
                break

            # New question starts
            if "of" in message and "|" in message and "Question" in message:
                if current_question:
                    questions.append(current_question)
                current_question = {
                    "id": None,
                    "metadata": {},
                    "logs": [],
                    "documents": [],
                    "gold_answer": None,
                    "predicted_answer": None,
                    "evaluations": {},
                }

                # Parse question header
                parts = message.split("|")
                if len(parts) >= 2:
                    metadata = parts[1].strip().split(",")
                    if len(metadata) >= 2:
                        current_question["metadata"] = {
                            "question_type": metadata[0].strip(),
                            "question_level": metadata[1].strip(),
                        }
                    else:
                        for i, metadata_item in enumerate(metadata):
                            current_question["metadata"][f"axes_{i}"] = metadata_item.strip()

                    # Extract question ID
                    if "Question" in parts[-1]:
                        parsing_id = parts[-1].split("Question")[-1].strip()
                        current_question["id"] = parsing_id.replace("=", "")

            # Store logs for current question
            if current_question:
                current_question["logs"].append(log)

                # Debug logging
                message = log.get("message", "")
                if message.startswith("Document:"):
                    # Extract document info from single line message
                    current_question["documents"].append(
                        {
                            "name": message.split("Document:")[-1].split("Original description:")[0].strip(),
                            "original_description": message.split("Original description:")[-1]
                            .split("New description:")[0]
                            .strip(),
                            "generated_description": message.split("New description:")[-1].strip(),
                        }
                    )

                # Extract answers and evaluations
                if "Gold Answer:" in message:
                    current_question["gold_answer"] = message.split("Gold Answer:")[-1].strip()
                elif "Predicted Answer:" in message:
                    current_question["predicted_answer"] = message.split("Predicted Answer:")[-1].strip()
                elif any(key in message for key in ["evaluator:", "LLMAsJudgeEvaluationTool"]):
                    for eval_type in ["soft_evaluator", "hard_evaluator", "loong_evaluator"]:
                        if eval_type in message:
                            # Extract score from message
                            score_text = message.split(":")[-1].strip()
                            # Convert score to consistent format
                            if score_text.lower() == "true":
                                score = True
                            elif score_text.lower() == "false":
                                score = False
                            else:
                                try:
                                    score = float(score_text)
                                except ValueError:
                                    score = score_text

                            # Store in same format as results
                            current_question["evaluations"][eval_type] = {
                                "score": score,
                                "explanation": "",  # Debug logs don't have explanations
                                "appeal_evaluation": None,
                            }

        # Add last question if exists and we haven't hit summary
        if current_question:
            questions.append(current_question)

        return questions

    def _calculate_statistics(self, logs: List[Dict]) -> Dict:
        """Calculate overall statistics for the experiment"""
        stats = {
            "total_questions": 0,
            "successful_runs": 0,
            "correct": 0,
            "errors": 0,
            "overall_accuracy": {},
            "metadata_breakdown": {},
        }

        # Find the experiment summary section
        summary_start = None
        for i, log in enumerate(logs):
            if log.get("message") == "=== EXPERIMENT COMPLETE ===":
                summary_start = i
                break

        if summary_start is None:
            return stats

        # Parse the summary statistics
        current_section = None  # 'overall', 'domain', or 'level'
        current_domain = None
        current_level = None

        for log in logs[summary_start:]:
            message = log.get("message", "")

            # Extract basic stats
            if "Total questions processed:" in message:
                stats["total_questions"] = int(message.split(":")[-1].strip())
            elif "Successful runs:" in message:
                stats["successful_runs"] = int(message.split(":")[-1].strip())
            elif "Errors:" in message:
                stats["errors"] = int(message.split(":")[-1].strip())

            # Track sections
            elif "Question Domain Breakdown:" in message:
                current_section = "domain"
                current_domain = None
            elif "--- question_domain:" in message:
                current_section = "domain"
                current_domain = None
            elif "--- question_level:" in message:
                current_section = "level"
                current_level = None

            # Parse domain/level headers
            elif current_section in ["domain", "level"] and ": ------------------------------------" in message:
                name = message.split(":")[0].strip()
                if "(N=" in message:
                    count = int(message.split("(N=")[1].split(")")[0])
                    if current_section == "domain":
                        current_domain = name
                        if current_domain not in stats["domain_breakdown"]:
                            stats["domain_breakdown"][current_domain] = {"count": count, "evaluators": {}}
                    else:
                        current_level = name
                        if current_level not in stats["level_breakdown"]:
                            stats["level_breakdown"][current_level] = {"count": count, "evaluators": {}}

            # Parse evaluator scores
            elif message.strip().startswith("LLMAsJudgeEvaluationTool"):
                parts = message.strip().split(":")
                if len(parts) == 2:
                    eval_name = parts[0].strip()
                    score_part = parts[1].strip()
                    try:
                        score = float(score_part.split("±")[0].strip())
                        if current_section == "domain" and current_domain:
                            stats["domain_breakdown"][current_domain]["evaluators"][eval_name] = score
                        elif current_section == "level" and current_level:
                            stats["level_breakdown"][current_level]["evaluators"][eval_name] = score
                        else:
                            stats["overall_accuracy"][eval_name] = score
                    except (ValueError, IndexError):
                        continue

        return stats

    def _clean_id(self, id_str: str) -> str:
        """Clean question ID by removing '=' characters"""
        return id_str.replace("=", "") if id_str else ""

    async def _get_question_summary(self, question: Dict) -> str:
        """Generate a summary of the question's performance using the logs"""
        # Determine if question was correct. be strict. if any of the evaluators is not 100, then the question is not correct.
        is_correct = True
        for result in question.get("evaluations", {}).values():
            if str(result["score"]).lower() != "true" and str(result["score"]) != "100":
                # check if appeal_evaluation is true
                if result.get("appeal_evaluation", {}).get("should_override", False):
                    continue
                else:
                    is_correct = False
                    break

        # If LLM is not available, return a basic summary
        if self.llm_error_analysis is None:
            logger.warning("LLM not available for generating summary - using basic template")
            if is_correct:
                return """**✅ Success Analysis (Basic - LLM Not Available):**
• Question was processed successfully
• Achieved passing scores on evaluations
• Matched expected answer format and content"""
            else:
                return """**❌ Error Analysis (Basic - LLM Not Available):**
• Question did not achieve passing scores
• Review logs above for specific error messages
• Compare gold and predicted answers for discrepancies"""

        # Format the question's logs for better context
        log_context = "\n".join(f"{log['level']} {log['name']} {log['message']}" for log in question["logs"])

        try:
            if not is_correct:
                # Get response from LLM
                response = await self.llm_error_analysis.ainvoke(
                    {
                        "logs": log_context,
                        "gold_answer": question["gold_answer"],
                        "predicted_answer": question["predicted_answer"],
                        "is_correct": is_correct,
                        "scores": {
                            evaluator: question["evaluations"][evaluator]["score"]
                            for evaluator in question["evaluations"].keys()
                        },
                    }
                )

                formatted_response = f"""
**Error Type:** {response.error_type}

**Pipeline Stage:** {response.pipeline_stage}

<details><summary>More Details</summary>
Within Reason: <span style="color: {"green" if response.within_reason else "red"}">{response.within_reason}</span>

{response.error_description}


{response.improvement_suggestion}
</details>"""
            else:
                formatted_response = f"""
**Agent Gave Correct Answer:**
{question["predicted_answer"]}"""
            return formatted_response

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Return placeholder if LLM fails
            if is_correct:
                return """**✅ Success Analysis (PLACEHOLDER - LLM Error):**
• Successfully processed all objectives without errors
• Achieved perfect scores across all evaluators
• Matched the expected answer format and content"""
            else:
                return """**❌ Error Analysis (PLACEHOLDER - LLM Error):**
• Found issues in data processing stage
• Encountered errors during objective merging
• Failed to match the expected answer format"""

    def _display_merged_tables(self, merged_dir: str) -> str:
        """Display merged tables from the specified directory"""
        try:
            if not self.current_experiment:
                return ""

            display = "\n### 💾 Schema and Merged Tables\n\n"

            # First display schema if available
            if self.current_experiment.get("questions", []):
                current_question = self.current_experiment["questions"][0]  # Get first question for schema
                if "schema" in current_question:
                    schema = current_question["schema"]
                    display += "#### 📋 Schema Information\n\n"
                    display += "<details>\n<summary><strong>Schema Details</strong></summary>\n\n"

                    # Add schema statistics
                    display += f"- Generated Classes: {schema.get('generated_classes', 'N/A')}\n"
                    display += f"- Total Fields: {schema.get('total_fields', 'N/A')}\n\n"

                    # Add schema object details if available
                    if "schema_object" in schema and "classes" in schema["schema_object"]:
                        display += "**Schema Classes:**\n\n"
                        for class_def in schema["schema_object"]["classes"]:
                            display += f"*Class: {class_def['name']}*\n\n"
                            display += "<table style='width:100%; border-collapse:collapse'>\n"
                            display += "<tr style='background-color:#374151'><th>Field</th><th>Type</th><th>Description</th><th>Extraction Guideline</th></tr>\n"
                            for field in class_def["fields"]:
                                display += f"<tr><td>{field['name']}</td><td><code>{field['data_type']}</code></td><td>{field['description']}</td><td>{field['extraction_guideline']}</td></tr>\n"
                            display += "</table>\n\n"

                    display += "</details>\n\n"
                    breakpoint()

            if not os.path.exists(merged_dir):
                return display

            display += "#### 📊 Extracted Tables\n\n"

            # Get all CSV files in the directory
            # if a folder, get all csv files in the folder.
            csv_files = glob.glob(os.path.join(merged_dir, "*.csv"))
            if not csv_files:
                return display

            for csv_file in sorted(csv_files):
                table_name = os.path.basename(csv_file).replace(".csv", "")
                try:
                    df = pd.read_csv(csv_file)
                    # if table name starts with "extracted_" process the table differently
                    if table_name.startswith("extracted_"):
                        # go through every cell in the dataframe, turn string into json
                        for col in df.columns:
                            for i, val in enumerate(df[col]):
                                if col != "__metadata__":
                                    try:
                                        json_obj = eval(val)  # keys: reasoning, value, citation, is_explicit
                                        cell_str = f"<div class='cell-value'>{json_obj['value']}</div><div class='cell-tag'>{'Explicit' if json_obj['is_explicit'] else 'not Explicit'}</div><div class='cell-reasoning'>{json_obj['reasoning']}</div>"
                                        df.at[i, col] = cell_str
                                    except Exception as e:
                                        logger.error(f"Error processing cell {col} in table {table_name}: {e}")
                                        pass
                    display += f"<details>\n<summary><strong>📄 {table_name} | {df.shape[0]} rows x {df.shape[1]} columns</strong></summary>\n\n"
                    # Convert DataFrame to markdown table with wrapped text
                    table_html = df.to_html(
                        index=False,
                        classes=["dataframe"],
                        float_format=lambda x: "%.3f" % x if isinstance(x, float) else x,
                        escape=False,
                    )

                    # Add CSS for table formatting
                    display += """<style>
                                .dataframe {
                                    width: 100%;
                                    border-collapse: collapse;
                                    margin: 10px 0;
                                }
                                .dataframe th, .dataframe td {
                                    padding: 12px;
                                    border: 1px solid #ddd;
                                    max-width: 300px;
                                    word-wrap: break-word;
                                    white-space: pre-wrap;
                                    vertical-align: top;
                                    line-height: 1.4;
                                }
                                .dataframe th {
                                    background-color: #374151;
                                    color: white;
                                    font-weight: bold;
                                    text-align: left;
                                }
                                .dataframe td {
                                    background-color: #1F2937;
                                }
                                .dataframe tr:hover td {
                                    background-color: #374151;
                                }
                                .cell-value {
                                    font-weight: bold;
                                    margin-bottom: 8px;
                                    color: #1a73e8;
                                }
                                .cell-tag {
                                    display: inline-block;
                                    padding: 2px 6px;
                                    background-color: #4F45E4;
                                    color: #1967d2;
                                    border-radius: 4px;
                                    font-size: 0.9em;
                                    margin-bottom: 8px;
                                }
                                .cell-reasoning {
                                    color: #3c4043;
                                    font-style: italic;
                                    border-left: 3px solid #dadce0;
                                    padding-left: 8px;
                                }
                                .cell-content {
                                    white-space: pre-wrap;
                                    word-break: break-word;
                                }
                                </style>"""
                    display += table_html + "\n\n"
                    display += "</details>\n\n"
                except Exception as e:
                    logger.error(f"Error reading table {table_name}: {e}")
                    continue

            return display
        except Exception as e:
            logger.error(f"Error displaying merged tables: {e}")
            return ""

    def citation_from_field(field, row):
        for field_name, field_extraction in row.items():
            if field_name == field:
                return field_extraction["citation"]
        return None

    async def get_question_info(self, question_idx: int) -> Tuple[str, str, str]:
        """Get detailed information about a specific question"""
        if not self.current_experiment or question_idx >= len(self.current_experiment["questions"]):
            return "❌ Invalid question number", "", "No data loaded"

        question = self.current_experiment["questions"][question_idx]

        # Create summary
        summary = f"## Question {question_idx + 1} Summary\n\n"

        summary += f"**ID:** {self._clean_id(question['id'])}\n\n"

        # Add metadata fields dynamically
        if question.get("metadata"):
            summary += "### Metadata\n\n"
            for key, value in question["metadata"].items():
                # Convert key from snake_case to Title Case for display
                display_key = key.replace("_", " ").title()
                summary += f"- **{display_key}:** {value}\n\n"

        # Add question text if available
        if question.get("question_text"):
            summary += "### ❓ Question\n\n"
            summary += "<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 8px; border-radius: 4px; color: #24292e; font-family: monospace;'>"
            summary += question["question_text"]
            summary += "</pre>\n\n"

        # Add answers in a clear format
        summary += "### 🧠 Answers\n\n"
        summary += "<table style='width: 100%; table-layout: fixed;'>\n"
        summary += "<tr><th style='width: 50%; padding: 8px;'>Gold Answer</th><th style='width: 50%; padding: 8px;'>Predicted Answer</th></tr>\n"
        summary += "<tr><td style='vertical-align: top; padding: 8px;'>\n\n"
        summary += (
            "<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 8px; border-radius: 4px; color: #24292e; font-family: monospace;'>"
            + question["gold_answer"]
            + "</pre>\n\n</td>"
        )
        summary += "<td style='vertical-align: top; padding: 8px;'>\n\n"
        summary += (
            "<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 8px; border-radius: 4px; color: #24292e; font-family: monospace;'>"
            + question["predicted_answer"]
            + "</pre>\n\n</td></tr>\n</table>\n\n"
        )

        # Add evaluations with color highlighting
        summary += "### 📋 Evaluations\n\n"
        for eval_type, eval_data in question["evaluations"].items():
            # Handle both old and new evaluation formats
            if isinstance(eval_data, dict):
                result = eval_data["score"]
                explanation = eval_data.get("explanation", "")
                engine = eval_data.get("engine", "")
                prompt_file = eval_data.get("prompt_file", "")
                appeal_evaluation = eval_data.get("appeal_evaluation", {})

            else:
                result = eval_data
                explanation = ""
                engine = ""
                prompt_file = ""

            result_str = str(result).lower()
            if result_str == "true" or result_str == "100" or result_str == "100.0":
                summary += f"**{eval_type}:** <span style='color: green; background-color: #e6ffe6; padding: 2px 6px; border-radius: 3px;'>{result}</span>\n\n"
            elif result_str == "false" or result_str == "0" or result_str == "0.0":
                summary += f"**{eval_type}:** <span style='color: red; background-color: #ffe6e6; padding: 2px 6px; border-radius: 3px;'>{result}</span>\n\n"
            else:
                summary += f"**{eval_type}:** {result}\n\n"

            if appeal_evaluation:
                if appeal_evaluation.get("should_override", False):
                    summary += "<span style='color: orange; padding: 2px 6px; border-radius: 3px;'>EVALUATION OVERRIDDEN</span>\n\n"
                else:
                    summary += (
                        "<span style='padding: 2px 6px; border-radius: 3px;'>EVALUATION NOT OVERRIDDEN</span>\n\n"
                    )

            if explanation:
                summary += f"<details><summary>Explanation</summary>\n\n{explanation}\n\n"

                if appeal_evaluation:
                    summary += f"*Appeal Explanation: {appeal_evaluation.get('explanation', '')}*\n\n"
                if engine:
                    summary += f"*Engine: {engine}*\n\n"
                if prompt_file:
                    summary += f"*Prompt: {prompt_file}*\n\n"
                summary += "</details>\n\n"

        # Add LLM summary
        summary += "### ✨ LLM Analysis\n\n"
        summary += await self._get_question_summary(question)
        summary += "\n\n"

        # Add merged tables if available
        merged_dir = None
        for log in question.get("logs", []):
            if log.get("name") == "sliders.system" and "Merged tables directory path:" in log.get("message", ""):
                merged_dir = log["message"].split("Merged tables directory path:")[1].strip()

        if merged_dir:
            summary += self._display_merged_tables(merged_dir)
        elif question.get("merging", {}).get("merged_tables_dir_path"):
            # Try to get merged tables from results data
            summary += self._display_merged_tables(question["merging"]["merged_tables_dir_path"])

        # Add performance and quality metrics in a table structure at the bottom
        metrics_table = "<table style='width: 100%; border-collapse: collapse; margin: 20px 0;'>\n"

        # Add performance metrics if available
        if question.get("timing"):
            metrics_table += "<tr><th colspan='2' style='text-align: left; padding: 10px; background-color: #374151; color: white;'>⚡ Performance Metrics</th></tr>\n"
            timing = question["timing"]

            if "total_duration" in timing:
                metrics_table += f"<tr><td style='padding: 8px; border: 1px solid #ddd; width: 30%;'><strong>Total Time</strong></td><td style='padding: 8px; border: 1px solid #ddd;'>{timing['total_duration']:.2f}s</td></tr>\n"

            for phase in ["schema_generation", "schema_extraction", "table_merging", "answer_generation"]:
                if phase in timing:
                    phase_data = timing[phase]
                    phase_display = phase.replace("_", " ").title()

                    if isinstance(phase_data, dict):
                        if "generation_time" in phase_data:
                            metrics_table += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>{phase_display}</strong></td><td style='padding: 8px; border: 1px solid #ddd;'>{phase_data['generation_time']:.2f}s</td></tr>\n"
                        elif "merging_time" in phase_data:
                            metrics_table += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>{phase_display}</strong></td><td style='padding: 8px; border: 1px solid #ddd;'>{phase_data['merging_time']:.2f}s</td></tr>\n"
                        elif "answer_time" in phase_data:
                            metrics_table += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>{phase_display}</strong></td><td style='padding: 8px; border: 1px solid #ddd;'>{phase_data['answer_time']:.2f}s</td></tr>\n"

        # Add quality metrics if available
        if question.get("quality"):
            metrics_table += "<tr><th colspan='2' style='text-align: left; padding: 10px; background-color: #374151; color: white;'>📈 Quality Metrics</th></tr>\n"
            quality = question["quality"]

            if "data_completeness_score" in quality:
                score = quality["data_completeness_score"] * 100
                metrics_table += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>Data Completeness</strong></td><td style='padding: 8px; border: 1px solid #ddd;'>{score:.2f}%</td></tr>\n"

            if "tables_with_data" in quality:
                metrics_table += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>Tables with Data</strong></td><td style='padding: 8px; border: 1px solid #ddd;'>{quality['tables_with_data']}</td></tr>\n"

            if "empty_tables" in quality:
                metrics_table += f"<tr><td style='padding: 8px; border: 1px solid #ddd;'><strong>Empty Tables</strong></td><td style='padding: 8px; border: 1px solid #ddd;'>{quality['empty_tables']}</td></tr>\n"

        metrics_table += "</table>\n\n"

        # Only add the metrics section if we have either performance or quality metrics
        if question.get("timing") or question.get("quality"):
            summary += "\n\n### 📊 Metrics\n\n"
            summary += metrics_table

        # Format logs
        logs_display = ""
        if question.get("logs", []):
            logs_display = "### 📝 Detailed Logs\n\n"

        # Organize logs by module name
        logs_by_module = {}
        for log in question["logs"]:
            module_name = log.get("name", "unknown")
            if module_name not in ["sliders.experiments.loong"]:  # All module names to skip
                if module_name not in logs_by_module:
                    logs_by_module[module_name] = []
                logs_by_module[module_name].append(log)

        # Sort modules to ensure consistent order
        for module_name in sorted(logs_by_module.keys()):
            module_logs = logs_by_module[module_name]

            # Create collapsible section for each module
            logs_display += f"<details>\n<summary><strong>📦 {module_name}</strong></summary>\n\n"

            # Add logs for this module
            for log in module_logs:
                time = log.get("time", "").split()[1]  # Just show time part
                level = log.get("level", "")
                level_emoji = {
                    "INFO": "ℹ️",
                    "WARNING": "⚠️",
                    "ERROR": "❌",
                }.get(level, "📝")

                if module_name == "sliders.llm_models":
                    message_text = log.get("message", "")
                    if message_text.startswith("def ExtractionOutput("):
                        message_text = message_text[len("def ExtractionOutput(") : -len("): pass")]
                        message_text = f"<span style='color: #007bff;'>Extracted Schema</span> \n\n<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 8px; border-radius: 4px; color: #24292e; font-family: monospace;'>{message_text}</pre>"
                        log["message"] = message_text

                if module_name == "sliders.modules.merge_schema":
                    message_text = log.get("message", "")
                    if message_text.startswith("Processing objective "):
                        message_text = message_text[len("Processing objective ") + 2 :]
                        message_text = f"***<span style='color: #007bff;'>OBJECTIVE</span>*** \n{message_text}"
                        log["message"] = message_text
                    elif "is necessary according to LLM check" in message_text:
                        if "yes" in message_text:
                            message_text = message_text.replace("yes", "<span style='color: green;'>yes</span>")
                        elif "no" in message_text:
                            message_text = message_text.replace("no", "<span style='color: red;'>no</span>")
                        log["message"] = message_text
                    elif message_text.startswith("Executing SQL"):
                        prefix = message_text.find(":")
                        message_text = f"{message_text[: prefix + 1]} \n\n<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f6f8fa; padding: 8px; border-radius: 4px; color: #24292e; font-family: monospace;'>{message_text[prefix + 1 :]}</pre>"
                        log["message"] = message_text
                    elif message_text.startswith("Debug objectives:"):
                        message_text = message_text[len("Debug objectives:") :]
                        message_text = (
                            f"**<span style='color: #007bff;'>Merged Table. </span>** Objectives: {message_text}"
                        )
                        log["message"] = message_text

                if module_name == "sliders.system":
                    message_text = log.get("message", "")
                    if message_text.startswith("Merged tables directory path:"):
                        message_text = message_text[len("Merged tables directory path:") :]
                        message_text = f"**<span style='color: #007bff;'>Tables Directory Path</span>** *<span style='color: #ffc600;'>{message_text}</span>*"
                        log["message"] = message_text

                logs_display += f"**{time} {level_emoji}** {log.get('message', '')}\n\n"

            logs_display += "</details>\n\n"

        status = f"Showing question {question_idx + 1} of {len(self.current_experiment['questions'])}"

        return summary, logs_display, status

    def get_statistics_display(self) -> str:
        """Get formatted statistics display"""
        if not self.current_experiment:
            return "No experiment loaded"

        stats = self.current_experiment["statistics"]
        display = f"""## 📈 Experiment Statistics

**Questions:**
- Total: {stats["total_questions"]}
- Successful: {stats["successful_runs"] if "successful_runs" in stats else "Information not available"}
- Errors: {stats["errors"] if "errors" in stats else "Information not available"}
- Fully Correct: {stats["correct"]}

**Overall Accuracy:**"""

        # Add overall evaluator scores with details
        for eval_name, eval_stats in stats["overall_accuracy"].items():
            if isinstance(eval_stats, dict):
                display += f"\n\n**{eval_name}:**"
                if "accuracy" in eval_stats:
                    display += f"\n- Accuracy: {eval_stats['accuracy']:.2f}%"
                if "average_score" in eval_stats:
                    display += f"\n- Average Score: {eval_stats['average_score']:.2f}"
                if "total" in eval_stats:
                    display += f"\n- Total Questions: {eval_stats['total']}"
                if "correct" in eval_stats:
                    display += f"\n- Correct Answers: {eval_stats['correct']}"
            else:
                display += f"\n- {eval_name}: {eval_stats:.2f}"

        # Add metadata breakdowns
        if "metadata_breakdown" in stats:
            for metadata_key, metadata_values in stats["metadata_breakdown"].items():
                # Convert metadata_key from snake_case to Title Case
                display_key = metadata_key.replace("_", " ").title()
                display += f"\n\n## {display_key} Breakdown"

                # Sort values by total count for better organization
                sorted_values = sorted(metadata_values.items(), key=lambda x: x[1]["total"], reverse=True)

                for value, value_stats in sorted_values:
                    display += f"\n\n### {value} (N={value_stats['total']})"
                    display += f"\n- Overall Accuracy: {value_stats.get('accuracy', 0):.2f}%"
                    display += f"\n- Correct: {value_stats['correct']} / {value_stats['total']}"

                    # Add evaluator scores
                    if value_stats.get("evaluator_scores"):
                        display += "\n\n**Evaluator Scores:**"
                        for eval_name, eval_scores in value_stats["evaluator_scores"].items():
                            if "accuracy" in eval_scores:
                                display += f"\n- {eval_name}: {eval_scores['accuracy']:.2f}%"

                    # Add examples
                    if value_stats.get("examples"):
                        examples = value_stats["examples"]

                        # Create a details section for examples
                        display += "\n\n<details><summary>📊 Examples</summary>\n\n"

                        if examples.get("success", []):
                            display += "**✅ Success Examples**\n\n"
                            for example in examples["success"]:
                                display += f"**Question:** {example['question']}\n\n"
                                display += f"**Gold Answer:** {example['gold_answer']}\n\n"
                                display += f"**Predicted Answer:** {example['predicted_answer']}\n\n"
                                display += "**Evaluations:**\n"
                                for eval_name, eval_data in example["evaluations"].items():
                                    display += f"- {eval_name}: {eval_data.get('correct')} "
                                    if eval_data.get("explanation"):
                                        display += f"({eval_data['explanation']})"
                                    display += "\n"
                                display += "\n---\n\n"

                        if examples.get("failure", []):
                            display += "**❌ Failure Examples**\n\n"
                            for example in examples["failure"]:
                                display += f"**Question:** {example['question']}\n\n"
                                display += f"**Gold Answer:** {example['gold_answer']}\n\n"
                                display += f"**Predicted Answer:** {example['predicted_answer']}\n\n"
                                display += "**Evaluations:**\n"
                                for eval_name, eval_data in example["evaluations"].items():
                                    display += f"- {eval_name}: {eval_data.get('correct')} "
                                    if eval_data.get("explanation"):
                                        display += f"({eval_data['explanation']})"
                                    display += "\n"
                                display += "\n---\n\n"

                        display += "</details>\n"

        return display

    def search_questions(self, search_term: str) -> str:
        """Search for questions containing specific terms"""
        if not self.current_experiment:
            return "❌ No experiment loaded"

        if not search_term.strip():
            return "Please enter a search term"

        matching_questions = []
        for i, question in enumerate(self.current_experiment["questions"]):
            # Search in question content and answers
            searchable_content = f"{question['id']} {' '.join(str(v) for v in question['metadata'].values())} {question['gold_answer']} {question['predicted_answer']}"
            if search_term.lower() in searchable_content.lower():
                # Determine status based on evaluations
                status = "❓"
                if question["evaluations"].get("hard_evaluator", "").lower() == "true":
                    status = "✅"
                elif question["evaluations"].get("hard_evaluator", "").lower() == "false":
                    status = "❌"

                # Format metadata for display
                metadata_str = ", ".join(f"{k}: {v}" for k, v in question["metadata"].items())
                matching_questions.append((i, f"[{metadata_str}] {question['id']}", status))

        if not matching_questions:
            return f"No questions found containing '{search_term}'"

        result = f"Found {len(matching_questions)} questions containing '{search_term}':\n\n"
        for idx, preview, status in matching_questions:
            result += f"**Question {idx + 1}** {status}: {preview}\n\n"

        return result

    def get_parse_errors(self) -> str:
        """Get formatted display of parse errors"""
        if not self.current_experiment or "parse_errors" not in self.current_experiment:
            return "No experiment loaded or no parse error information available"

        errors = self.current_experiment["parse_errors"]
        if not errors:
            return "✅ No parsing errors found"

        display = f"## ⚠️ Found {len(errors)} Parse Errors\n\n"
        for i, err in enumerate(errors[:10], 1):  # Show first 10 errors
            display += f"### Error {i}\n"
            display += f"**Lines {err['start_line']}-{err['end_line']}:** {err['error']}\n"
            display += f"**Preview:**\n```json\n{err['text_preview']}\n```\n\n"

        if len(errors) > 10:
            display += f"\n... and {len(errors) - 10} more errors"

        return display

    def get_question_choices(self) -> List[str]:
        """Get list of questions with their evaluator scores"""
        if not self.current_experiment:
            logger.warning("No experiment loaded")
            return []

        choices = []
        logger.info(f"Found {len(self.current_experiment['questions'])} questions")

        for i, question in enumerate(self.current_experiment["questions"]):
            # Get evaluator score
            evaluations = question.get("evaluations", {})
            scores = [str(eval_info.get("score", "N/A")) for eval_info in evaluations.values()]

            # Format the choice string
            choice = f"Question {i + 1} (Score: {'/'.join(scores)})"
            choices.append(choice)

        logger.info(f"Returning {len(choices)} choices: {choices}")
        return choices


# Directory scanning functions
def get_experiment_files(require_debug_log: bool = False, require_results: bool = False) -> List[ExperimentFiles]:
    """Get list of experiment files from both logs and results directories

    Args:
        require_debug_log: If True, only return experiments with debug logs
        require_results: If True, only return experiments with results files

    Returns:
        List of ExperimentFiles objects containing matched file information
    """
    try:
        # Get paths
        log_dir = os.path.join(CURRENT_DIR, "../logs/experiments")
        results_dir = os.path.join(CURRENT_DIR, "../results")

        # Initialize empty experiment files dict
        experiment_files: Dict[str, ExperimentFiles] = {}

        # Helper function to extract timestamp from filename
        def extract_timestamp(filename: str) -> str:
            """Extract timestamp from end of filename (e.g. 20240322_123456)"""
            # Remove extension
            base = os.path.splitext(filename)[0]
            # Find last sequence of digits with optional underscore
            parts = base.split("_")
            if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                return f"{parts[-2]}_{parts[-1]}"
            elif parts[-1].isdigit():
                return parts[-1]
            return base  # Fallback to full base name

        # Process debug logs
        if os.path.exists(log_dir):
            debug_logs = glob.glob(os.path.join(log_dir, "*.jsonl"))
            for log_path in debug_logs:
                # Extract timestamp from filename
                timestamp = extract_timestamp(os.path.basename(log_path))
                experiment_files[timestamp] = ExperimentFiles(timestamp=timestamp, debug_log=log_path)

        # Process results files
        if os.path.exists(results_dir):
            results = glob.glob(os.path.join(results_dir, "*.json"))
            for result_path in results:
                # Extract timestamp from filename
                timestamp = extract_timestamp(os.path.basename(result_path))
                if timestamp in experiment_files:
                    experiment_files[timestamp].results_file = result_path
                else:
                    experiment_files[timestamp] = ExperimentFiles(timestamp=timestamp, results_file=result_path)

        # Filter based on requirements
        filtered_files = []
        for exp_file in experiment_files.values():
            if require_debug_log and not exp_file.has_debug_log:
                continue
            if require_results and not exp_file.has_results:
                continue
            filtered_files.append(exp_file)

        if not filtered_files:
            if require_debug_log and require_results:
                return [ExperimentFiles(timestamp="No experiments with both debug logs and results found")]
            elif require_debug_log:
                return [ExperimentFiles(timestamp="No experiments with debug logs found")]
            elif require_results:
                return [ExperimentFiles(timestamp="No experiments with results found")]
            else:
                return [ExperimentFiles(timestamp="No experiments found")]

        return sorted(filtered_files, key=lambda x: x.timestamp, reverse=True)

    except Exception as e:
        logger.error(f"Error scanning directories: {e}")
        return [ExperimentFiles(timestamp=f"Error scanning directories: {str(e)}")]


# Initialize viewer
viewer = ErrorAnalysisViewer()

# Create Gradio interface
with gr.Blocks(title="Error Analysis Viewer", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🔍 Error Analysis Viewer")
    gr.Markdown("Analyze experiment results and debug logs")

    with gr.Tab("📁 Load Experiment"):
        gr.Markdown("## Select Experiment")
        with gr.Row():
            with gr.Column(scale=3):
                experiment_dropdown = gr.Dropdown(
                    label="Select Experiment",
                    choices=[exp.display_name for exp in get_experiment_files()],
                    value=None,
                    interactive=True,
                )
            with gr.Column(scale=1):
                with gr.Row():
                    require_debug = gr.Checkbox(label="Require Debug Logs", value=False)
                    require_results = gr.Checkbox(label="Require Results", value=False)

        with gr.Row():
            load_experiment_btn = gr.Button("Load Selected Experiment", variant="primary")
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm")

        experiment_status = gr.Textbox(label="Load Status", interactive=False)
        parse_errors_output = gr.Markdown(label="Parse Errors")

        # Event handlers
        def refresh_experiments(require_debug: bool, require_results: bool):
            experiments = get_experiment_files(require_debug_log=require_debug, require_results=require_results)
            return gr.Dropdown(choices=[exp.display_name for exp in experiments])

        def load_experiment_file(filename: str, require_debug: bool, require_results: bool):
            if not filename or filename.startswith(("No ", "Error")):
                return "❌ No valid file selected", ""

            try:
                # Get the experiment files again to find the selected one
                experiments = get_experiment_files(require_debug_log=require_debug, require_results=require_results)
                selected_exp = next((exp for exp in experiments if exp.display_name == filename), None)

                if not selected_exp:
                    return "❌ Could not find selected experiment", ""

                # Load experiment with both debug log and results if available
                status = viewer.load_experiment(selected_exp)

                errors = viewer.get_parse_errors() if selected_exp.has_debug_log else ""
                return status, errors

            except Exception as e:
                return f"❌ Error loading experiment: {str(e)}", ""

        # Connect event handlers
        refresh_btn.click(
            fn=refresh_experiments, inputs=[require_debug, require_results], outputs=[experiment_dropdown]
        )

        require_debug.change(
            fn=refresh_experiments, inputs=[require_debug, require_results], outputs=[experiment_dropdown]
        )

        require_results.change(
            fn=refresh_experiments, inputs=[require_debug, require_results], outputs=[experiment_dropdown]
        )

        load_experiment_btn.click(
            fn=load_experiment_file,
            inputs=[experiment_dropdown, require_debug, require_results],
            outputs=[experiment_status, parse_errors_output],
        )

    with gr.Tab("📊 Statistics"):
        stats_button = gr.Button("🔄 Refresh Statistics", variant="primary")
        stats_output = gr.Markdown(label="Statistics")

        with gr.Row():
            debug_logs_btn = gr.Button("🔍 Debug Logs in Breakpoint", variant="secondary")
            debug_results_btn = gr.Button("🔍 Debug Results in Breakpoint", variant="secondary")

        def debug_logs():
            """Debug function to examine logs"""
            if not viewer.current_experiment:
                return "No experiment loaded"

            logger.info("Examining debug logs... use 'c' to exit")
            # all_questions = viewer.current_experiment["questions"]
            breakpoint()
            return "Debug session completed"

        def debug_results():
            """Debug function to examine results"""
            if not viewer.current_experiment:
                return "No experiment loaded"

            logger.info("Examining results... use 'c' to exit")
            if viewer.current_results:
                breakpoint()
            return "Debug session completed"

        debug_logs_btn.click(fn=debug_logs, outputs=[stats_output])
        debug_results_btn.click(fn=debug_results, outputs=[stats_output])
        stats_button.click(fn=viewer.get_statistics_display, outputs=[stats_output])

    with gr.Tab("📂 Browse Questions"):
        with gr.Row():
            with gr.Column(scale=1):
                question_dropdown = gr.Dropdown(
                    label="Select Question",
                    choices=["No experiment loaded"],
                    value="No experiment loaded",
                    interactive=True,
                    allow_custom_value=False,
                )
                view_button = gr.Button("🔍 View Question", variant="primary")

            with gr.Column(scale=3):
                status_output = gr.Textbox(label="Status", interactive=False, max_lines=1)

        summary_output = gr.Markdown(label="Question Summary")
        logs_output = gr.Markdown(label="Detailed Logs")

        # Update question choices when experiment is loaded
        async def load_experiment_and_update(filename: str, require_debug: bool, require_results: bool):
            # First load the experiment
            status, errors = load_experiment_file(filename, require_debug, require_results)
            logger.info(f"Load experiment status: {status}")

            # Then get the question choices
            choices = viewer.get_question_choices()
            logger.info(f"Got choices: {choices}")

            if not choices:
                choices = ["No questions available"]
                logger.warning("No choices available, using default")

            # Return updates
            return (
                status,  # experiment_status
                errors,  # parse_errors_output
                {
                    "choices": choices,
                    "value": choices[0] if choices else "No questions available",
                    "__type__": "update",
                },  # question_dropdown
                "",  # summary_output
                "",  # logs_output
                "Please select a question",  # status_output
            )

        def view_selected_question(selected):
            logger.info(f"Selected question: {selected}")
            if not selected or selected in ["No experiment loaded", "No questions available"]:
                return "", "", "No question selected"
            # Extract question number from the selection string
            try:
                question_idx = int(selected.split()[1]) - 1
                logger.info(f"Extracted question index: {question_idx}")
                return asyncio.run(viewer.get_question_info(question_idx))
            except Exception as e:
                logger.error(f"Error processing selection: {e}")
                return "", "", f"Error processing selection: {str(e)}"

        # Update choices when experiment is loaded
        load_experiment_btn.click(
            fn=load_experiment_and_update,
            inputs=[experiment_dropdown, require_debug, require_results],
            outputs=[
                experiment_status,
                parse_errors_output,
                question_dropdown,
                summary_output,
                logs_output,
                status_output,
            ],
        )

        view_button.click(
            fn=view_selected_question, inputs=[question_dropdown], outputs=[summary_output, logs_output, status_output]
        )

    with gr.Tab("🔎 Search"):
        with gr.Row():
            search_input = gr.Textbox(label="Search Term", placeholder="Enter keywords to search in questions", lines=1)
            search_button = gr.Button("🔍 Search", variant="primary")

        search_output = gr.Markdown(label="Search Results")

        search_button.click(fn=viewer.search_questions, inputs=[search_input], outputs=[search_output])

    with gr.Tab("ℹ️ Info"):
        gr.Markdown(f"""
        ## About This Tool
        
        This Error Analysis Viewer helps you analyze experiment results and debug logs with the following features:
        
        **Features:**
        - **📁 Experiment Loading**: Load and analyze experiment debug logs
        - **📊 Statistics**: View overall experiment performance metrics
        - **🔍 Question Analysis**: Examine individual questions in detail
          - ** ✏️ LLM Error Summary**: Detailed explanations of the error and where it may have occured.
        - **🔎 Search**: Find specific questions or patterns
        
        **Available Experiments:**
        - {len(get_experiment_files())} experiment logs in `logs/experiments/`
        
        **Data Sources:**
        You can load experiments using:
        - **📊 Results File (.json)**: Contains final outcomes and evaluations
          - Question metadata and classification
          - Gold and predicted answers
          - Evaluation scores and explanations
          - Performance metrics and quality scores
          
          - Best for quick analysis of outcomes
        
        - **📝 Debug Log File (.jsonl)**: Contains detailed execution logs
          - Step-by-step pipeline execution
          - Intermediate outputs and decisions
          - Error messages and warnings
          - Schema generation and extraction details
          
          - Best for debugging and understanding the process
          - Logs are converted from strings to JSON objects. This is has a 97-99% parse rate and can lead to missing information.
          - Only generates experiment results if the entire experiment is run successfully.
        
        - **Combined Analysis**: If both files are available, both files are used to generate the results.
        
        **Data Organization:**
        - Each experiment contains multiple questions
        - Questions are analyzed with different evaluators
        - Detailed logs are available for debugging
        
        **How to Use:**
        1. Load an experiment from the dropdown menu
           - Use checkboxes to filter for specific file types
           - Results only: Quick analysis of outcomes and experiment-level statistics
           - Debug logs only: Detailed process inspection
           - Both: Complete analysis capabilities
        2. View overall statistics and performance breakdowns
        3. Browse individual questions to see:
           - Question details and metadata
           - Gold vs predicted answers
           - LLM error analysis
           - Performance and quality metrics (if available)
           - Detailed logs by module (if available)
        4. Search for specific patterns or issues
        """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7862, share=False, debug=True)
