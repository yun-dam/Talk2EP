import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sliders.log_utils import logger
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(CURRENT_DIR, ".env"))
RESULTS_DIR = os.environ["SLIDERS_RESULTS"]
PROMPT_LOGS_DIR = os.path.join(os.environ["SLIDERS_LOGS_DIR"], "prompt_logs")


def extract_action_columns(raw: str) -> List[Dict[str, Any]]:
    """Extract table data from tuple-formatted text.

    Converts lines like:
        ('col1', 'col2', 'col3')
        (val1, val2, val3)
        (val4, val5, val6)

    Into a list of dictionaries for DataFrame creation.
    """
    # Keep only tuple lines
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip().startswith("(")]

    if not lines:
        return []

    try:
        # First tuple is the header
        header = eval(lines[0], {"__builtins__": {}})
        row_lines = lines[1:]  # the data rows

        if not row_lines:
            return []

        # Parse each row individually to handle errors gracefully
        env = {"__builtins__": {}, "nan": math.nan, "list": list}
        records = []

        for row_line in row_lines:
            try:
                row = eval(row_line, env, {})
                records.append(dict(zip(header, row)))
            except Exception as e:
                # Skip malformed rows
                logger.debug(f"Skipping malformed row: {row_line[:100]}... Error: {e}")
                continue

        return records
    except Exception as e:
        logger.error(f"Error parsing table data: {e}")
        return []


def parse_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                obj["__line_number"] = line_number
                events.append(obj)
            except json.JSONDecodeError as e:
                events.append(
                    {
                        "timestamp": None,
                        "level": "ERROR",
                        "event": "parse_error",
                        "error": str(e),
                        "raw": line,
                        "__line_number": line_number,
                    }
                )
    return events


def iso_to_dt(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # 2025-09-25T00:26:38.132522Z
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def load_run_data(
    results_file: str,
) -> Tuple[Optional[Dict[str, Dict[str, Any]]], Optional[pd.DataFrame], Optional[str]]:
    """Load results and prompt logs for a single run.

    Returns:
        Tuple of (results_summary_by_question, paired_df, run_label)
    """
    if not os.path.exists(results_file):
        return None, None, None

    results_summary_by_question: Dict[str, Dict[str, Any]] = {}

    # Load results file
    try:
        with open(results_file, "r", encoding="utf-8") as rf:
            results_payload = json.load(rf)

        # Normalize different result file formats
        results_list = []
        metadata_list = []

        if isinstance(results_payload, list):
            if results_payload and isinstance(results_payload[0], list):
                results_list = [item for item in results_payload[0] if isinstance(item, dict)]
                # Check if second element contains metadata array
                if len(results_payload) > 1:
                    if isinstance(results_payload[1], list):
                        # Metadata is an array aligned by index with results
                        metadata_list = results_payload[1]
                    # elif isinstance(results_payload[1], dict):
                    #     # Metadata is a dict (keyed by id or question)
                    #     metadata_dict = results_payload[1]
            else:
                results_list = [item for item in results_payload if isinstance(item, dict)]
        elif isinstance(results_payload, dict):
            results_list = results_payload.get("results", [])
            metadata_payload = results_payload.get("metadata", {})
            if isinstance(metadata_payload, list):
                metadata_list = metadata_payload
            if results_list and len(results_list) > 0 and isinstance(results_list[0], list):
                results_list = [item for item in results_list[0] if isinstance(item, dict)]

        for idx, item in enumerate(results_list):
            if not isinstance(item, dict):
                continue
            q_text = item.get("question_id")
            if not q_text:
                continue

            if q_text in results_summary_by_question:
                continue

            # Extract metadata for this question
            q_metadata = {}

            # Try metadata_list first (aligned by index)
            if metadata_list and idx < len(metadata_list):
                metadata_item = metadata_list[idx]
                if isinstance(metadata_item, dict):
                    # Check for misc_question_metadata key
                    q_metadata = metadata_item.get("misc_question_metadata", metadata_item)

            # Fallback: metadata might be directly in the item
            if not q_metadata:
                q_metadata = item.get("metadata", {})

            # Extract common metadata fields
            domain = q_metadata.get("domain") or item.get("domain")
            question_type = q_metadata.get("question_type") or q_metadata.get("type") or item.get("question_type")
            question_level = q_metadata.get("question_level") or item.get("question_level")

            results_summary_by_question[q_text] = {
                "gold_answer": item.get("gold_answer"),
                "predicted_answer": item.get("predicted_answer"),
                "evaluation_tools": item.get("evaluation_tools") or {},
                "question_id": item.get("question_id"),
                "metadata": {
                    "domain": domain,
                    "question_type": question_type,
                    "question_level": question_level,
                },
            }

    except Exception as e:
        logger.error(f"Error loading results file {results_file}: {e}")
        return None, None, None

    # Locate corresponding prompt log file
    prompt_log_file: Optional[str] = None
    try:
        results_basename = os.path.basename(results_file).replace(".json", "")
        results_name_no_ext = results_basename.split("_")[-1]

        logs_dir = PROMPT_LOGS_DIR
        if os.path.isdir(logs_dir):
            prompt_log_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".jsonl")]

            exact_matches = [f for f in prompt_log_files if os.path.basename(f) == f"{results_name_no_ext}.jsonl"]

            if exact_matches:
                prompt_log_file = exact_matches[0]

    except Exception as e:
        logger.error(f"Error locating prompt log for {results_file}: {e}")
        return results_summary_by_question, None, os.path.basename(results_file)

    # Load prompt logs if found
    paired_df = None
    if prompt_log_file and os.path.exists(prompt_log_file):
        try:
            events = parse_jsonl_file(prompt_log_file)

            paired_df, _ = pair_calls(events)
        except Exception as e:
            logger.error(f"Error parsing prompt log {prompt_log_file}: {e}")

    run_label = os.path.basename(results_file).replace(".json", "")
    return results_summary_by_question, paired_df, run_label


def pair_calls(events: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    starts: Dict[str, Dict[str, Any]] = {}
    ends: Dict[str, Dict[str, Any]] = {}
    dataframe_logs: List[Dict[str, Any]] = []

    for e in events:
        request_id = e.get("request_id") or (e.get("metadata") or {}).get("request_id")
        if not request_id:
            # Try to infer from parent_run_id if present
            request_id = (e.get("metadata") or {}).get("parent_run_id")

        evt = e.get("event")

        # Handle dataframe_log events separately
        if evt == "dataframe_log":
            dataframe_logs.append(e)
            continue

        if not request_id:
            continue

        if evt == "llm_call_start":
            starts[request_id] = e
        elif evt == "llm_call_end":
            ends[request_id] = e

    paired_rows: List[Dict[str, Any]] = []
    unmatched_rows: List[Dict[str, Any]] = []

    # Build pairs
    all_ids = set(starts.keys()) | set(ends.keys())
    for rid in sorted(all_ids):
        start = starts.get(rid)
        end = ends.get(rid)
        if start and end:
            start_ts = iso_to_dt(start.get("timestamp"))
            end_ts = iso_to_dt(end.get("timestamp"))
            duration_ms: Optional[float] = None
            if start_ts and end_ts:
                duration_ms = (end_ts - start_ts).total_seconds() * 1000.0

            # Prompt extraction preference: prompt_file path; else use system/user/metadata.question
            prompt_file = start.get("prompt_file")
            system_message = start.get("system_message") or ""
            user_message = start.get("user_message") or ""
            question = (start.get("metadata") or {}).get("question") or ""
            question_id = (start.get("metadata") or {}).get("question_id") or ""
            # Prefer explicit messages; fallback to question if lengthy
            # combined = "\n\n".join([p for p in [system_message, user_message] if p])
            combined = f"System1: {system_message}\n\nHuman1: {user_message}"
            prompt_preview = combined or "(no prompt content available)"

            # Output from end
            output = end.get("llm_output")

            row = {
                "request_id": rid,
                "start_time": start.get("timestamp"),
                "end_time": end.get("timestamp"),
                "duration_ms": duration_ms,
                "level": end.get("level") or start.get("level"),
                "stage": (start.get("metadata") or {}).get("stage"),
                "objective": (start.get("metadata") or {}).get("objective"),
                "question": question,
                "prompt_file": prompt_file,
                "prompt": prompt_preview,
                "output": output,
                "model": start.get("model") or end.get("model"),
                "provider": start.get("provider") or end.get("provider"),
                "start_line": start.get("__line_number"),
                "end_line": end.get("__line_number"),
                "question_id": question_id,
            }
            paired_rows.append(row)
        else:
            e = start or end
            unmatched_rows.append(
                {
                    "request_id": rid,
                    "event": e.get("event"),
                    "timestamp": e.get("timestamp"),
                    "level": e.get("level"),
                    "stage": (e.get("metadata") or {}).get("stage"),
                    "objective": (e.get("metadata") or {}).get("objective"),
                    "line_number": e.get("__line_number"),
                    "question_id": (e.get("metadata") or {}).get("question_id") or "",
                }
            )

    # Add dataframe_log events as standalone entries
    for df_log in dataframe_logs:
        timestamp = df_log.get("timestamp")
        prompt_file = df_log.get("prompt_file")
        user_message = df_log.get("user_message") or ""
        llm_output = df_log.get("llm_output")
        metadata = df_log.get("metadata") or {}

        row = {
            "request_id": f"dataframe_log_{df_log.get('__line_number')}",
            "start_time": timestamp,
            "end_time": timestamp,
            "duration_ms": 0,
            "level": df_log.get("level"),
            "stage": metadata.get("stage"),
            "objective": metadata.get("objective"),
            "question": metadata.get("question"),
            "prompt_file": prompt_file,
            "prompt": user_message,
            "output": llm_output,
            "model": "dataframe_log",
            "provider": "dataframe_log",
            "start_line": df_log.get("__line_number"),
            "end_line": df_log.get("__line_number"),
            "question_id": metadata.get("question_id") or "",
        }
        paired_rows.append(row)

    paired_df = pd.DataFrame(paired_rows)
    unmatched_df = pd.DataFrame(unmatched_rows)
    if not paired_df.empty:
        paired_df = paired_df.sort_values(by=["start_time", "end_time"], ascending=[True, True])
    if not unmatched_df.empty:
        unmatched_df = unmatched_df.sort_values(by=["timestamp"], ascending=[True])
    return paired_df, unmatched_df


def create_questions_dataframe(results_summary_by_question: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Create a DataFrame from results summary for query-based filtering.

    Args:
        results_summary_by_question: Dictionary mapping questions to their results and metadata

    Returns:
        DataFrame with columns for metadata and evaluation results
    """
    rows = []
    for question_id, summary in results_summary_by_question.items():
        row = {
            "question": summary.get("question"),
            "question_id": question_id,
        }

        # Add metadata fields
        metadata = summary.get("metadata", {})
        row["domain"] = metadata.get("domain")
        row["question_type"] = metadata.get("question_type")
        row["question_level"] = metadata.get("question_level")

        # Add evaluation results (one column per tool)
        eval_tools = summary.get("evaluation_tools", {})
        for tool_name, tool_result in eval_tools.items():
            if tool_result:
                correct = tool_result.get("correct")
                # Simplify tool name (remove "LLMAsJudgeEvaluationTool" prefix if present)
                clean_tool_name = tool_name.replace("LLMAsJudgeEvaluationTool", "")

                # Store boolean or numeric correctness
                if isinstance(correct, bool):
                    row[clean_tool_name] = correct
                elif isinstance(correct, (int, float)):
                    row[clean_tool_name] = correct
                    row[f"{clean_tool_name}_correct"] = correct == 100
                else:
                    row[clean_tool_name] = False

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    st.set_page_config(page_title="LLM Log Viewer", layout="wide")
    st.title("LLM Logs Viewer")

    # File picker - Start with results directory
    default_results_dir = RESULTS_DIR
    st.sidebar.header("Data source")

    # Mode selector
    view_mode = st.sidebar.radio("View mode", ["Single File", "Compare Runs"], index=0)

    results_dir = st.sidebar.text_input("Results directory", value=default_results_dir)

    result_files: List[str] = []
    if os.path.isdir(results_dir):
        for name in sorted(os.listdir(results_dir)):
            if name.endswith(".json"):
                result_files.append(os.path.join(results_dir, name))

    if view_mode == "Compare Runs":
        # Multiple file selection for comparison
        if result_files:
            file_names = [os.path.basename(p) for p in result_files]
            selected_indices = st.sidebar.multiselect(
                "Select result files to compare",
                options=list(range(len(result_files))),
                format_func=lambda i: file_names[i],
                default=[],
            )
            selected_results_files = [result_files[i] for i in selected_indices]
        else:
            selected_results_files = []
            st.sidebar.warning("No result files found in directory")
    else:
        # Single file selection
        file_idx = 0
        if result_files:
            file_names = [os.path.basename(p) for p in result_files]
            file_idx = st.sidebar.selectbox(
                "Select results file", options=list(range(len(result_files))), format_func=lambda i: file_names[i]
            )
            selected_results_file = result_files[file_idx]
        else:
            selected_results_file = st.sidebar.text_input("Results file path", value="")

    # Route to appropriate view
    if view_mode == "Compare Runs":
        if not selected_results_files or len(selected_results_files) < 2:
            st.info("Please select at least 2 result files to compare.")
            return
        render_compare_view(selected_results_files)
    else:
        if not selected_results_file or not os.path.exists(selected_results_file):
            st.info("Select a valid results JSON file to view paired calls.")
            return
        render_single_file_view(selected_results_file)


def render_compare_view(selected_results_files: List[str]) -> None:
    """Render the comparison view for multiple runs."""
    st.subheader(f"Comparing {len(selected_results_files)} runs")

    # Load data for all runs
    runs_data = []
    for results_file in selected_results_files:
        results_summary, paired_df, run_label = load_run_data(results_file)

        if results_summary is not None:
            runs_data.append(
                {
                    "results_file": results_file,
                    "results_summary": results_summary,
                    "paired_df": paired_df,
                    "run_label": run_label,
                }
            )
            st.caption(f"✅ Loaded: {run_label}")
        else:
            st.warning(f"❌ Failed to load: {os.path.basename(results_file)}")

    if not runs_data:
        st.error("No runs loaded successfully.")
        return

    # Find common questions across all runs
    all_question_ids = set()
    for run in runs_data:
        all_question_ids.update([s["question_id"] for s in run["results_summary"].values() if s.get("question_id")])

    common_question_ids = all_question_ids.copy()
    for run in runs_data:
        run_qids = {s["question_id"] for s in run["results_summary"].values() if s.get("question_id")}
        common_question_ids.intersection_update(run_qids)

    st.metric("Total unique questions", len(all_question_ids))
    st.metric("Questions in all runs", len(common_question_ids))

    # Build combined DataFrame for filtering
    combined_questions_data = []
    for question_id in all_question_ids:
        row = {"question_id": question_id}

        # Get metadata from first run that has this question
        for run_idx, run in enumerate(runs_data):
            if question_id in run["results_summary"]:
                summary = run["results_summary"][question_id]
                metadata = summary.get("metadata", {})
                row["domain"] = metadata.get("domain")
                row["question_type"] = metadata.get("question_type")
                row["question_level"] = metadata.get("question_level")
                row["question_id"] = summary.get("question_id")
                row["question"] = summary.get("question")
                break

        # Add correctness for each run
        for run_idx, run in enumerate(runs_data):
            summary = run["results_summary"].get(question_id)
            if summary:
                eval_tools = summary.get("evaluation_tools", {})
                # Check if any tool says it's correct
                all_correct = True
                any_correct = False
                for tool_name, tool_result in eval_tools.items():
                    if tool_result:
                        correct = tool_result.get("correct")
                        if isinstance(correct, bool):
                            is_corr = correct
                        elif isinstance(correct, (int, float)):
                            is_corr = correct == 100
                        else:
                            is_corr = False

                        if is_corr:
                            any_correct = True
                        else:
                            all_correct = False

                row[f"run_{run_idx}_correct"] = all_correct if eval_tools else None
                row[f"run_{run_idx}_any_correct"] = any_correct if eval_tools else None
            else:
                row[f"run_{run_idx}_correct"] = None
                row[f"run_{run_idx}_any_correct"] = None

        combined_questions_data.append(row)

    combined_df = pd.DataFrame(combined_questions_data)

    # Pandas query filtering
    st.sidebar.header("Advanced Filtering")

    # Show available columns for querying
    if not combined_df.empty:
        with st.sidebar.expander("Available Query Columns", expanded=False):
            st.markdown("**Metadata columns:**")
            metadata_cols = [
                col
                for col in ["domain", "question_type", "question_level", "question_id"]
                if col in combined_df.columns
            ]
            for col in metadata_cols:
                unique_vals = combined_df[col].dropna().unique()
                if len(unique_vals) <= 10:
                    st.markdown(f"- `{col}`: {list(unique_vals)}")
                else:
                    st.markdown(f"- `{col}` ({len(unique_vals)} unique values)")

            st.markdown("**Per-run correctness columns:**")
            run_cols = [col for col in combined_df.columns if col.startswith("run_")]
            for col in sorted(run_cols):
                st.markdown(f"- `{col}`")

    with st.sidebar.expander("Query Examples", expanded=False):
        st.markdown("""
        **Example queries:**
        - `domain == 'NLP'`
        - `question_level >= 3`
        - `run_0_correct == True and run_1_correct == False`
        - `run_0_correct == False and run_1_correct == False`
        - `domain.str.contains('Computer')`
        - `question_type == 'paper' and question_level >= 4`
        """)

    # Preset filters
    st.sidebar.markdown("**Quick filters:**")
    col1, col2 = st.sidebar.columns(2)
    quick_filter = None
    if col1.button("Run 0 ✅, Run 1 ❌"):
        quick_filter = "run_0_correct == True and run_1_correct == False"
    if col2.button("All runs ❌"):
        quick_filter = " and ".join([f"run_{i}_correct == False" for i in range(len(runs_data))])
    if col1.button("Any run ✅"):
        quick_filter = " or ".join([f"run_{i}_correct == True" for i in range(len(runs_data))])
    if col2.button("Clear filter"):
        quick_filter = ""

    # Initialize session state for pandas query
    if "pandas_query_compare" not in st.session_state:
        st.session_state.pandas_query_compare = ""

    # Update session state if quick filter is applied
    if quick_filter is not None:
        st.session_state.pandas_query_compare = quick_filter

    pandas_query = st.sidebar.text_input(
        "Pandas query",
        value=st.session_state.pandas_query_compare,
        help="Enter a pandas query expression to filter questions",
        key="pandas_query_input_compare",
    )

    # Update session state when user types in the input
    if pandas_query != st.session_state.pandas_query_compare:
        st.session_state.pandas_query_compare = pandas_query

    # Apply query filter
    filtered_question_ids = set(all_question_ids)
    if pandas_query and pandas_query.strip():
        try:
            filtered_combined_df = combined_df.query(pandas_query)
            filtered_question_ids = set(filtered_combined_df["question_id"].tolist())
            st.sidebar.success(f"Query matched {len(filtered_question_ids)} questions")
        except Exception as e:
            st.sidebar.error(f"Invalid query: {e}")

    # Question selector
    question_options = sorted(list(filtered_question_ids))

    selected_question = None
    if question_options:
        selected_question = st.selectbox("Select a question to compare", options=question_options)
    else:
        st.info("No questions found.")
        return

    if selected_question:
        selected_qid = combined_df.loc[combined_df["question_id"] == selected_question, "question_id"].iloc[0]
        st.subheader(f"Question: {selected_qid[:100]}...")

        # Create tabs for each run
        tab_labels = [run["run_label"] for run in runs_data]
        tabs = st.tabs(tab_labels)

        for tab_idx, (tab, run) in enumerate(zip(tabs, runs_data)):
            with tab:
                summary = run["results_summary"].get(selected_qid)

                if not summary:
                    st.warning("Question not found in this run")
                    continue

                # Show question details
                with st.container(border=True):
                    st.markdown("**Question**")
                    st.code(summary.get("question"), language="")

                    gold_answer = summary.get("gold_answer")
                    predicted_answer = summary.get("predicted_answer")

                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("**Gold answer**")
                        if gold_answer is None:
                            st.warning("Gold answer is None")
                        elif isinstance(gold_answer, (list, dict)):
                            st.json(gold_answer)
                        else:
                            st.code(str(gold_answer), language="")
                    with cols[1]:
                        st.markdown("**Predicted answer**")
                        if predicted_answer is None:
                            st.warning("Predicted answer is None")
                        elif isinstance(predicted_answer, (list, dict)):
                            st.json(predicted_answer)
                        else:
                            st.code(str(predicted_answer), language="")

                    # Evaluation tools
                    eval_tools = summary.get("evaluation_tools") or {}
                    if eval_tools:
                        with st.expander("Evaluation tools results", expanded=True):
                            for tool_name, tool_res in eval_tools.items():
                                correct = tool_res.get("correct")
                                if isinstance(correct, bool):
                                    status = "✅ correct" if correct else "❌ incorrect"
                                elif isinstance(correct, (int, float)):
                                    status = f"Score: {correct}" + (" ✅" if correct == 100 else "")
                                else:
                                    status = "❌ incorrect"
                                st.markdown(f"- `{tool_name}`: {status}")

                # Show prompts from paired_df
                if run["paired_df"] is not None and not run["paired_df"].empty:
                    question_df = run["paired_df"][run["paired_df"]["question_id"] == selected_qid]

                    if not question_df.empty:
                        st.markdown(f"**Prompts ({len(question_df)} calls)**")

                        for _, row in question_df.iterrows():
                            with st.container(border=True):
                                header_cols = st.columns([2, 2, 1, 1])
                                header_cols[0].markdown(f"**request_id**: `{row['request_id']}`")
                                header_cols[1].markdown(
                                    f"**stage/objective**: `{row.get('stage')}` / `{row.get('objective')}`"
                                )
                                header_cols[2].markdown(
                                    f"**duration (ms)**: `{int(row['duration_ms']) if pd.notna(row['duration_ms']) else 'N/A'}`"
                                )
                                header_cols[3].markdown(f"**model**: `{row.get('model')}`")

                                if row.get("prompt_file"):
                                    st.caption(f"prompt_file: {row.get('prompt_file')}")

                                with st.expander("Prompt", expanded=False):
                                    prompt_text = row.get("prompt") or ""
                                    prompt_file = row.get("prompt_file") or ""

                                    has_table = "(" in prompt_text and "\n(" in prompt_text
                                    should_extract_table = has_table and any(
                                        [
                                            "objective_necessary" in prompt_file.lower(),
                                            "run_objectives" in prompt_file.lower(),
                                            "merging" in prompt_file.lower(),
                                        ]
                                    )

                                    if should_extract_table:
                                        try:
                                            table_data = extract_action_columns(prompt_text)
                                            if table_data:
                                                df = pd.DataFrame(table_data)
                                                cols_to_drop = [
                                                    col for col in df.columns if col.endswith(("_quote", "_rationale"))
                                                ]
                                                if cols_to_drop:
                                                    df = df.drop(columns=cols_to_drop)

                                                st.markdown("**📊 Extracted Table Data:**")
                                                st.dataframe(df, use_container_width=True)
                                                st.caption(f"Table: {len(df)} rows × {len(df.columns)} columns")

                                                with st.expander("Show full prompt text"):
                                                    st.code(prompt_text, language="")
                                            else:
                                                st.code(prompt_text, language="")
                                        except Exception as e:
                                            logger.error(f"Error extracting table: {e}")
                                            st.code(prompt_text, language="")
                                    else:
                                        st.code(prompt_text, language="")

                                with st.expander("Output", expanded=False):
                                    output_val = row.get("output")
                                    if isinstance(output_val, (dict, list)):
                                        st.json(output_val)
                                    elif output_val is not None:
                                        try:
                                            parsed_json = json.loads(str(output_val))
                                            st.json(parsed_json)
                                        except (json.JSONDecodeError, ValueError, TypeError):
                                            st.code(str(output_val), language="")
                                    else:
                                        st.code("", language="")
                    else:
                        st.info("No prompt logs found for this question in this run")
                else:
                    st.info("No prompt logs available for this run")


def render_single_file_view(selected_results_file: str) -> None:
    """Render the single file view."""
    st.caption(f"Reading results: {selected_results_file}")

    # Load results file
    results_summary_by_question: Dict[str, Dict[str, Any]] = {}
    try:
        with open(selected_results_file, "r", encoding="utf-8") as rf:
            results_payload = json.load(rf)

        # Normalize different result file formats
        results_list = []

        if isinstance(results_payload, list):
            # Check if it's a list of lists (first element is results, second might be metadata)
            if results_payload and isinstance(results_payload[0], list):
                # Use only the first element which contains the actual results
                results_list = [item for item in results_payload[0] if isinstance(item, dict)]
                st.caption(f"Using first element of results array (length: {len(results_list)})")
            else:
                # Simple list of dicts
                results_list = [item for item in results_payload if isinstance(item, dict)]
        elif isinstance(results_payload, dict):
            # If it's a dict, look for "results" key
            results_list = results_payload.get("results", [])
            # Handle case where results might be nested
            if results_list and len(results_list) > 0 and isinstance(results_list[0], list):
                # Use only the first element
                results_list = [item for item in results_list[0] if isinstance(item, dict)]

        # Extract metadata (might be in second element of list or in payload)
        metadata_list = []
        if isinstance(results_payload, list) and len(results_payload) > 1:
            if isinstance(results_payload[1], list):
                # Metadata is an array aligned by index with results
                metadata_list = results_payload[1]
            # elif isinstance(results_payload[1], dict):
            #     # Metadata is a dict (keyed by id or question)
            #     metadata_dict = results_payload[1]
        elif isinstance(results_payload, dict):
            metadata_payload = results_payload.get("metadata", {})
            if isinstance(metadata_payload, list):
                metadata_list = metadata_payload

        for idx, item in enumerate(results_list):
            if not isinstance(item, dict):
                continue
            q_text = item.get("question_id")
            if not q_text:
                continue

            # Skip duplicates (keep first occurrence)
            if q_text in results_summary_by_question:
                continue

            # Extract metadata for this question
            q_metadata = {}

            # Try metadata_list first (aligned by index)
            if metadata_list and idx < len(metadata_list):
                metadata_item = metadata_list[idx]
                if isinstance(metadata_item, dict):
                    # Check for misc_question_metadata key
                    q_metadata = metadata_item.get("misc_question_metadata", metadata_item)

            # Fallback: metadata might be directly in the item
            if not q_metadata:
                q_metadata = item.get("metadata", {})

            # Extract common metadata fields
            domain = q_metadata.get("domain") or item.get("domain")
            question_type = q_metadata.get("question_type") or q_metadata.get("type") or item.get("question_type")
            question_level = q_metadata.get("question_level") or item.get("question_level")

            results_summary_by_question[q_text] = {
                "question": item.get("question"),
                "gold_answer": item.get("gold_answer"),
                "predicted_answer": item.get("predicted_answer"),
                "evaluation_tools": item.get("evaluation_tools") or {},
                "question_id": item.get("question_id"),
                "metadata": {
                    "domain": domain,
                    "question_type": question_type,
                    "question_level": question_level,
                },
            }

        st.caption(f"✅ Loaded {len(results_summary_by_question)} questions from results file")

    except Exception as e:
        # Non-fatal; continue without results summaries
        import traceback

        logger.error(traceback.format_exc())
        logger.error(f"Error loading results summary: {e}")
        st.error(f"Error loading results file: {e}")
        return

    # Locate corresponding prompt log file using the basename of the results file
    selected_file: Optional[str] = None
    try:
        # Extract experiment ID from results filename
        # Expected format: <prefix>_<experiment_id>.json
        results_basename = os.path.basename(selected_results_file)
        # Remove .json extension
        results_name_no_ext = os.path.splitext(results_basename)[0]

        # Try to extract experiment ID (assuming format like "final_<exp_id>" or "sample_<exp_id>")
        # Split by underscore and get the last part as potential experiment ID
        parts = results_name_no_ext.split("_")
        potential_exp_id = parts[-1] if parts else results_name_no_ext

        logs_dir = PROMPT_LOGS_DIR
        if os.path.isdir(logs_dir):
            prompt_log_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir) if f.endswith(".jsonl")]

            # Try exact match first (experiment_id.jsonl)
            exact_matches = [f for f in prompt_log_files if os.path.basename(f) == f"{potential_exp_id}.jsonl"]

            # Try partial match (contains experiment_id)
            partial_matches = (
                [f for f in prompt_log_files if potential_exp_id in os.path.basename(f)] if not exact_matches else []
            )

            # Pick the best match
            if exact_matches:
                selected_file = exact_matches[0]
            elif partial_matches:
                # Prefer most recent file
                selected_file = max(partial_matches, key=os.path.getmtime)

        if selected_file and os.path.exists(selected_file):
            st.caption(f"Matched prompt log: {selected_file}")
        else:
            st.warning(f"No matching prompt log found for experiment ID: {potential_exp_id}")
            return

    except Exception as e:
        # Non-fatal; continue without prompt logs
        import traceback

        logger.error(traceback.format_exc())
        logger.error(f"Error locating prompt log: {e}")
        st.error(f"Error locating prompt log: {e}")
        return

    events = parse_jsonl_file(selected_file)
    paired_df, unmatched_df = pair_calls(events)

    # Filters
    st.sidebar.header("Filters")
    rid_query = st.sidebar.text_input("Filter by request_id contains")
    q_query = st.sidebar.text_input("Filter by question contains")
    qid_query = st.sidebar.text_input("Filter by question_id contains")

    filtered_df = paired_df
    if rid_query:
        filtered_df = filtered_df[filtered_df["request_id"].astype(str).str.contains(rid_query, na=False)]
    if q_query:
        filtered_df = filtered_df[filtered_df["question"].fillna("").str.contains(q_query, case=False, na=False)]
    if qid_query:
        filtered_df = filtered_df[filtered_df["question_id"].fillna("").str.contains(qid_query, case=False, na=False)]

    # Pandas query filtering
    st.sidebar.header("Advanced Filtering")

    # Show available columns for querying
    if results_summary_by_question:
        questions_df = create_questions_dataframe(results_summary_by_question)
        if not questions_df.empty:
            with st.sidebar.expander("Available Query Columns", expanded=False):
                st.markdown("**Metadata columns:**")
                metadata_cols = [
                    col
                    for col in ["domain", "question_type", "question_level", "question_id"]
                    if col in questions_df.columns
                ]
                for col in metadata_cols:
                    unique_vals = questions_df[col].dropna().unique()
                    if len(unique_vals) <= 10:
                        st.markdown(f"- `{col}`: {list(unique_vals)}")
                    else:
                        st.markdown(f"- `{col}` ({len(unique_vals)} unique values)")

                st.markdown("**Evaluation columns:**")
                eval_cols = [
                    col
                    for col in questions_df.columns
                    if col not in ["question", "question_id", "domain", "question_type", "question_level"]
                ]
                for col in eval_cols:
                    st.markdown(f"- `{col}`")

    with st.sidebar.expander("Query Examples", expanded=False):
        st.markdown("""
        **Example queries:**
        - `domain == 'NLP'`
        - `question_level >= 3`
        - `question_level > 2 and loong_evaluator_correct == True`
        - `soft_evaluator == False`
        - `domain.str.contains('Computer')`
        - `question_type.notna()`
        """)

    # Initialize session state for pandas query
    if "pandas_query_single" not in st.session_state:
        st.session_state.pandas_query_single = ""

    pandas_query = st.sidebar.text_input(
        "Pandas query",
        value=st.session_state.pandas_query_single,
        help="Enter a pandas query expression to filter questions",
        key="pandas_query_input_single",
    )

    # Update session state when user types in the input
    if pandas_query != st.session_state.pandas_query_single:
        st.session_state.pandas_query_single = pandas_query

    # Apply pandas query filter
    if pandas_query and pandas_query.strip() and results_summary_by_question:
        try:
            questions_df = create_questions_dataframe(results_summary_by_question)
            if not questions_df.empty:
                # Apply the query
                filtered_questions_df = questions_df.query(pandas_query)
                allowed_questions_from_query = set(filtered_questions_df["question_id"].tolist())

                # Filter the main dataframe
                filtered_df = filtered_df[filtered_df["question_id"].isin(allowed_questions_from_query)]

                st.sidebar.success(f"Query matched {len(allowed_questions_from_query)} questions")
        except Exception as e:
            st.sidebar.error(f"Invalid query: {e}")

    # KPIs
    total_calls = len(paired_df)
    matched_calls = len(filtered_df)
    unmatched_count = len(unmatched_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total paired calls", f"{total_calls}")
    col2.metric("Visible after filters", f"{matched_calls}")
    col3.metric("Unmatched events", f"{unmatched_count}")

    # Helper function for checking correctness
    def is_correct(correct_value: Any) -> bool:
        """Check if a correct value indicates correctness, handling both bool and int types."""
        if isinstance(correct_value, bool):
            return correct_value
        elif isinstance(correct_value, (int, float)):
            return correct_value == 100
        else:
            return False

    # Results summary (from results_summary_by_question)
    if results_summary_by_question:
        # Overall summary
        total_questions = len(results_summary_by_question)
        all_tools_correct = 0
        any_tool_incorrect = 0
        for s in results_summary_by_question.values():
            tools = s.get("evaluation_tools") or {}
            if tools:
                flags = [is_correct((res or {}).get("correct")) for res in tools.values()]
                if all(flags):
                    all_tools_correct += 1
                if any(not f for f in flags):
                    any_tool_incorrect += 1

        s1, s2, s3 = st.columns(3)
        s1.metric("Total questions (results)", f"{total_questions}")
        s2.metric("All tools correct", f"{all_tools_correct}")
        s3.metric("Any tool incorrect", f"{any_tool_incorrect}")

        # Per-tool breakdown
        # Collect tool names
        tool_name_set = set()
        for s in results_summary_by_question.values():
            for tn in (s.get("evaluation_tools") or {}).keys():
                tool_name_set.add(tn)
        tool_names = sorted(tool_name_set)

        if tool_names:
            with st.expander("Per-tool correctness breakdown", expanded=False):
                for tn in tool_names:
                    tot = 0
                    corr = 0
                    for s in results_summary_by_question.values():
                        tools = s.get("evaluation_tools") or {}
                        if tn in tools:
                            tot += 1
                            if is_correct(tools[tn].get("correct")):
                                corr += 1
                    pct = (corr / tot * 100.0) if tot else 0.0
                    st.markdown(f"- `{tn}`: {corr}/{tot} correct ({pct:.1f}%)")

    # Question selector and prompt rendering
    if filtered_df.empty:
        st.warning("No paired calls after filters.")
    else:
        question_options = sorted([q for q in filtered_df["question_id"].dropna().unique().tolist() if str(q).strip()])

        selected_question = None
        if question_options:
            # Create display options with question_id: question[:100] format
            display_options = []
            question_id_to_question = {}

            for qid in question_options:
                # Find the corresponding question text
                question_text = (
                    filtered_df[filtered_df["question_id"] == qid]["question"].iloc[0]
                    if not filtered_df[filtered_df["question_id"] == qid].empty
                    else str(qid)
                )
                display_text = f"{qid}: {question_text[:100]}"
                display_options.append(display_text)
                question_id_to_question[display_text] = qid

            selected_display = st.selectbox("Select a question", options=display_options)
            selected_question = question_id_to_question[selected_display]
        else:
            st.info("No questions found in the current selection.")

        if selected_question is not None:
            question_df = filtered_df[filtered_df["question_id"] == selected_question]

            st.subheader(f"Prompts for selected question ({len(question_df)})")

            # Render question, gold/predicted answers, and evaluation results (if available)
            summary = results_summary_by_question.get(selected_question)

            # Debug: Check if we found the summary
            if summary is None and results_summary_by_question:
                st.error("Summary is None! Debugging key matching...")
                with st.expander("Debug: Key matching", expanded=True):
                    st.write(f"**Selected question (repr):** {repr(selected_question[:200])}")
                    st.write(f"**Total keys in results:** {len(results_summary_by_question)}")
                    # Check for exact match
                    exact_match = selected_question in results_summary_by_question
                    st.write(f"**Exact match found:** {exact_match}")

                    # Try to find partial matches
                    partial_matches = []
                    for key in list(results_summary_by_question.keys())[:5]:
                        if selected_question[:50] in key or key[:50] in selected_question:
                            partial_matches.append(key)

                    if partial_matches:
                        st.write(f"**Found {len(partial_matches)} partial matches**")
                        for pm in partial_matches[:2]:
                            st.write(f"**Matching key (repr):** {repr(pm[:200])}")
                    else:
                        st.write("**No partial matches found. Showing first key:**")
                        first_key = list(results_summary_by_question.keys())[0]
                        st.write(f"**First key (repr):** {repr(first_key[:200])}")

            if summary:
                # Debug: show what's in the summary
                with st.expander("Debug: Summary contents", expanded=False):
                    st.json(summary)

                with st.container(border=True):
                    st.markdown("**Question**")
                    st.code(summary.get("question"), language="")
                    gold_answer = summary.get("gold_answer")
                    predicted_answer = summary.get("predicted_answer")

                    # Debug info
                    st.caption(
                        f"Gold answer type: {type(gold_answer)}, Predicted answer type: {type(predicted_answer)}"
                    )

                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown("**Gold answer**")
                        if gold_answer is None:
                            st.warning("Gold answer is None")
                        elif isinstance(gold_answer, (list, dict)):
                            st.json(gold_answer)
                        else:
                            st.code(str(gold_answer), language="")
                    with cols[1]:
                        st.markdown("**Predicted answer**")
                        if predicted_answer is None:
                            st.warning("Predicted answer is None")
                        elif isinstance(predicted_answer, (list, dict)):
                            st.json(predicted_answer)
                        else:
                            st.code(str(predicted_answer), language="")

                    eval_tools = summary.get("evaluation_tools") or {}
                    if eval_tools:
                        with st.expander("Evaluation tools results", expanded=True):
                            for tool_name, tool_res in eval_tools.items():
                                correct = tool_res.get("correct")
                                if isinstance(correct, bool):
                                    status = "✅ correct" if correct else "❌ incorrect"
                                elif isinstance(correct, (int, float)):
                                    status = f"Score: {correct}" + (" ✅" if correct == 100 else "")
                                else:
                                    status = "❌ incorrect"
                                st.markdown(f"- `{tool_name}`: {status}")
                    else:
                        st.warning("No evaluation tools found in summary")
            else:
                st.warning(
                    f"No results found for this question. Questions in results: {len(results_summary_by_question)}"
                )
                with st.expander("Debug: Show question comparison", expanded=True):
                    st.write(f"**Selected question length:** {len(summary.get('question'))}")
                    st.write(f"**Selected question (first 200 chars):** {summary.get('question')[:200]}")
                    if results_summary_by_question:
                        first_result_q = list(results_summary_by_question.keys())[0]
                        st.write(f"**First result question length:** {len(first_result_q)}")
                        st.write(f"**First result question (first 200 chars):** {first_result_q[:200]}")
                        # Try to find closest match
                        for result_q in list(results_summary_by_question.keys())[:3]:
                            if summary.get("question")[:100] in result_q or result_q[:100] in summary.get("question"):
                                st.write(f"**Potential match found! Length:** {len(result_q)}")
                                st.code(result_q[:500], language="")

            for _, row in question_df.iterrows():
                with st.container(border=True):
                    header_cols = st.columns([2, 2, 1, 1, 1])
                    header_cols[0].markdown(f"**request_id**: `{row['request_id']}`")
                    header_cols[1].markdown(f"**stage/objective**: `{row.get('stage')}` / `{row.get('objective')}`")
                    header_cols[2].markdown(
                        f"**duration (ms)**: `{int(row['duration_ms']) if pd.notna(row['duration_ms']) else 'N/A'}`"
                    )
                    header_cols[3].markdown(f"**model**: `{row.get('model')}`")
                    header_cols[4].markdown(f"**level**: `{row.get('level')}`")

                    if row.get("prompt_file"):
                        st.caption(f"prompt_file: {row.get('prompt_file')}")

                    with st.expander("Prompt", expanded=False):
                        prompt_text = row.get("prompt") or ""
                        prompt_file = row.get("prompt_file") or ""

                        # Check if this prompt contains table data (for objective necessity checks, merging, etc.)
                        has_table = "(" in prompt_text and "\n(" in prompt_text

                        # Display tables for objective_necessary, run_objectives, and merging prompts
                        should_extract_table = has_table and any(
                            [
                                "objective_necessary" in prompt_file.lower(),
                                "run_objectives" in prompt_file.lower(),
                                "merging" in prompt_file.lower(),
                            ]
                        )

                        if should_extract_table:
                            # Try to extract and display table
                            try:
                                table_data = extract_action_columns(prompt_text)
                                if table_data:
                                    df = pd.DataFrame(table_data)
                                    # Remove citation and rationale columns for cleaner display
                                    cols_to_drop = [col for col in df.columns if col.endswith(("_quote", "_rationale"))]
                                    if cols_to_drop:
                                        df = df.drop(columns=cols_to_drop)

                                    st.markdown("**📊 Extracted Table Data:**")
                                    st.dataframe(df, use_container_width=True)
                                    st.caption(f"Table: {len(df)} rows × {len(df.columns)} columns")

                                    # Show full prompt in a nested expander
                                    with st.expander("Show full prompt text"):
                                        st.code(prompt_text, language="")
                                else:
                                    # No table extracted, show normally
                                    st.code(prompt_text, language="")
                            except Exception as e:
                                logger.error(f"Error extracting table: {e}")
                                st.code(prompt_text, language="")
                        else:
                            # No table data, show normally
                            st.code(prompt_text, language="")

                    with st.expander("Output", expanded=False):
                        output_val = row.get("output")
                        if isinstance(output_val, (dict, list)):
                            # Already a dict or list, display as JSON
                            st.json(output_val)
                        elif output_val is not None:
                            # Try to parse as JSON first
                            try:
                                parsed_json = json.loads(str(output_val))
                                st.json(parsed_json)
                            except (json.JSONDecodeError, ValueError, TypeError):
                                # Not valid JSON, display as code
                                st.code(str(output_val), language="")
                        else:
                            st.code("", language="")
        else:
            st.info("Select a question to view its prompts.")

    # Unmatched section
    with st.expander("Unmatched events"):
        if unmatched_df.empty:
            st.write("None")
        else:
            st.dataframe(unmatched_df)

    # Export
    if not paired_df.empty:
        export_cols = [
            "request_id",
            "start_time",
            "end_time",
            "duration_ms",
            "stage",
            "objective",
            "question",
            "question_id",
            "prompt_file",
            "prompt",
            "output",
            "model",
            "provider",
        ]
        export_df = paired_df[export_cols].copy()
        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV (paired)", csv, file_name="paired_calls.csv", mime="text/csv")


if __name__ == "__main__":
    main()
