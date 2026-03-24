import os
import pandas as pd
import numpy as np
from datetime import datetime
from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated
from sliders.document import Document
from sliders.llm_models import Tables
from langchain_openai import AzureChatOpenAI
import json
from sliders.llm.prompts import load_fewshot_prompt_template

import duckdb
from pydantic import BaseModel


llm_client = AzureChatOpenAI(
    model="gpt-4.1",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZARE_URL_ENDPOINT"),
    api_version="2024-12-01-preview",
    temperature=0,
)


class ActionOutput(BaseModel):
    action: str
    output: str
    success: bool
    error: str


def run_sql(sql: str, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    # Create a DuckDB connection
    conn = duckdb.connect()

    # Register the DataFrame as a table in DuckDB
    conn.register(table_name, df)

    # Execute the SQL query
    result = conn.execute(sql).fetchdf()

    # Close the connection
    conn.close()

    return result


async def resolve_conflicts(
    rows_to_resolve: pd.DataFrame, conflicting_columns: list[str], table_name: str
) -> tuple[pd.DataFrame, str]:
    """Resolve the conflicts in the rows."""
    resolve_conflicts_prompt_template = load_fewshot_prompt_template(
        template_file="sliders/merge_agent_conflict_resolution.prompt",
        template_blocks=[],
        is_distilled=False,
        keep_indentation=True,
    )
    resolve_conflicts_chain = resolve_conflicts_prompt_template | llm_client
    sql_query = resolve_conflicts_chain.ainvoke(
        {"rows_to_resolve": rows_to_resolve, "conflicting_columns": conflicting_columns, "table_name": table_name}
    )

    try:
        df = run_sql(sql_query, rows_to_resolve, table_name)
    except Exception as e:
        return rows_to_resolve, str(e)
    return df, None


@tool
def deduplicate_rows(
    reasoning: Annotated[str, "Reason for choosing this tool and how it will help deduplicate the rows."],
    rows_query: Annotated[str, "DuckDB SQL query that selects rows that are duplicate"],
    df: Annotated[pd.DataFrame, InjectedToolArg],
    table_name: Annotated[str, InjectedToolArg],
) -> tuple[pd.DataFrame, str]:
    """Deduplicate rows with similar values."""
    try:
        duplicate_df = run_sql(rows_query, df, table_name)
    except Exception as e:
        return df, str(e)

    if duplicate_df.empty:
        return df, None

    # Normalize cells so that unhashable objects (dict/list/sets/tuples/ndarray) become strings for duplicate detection
    def _normalize_cell(value):
        try:
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, np.ndarray):
                value = value.tolist()
            if isinstance(value, set):
                value = sorted(list(value))
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, sort_keys=True, default=str)
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return value
        except Exception:
            return str(value)

    try:
        normalized_duplicate_df = duplicate_df.applymap(_normalize_cell)
        # Compute deduplicated subset indices based on normalized view
        dedup_indices = normalized_duplicate_df.drop_duplicates(keep="first").index
        dedup_subset = duplicate_df.loc[dedup_indices]

        # Merge the deduplicated rows back to the original df
        if "row_id" in duplicate_df.columns and "row_id" in df.columns:
            duplicate_row_ids = set(duplicate_df["row_id"].tolist())
            # Remove all original duplicates using row_id and append the deduped subset
            base_df = df[~df["row_id"].isin(duplicate_row_ids)].copy()

            # Align columns in both DataFrames
            for col in dedup_subset.columns:
                if col not in base_df.columns:
                    base_df[col] = None
            for col in base_df.columns:
                if col not in dedup_subset.columns:
                    dedup_subset[col] = None

            df = pd.concat([base_df, dedup_subset[base_df.columns]], ignore_index=True)
        else:
            # Fallback: concatenate and drop duplicates across the combined DataFrame using normalized view
            combined = pd.concat([df, dedup_subset], ignore_index=True)
            normalized_combined = combined.applymap(_normalize_cell)
            keep_indices = normalized_combined.drop_duplicates(keep="first").index
            df = combined.loc[keep_indices].reset_index(drop=True)
    except Exception as e:
        return df, str(e)

    return df, None


@tool
def canonicalize_column_values(
    reasoning: Annotated[
        str, "Reasoning for choosing this tool and how it will help canonicalize the values in the column."
    ],
    rows_query: Annotated[str, "DuckDB SQL query to select rows to canonicalize"],
    column_to_canonicalize: Annotated[str, "Column name to canonicalize"],
    canonical_value: Annotated[str, "The canonical value to use"],
    df: Annotated[pd.DataFrame, InjectedToolArg],
    table_name: Annotated[str, InjectedToolArg],
) -> tuple[pd.DataFrame, str]:
    """Canonicalize the values in the column."""
    # Get the rows to canonicalize
    try:
        rows_to_canonicalize = run_sql(rows_query, df, table_name)
    except Exception as e:
        return df, str(e)

    # Update the specified column with the canonical value
    rows_to_canonicalize[column_to_canonicalize] = canonical_value

    # Get the row indices to update in the original dataframe
    row_indices = rows_to_canonicalize.index

    # Update the original dataframe with canonicalized values
    df.loc[row_indices, column_to_canonicalize] = canonical_value

    return df, None


@tool
async def resolve_conflicting_rows(
    reasoning: Annotated[str, "Reasoning for using this tool and how it will help resolve the conflicts in the rows."],
    rows_query: Annotated[str, "DuckDB SQL query to select the rows to resolve conflicts"],
    conflicting_columns: Annotated[list[str], "List of column names with conflicting values that need resolution"],
    df: Annotated[pd.DataFrame, InjectedToolArg],
    table_name: Annotated[str, InjectedToolArg],
) -> tuple[pd.DataFrame, str]:
    """Resolve the conflicting rows in the table."""
    # Get the rows to resolve conflicts

    try:
        rows_to_resolve = run_sql(rows_query, df, table_name)
    except Exception as e:
        return df, str(e)

    # Resolve the conflicts
    resolved_rows, error = await resolve_conflicts(rows_to_resolve, conflicting_columns, table_name)

    if not error:
        # Get the indices of rows that were resolved
        resolved_indices = rows_to_resolve.index

        # Remove the original conflicting rows from the dataframe
        df = df.drop(resolved_indices)

        # Add the resolved rows back to the dataframe
        df = pd.concat([df, resolved_rows], ignore_index=True)

    return df, error


@tool
async def aggregate_rows(
    reasoning: Annotated[str, "Reasoning for using this tool and how it will help aggregate the rows into one row."],
    rows_query: Annotated[str, "DuckDB SQL query to select the rows to aggregate"],
    aggregation_query: Annotated[str, "DuckDB SQL query for how to aggregate the rows into one row"],
    df: Annotated[pd.DataFrame, InjectedToolArg],
    table_name: Annotated[str, InjectedToolArg],
) -> tuple[pd.DataFrame, str]:
    """Aggregate the rows in the table into one row."""
    # Get the rows to aggregate and the aggregated result
    try:
        rows_to_aggregate = run_sql(rows_query, df, table_name)
    except Exception as e:
        return df, str(e)

    try:
        aggregated_rows = run_sql(aggregation_query, df, table_name)
    except Exception as e:
        return df, str(e)

    # Ask LLM to generate a single SELECT that returns the aggregated row with provenance columns
    try:
        provenance_prompt_template = load_fewshot_prompt_template(
            template_file="sliders/merge_agent_aggregate_provenance.prompt",
            template_blocks=[],
            is_distilled=False,
            keep_indentation=True,
        )
        provenance_chain = provenance_prompt_template | llm_client
        provenance_sql_response = await provenance_chain.ainvoke(
            {
                "rows_to_aggregate": rows_to_aggregate,
                "aggregated_rows": aggregated_rows,
                "table_name": table_name,
                "aggregation_query": aggregation_query,
            }
        )

        provenance_sql = (
            provenance_sql_response.content
            if hasattr(provenance_sql_response, "content")
            else str(provenance_sql_response)
        ).strip()

        aggregated_rows_with_provenance = run_sql(provenance_sql, df, table_name)

        # Ensure df has any new provenance columns and align columns
        for col in aggregated_rows_with_provenance.columns:
            if col not in df.columns:
                df[col] = None

        for col in df.columns:
            if col not in aggregated_rows_with_provenance.columns:
                aggregated_rows_with_provenance[col] = None

        aggregated_rows = aggregated_rows_with_provenance[df.columns]

    except Exception as e:
        return df, str(e)

    # Replace original rows with aggregated row(s)
    try:
        df = df.drop(rows_to_aggregate.index)
        df = pd.concat([df, aggregated_rows], ignore_index=True)
    except Exception as e:
        return df, str(e)

    return df, None


@tool
def reset_to_original_table(
    reasoning: Annotated[
        str, "Reasoning for using this tool and how it will help reset the table to the original state."
    ],
    df: Annotated[pd.DataFrame, InjectedToolArg],
) -> tuple[pd.DataFrame, str]:
    """Reset the table to the original state."""
    return df, None


def format_action_history(action_history: list[ActionOutput]) -> str:
    if len(action_history) == 0:
        return "No action history"

    action_history_str = ""
    for step, action in enumerate(action_history):
        action_history_str += f"Step {step + 1}:\n"
        action_history_str += f"Action: {action.action}\n"
        action_history_str += f"Success: {action.success}\n"
        if action.error:
            action_history_str += f"Error: {action.error}\n"
        action_history_str += "\n"

    return action_history_str


def format_table(table_data):
    return (
        str(tuple(table_data.columns.to_list()))
        + "\n"
        + "\n".join([str(row) for row in table_data.to_records(index=False)])
    )


async def run_merge_agent(
    question: str,
    documents: list[Document],
    schema: Tables,
    table_data: pd.DataFrame,
    table_name: str,
    original_table_name: str,
    metadata: dict,
    model_config: dict,
) -> pd.DataFrame:
    controller_prompt_template = load_fewshot_prompt_template(
        template_file="sliders/merge_agent_controller.prompt",
        template_blocks=[],
        is_distilled=False,
        keep_indentation=True,
    )
    original_table_data = table_data.copy()
    controller_chain = controller_prompt_template | llm_client.bind_tools(
        [deduplicate_rows, canonicalize_column_values, resolve_conflicting_rows, aggregate_rows]
    )

    # Prepare logging (single JSON file per run)
    try:
        question_id = str(metadata.get("question_id", "unknown"))
    except Exception:
        question_id = "unknown"
    run_ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_log_dir = os.path.join("logs", "merge_agent", question_id)
    try:
        os.makedirs(run_log_dir, exist_ok=True)
    except Exception:
        pass
    run_log_file = os.path.join(run_log_dir, f"merge_agent_run_{question_id}_{run_ts}.json")

    # Initialize the run log file if missing
    try:
        if not os.path.exists(run_log_file):
            with open(run_log_file, "w", encoding="utf-8") as f:
                f.write("[]")
    except Exception:
        pass

    def _append_run_log(entry: dict):
        try:
            # Read, append, write back array
            with open(run_log_file, "r", encoding="utf-8") as f:
                arr = json.load(f)
            arr.append(entry)
            with open(run_log_file, "w", encoding="utf-8") as f:
                json.dump(arr, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            # Best-effort: fall back to line-wise JSON
            try:
                with open(run_log_file, "a", encoding="utf-8") as f:
                    f.write("\n" + json.dumps(entry, ensure_ascii=False, default=str))
            except Exception:
                pass

    max_steps = 20
    action_history = []
    for i in range(max_steps):
        prompt_payload = {
            "question": question,
            "schema": schema,
            "extracted_tables": format_table(table_data),
            "table_name": table_name,
            "action_history": format_action_history(action_history),
            "step_number": i,
            "max_steps": max_steps,
        }

        model_response = await controller_chain.ainvoke(prompt_payload)

        # Log prompt and response for this step
        try:
            response_payload = {
                "content": getattr(model_response, "content", str(model_response)),
                "tool_calls": getattr(model_response, "additional_kwargs", {}).get("tool_calls"),
            }
            _append_run_log(
                {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "question_id": question_id,
                    "table_name": table_name,
                    "step_number": i,
                    "prompt": prompt_payload,
                    "response": response_payload,
                }
            )
        except Exception:
            pass

        if model_response.content == "":
            tool_calls = model_response.additional_kwargs.get("tool_calls")
            if tool_calls:
                function_call = tool_calls[0]["function"]["name"]
                function_args = json.loads(tool_calls[0]["function"]["arguments"])
                if function_call == "deduplicate_rows":
                    table_data, error = deduplicate_rows.invoke(
                        {
                            "reasoning": function_args["reasoning"],
                            "rows_query": function_args["rows_query"],
                            "df": table_data,
                            "table_name": table_name,
                        }
                    )
                elif function_call == "canonicalize_column_values":
                    table_data, error = canonicalize_column_values.invoke(
                        {
                            "reasoning": function_args["reasoning"],
                            "rows_query": function_args["rows_query"],
                            "column_to_canonicalize": function_args["column_to_canonicalize"],
                            "canonical_value": function_args["canonical_value"],
                            "df": table_data,
                            "table_name": table_name,
                        }
                    )
                elif function_call == "resolve_conflicting_rows":
                    table_data, error = await resolve_conflicting_rows.ainvoke(
                        {
                            "reasoning": function_args["reasoning"],
                            "rows_query": function_args["rows_query"],
                            "conflicting_columns": function_args["conflicting_columns"],
                            "df": table_data,
                            "table_name": table_name,
                        }
                    )
                elif function_call == "aggregate_rows":
                    table_data, error = await aggregate_rows.ainvoke(
                        {
                            "reasoning": function_args["reasoning"],
                            "rows_query": function_args["rows_query"],
                            "aggregation_query": function_args["aggregation_query"],
                            "df": table_data,
                            "table_name": table_name,
                        }
                    )
                elif function_call == "reset_to_original_table":
                    table_data = original_table_data
            else:
                return table_data
        else:
            return table_data

        action_history.append(
            ActionOutput(
                action=function_call + " (" + ", ".join([k + "=" + repr(v) for k, v in function_args.items()]) + ")",
                output=format_table(table_data),
                success=not bool(error),
                error=str(error),
            )
        )

    return table_data
