import asyncio
import json
import os
import pandas as pd
from datetime import datetime
from sliders.document import Document
from sliders.llm_models import Tables, ObjectiveNecessity, TableOperation, ProvenanceSQL
from sliders.llm_models import ExtractedTable
from sliders.callbacks.logging import LoggingHandler
from sliders.log_utils import logger
from sliders.globals import SlidersGlobal
from sliders.modules.merge_techniques.utils import format_table

import duckdb
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template


def log_dataframe(
    df: pd.DataFrame,
    stage: str,
    metadata: dict = None,
    prompt_file: str = None,
) -> None:
    """
    Log a generated dataframe using the same format as the LoggingHandler.

    Args:
        df: The dataframe to log
        stage: The stage name (e.g., 'run_objective', 'get_provenance', 'final_result')
        metadata: Additional metadata to include in the log
        prompt_file: Optional prompt file name for context
    """
    # Ensure log directory exists
    log_dir = os.path.join(os.environ.get("SLIDERS_LOGS_DIR"), "prompt_logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating log directory: {e}")
        return

    # Determine log file path
    if SlidersGlobal.experiment_id:
        filename = f"{SlidersGlobal.experiment_id}.jsonl"
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%fZ")
        filename = f"{ts}.jsonl"
        logger.warning("Experiment id is not set, using timestamp instead")

    log_path = os.path.join(log_dir, filename)

    # Prepare dataframe information
    df_info = {
        "shape": df.shape if df is not None else None,
        "columns": df.columns.tolist() if df is not None else None,
        "formatted_table": format_table(df) if df is not None else None,
    }

    try:
        formatted_table = df.to_markdown() if df is not None else None
    except Exception:
        formatted_table = format_table(df) if df is not None else None

    # Construct the payload in the same format as LoggingHandler
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": "INFO",
        "event": "dataframe_log",
        "prompt_file": prompt_file or "unknown",
        "experiment_id": SlidersGlobal.experiment_id,
        "system_message": None,
        "user_message": "Table converted to markdown",
        "llm_output": formatted_table,
        "metadata": {
            "stage": stage,
            "dataframe_info": df_info,
            **(metadata or {}),
        },
    }

    # Append to JSONL file
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.error(f"Error appending dataframe log to JSONL: {e}")


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


async def run_objective(
    objective: str,
    reasoning: str,
    table_data: pd.DataFrame,
    schema: Tables,
    documents: list[Document],
    question: str,
    table_name: str,
    metadata: dict,
) -> pd.DataFrame:
    max_tries = 5
    feedback = None
    actions = []
    for _ in range(max_tries):
        llm_client = get_llm_client(model="gpt-4.1")

        objectives_template = load_fewshot_prompt_template(
            template_file="sliders/merge_seq_agent/run_objectives.prompt",
            template_blocks=[],
        )

        objectives_chain = objectives_template | llm_client.with_structured_output(TableOperation)

        handler = LoggingHandler(
            prompt_file="sliders/merge_seq_agent/run_objectives.prompt",
            metadata={
                "question": question,
                "stage": "run_objective",
                "objective": objective,
                **(metadata or {}),
            },
        )

        table_operation = await objectives_chain.ainvoke(
            {
                "objective_name": objective,
                "reasoning": reasoning,
                "table_data": format_table(table_data),
                "schema": schema,
                "question": question,
                "table_name": table_name,
                "actions": json.dumps(actions, indent=4),
            },
            config={"callbacks": [handler]},
        )

        error = None
        missing_columns = None
        try:
            df = run_sql(table_operation.sql_query, table_data, table_name)
            # Check if all columns from the original table are present in the new dataframe
            missing_columns = list(set(table_data.columns) - set(df.columns))
        except Exception as e:
            error = str(e)

        if error:
            feedback = f"Error running sql query: {table_operation.sql_query}. Got the error: {error}"
            actions.append(
                {
                    "action": table_operation.reasoning,
                    "sql_query": table_operation.sql_query,
                    "error": error,
                    "missing_columns": None,
                    "feedback": feedback,
                }
            )
        elif missing_columns:
            feedback = f"The sql query {table_operation.sql_query} gives the following columns: {df.columns}. The original table has the following columns: {table_data.columns}. The sql query is missing the following columns: {missing_columns}"
            actions.append(
                {
                    "action": table_operation.reasoning,
                    "sql_query": table_operation.sql_query,
                    "error": None,
                    "missing_columns": missing_columns,
                    "feedback": feedback,
                }
            )
            log_dataframe(df, "missing_columns", metadata, "sliders/merge_seq_agent/run_objectives.prompt")
        else:
            feedback = None
            actions.append(
                {
                    "action": table_operation.reasoning,
                    "sql_query": table_operation.sql_query,
                    "error": None,
                    "missing_columns": None,
                    "feedback": feedback,
                }
            )
            log_dataframe(df, "success", metadata, "sliders/merge_seq_agent/run_objectives.prompt")
            break

    if feedback:
        return None, actions
    else:
        return df, actions


async def get_provenance(
    objective: str,
    run_objectives_query: list[dict],
    df: pd.DataFrame,
    schema: Tables,
    table_name: str,
    metadata: dict,
) -> pd.DataFrame:
    """
    Given the objective, new_table_data and the sql, ask the llm to generate a sql query which gets the provenance of the new_table_data from the <column_name>_quote and <column_name>_rationale columns.
    """

    llm_client = get_llm_client(model="gpt-4.1")

    provenance_template = load_fewshot_prompt_template(
        template_file="sliders/merge_seq_agent/get_provenance.prompt",
        template_blocks=[],
    )
    provenance_chain = provenance_template | llm_client.with_structured_output(ProvenanceSQL)

    max_tries = 3
    feedback = None
    actions = []
    for _ in range(max_tries):
        handler = LoggingHandler(
            prompt_file="sliders/merge_seq_agent/get_provenance.prompt",
            metadata={
                "question": metadata.get("question"),
                "stage": "get_provenance",
                "objective": objective,
                **(metadata or {}),
            },
        )

        provenance_sql = await provenance_chain.ainvoke(
            {
                "objective_name": objective,
                "reasoning_for_sql_query": run_objectives_query[-1]["action"],
                "sql_operation": run_objectives_query[-1]["sql_query"],
                "new_table": format_table(df),
                "schema": schema,
                "actions": json.dumps(actions, indent=4),
            },
            config={"callbacks": [handler]},
        )

        error = None
        try:
            df = run_sql(provenance_sql.sql_query, df, table_name)
        except Exception as e:
            df = None
            error = str(e)

        if error:
            feedback = f"Error running sql query: {provenance_sql.sql_query}. Got the error: {error}"
            actions.append(
                {
                    "action": provenance_sql.reasoning,
                    "sql_query": provenance_sql.sql_query,
                    "error": error,
                    "feedback": feedback,
                }
            )
        else:
            feedback = None
            actions.append(
                {
                    "action": provenance_sql.reasoning,
                    "sql_query": provenance_sql.sql_query,
                    "error": None,
                    "feedback": feedback,
                }
            )
            break

    return df, actions


def merge_df_with_provenance(df: pd.DataFrame, df_provenance: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the main dataframe with its provenance dataframe.
    Join on row_id and keep the <column_name>_quote and <column_name>_rationale columns from df_provenance
    and everything else from df.
    """
    if df_provenance is None or df_provenance.empty:
        return df

    # Get the provenance columns (those ending with _quote or _rationale)
    provenance_columns = [col for col in df_provenance.columns if col.endswith("_quote") or col.endswith("_rationale")]

    # Include row_id for joining
    columns_to_keep_from_provenance = ["row_id"] + provenance_columns

    # Select only the columns we need from df_provenance
    df_provenance_subset = df_provenance[columns_to_keep_from_provenance]

    # Merge on row_id, keeping all rows from df and adding provenance columns
    merged_df = df.merge(df_provenance_subset, on="row_id", how="left")

    return merged_df


async def run_merge_objectives_sequentially(
    question: str,
    documents: list[Document],
    schema: Tables,
    table_data: pd.DataFrame,
    table_name: str,
    original_table_name: str,
    run_provenance: bool,
    metadata: dict,
    model_config: dict,
) -> pd.DataFrame:
    """
    This function runs all the merge objectives sequentially. It first asks if the objective is necessary. If it is, it runs the objective. If it is not, it skips it.
    Its similar to the merge_objectives_sql_generation function, but after each merge, it makes another llm call to get the provenance.

    Logic:
    ```
    for objective in objectives:
        if is_objective_necessary(objective, table_data, **kwargs):
            df = run_objective(objective, table_data, **kwargs)
            df_provenance = get_provenance(objective, table_data, df, **kwargs)
            df = merge_df_with_provenance(df, df_provenance)
            table_data = df
    ```

    Goal: to get the correct answer from the final table.
    Stretch goal: the final table should have the same columns as the original table.
    """

    llm_client = get_llm_client(model="gpt-4.1")

    necessary_chain_template = load_fewshot_prompt_template(
        template_file="sliders/merge_seq_agent/objective_necessary.prompt",
        template_blocks=[],
    )
    necessary_chain = necessary_chain_template | llm_client.with_structured_output(ObjectiveNecessity)

    objectives = ["clear_nan", "deduplication", "aggregation", "conflict_resolution"]

    max_loops = 3
    for _ in range(max_loops):
        necessary_tasks = []
        for objective in objectives:
            handler = LoggingHandler(
                prompt_file="sliders/merge_seq_agent/objective_necessary.prompt",
                metadata={
                    "question": question,
                    "stage": "objective_necessary",
                    "objective": objective,
                    **(metadata or {}),
                },
            )
            necessary_tasks.append(
                necessary_chain.ainvoke(
                    {
                        "objective_name": objective,
                        "table_data": format_table(table_data),
                        "schema": schema,
                        "question": question,
                        "other_operations": json.dumps([op for op in objectives if op != objective], indent=4),
                    },
                    config={"callbacks": [handler]},
                )
            )

        necessary_tasks = await asyncio.gather(*necessary_tasks)

        found_at_least_one_necessary_task = False
        run_objectives_queries = []
        get_provenance_queries = []
        for objective, necessary_task in zip(objectives, necessary_tasks):
            if necessary_task.required:
                found_at_least_one_necessary_task = True
                df, objective_actions = await run_objective(
                    objective,
                    necessary_task.reasoning,
                    table_data,
                    schema,
                    documents,
                    question,
                    table_name,
                    metadata,
                )
                if run_provenance:
                    df_provenance, provenance_actions = await get_provenance(
                        objective,
                        objective_actions,
                        df,
                        schema,
                        table_name,
                        metadata,
                    )
                    df = merge_df_with_provenance(df, df_provenance)
                if df is not None:
                    table_data = df

                run_objectives_queries.extend(objective_actions)
                if run_provenance:
                    get_provenance_queries.extend(provenance_actions)

        if len(table_data) <= 1:
            break
        if not found_at_least_one_necessary_task:
            break

    return ExtractedTable(
        name=original_table_name,
        tables=schema,
        sql_query=None,
        dataframe=table_data,
        dataframe_table_name=table_name,
        table_str=format_table(table_data),
        actions=run_objectives_queries + get_provenance_queries if run_provenance else run_objectives_queries,
    )
