import time
import traceback
import pandas as pd

from sliders.document import Document
from sliders.llm_models import Tables, ExtractedTable, Output
from sliders.llm_tools.sql import DuckSQLBasic
from sliders.log_utils import logger
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.modules.merge_techniques.utils import get_table_schema


def create_merge_chain(model: str, temperature: float, **kwargs):
    llm_client = get_llm_client(model=model, temperature=temperature, **kwargs)
    merge_chain = load_fewshot_prompt_template(
        template_file="sliders/schema_merging_sql.prompt",
        template_blocks=[],
    )
    return merge_chain | llm_client.with_structured_output(Output)


async def run_merge_simple_sql_generation(
    original_table_name: str,
    table_name: str,
    table_data: pd.DataFrame,
    schema: Tables,
    documents: list[Document],
    question: str,
    duck_sql: DuckSQLBasic,
    metadata: dict,
    model_config: dict,
):
    merge_start_time = time.time()
    merge_chain = create_merge_chain(
        **model_config["merge_tables"],
    )
    sql_attempts = 0

    format_table = (
        str(tuple(table_data.columns.to_list()))
        + "\n"
        + "\n".join([str(row) for row in table_data.to_records(index=False)])
    )

    # Track input data size
    input_rows = len(table_data)
    metadata["merging"]["total_rows_processed"] = metadata["merging"].get("total_rows_processed", 0) + input_rows
    successful_merge = False

    try:
        # First try to generate a SQL query
        sql_decision = await merge_chain.ainvoke(
            {
                "extracted_value_table": format_table,
                "fields": get_table_schema(original_table_name, schema),
                "table_name": table_name,
                "feedback": None,
                "question": question,
            }
        )
        sql_attempts += 1
        metadata["merging"]["sql_queries_executed"] = metadata["merging"].get("sql_queries_executed", 0) + 1

        df = None
        error_message = ""

        try:
            # Try to execute the SQL query
            result = duck_sql.sql(sql_decision.sql_query)
            if result:
                df = result.to_df()
                successful_merge = True
            else:
                error_message = f"Couldn't execute SQL query: {sql_decision.sql_query}"
                metadata["errors"].append(
                    {
                        "stage": "merge_sql_execution",
                        "error": error_message,
                        "table_name": table_name,
                        "sql_query": sql_decision.sql_query,
                    }
                )
        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            logger.error(traceback.format_exc())
            error_message = f"Error executing SQL query: {e}. Please try again. The generated SQL query was: {sql_decision.sql_query}"
            metadata["errors"].append(
                {
                    "stage": "merge_sql_execution",
                    "error": str(e),
                    "table_name": table_name,
                    "sql_query": sql_decision.sql_query,
                    "table_data_preview": format_table[:500],  # First 500 chars for debugging
                }
            )

        # Retry logic with feedback
        if df is None:
            for retry_count in range(2):
                try:
                    # Try to generate a SQL query with feedback
                    sql_decision = await merge_chain.ainvoke(
                        {
                            "extracted_value_table": format_table,
                            "fields": get_table_schema(original_table_name, schema),
                            "table_name": table_name,
                            "feedback": error_message,
                            "question": question,
                        }
                    )
                    sql_attempts += 1
                    metadata["merging"]["sql_queries_executed"] = metadata["merging"].get("sql_queries_executed", 0) + 1

                    result = duck_sql.sql(sql_decision.sql_query)
                    if result:
                        df = result.to_df()
                        successful_merge = True
                        break
                    else:
                        error_message = f"Couldn't execute SQL query: {sql_decision.sql_query}"
                        metadata["errors"].append(
                            {
                                "stage": f"merge_sql_retry_{retry_count + 1}",
                                "error": error_message,
                                "table_name": table_name,
                                "sql_query": sql_decision.sql_query,
                            }
                        )
                except Exception as e:
                    logger.error(f"Error in retry {retry_count + 1}: {e}")
                    logger.error(traceback.format_exc())
                    error_message = f"Error in retry {retry_count + 1}: {e}"
                    metadata["errors"].append(
                        {
                            "stage": f"merge_sql_retry_{retry_count + 1}",
                            "error": str(e),
                            "table_name": table_name,
                            "sql_query": sql_decision.sql_query
                            if "sql_query" in locals()
                            else "Query generation failed",
                        }
                    )

        # Update metadata based on results
        if successful_merge:
            metadata["merging"]["tables_created"] = metadata["merging"].get("tables_created", 0) + 1
            if df is not None:
                output_rows = len(df)
                metadata["merging"][f"{table_name}_input_rows"] = input_rows
                metadata["merging"][f"{table_name}_output_rows"] = output_rows
                metadata["merging"][f"{table_name}_data_reduction_ratio"] = (
                    (input_rows - output_rows) / input_rows if input_rows > 0 else 0
                )
        else:
            metadata["merging"]["merge_failures"] = metadata["merging"].get("merge_failures", 0) + 1

        merge_time = time.time() - merge_start_time
        metadata["merging"]["merging_time"] = metadata["merging"].get("merging_time", 0) + merge_time
        metadata["merging"][f"{table_name}_merge_time"] = merge_time
        metadata["merging"][f"{table_name}_sql_attempts"] = sql_attempts

        return ExtractedTable(
            name=table_name,
            tables=schema,
            sql_query=sql_decision,
            dataframe=df,
            dataframe_table_name=table_name + "_dataframe",
        )

    except Exception as e:
        logger.error(f"Error merging table {table_name}: {e}")
        logger.error(traceback.format_exc())
        metadata["errors"].append(
            {"stage": "merge_sql_generation", "error": str(e), "table_name": table_name, "question": question}
        )
        metadata["merging"]["merge_failures"] = metadata["merging"].get("merge_failures", 0) + 1
        metadata["merging"]["merging_time"] = (
            metadata["merging"].get("merging_time", 0) + time.time() - merge_start_time
        )

        # Return empty table on failure
        return ExtractedTable(
            name=table_name,
            tables=schema,
            sql_query=None,
            dataframe=None,
            dataframe_table_name=table_name + "_dataframe",
        )
