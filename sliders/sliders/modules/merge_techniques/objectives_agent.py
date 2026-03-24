import time
import traceback
import pandas as pd
from sliders.document import Document
from sliders.llm_models import Tables, ExtractedTable, Output, ObjectiveNecessity
from sliders.llm_tools.sql import DuckSQLBasic
from sliders.log_utils import logger
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.modules.merge_techniques.utils import get_table_schema


def create_objectives_merge_chain(objective_key: str, model: str, temperature: float, **kwargs):
    llm_client = get_llm_client(model=model, temperature=temperature, **kwargs)
    template_file = f"sliders/objectives_merging_sql_{objective_key}.prompt"
    objectives_merge_chain = load_fewshot_prompt_template(
        template_file=template_file,
        template_blocks=[],
    )
    return objectives_merge_chain | llm_client.with_structured_output(Output)


def create_objective_necessity_check_chain(model: str, temperature: float, **kwargs):
    llm_client = get_llm_client(model=model, temperature=temperature, **kwargs)
    template_file = "sliders/check_objective_necessity.prompt"
    objective_necessity_check_chain = load_fewshot_prompt_template(
        template_file=template_file,
        template_blocks=[],
    )
    return objective_necessity_check_chain | llm_client.with_structured_output(ObjectiveNecessity)


async def is_objective_necessary(
    self,
    objective_name: str,
    objective: str,
    format_table: str,
    current_fields: list,
    question: str,
    metadata: dict,
    model_config: dict,
) -> bool:
    """Check if the given objective is necessary for the current table state."""
    try:
        # Create the necessity check chain
        necessity_chain = create_objective_necessity_check_chain(
            **model_config["check_objective_necessity"],
        )

        # Ask the LLM if this objective is necessary
        necessity_response = await necessity_chain.ainvoke(
            {
                "extracted_value_table": format_table,
                "fields": current_fields,
                "objective": objective,
                "objective_name": objective_name,
                "question": question,
            }
        )

        # Parse the text response to determine if objective is necessary

        # Check if the LLM thinks the objective is necessary
        logger.info(f"Objective {objective} | necessity check response: {necessity_response}")
        return necessity_response.required

    except Exception as e:
        logger.warning(f"Error checking objective necessity for {objective}: {e}, proceeding with objective")
        # If we can't determine necessity, assume it's necessary to be safe
        return True


async def run_merge_objectives_sql_generation(
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
    merge_chain = create_objectives_merge_chain(
        objective_key="general",
        **model_config["merge_tables"],
    )

    # Define the objectives to process
    objectives = ["dedup_numerical", "dedup_qualitative", "conflict_resolution", "aggregation"]

    # Start with the original table
    current_df = table_data
    backup_merged_table = current_df
    current_table_name = table_name
    sql_attempts = 0
    final_sql_query = None

    # Track input data size
    input_rows = len(table_data)
    metadata["merging"]["total_rows_processed"] = metadata["merging"].get("total_rows_processed", 0) + input_rows

    # Process each objective sequentially in a loop
    debug_objectives = []
    iteration_count = 0
    max_iterations = 15  # Prevent infinite loops
    consecutive_no_improvement = 0
    max_no_improvement = len(objectives)  # Stop if no improvement during all objectives
    actions = []
    original_table = table_data.copy()
    original_table_format = (
        str(tuple(original_table.columns.to_list()))
        + "\n"
        + "\n".join([str(row) for row in original_table.to_records(index=False)])
    )

    try:
        while (
            len(objectives) > 0 and iteration_count < max_iterations and consecutive_no_improvement < max_no_improvement
        ):
            iteration_count += 1
            objective = objectives.pop(0)

            logger.info(
                f"Processing objective {iteration_count}: {objective} (consecutive no improvement: {consecutive_no_improvement})"
            )
            logger.info(f"Current dataframe shape: {current_df.shape}")

            # Create a descriptive name for this objective
            objective_name = objective
            if objective == "dedup_numerical":
                objective_name = "Numerical Deduplication"
            elif objective == "dedup_qualitative":
                objective_name = "Qualitative Deduplication"
            elif objective == "conflict_resolution":
                objective_name = "Conflict Resolution"
            elif objective == "aggregation":
                objective_name = "Aggregation"
            else:
                continue

            # Generate SQL query for this specific objective
            # First check if this objective is necessary before proceeding
            try:
                # Format the table data for the LLM
                format_table = (
                    str(tuple(current_df.columns.to_list()))
                    + "\n"
                    + "\n".join([str(row) for row in current_df.to_records(index=False)])
                )

                # Map current columns to field descriptions, preserving original field info where possible
                original_fields = get_table_schema(original_table_name, schema) or []
                current_columns = current_df.columns.tolist()

                # Create a mapping from original field names to field objects
                original_field_map = {field.name: field for field in original_fields}

                # Build current fields list, trying to preserve original descriptions
                current_fields = []
                for col in current_columns:
                    # Try to find a matching original field (exact match or partial match)
                    matching_field = None

                    # First try exact match
                    if col in original_field_map:
                        matching_field = original_field_map[col]
                    else:
                        # Try to find partial matches (e.g., company_name_dedup_numerical -> company_name)
                        for orig_name, orig_field in original_field_map.items():
                            if orig_name in col or col.startswith(orig_name):
                                matching_field = orig_field
                                break

                    if matching_field:
                        # Use original field info but update the name
                        current_fields.append(
                            {
                                "name": col,
                                "description": matching_field.description,
                                "data_type": matching_field.data_type,
                                "unit": matching_field.unit,
                                "scale": matching_field.scale,
                            }
                        )
                    else:
                        # Create a new field with basic info
                        current_fields.append(
                            {
                                "name": col,
                                "description": f"Transformed column: {col}",
                                "data_type": str(current_df[col].dtype),
                                "unit": None,
                                "scale": None,
                            }
                        )

                # logger.info(f"Current table columns for objective {objective}: {current_columns}")
                # logger.info(f"Mapped {len(current_fields)} fields for LLM")

            except Exception as e:
                logger.error(f"Error preparing data for objective {objective}: {e}")
                continue

            # Check if this objective is necessary before generating SQL
            if not await is_objective_necessary(
                objective_name, objective, format_table, current_fields, question, metadata, model_config
            ):
                logger.info(f"Skipping objective {objective} as it's not necessary")
                consecutive_no_improvement += 1
                continue

            # Track that we're processing this objective
            debug_objectives.append({"objective": objective, "status": "EXECUTED"})

            def format_actions(actions: list[dict]) -> str:
                repr_action = ""
                for step_num, action in enumerate(actions):
                    repr_action += f"Step {step_num + 1}:\n"
                    repr_action += f"Action: {action['action']}\n"
                    repr_action += f"SQL Query:\n```{action['sql_query']}```\n\n"
                return repr_action

            # Now proceed with SQL generation and execution
            try:
                sql_decision = await merge_chain.ainvoke(
                    {
                        "extracted_value_table": format_table,
                        "fields": current_fields,  # Use mapped fields with proper structure
                        "table_name": current_table_name,
                        "objective": objective,
                        "objective_name": objective_name,
                        "question": question,
                        "feedback": None,
                        "actions": format_actions(actions),
                        "original_table": original_table_format,
                    }
                )
                sql_attempts += 1
                actions.append(
                    {
                        "action": sql_decision.decision.reasoning,
                        "sql_query": sql_decision.sql_query,
                    }
                )
                metadata["merging"]["sql_queries_executed"] = metadata["merging"].get("sql_queries_executed", 0) + 1

                if not sql_decision.sql_query or sql_decision.sql_query.strip() == "":
                    logger.warning(f"Did not extract SQL query for objective {objective}, skipping")
                    continue

                # logger.info(f"Objective {objective} SQL Query: {sql_decision.sql_query}")

                # Execute the objective-specific query
                logger.info(
                    f"Executing SQL for objective {objective} on table {current_table_name}: {sql_decision.sql_query}"
                )
                result = duck_sql.sql(sql_decision.sql_query)

                if result:
                    new_df = result.to_df()

                    # Check if the result is empty
                    if new_df.empty:
                        logger.warning(
                            f"Objective {objective} query returned empty dataframe: {sql_decision.sql_query}, skipping"
                        )
                        continue

                    # Check if we made meaningful progress (reduced row or column count)
                    if len(new_df) < len(current_df) or len(new_df.columns) < len(current_df.columns):
                        current_df = new_df
                        backup_merged_table = current_df

                        # Update the existing table instead of creating a new one
                        duck_sql.register(current_df, current_table_name)

                        # Reset consecutive no-improvement counter
                        consecutive_no_improvement = 0

                        logger.info(f"Objective {objective} successful. New dataframe shape: {current_df.shape}")
                        debug_objectives[-1]["status"] = "SUCCESS"
                    else:
                        consecutive_no_improvement += 1
                        # logger.info(f"Objective {objective} did not improve data quality, skipping (consecutive no improvement: {consecutive_no_improvement})")
                else:
                    logger.warning(f"Objective {objective} query returned no results: {sql_decision.sql_query}")

            except Exception as e:
                logger.error(traceback.format_exc())
                debug_objectives[-1]["status"] = "FAILED"
                metadata["errors"].append(
                    {
                        "stage": f"merge_objective_{objective}",
                        "error": str(e),
                        "table_name": current_table_name,
                        "objective": objective,
                    }
                )
                # Retry the objective if only fails once
                if len(debug_objectives) >= 2 and debug_objectives[-2]["objective"] != objective:
                    logger.error(f"Error processing objective {objective}: {e}. Retrying...")
                    objectives.insert(0, objective)
                else:
                    consecutive_no_improvement += 1
                continue

            if len(current_df) <= 1:
                break

            # Add objective back to queue to retry later
            objectives.append(objective)

        # Log loop termination reason
        if iteration_count >= max_iterations:
            logger.info(f"Loop terminated: reached maximum iterations ({max_iterations})")
        elif consecutive_no_improvement >= max_no_improvement:
            logger.info(f"Loop terminated: no improvement for {consecutive_no_improvement} consecutive objectives")
        elif len(objectives) == 0:
            logger.info("Loop terminated: all objectives processed")
        elif len(current_df) <= 1:
            logger.info("Loop terminated: dataframe has reached the minimum size")

        logger.info(f"Debug objectives: {debug_objectives}")
        if current_df is not None:
            logger.info(f"Final merged dataframe shape: {current_df.shape}")
            logger.info(f"Final merged dataframe: {current_df}")

        # Check if we have a valid merged dataframe
        if current_df is not None and not current_df.empty and len(current_df) < input_rows:
            final_sql_query = sql_decision

            # Update metadata based on results
            metadata["merging"]["tables_created"] = metadata["merging"].get("tables_created", 0) + 1
            output_rows = len(current_df)
            metadata["merging"][f"{table_name}_input_rows"] = input_rows
            metadata["merging"][f"{table_name}_output_rows"] = output_rows
            metadata["merging"][f"{table_name}_data_reduction_ratio"] = (
                (input_rows - output_rows) / input_rows if input_rows > 0 else 0
            )

            # Add objectives-specific metadata
            metadata["merging"][f"{table_name}_objectives_processed"] = debug_objectives
            metadata["merging"][f"{table_name}_iterations"] = iteration_count
            metadata["merging"][f"{table_name}_final_objective_count"] = len(objectives)
            metadata["merging"][f"{table_name}_consecutive_no_improvement"] = consecutive_no_improvement
        else:
            metadata["merging"]["merge_failures"] = metadata["merging"].get("merge_failures", 0) + 1
            logger.warning(
                f"Warning: No valid merged data for table {table_name}, using previously successful dataframe"
            )
            current_df = backup_merged_table
            logger.info(f"Final merged dataframe: {current_df}")

        merge_time = time.time() - merge_start_time
        metadata["merging"]["merging_time"] = metadata["merging"].get("merging_time", 0) + merge_time
        metadata["merging"][f"{table_name}_merge_time"] = merge_time
        metadata["merging"][f"{table_name}_sql_attempts"] = sql_attempts

        return ExtractedTable(
            name=table_name,
            tables=schema,
            sql_query=final_sql_query,
            dataframe=current_df,
            dataframe_table_name=table_name + "_dataframe",
        )

    except Exception as e:
        logger.error(f"Error merging table {table_name}: {e}")
        logger.error(traceback.format_exc())
        metadata["errors"].append(
            {
                "stage": "merge_objectives_sql_generation",
                "error": str(e),
                "table_name": table_name,
                "question": question,
            }
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
