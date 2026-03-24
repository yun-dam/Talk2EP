import pandas as pd
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from sliders.document import Document
from sliders.llm_models import Tables
from sliders.llm_models import ExtractedTable
from sliders.llm import get_llm_client, load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler


class Action(BaseModel):
    reasoning: str
    action: str
    arguments: dict


class ActionResult(TypedDict):
    success: bool
    error: str


class MergeAgentState(TypedDict):
    question: str
    documents: list[Document]
    schema: Tables
    original_table: pd.DataFrame
    current_table: pd.DataFrame
    action_history: list[Action | ActionResult]
    conversation_history: list[str]
    final_table: ExtractedTable


def create_controller_chain():
    llm_client = get_llm_client(model="gpt-4.1", temperature=0.0)
    controller_template = load_fewshot_prompt_template(
        template_file="sliders/merge_agent_controller.prompt", template_blocks=[]
    )
    return controller_template | llm_client.with_structured_output(Action)


def create_conflict_resolution_chain():
    llm_client = get_llm_client(model="gpt-4.1", temperature=0.0)
    conflict_resolution_template = load_fewshot_prompt_template(
        template_file="sliders/merge_agent_conflict_resolution.prompt", template_blocks=[]
    )
    return conflict_resolution_template | llm_client.with_structured_output(method="json_mode")


def resolve_entities(
    table: pd.DataFrame, resolutions: list[tuple[list[int], str, str]]
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Resolve entities by updating specific column values for given rows.

    Args:
        table: DataFrame to update
        resolutions: List of tuples (row_ids, column_name, resolved_value)

    Returns:
        Tuple of (updated_table, error_message). If successful, error_message is None.
    """
    try:
        if not resolutions:
            return None, "No resolutions provided"

        # Create a copy of the table to avoid modifying the original
        updated_table = table.copy()

        for row_ids, column_name, resolved_value in resolutions:
            # Validate row IDs exist in the table
            invalid_ids = [rid for rid in row_ids if rid not in updated_table.index]
            if invalid_ids:
                return None, f"Invalid row IDs: {invalid_ids}"

            # Validate column exists
            if column_name not in updated_table.columns:
                return None, f"Column '{column_name}' does not exist in the table"

            # Update the specified rows and column with the resolved value
            updated_table.loc[row_ids, column_name] = resolved_value

        return updated_table, None

    except Exception as e:
        return None, f"Error resolving entities: {str(e)}"


def merge_rows(table: pd.DataFrame, row_ids: list[int]) -> tuple[pd.DataFrame | None, str | None]:
    """
    Merge multiple rows into a single row.

    For columns ending with '_quote' or '_rationale' and metadata columns (row_number, page_number),
    concatenate the values. For other columns, use the first row's values.

    Args:
        table: DataFrame to merge rows from
        row_ids: List of row indices to merge

    Returns:
        Tuple of (merged_table, error_message). If successful, error_message is None.
    """
    try:
        if not row_ids:
            return None, "No row IDs provided for merging"

        # Validate row IDs exist in the table
        invalid_ids = [rid for rid in row_ids if rid not in table.index]
        if invalid_ids:
            return None, f"Invalid row IDs: {invalid_ids}"

        if len(row_ids) < 2:
            return None, "At least 2 rows are required for merging"

        # Get the rows to merge
        rows_to_merge = table.loc[row_ids]

        # Create the merged row
        merged_row = {}
        metadata_columns = ["row_number", "page_number"]

        for column in table.columns:
            column_lower = column.lower()

            # Check if this is a column that should be concatenated
            should_concatenate = (
                column_lower.endswith("_quote")
                or column_lower.endswith("_rationale")
                or column_lower in metadata_columns
            )

            if should_concatenate:
                # Concatenate non-null values with a separator
                values = rows_to_merge[column].dropna().astype(str)
                if len(values) > 0:
                    merged_row[column] = " | ".join(values)
                else:
                    merged_row[column] = None
            else:
                # Use the first non-null value, or the first value if all are null
                non_null_values = rows_to_merge[column].dropna()
                if len(non_null_values) > 0:
                    merged_row[column] = non_null_values.iloc[0]
                else:
                    merged_row[column] = rows_to_merge[column].iloc[0]

        # Create new table with merged row
        new_table = table.drop(index=row_ids).copy()
        merged_row_df = pd.DataFrame([merged_row], index=[min(row_ids)])
        new_table = pd.concat([new_table, merged_row_df]).sort_index()

        return new_table, None

    except Exception as e:
        return None, f"Error merging rows: {str(e)}"


def aggregate_table(table: pd.DataFrame, sql_query: str) -> tuple[pd.DataFrame | None, str | None]:
    """Execute SQL query on the table using DuckDB"""
    try:
        import duckdb

        # Create a DuckDB connection
        conn = duckdb.connect()

        # Register the DataFrame as a table in DuckDB
        conn.register("table", table)

        # Execute the SQL query
        result = conn.execute(sql_query).fetchdf()

        # Close the connection
        conn.close()

        return result, None

    except Exception as e:
        return None, f"Error executing SQL query: {str(e)}"


async def resolve_conflicts(
    table: pd.DataFrame, row_ids: list[int], conflicting_column_names: list[str], state: MergeAgentState
) -> tuple[pd.DataFrame | None, str | None]:
    """Resolve conflicts in specified rows and columns by using LLM to determine which values to keep.

    Preserves non-conflicting column values using a deterministic strategy (first non-null),
    concatenates quote/rationale/metadata columns similar to merge_rows, and inserts the
    resolved row at the smallest original index to keep row-id semantics consistent.
    """
    try:
        conflict_resolution_handler = LoggingHandler(
            prompt_file="sliders/merge_agent_conflict_resolution.prompt",
            metadata={
                "conflicting_column_names": conflicting_column_names,
                "question": state.get("question", ""),
                "schema": state.get("schema", {}),
            },
        )
        conflict_resolution_chain = create_conflict_resolution_chain()
        # Validate inputs
        if not row_ids:
            return None, "No row IDs provided"

        if not conflicting_column_names:
            return None, "No conflicting column names provided"

        # Check if all row IDs exist in the table
        missing_rows = [row_id for row_id in row_ids if row_id not in table.index]
        if missing_rows:
            return None, f"Row IDs not found in table: {missing_rows}"

        # Check if all column names exist in the table
        missing_columns = [col for col in conflicting_column_names if col not in table.columns]
        if missing_columns:
            return None, f"Column names not found in table: {missing_columns}"

        # Get the conflicting rows
        conflicting_rows = table.loc[row_ids]

        # Ask LLM for resolutions for the specified columns
        prompt_inputs = {
            "conflicting_column_names": conflicting_column_names,
            "conflicting_rows": conflicting_rows.to_dict(orient="records"),
            "question": state.get("question", ""),
            "schema": state.get("schema", {}),
        }

        response = await conflict_resolution_chain.ainvoke(
            prompt_inputs, config={"callbacks": [conflict_resolution_handler]}
        )

        # Build a single merged row
        merged_row: dict = {}
        metadata_columns = ["row_number", "page_number"]

        for column in table.columns:
            column_lower = column.lower()
            should_concatenate = (
                column_lower.endswith("_quote")
                or column_lower.endswith("_rationale")
                or column_lower in metadata_columns
            )

            if column in conflicting_column_names:
                # Use LLM provided value if present; otherwise fall back to first non-null
                resolved_value = response.get(column)
                if resolved_value is not None:
                    merged_row[column] = resolved_value
                else:
                    non_null_values = conflicting_rows[column].dropna()
                    merged_row[column] = (
                        non_null_values.iloc[0] if len(non_null_values) > 0 else conflicting_rows[column].iloc[0]
                    )
            elif should_concatenate:
                values = conflicting_rows[column].dropna().astype(str)
                merged_row[column] = " | ".join(values) if len(values) > 0 else None
            else:
                non_null_values = conflicting_rows[column].dropna()
                merged_row[column] = (
                    non_null_values.iloc[0] if len(non_null_values) > 0 else conflicting_rows[column].iloc[0]
                )

        # Create resolved table by removing originals and inserting merged row at min index
        resolved_table = table.drop(index=row_ids).copy()
        merged_row_df = pd.DataFrame([merged_row], index=[min(row_ids)])
        resolved_table = pd.concat([resolved_table, merged_row_df]).sort_index()

        return resolved_table, None

    except Exception as e:
        return None, f"Error resolving conflicts: {str(e)}"


class MergeAgent:
    def __init__(self):
        workflow = StateGraph(MergeAgentState)

        workflow.add_node("controller", self.controller)
        workflow.add_node("deduplicate", self.deduplicate)
        workflow.add_node("entity_resolution", self.entity_resolution)
        workflow.add_node("conflict_resolution", self.conflict_resolution)
        workflow.add_node("aggregation", self.aggregation)
        workflow.add_node("finalize", self.finalize)

        workflow.add_edge(START, "controller")
        workflow.add_conditional_edges(
            "controller",
            self.router,
            {
                "deduplicate": "deduplicate",
                "entity_resolution": "entity_resolution",
                "conflict_resolution": "conflict_resolution",
                "aggregation": "aggregation",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("deduplicate", "controller")
        workflow.add_edge("entity_resolution", "controller")
        workflow.add_edge("conflict_resolution", "controller")
        workflow.add_edge("aggregation", "controller")
        workflow.add_edge("finalize", END)

        self.chain = workflow.compile()
        with open("graph.png", "wb") as f:
            f.write(self.chain.get_graph().draw_mermaid_png())

    async def ainvoke(self, state: MergeAgentState) -> MergeAgentState:
        return await self.chain.ainvoke(state)

    @staticmethod
    async def controller(state: MergeAgentState) -> dict:
        # Initialize defaults
        if "current_table" not in state or state["current_table"] is None:
            state["current_table"] = state["original_table"].copy()
        if "action_history" not in state or state["action_history"] is None:
            state["action_history"] = []
        if "conversation_history" not in state or state["conversation_history"] is None:
            state["conversation_history"] = []

        # Prepare the prompt inputs
        prompt_inputs = {
            "question": state.get("question", ""),
            "documents": [doc.document_name for doc in state.get("documents", [])],
            "schema": state["schema"].model_dump()
            if hasattr(state.get("schema"), "model_dump")
            else state.get("schema", {}),
            "extracted_tables": state["current_table"].to_string()
            if isinstance(state.get("current_table"), pd.DataFrame)
            else "",
            "action_history": state.get("action_history", []),
        }
        controller_handler = LoggingHandler(
            prompt_file="sliders/merge_agent_controller.prompt",
            metadata={
                "question": state.get("question", ""),
                "schema": state.get("schema", {}),
                "extracted_tables": state["current_table"].to_string()
                if isinstance(state.get("current_table"), pd.DataFrame)
                else "",
                "action_history": state.get("action_history", []),
            },
        )
        controller_chain = create_controller_chain()

        # Generate the next action using the controller chain
        try:
            response = await controller_chain.ainvoke(prompt_inputs, config={"callbacks": [controller_handler]})
            action = {
                "reasoning": response.get("reasoning", ""),
                "action": response.get("action", ""),
                "arguments": response.get("arguments", {}),
            }

            prior_history = state.get("action_history", [])
            return {
                "action_history": prior_history + [action],
            }
        except Exception as e:
            # If there's an error, add an error action result
            prior_history = state.get("action_history", [])
            return {
                "action_history": prior_history
                + [
                    {
                        "success": False,
                        "error": f"Controller failed to generate action: {str(e)}",
                    }
                ],
            }

    @staticmethod
    def deduplicate(state: MergeAgentState) -> dict:
        action = state["action_history"][-1]

        # Accept both 'row_ids' and legacy 'rows'
        arguments = action.get("arguments", {})
        row_ids = arguments.get("row_ids", arguments.get("rows"))

        # Validate that the rows are provided
        if row_ids is None:
            return {
                "action_history": state["action_history"]
                + [
                    {
                        "success": False,
                        "error": "No row ids were specified to deduplicate. Please select the rows to deduplicate from the table.",
                    }
                ],
            }

        # Validate that the rows are not empty
        if len(row_ids) == 0:
            return {
                "action_history": state["action_history"]
                + [
                    {
                        "success": False,
                        "error": "No row ids were specified to deduplicate. Please select the rows to deduplicate from the table.",
                    }
                ],
            }

        # Merge the rows (Deduplication)
        new_table, error = merge_rows(state["current_table"], row_ids)
        if new_table is not None:
            return {
                "current_table": new_table,
                "action_history": state["action_history"] + [{"success": True, "error": ""}],
            }
        else:
            return {
                "action_history": state["action_history"] + [{"success": False, "error": error}],
            }

    @staticmethod
    def entity_resolution(state: MergeAgentState) -> dict:
        action = state["action_history"][-1]

        # Validate that the resolutions are provided
        if "resolutions" not in action["arguments"]:
            return {
                "action_history": state["action_history"]
                + [
                    {
                        "success": False,
                        "error": "No resolutions were provided to resolve entities. Please provide a list of tuples with three elements: (row_ids: list[int], column_name: str, resolved_value: str).",
                    }
                ],
            }

        # Validate that the resolutions are not empty
        if len(action["arguments"]["resolutions"]) == 0:
            return {
                "action_history": state["action_history"]
                + [
                    {
                        "success": False,
                        "error": "No resolutions were provided to resolve entities. Please provide a list of tuples with three elements: (row_ids: list[int], column_name: str, resolved_value: str).",
                    }
                ],
            }

        # Validate the format of the resolutions
        for resolution in action["arguments"]["resolutions"]:
            if len(resolution) != 3:
                return {
                    "action_history": state["action_history"]
                    + [
                        {
                            "success": False,
                            "error": "Invalid resolution format. Please provide a list of tuples with three elements: (row_ids: list[int], column_name: str, resolved_value: str).",
                        }
                    ],
                }

        # Resolve the entities
        new_table, error = resolve_entities(state["current_table"], action["arguments"]["resolutions"])
        if new_table is not None:
            return {
                "current_table": new_table,
                "action_history": state["action_history"] + [{"success": True, "error": ""}],
            }
        else:
            return {
                "action_history": state["action_history"] + [{"success": False, "error": error}],
            }

    @staticmethod
    def conflict_resolution(state: MergeAgentState) -> dict:
        action = state["action_history"][-1]
        arguments = action.get("arguments", {})
        row_ids = arguments.get("row_ids")
        column_names = arguments.get("column_names")
        # Backward compat for older keys
        if row_ids is None and "rows" in arguments:
            row_ids = arguments.get("rows")
        if column_names is None and "conflicting_column_names" in arguments:
            column_names = arguments.get("conflicting_column_names")

        if row_ids is None:
            return {
                "action_history": state["action_history"]
                + [
                    {
                        "success": False,
                        "error": "No row ids were provided to resolve the conflicts. Please provide a tuple as the parameter: (row_ids: list[int], conflicting_column_names: list[str]).",
                    }
                ],
            }
        if column_names is None:
            return {
                "action_history": state["action_history"]
                + [
                    {
                        "success": False,
                        "error": "No conflicting column names were provided to resolve the conflicts. Please provide a tuple as the parameter: (row_ids: list[int], conflicting_column_names: list[str]).",
                    }
                ],
            }

        if len(row_ids) == 0:
            return {
                "action_history": state["action_history"]
                + [
                    {
                        "success": False,
                        "error": "No row ids were provided to resolve the conflicts. Please provide a tuple as the parameter: (row_ids: list[int], conflicting_column_names: list[str]).",
                    }
                ],
            }
        if len(column_names) == 0:
            return {
                "action_history": state["action_history"]
                + [
                    {
                        "success": False,
                        "error": "No conflicting column names were provided to resolve the conflicts. Please provide a tuple as the parameter: (row_ids: list[int], conflicting_column_names: list[str]).",
                    }
                ],
            }

        new_table, error = resolve_conflicts(state["current_table"], row_ids, column_names)
        if new_table is not None:
            return {
                "current_table": new_table,
                "action_history": state["action_history"] + [{"success": True, "error": ""}],
            }
        else:
            return {
                "action_history": state["action_history"] + [{"success": False, "error": error}],
            }

    @staticmethod
    def aggregation(state: MergeAgentState) -> dict:
        action = state["action_history"][-1]

        # Validate that the sql query is provided
        if "sql_query" not in action["arguments"]:
            return {
                "action_history": state["action_history"]
                + [
                    {
                        "success": False,
                        "error": "No sql query was provided to aggregate the table. Please provide a sql query to aggregate the table.",
                    }
                ],
            }

        new_table, error = aggregate_table(state["current_table"], action["arguments"]["sql_query"])
        if new_table is not None:
            return {
                "current_table": new_table,
                "action_history": state["action_history"] + [{"success": True, "error": ""}],
            }
        else:
            return {
                "action_history": state["action_history"] + [{"success": False, "error": error}],
            }

    @staticmethod
    def router(state: MergeAgentState) -> str:
        # Get the last action from the action history
        if not state["action_history"]:
            return "controller"

        last_action = state["action_history"][-1]

        # If the last entry is an ActionResult (TypedDict), detect via key presence
        if isinstance(last_action, dict) and ("success" in last_action):
            if len(state["action_history"]) < 2:
                return "controller"

            # Find the most recent Action (object with 'action' key)
            for i in range(len(state["action_history"]) - 1, -1, -1):
                candidate = state["action_history"][i]
                if isinstance(candidate, dict) and ("action" in candidate):
                    last_action = candidate
                    break
            else:
                return "controller"

        # Check if we should stop
        if last_action["action"] == "stop":
            return "finalize"

        # Check for repeated actions (last two actions are the same)
        if len(state["action_history"]) >= 4:  # Need at least 2 Action + 2 ActionResult pairs
            # Find the last two Actions (objects with 'action' key)
            actions = [item for item in state["action_history"] if isinstance(item, dict) and ("action" in item)]
            if len(actions) >= 2:
                last_two_actions = actions[-2:]
                if (
                    last_two_actions[0]["action"] == last_two_actions[1]["action"]
                    and last_two_actions[0]["arguments"] == last_two_actions[1]["arguments"]
                ):
                    return "controller"  # Send back to controller if actions are repeated

        # Route to the appropriate node based on the action
        action_name = last_action["action"]
        if action_name == "deduplicate":
            return "deduplicate"
        elif action_name == "entity_resolution":
            return "entity_resolution"
        elif action_name == "conflict_resolution":
            return "conflict_resolution"
        elif action_name == "aggregation":
            return "aggregation"
        else:
            return "controller"

    @staticmethod
    def summarize_actions(state: MergeAgentState) -> str:
        history = state.get("action_history", [])
        actions = [str(item.get("action")) for item in history if isinstance(item, dict) and ("action" in item)]
        return ", ".join(actions)

    @staticmethod
    def finalize(state: MergeAgentState) -> dict:
        # Build a Table object from current state
        try:
            current_df = state.get("current_table")
            table_obj = ExtractedTable(
                name="merged_table",
                tables=state.get("schema"),
                sql_query=None,
                dataframe=current_df,
                dataframe_table_name="merged_table_dataframe",
            )
            return {"final_table": table_obj}
        except Exception:
            # Fallback: leave final_table unset on error
            return {}


async def run_merge_agent(
    question: str, documents: list[Document], schema: Tables, extracted_tables: pd.DataFrame
) -> ExtractedTable | None:
    # Initialize state with required defaults
    initial_state: MergeAgentState = {
        "question": question,
        "documents": documents,
        "schema": schema,
        "original_table": extracted_tables,
        "current_table": extracted_tables.copy() if isinstance(extracted_tables, pd.DataFrame) else extracted_tables,
        "action_history": [],
        "conversation_history": [],
        "final_table": None,  # type: ignore
    }
    agent = MergeAgent()

    state = await agent.ainvoke(initial_state)
    return state.get("final_table")
