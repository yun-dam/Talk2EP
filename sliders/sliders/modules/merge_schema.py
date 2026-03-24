import time
import traceback
import uuid
import json
import pandas as pd

from sliders.document import Document
from sliders.llm_models import Tables, ExtractedTable
from sliders.llm_tools.sql import DuckSQLBasic
from sliders.log_utils import logger
from sliders.modules.merge_techniques import (
    run_merge_objectives_sequentially,
    run_merge_objectives_sql_generation,
    run_merge_simple_sql_generation,
    run_merge_agent,
)
from sliders.modules.merge_techniques.utils import sanitize_table_name


class MergedTables:
    def __init__(self, config: dict, model_config: dict):
        self.config = config
        self.model_config = model_config

    @staticmethod
    def create_table_data(table_name: str, table_data: list[dict], sid: str) -> tuple[pd.DataFrame, str]:
        rows = []

        for i, row in enumerate(table_data):
            processed_row = {}
            processed_row["row_id"] = i
            processed_row["page_number"] = row["__metadata__"].get("chunk_id")
            processed_row["document_name"] = row["__metadata__"].get("document_name")
            processed_row["text_header"] = row["__metadata__"].get("chunk_header")
            for field_name, field_data in row["fields"].items():
                processed_row[f"{field_name}_quote"] = field_data.get("quote")
                processed_row[f"{field_name}_rationale"] = field_data.get("rationale")
                val = field_data.get("value", "")
                if isinstance(val, list) or isinstance(val, tuple) or isinstance(val, dict):
                    processed_row[field_name] = json.dumps(val)
                else:
                    processed_row[field_name] = val
                processed_row[f"{field_name}_is_explicit"] = field_data.get("is_explicit", True)

            # Check if all field values are None or empty string
            has_meaningful_data = False
            for field_name in row["fields"].keys():
                field_value = processed_row[field_name]
                if field_value is not None and field_value != "":
                    has_meaningful_data = True
                    break

            # Skip this row if no meaningful data found
            if not has_meaningful_data:
                continue
            rows.append(processed_row)

        table_data = pd.DataFrame(rows)
        new_table_name = f"{sanitize_table_name(table_name)}_{sid}"

        return table_data, new_table_name

    async def run_merge_sql(
        self,
        original_table_name: str,
        table_name: str,
        table_data: pd.DataFrame,
        schema: Tables,
        documents: list[Document],
        question: str,
        duck_sql: DuckSQLBasic,
        metadata: dict,
    ) -> ExtractedTable:
        if self.config.get("merge_strategy", "simple") == "simple":
            return await run_merge_simple_sql_generation(
                original_table_name,
                table_name,
                table_data,
                schema,
                documents,
                question,
                duck_sql,
                metadata,
                self.model_config,
            )

        elif self.config.get("merge_strategy", "simple") == "objectives_based":
            return await run_merge_objectives_sql_generation(
                original_table_name,
                table_name,
                table_data,
                schema,
                documents,
                question,
                duck_sql,
                metadata,
                self.model_config,
            )
        elif self.config.get("merge_strategy", "simple") == "agentic":
            return await run_merge_agent(
                original_table_name,
                table_name,
                table_data,
                schema,
                documents,
                question,
                duck_sql,
                metadata,
                self.model_config,
            )
        elif self.config.get("merge_strategy", "simple") == "seq_agent":
            return await run_merge_objectives_sequentially(
                question=question,
                documents=documents,
                schema=schema,
                table_data=table_data,
                table_name=table_name,
                original_table_name=original_table_name,
                run_provenance=self.config.get("run_provenance", False),
                metadata=metadata,
                model_config=self.model_config,
            )
        else:
            raise ValueError(f"Invalid merge strategy: {self.config['merge_strategy']}")

    def _has_improved_data_quality(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> bool:
        """Check if the new dataframe has improved data quality compared to the old one."""
        if len(new_df) < len(old_df):
            return True

        # Check for reduced NULL values
        old_null_count = old_df.isnull().sum().sum()
        new_null_count = new_df.isnull().sum().sum()
        if new_null_count < old_null_count:
            logger.info(f"Data quality improved: NULL values reduced from {old_null_count} to {new_null_count}")
            return True

        # Check for more unique values (indicating deduplication)
        old_unique_ratio = old_df.nunique().sum() / len(old_df.columns)
        new_unique_ratio = new_df.nunique().sum() / len(new_df.columns)
        if new_unique_ratio > old_unique_ratio:
            logger.info(
                f"Data quality improved: unique value ratio increased from {old_unique_ratio:.3f} to {new_unique_ratio:.3f}"
            )
            return True

        # Check for more complete rows (fewer NULLs per row)
        old_completeness = 1 - (old_df.isnull().sum(axis=1).sum() / (len(old_df) * len(old_df.columns)))
        new_completeness = 1 - (new_df.isnull().sum(axis=1).sum() / (len(new_df) * len(new_df.columns)))
        if new_completeness > old_completeness:
            logger.info(
                f"Data quality improved: row completeness increased from {old_completeness:.3f} to {new_completeness:.3f}"
            )
            return True

        return False

    async def merge_single_table(
        self,
        table_name: str,
        table_data: list[dict],
        schema: Tables,
        documents: list[Document],
        question: str,
        metadata: dict,
    ) -> ExtractedTable:
        # Create a table for this class
        with DuckSQLBasic() as duck_sql:
            sid = uuid.uuid4().hex[:8]
            table_data, new_table_name = MergedTables.create_table_data(table_name, table_data, sid)

            if table_data.empty:
                logger.warning(f"Table {table_name} is empty, skipping")
                return ExtractedTable(
                    name=new_table_name,
                    tables=schema,
                    sql_query=None,
                    dataframe=table_data,
                    dataframe_table_name=table_name + "_dataframe",
                )
            try:
                duck_sql.register(table_data, new_table_name)

                return await self.run_merge_sql(
                    table_name,
                    new_table_name,
                    table_data,
                    schema,
                    documents,
                    question,
                    duck_sql,
                    metadata,
                )
            finally:
                try:
                    duck_sql.unregister(table_name)
                except Exception as e:
                    logger.error(f"Error unregistering table {table_name}: {e}")
                    logger.error(traceback.format_exc())

    async def merge_chunks_tables(
        self,
        extracted_tables: dict,
        documents: list[Document],
        question: str,
        schema: Tables,
        metadata: dict,
    ) -> list[ExtractedTable]:
        """Merge data extracted from multiple chunks with configurable models"""
        merge_start_time = time.time()
        tables = []

        try:
            for table_name, table_data in extracted_tables.items():
                # extracted table to csv
                if table_name.startswith("AdditionalInformation"):
                    tables.append(
                        ExtractedTable(
                            name=table_name,
                            tables=schema,
                            sql_query=None,
                            dataframe=pd.DataFrame(table_data),
                            dataframe_table_name=table_name + "_dataframe",
                        )
                    )
                    continue
                tables.append(
                    await self.merge_single_table(table_name, table_data, schema, documents, question, metadata)
                )

            # Update timing metadata
            metadata["timing"]["table_merging"]["merging_time"] = time.time() - merge_start_time

            # Calculate overall merging statistics
            successful_tables = len([t for t in tables if t.dataframe is not None])
            failed_tables = len([t for t in tables if t.dataframe is None])

            metadata["merging"]["successful_merges"] = successful_tables
            metadata["merging"]["failed_merges"] = failed_tables
            metadata["merging"]["merge_success_rate"] = successful_tables / len(tables) if len(tables) > 0 else 0

        except Exception as e:
            logger.error(f"Error merging tables: {e}")
            logger.error(traceback.format_exc())
            metadata["errors"].append(
                {
                    "stage": "merge_chunks_tables",
                    "error": str(e),
                    "question": question,
                    "num_extracted_tables": len(extracted_tables),
                }
            )
            metadata["timing"]["table_merging"]["merging_time"] = time.time() - merge_start_time

        return tables
