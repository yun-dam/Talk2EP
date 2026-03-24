from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import uuid

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from sliders.callbacks.logging import LoggingHandler
from sliders.document import Document
from sliders.llm_models import Action, RephrasedQuestion, ExtractedTable, SQLAnswer, TableProcessingNeeded
from sliders.llm_tools.sql import DuckSQLBasic, run_sql_query
from sliders.log_utils import logger
from sliders.modules.extract_schema import ExtractSchema
from sliders.modules.generate_schema import GenerateSchema
from sliders.modules.merge_schema import MergedTables
from sliders.utils import prepare_schema_repr
from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.modules.merge_techniques.utils import format_table

if TYPE_CHECKING:
    from sliders.llm_models import Tables, Table


class System(ABC):
    def __init__(self, config):
        self.config = config
        self._setup_chains()

    @abstractmethod
    async def run(self, question: str, documents: list[Document], question_id: str = "") -> tuple[str, dict]:
        pass

    @abstractmethod
    def _setup_chains(self):
        pass


class SlidersAgent(System):
    def __init__(self, config):
        super().__init__(config)
        self._setup_modules()

    def _setup_modules(self):
        self.extract_schema = ExtractSchema(self.config.get("extract_schema", {}), model_config=self.config["models"])
        self.generate_schema = GenerateSchema(
            self.config.get("generate_schema", {}), model_config=self.config["models"]
        )
        self.merge_tables = MergedTables(self.config.get("merge_tables", {}), model_config=self.config["models"])

    def _setup_chains(self):
        # Answer question from schema when we find tables
        answer_llm_client = get_llm_client(**self.config["models"]["answer"])
        answer_template = load_fewshot_prompt_template(
            template_file="sliders/answer_from_schema_tool_use.prompt",
            template_blocks=[],
        )
        self.answer_question_chain = answer_template | answer_llm_client.with_structured_output(Action)

        force_answer_llm_client = get_llm_client(**self.config["models"]["force_answer"])
        force_answer_template = load_fewshot_prompt_template(
            template_file="sliders/force_answer_from_schema_tool_use.prompt",
            template_blocks=[],
        )
        self.force_answer_question_chain = force_answer_template | force_answer_llm_client.with_structured_output(
            SQLAnswer
        )

        direct_answer_llm_client = get_llm_client(**self.config["models"]["direct_answer"])
        direct_answer_template = load_fewshot_prompt_template(
            template_file="sliders/direct_answer_from_tables.prompt",
            template_blocks=[],
        )
        self.direct_answer_chain = direct_answer_template | direct_answer_llm_client | StrOutputParser()

        check_if_merge_needed_llm_client = get_llm_client(**self.config["models"]["check_if_merge_needed"])
        check_if_merge_needed_template = load_fewshot_prompt_template(
            template_file="sliders/check_if_merge_needed.prompt",
            template_blocks=[],
        )
        self.check_if_merge_needed_chain = (
            check_if_merge_needed_template
            | check_if_merge_needed_llm_client.with_structured_output(TableProcessingNeeded)
        )

        # Answer question from schema when we don't find any tables
        answer_no_table_llm_client = get_llm_client(**self.config["models"]["answer_no_table"])
        answer_no_table_template = load_fewshot_prompt_template(
            template_file="sliders/answer_from_schema_no_table.prompt",
            template_blocks=[],
        )
        self.answer_question_no_table_chain = (
            answer_no_table_template | answer_no_table_llm_client.with_structured_output(Action)
        )

        # Generate final answer from SQL output
        tool_output_llm_client = get_llm_client(**self.config["models"]["answer_tool_output"])
        tool_output_template = load_fewshot_prompt_template(
            template_file="sliders/answer_with_tool_use_output.prompt",
            template_blocks=[],
        )
        self.tool_output_chain = tool_output_template | tool_output_llm_client | StrOutputParser()

        if self.config.get("generate_task_guidelines", False):
            task_guidelines_llm_client = get_llm_client(**self.config["models"]["task_guidelines"])
            task_guidelines_template = load_fewshot_prompt_template(
                template_file="sliders/task_guidelines.prompt",
                template_blocks=[],
            )
            self.create_task_guidelines_chain = task_guidelines_template | task_guidelines_llm_client

        if self.config.get("rephrase_question", False):
            rephrase_question_llm_client = get_llm_client(**self.config["models"]["rephrase_question"])
            rephrase_question_template = load_fewshot_prompt_template(
                template_file="sliders/rephrase_question.prompt",
                template_blocks=[],
            )
            self.rephrase_question_chain = (
                rephrase_question_template | rephrase_question_llm_client.with_structured_output(RephrasedQuestion)
            )

    def _initialize_metadata(
        self, question: str, documents: list[Document], start_time: float, question_id: str = ""
    ) -> dict:
        return {
            # Basic request information
            "question": question,
            "num_documents": len(documents),
            "document_names": [doc.document_name for doc in documents],
            "document_sizes": [len(doc.content) for doc in documents],
            "total_chunks": sum(len(doc.chunks) for doc in documents),
            # Processing pipeline timing
            "timing": {
                "start_time": start_time,
                "schema_generation": {},
                "schema_extraction": {},
                "table_merging": {},
                "answer_generation": {},
                "total_duration": None,
            },
            # Schema and extraction metrics
            "schema": {"generated_classes": 0, "total_fields": 0, "generation_tokens": 0, "generation_time": 0},
            # Extraction statistics
            "extraction": {
                "chunks_processed": 0,
                "successful_extractions": 0,
                "failed_extractions": 0,
                "retry_attempts": 0,
                "extraction_time": 0,
            },
            # Table merging information
            "merging": {
                "tables_created": 0,
                "sql_queries_executed": 0,
                "merge_failures": 0,
                "total_rows_processed": 0,
                "merging_time": 0,
                "merged_tables_dir_path": "",
            },
            # Answer generation metrics
            "answer_generation": {
                "sql_execution_attempts": 0,
                "sql_execution_errors": 0,
                "final_answer_tokens": 0,
                "used_tables": False,
                "answer_time": 0,
            },
            # Quality and performance metrics
            "quality": {"tables_with_data": 0, "empty_tables": 0, "data_completeness_score": 0.0},
            # Error tracking
            "errors": [],
            # Question ID
            "question_id": question_id,
        }

    def _finalize_metadata(self, metadata: dict, tables: list["Table"], start_time: float) -> dict:
        metadata["timing"]["total_duration"] = time.time() - start_time
        metadata["quality"]["tables_with_data"] = len(
            [t for t in tables if t.dataframe is not None and not t.dataframe.empty]
        )
        metadata["quality"]["empty_tables"] = len([t for t in tables if t.dataframe is None or t.dataframe.empty])

        # Calculate data completeness score
        if metadata["quality"]["tables_with_data"] + metadata["quality"]["empty_tables"] > 0:
            metadata["quality"]["data_completeness_score"] = metadata["quality"]["tables_with_data"] / (
                metadata["quality"]["tables_with_data"] + metadata["quality"]["empty_tables"]
            )
        return metadata

    async def run(self, question: str, documents: list[Document], question_id: str = "") -> tuple[str, dict]:
        start_time = time.time()

        # Initialize comprehensive metadata structure
        metadata = self._initialize_metadata(question, documents, start_time, question_id)

        # Generate task guidelines internally
        if self.config.get("generate_task_guidelines", False):
            task_guidelines_handler = LoggingHandler(
                prompt_file="sliders/task_guidelines.prompt",
                metadata={
                    "question": question,
                    "document_descriptions": [doc.description for doc in documents],
                    "question_id": question_id,
                },
            )
            task_guidelines = await self.create_task_guidelines_chain.ainvoke(
                {"question": question, "document_descriptions": [doc.description for doc in documents]},
                config={"callbacks": [task_guidelines_handler]},
            )
        else:
            task_guidelines = None

        if self.config.get("rephrase_question", False):
            rephrase_question_handler = LoggingHandler(
                prompt_file="sliders/rephrase_question.prompt",
                metadata={
                    "question": question,
                    "question_id": question_id,
                },
            )
            rephrase_question = await self.rephrase_question_chain.ainvoke(
                {"instruction": question}, config={"callbacks": [rephrase_question_handler]}
            )
        else:
            rephrase_question = None

        if rephrase_question:
            rephrased_question = rephrase_question.question
        else:
            rephrased_question = question

        logger.info("Generating schema...")
        # Generate schema
        schema = await self.generate_schema.generate(
            rephrased_question,
            documents,
            metadata,
            task_guidelines,
        )

        logger.info("Extracting tables from documents...")
        # Extract tables from documents
        extracted_tables = await self.extract_schema.extract(
            rephrased_question, schema, documents, metadata, task_guidelines
        )

        logger.info(f"Perform merge: {self.config.get('perform_merge', True)}")
        if self.config.get("perform_merge", True):
            merge_needed_dict = {}
            check_if_merge_needed = True
            logger.info(f"Check if merge needed: {self.config.get('check_if_merge_needed', False)}")
            if self.config.get("check_if_merge_needed", False):
                for table_name, table_data in extracted_tables.items():
                    sid = uuid.uuid4().hex[:8]
                    table_data_df, new_table_name = MergedTables.create_table_data(
                        table_name, extracted_tables[table_name], sid
                    )
                    formatted_table = format_table(table_data_df)
                    check_if_merge_needed_handler = LoggingHandler(
                        prompt_file="sliders/check_if_merge_needed.prompt",
                        metadata={
                            "question": question,
                            "table": formatted_table,
                            "schema": prepare_schema_repr(schema),
                            "question_id": question_id,
                        },
                    )
                    check_if_merge_needed_output = await self.check_if_merge_needed_chain.ainvoke(
                        {"question": question, "table": formatted_table, "schema": prepare_schema_repr(schema)},
                        config={"callbacks": [check_if_merge_needed_handler]},
                    )
                    check_if_merge_needed = check_if_merge_needed_output.processing_needed
                    merge_needed_dict[table_name] = check_if_merge_needed
                    logger.info(f"Table {table_name} merge needed: {check_if_merge_needed}")
            else:
                for table_name, table_data in extracted_tables.items():
                    merge_needed_dict[table_name] = True

            tables = []
            for table_name, merge_needed in merge_needed_dict.items():
                if merge_needed:
                    # Merge tables from different documents
                    logger.info("Merging tables from different documents...")
                    tables.extend(
                        await self.merge_tables.merge_chunks_tables(
                            {table_name: extracted_tables[table_name]},
                            documents,
                            question,
                            schema,
                            metadata,
                        )
                    )

                    logger.info(f"Merged tables directory path: {metadata['merging']['merged_tables_dir_path']}")
                else:
                    sid = uuid.uuid4().hex[:8]
                    table_data_df, new_table_name = MergedTables.create_table_data(
                        table_name, extracted_tables[table_name], sid
                    )
                    # Convert extracted_tables to ExtractedTable objects without merging
                    tables.append(
                        ExtractedTable(
                            name=table_name,
                            tables=schema,
                            sql_query=None,
                            dataframe=table_data_df,
                            dataframe_table_name=new_table_name,
                            table_str=format_table(table_data_df),
                        )
                    )
                    logger.info(f"Created {len(tables)} tables without merging for table {table_name}")
        else:
            tables = []
            for table_name in extracted_tables.keys():
                sid = uuid.uuid4().hex[:8]
                table_data_df, new_table_name = MergedTables.create_table_data(
                    table_name, extracted_tables[table_name], sid
                )
                tables.append(
                    ExtractedTable(
                        name=table_name,
                        tables=schema,
                        sql_query=None,
                        dataframe=table_data_df,
                        dataframe_table_name=new_table_name,
                        table_str=format_table(table_data_df),
                    )
                )

        # Answer question
        logger.info("Answering question from tables...")
        if self.config.get("force_sql", False):
            answer = await self._force_answer_question_from_tables(question, tables, schema, metadata)
        else:
            answer = await self._answer_question_from_tables(question, tables, schema, metadata)

        # Finalize metadata
        metadata = self._finalize_metadata(metadata, tables, start_time)

        if isinstance(answer, AIMessage):
            answer = answer.content

        return answer, metadata

    async def _force_answer_question_from_tables(
        self, question: str, tables: list["Table"], schema: Tables, metadata: dict
    ) -> str:
        answer_start_time = time.time()

        with DuckSQLBasic() as duck_sql_conn:
            for table in tables:
                if table.dataframe is not None:
                    duck_sql_conn.register(table.dataframe, table.dataframe_table_name)
                    if table.dataframe is not None:
                        table.table_str = (
                            str(tuple(table.dataframe.columns.to_list()))
                            + "\n"
                            + "\n".join([str(row) for row in table.dataframe.to_records(index=False)])
                        )
                else:
                    table.table_str = None

            tables = [table for table in tables if table.table_str is not None]

            # If no tables, answer question from schema
            if len(tables) == 0:
                metadata["answer_generation"]["used_tables"] = False
                answer_question_no_table_handler = LoggingHandler(
                    prompt_file="sliders/answer_from_schema_no_table.prompt",
                    metadata={
                        "question": question,
                        "classes": tables,
                        "feedback": None,
                        "question_id": metadata.get("question_id", None),
                        "stage": "answer_question_no_table",
                    },
                )
                result = await self.answer_question_no_table_chain.ainvoke(
                    {"question": question, "classes": tables, "feedback": None},
                    config={"callbacks": [answer_question_no_table_handler]},
                )
                final_answer = result.answer
            # If tables, answer question from tables
            else:
                metadata["answer_generation"]["used_tables"] = True

                async def query_llm_for_sql(feedback: str = None) -> SQLAnswer:
                    force_answer_question_handler = LoggingHandler(
                        prompt_file="sliders/force_answer_from_schema_tool_use.prompt",
                        metadata={
                            "question": question,
                            "classes": tables,
                            "feedback": feedback,
                            "question_id": metadata.get("question_id", None),
                            "stage": "generate_sql_answer",
                        },
                    )
                    result = await self.force_answer_question_chain.ainvoke(
                        {
                            "tables": tables,
                            "question": question,
                            "feedback": feedback,
                            "classes": prepare_schema_repr(schema),
                        },
                        config={"callbacks": [force_answer_question_handler]},
                    )

                    tool_output, error = run_sql_query(result.sql_query, duck_sql_conn)

                    if error:
                        logger.error(f"Error running SQL query: {error}")
                        logger.error(f"Tool output: {tool_output}")
                        logger.error(f"SQL query: {result.sql_query}")
                    return result.sql_query, tool_output, error

                sql_attempts = 0
                max_force_sql_attempts = 3
                error = True
                tool_output = None
                while sql_attempts < max_force_sql_attempts and error:
                    sql_attempts += 1

                    sql_query, tool_output, error = await query_llm_for_sql(feedback=tool_output)
                    if not error:
                        break

                if not error:
                    tool_output_handler = LoggingHandler(
                        prompt_file="sliders/answer_with_tool_use_output.prompt",
                        metadata={
                            "question": question,
                            "tool_call": sql_query,
                            "tool_output": json.dumps(tool_output),
                            "question_id": metadata.get("question_id", None),
                            "stage": "tool_output",
                        },
                    )
                    final_answer = await self.tool_output_chain.ainvoke(
                        {
                            "question": question,
                            "tool_call": sql_query,
                            "tool_output": json.dumps(tool_output),
                            "tables": tables,
                            "classes": prepare_schema_repr(schema),
                        },
                        config={"callbacks": [tool_output_handler]},
                    )

                else:
                    logger.info("Error running SQL query, generating direct answer...")
                    # direct_answer_handler = LoggingHandler(
                    #     prompt_file="sliders/direct_answer_from_tables.prompt",
                    #     metadata={
                    #         "question": question,
                    #         "classes": prepare_schema_repr(schema),
                    #         "question_id": metadata.get("question_id", None),
                    #         "stage": "direct_answer",
                    #     },
                    # )
                    # final_answer = await self.direct_answer_chain.ainvoke(
                    #     {"question": question, "classes": prepare_schema_repr(schema), "tables": tables},
                    #     config={"callbacks": [direct_answer_handler]},
                    # )
                    final_answer = "Error running SQL query"

        # Record answer generation timing
        metadata["timing"]["answer_generation"]["answer_time"] = time.time() - answer_start_time

        # Estimate final answer tokens (rough approximation)
        if isinstance(final_answer, str):
            metadata["answer_generation"]["final_answer_tokens"] = len(final_answer.split())

        return final_answer

    async def _answer_question_from_tables(
        self, question: str, tables: list["Table"], schema: Tables, metadata: dict
    ) -> str:
        answer_start_time = time.time()

        with DuckSQLBasic() as duck_sql_conn:
            for table in tables:
                if table.dataframe is not None:
                    duck_sql_conn.register(table.dataframe, table.dataframe_table_name)
                    if table.dataframe is not None:
                        table.table_str = (
                            str(tuple(table.dataframe.columns.to_list()))
                            + "\n"
                            + "\n".join([str(row) for row in table.dataframe.to_records(index=False)])
                        )
                else:
                    table.table_str = None

            tables = [table for table in tables if table.table_str is not None]

            # If no tables, answer question from schema
            if len(tables) == 0:
                metadata["answer_generation"]["used_tables"] = False
                answer_question_no_table_handler = LoggingHandler(
                    prompt_file="sliders/answer_from_schema_no_table.prompt",
                    metadata={
                        "question": question,
                        "classes": tables,
                        "feedback": None,
                        "question_id": metadata.get("question_id", None),
                        "stage": "answer_question_no_table",
                    },
                )
                result = await self.answer_question_no_table_chain.ainvoke(
                    {"question": question, "classes": tables, "feedback": None},
                    config={"callbacks": [answer_question_no_table_handler]},
                )
                final_answer = result.answer
            # If tables, answer question from tables
            else:

                async def query_llm_for_sql(feedback: str = None):
                    answer_question_handler = LoggingHandler(
                        prompt_file="sliders/answer_from_schema_tool_use.prompt",
                        metadata={
                            "question": question,
                            "classes": tables,
                            "feedback": feedback,
                            "question_id": metadata.get("question_id", None),
                            "stage": "answer_question",
                        },
                    )
                    result = await self.answer_question_chain.ainvoke(
                        {
                            "tables": tables,
                            "question": question,
                            "feedback": feedback,
                            "classes": prepare_schema_repr(schema),
                        },
                        config={"callbacks": [answer_question_handler]},
                    )

                    if result.run_sql:
                        tool_output, error = run_sql_query(result.sql_query, duck_sql_conn)
                        if error:
                            logger.error(f"Error running SQL query: {error}")
                            logger.error(f"SQL query: {result.sql_query}")
                        return result, tool_output, error
                    else:
                        return result, None, False

                metadata["answer_generation"]["used_tables"] = True

                max_answer_question_attempts = 3
                error = True
                tool_output = None
                answer_question_attempts = 0
                while answer_question_attempts < max_answer_question_attempts and error:
                    answer_question_attempts += 1
                    logger.info(f"Answer question attempts: {answer_question_attempts}")

                    result, tool_output, error = await query_llm_for_sql(feedback=tool_output)
                    if not error:
                        break

                if error:
                    direct_answer_handler = LoggingHandler(
                        prompt_file="sliders/direct_answer_from_tables.prompt",
                        metadata={
                            "question": question,
                            "classes": prepare_schema_repr(schema),
                            "question_id": metadata.get("question_id", None),
                            "stage": "direct_answer",
                        },
                    )
                    final_answer = await self.direct_answer_chain.ainvoke(
                        {"question": question, "classes": prepare_schema_repr(schema), "tables": tables},
                        config={"callbacks": [direct_answer_handler]},
                    )
                elif result.run_sql:
                    tool_output_handler = LoggingHandler(
                        prompt_file="sliders/answer_with_tool_use_output.prompt",
                        metadata={
                            "question": question,
                            "tool_call": result.sql_query,
                            "tool_output": json.dumps(tool_output),
                            "question_id": metadata.get("question_id", None),
                            "stage": "tool_output",
                        },
                    )
                    final_answer = await self.tool_output_chain.ainvoke(
                        {
                            "question": question,
                            "tool_call": result.sql_query,
                            "tool_output": json.dumps(tool_output),
                            "tables": tables,
                            "classes": prepare_schema_repr(schema),
                        },
                        config={"callbacks": [tool_output_handler]},
                    )
                else:
                    final_answer = result.answer

        # Record answer generation timing
        metadata["timing"]["answer_generation"]["answer_time"] = time.time() - answer_start_time

        # Estimate final answer tokens (rough approximation)
        if isinstance(final_answer, str):
            metadata["answer_generation"]["final_answer_tokens"] = len(final_answer.split())

        return final_answer
