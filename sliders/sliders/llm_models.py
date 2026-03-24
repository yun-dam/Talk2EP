from typing import Optional, Type, get_args, get_origin
from dataclasses import dataclass
import pandas as pd
from pydantic import BaseModel, create_model
from pydantic import Field as PydanticField

from sliders.log_utils import logger
from sliders.utils import pydantic_model_to_signature, string_to_type


class IsRelevantPage(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for the decision if the page is relevant to the question.",
    )
    is_relevant: bool = PydanticField(
        ...,
        description="Whether the page is relevant to the question.",
    )


class Evaluation(BaseModel):
    explanation: str
    correct: bool


class EvaluationScore(BaseModel):
    explanation: str
    correct: int


class SequentialAnswer(BaseModel):
    scratchpad: str
    answer: str
    found_answer: bool


class ChunkAnswer(BaseModel):
    answer: str
    found_answer: bool


class Action(BaseModel):
    reasoning: str
    run_sql: bool
    answer: str | None
    sql_query: str | None


class SQLAnswer(BaseModel):
    reasoning: str
    sql_query: str


class Normalization(BaseModel):
    currency: Optional[str]
    percent: Optional[str]
    date_format: Optional[str]


class Field(BaseModel):
    name: str
    data_type: str
    enum_values: Optional[list[str]]
    unit: Optional[str]
    scale: Optional[str]
    description: str
    required: bool
    normalization: Optional[Normalization]


class Table(BaseModel):
    name: str
    description: str
    fields: list[Field]


class Tables(BaseModel):
    tables: list[Table]


# class Class(BaseModel):
#     name: str
#     fields: list[Field]


# class Classes(BaseModel):
#     classes: list[Class]


class NewFieldValues(BaseModel):
    row_number: int
    name: str
    reasoning: str
    value: str | float | int | bool | None
    field: str


class NewField(BaseModel):
    name: str
    description: str
    extraction_guideline: str
    data_type: str
    unit: str
    scale: str


class NewFields(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for which new fields are required. Each new field should be atomic and not be a combination of other fields.",
    )
    fields: list[NewField]
    # values: list[NewFieldValues]


class Column(BaseModel):
    reason: str = PydanticField(description="The reason for the decision to compute the new value for the field.")
    field_name: str = PydanticField(description="The name of the field that is being computed.")
    row_ids: list[int] = PydanticField(
        description="The row ids of the rows that are used to compute the new value for the field."
    )
    new_column_name: str = PydanticField(
        description="After the SQL query is executed, if there is a new column name for the exisiting field, then this should be the name of the new column."
    )


class Decision(BaseModel):
    reasoning: str = PydanticField(description="The reasoning for the decision to compute the new value for fields.")
    fields: list[Column] = PydanticField(description="The fields that are being computed and the new values for them.")


class Output(BaseModel):
    decision: Decision = PydanticField(description="The decision to compute the new value for fields.")
    sql_query: str = PydanticField(description="The SQL query to compute the new value for fields.")


class ObjectiveNecessity(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for the decision if the objective is necessary cleaning the data.",
    )
    required: bool = PydanticField(
        ...,
        description="Whether the objective is necessary for the answer.",
    )


class TableOperation(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for how you will construct the sql query to perform the table operation.",
    )
    sql_query: str = PydanticField(
        ...,
        description="The SQL query to perform the table operation.",
    )


class ProvenanceSQL(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for how you will construct the sql query to get the provenance of the new table data.",
    )
    sql_query: str = PydanticField(
        ...,
        description="The SQL query to get the provenance of the new table data.",
    )


def create_dynamic_extraction_relation_model(
    tables: list[Table],
) -> Type[BaseModel]:
    relation_models = []
    for table in tables:
        field_models = {}
        for field in table.fields:
            # this data_type is different since we want to allow lists of types
            data_type = string_to_type(field.data_type)

            # get the list of types
            origin = get_origin(data_type)
            args = get_args(data_type)
            if origin is list and args:
                value_type = args[0]
            else:
                value_type = data_type

            field_models[field.name] = create_model(
                "Extracted",
                reasoning=(str, ...),
                value=(Optional[value_type], ...),
                quote=(Optional[list[str]], ...),
                is_explicit=(bool, ...),
            )

        relation_models.append(create_model(table.name, **field_models))

    model = create_model(
        "ExtractionOutput",
        extraction_plan=(str, ...),
        **{model_name.__name__: (list[model_name], ...) for model_name in relation_models},
    )
    logger.info(pydantic_model_to_signature(model, func_name="ExtractionOutput"))
    return model


class DocumentDescriptions(BaseModel):
    descriptions: list[str] = PydanticField(
        ...,
        description="List of document descriptions, one per document.",
        min_items=1,
    )

    def __iter__(self):
        return iter(self.descriptions)

    def __len__(self):
        return len(self.descriptions)

    def __getitem__(self, idx):
        return self.descriptions[idx]


class DocumentTitle(BaseModel):
    thought: str = PydanticField(
        ...,
        description="The thought process for the decision to extract/generate the title for the document.",
    )
    title: str = PydanticField(
        ...,
        description="The title of the document.",
    )


class ErrorAnalysisResponse(BaseModel):
    """Model for error analysis of a question's execution."""

    within_reason: bool = PydanticField(
        ...,
        description="Regardless if the actual evaluator marked this as correct or not, is the difference between the gold answers and the predicted answer within reason? Check if the predicted answer correctly captures the meaning of the gold answer, even if it provides more or fewer details, uses different wording, or varies in specificity.",
    )
    error_type: str = PydanticField(
        ...,
        description="The high-level categorization of the error (e.g., 'Schema Mismatch', 'Data Quality Issue', 'Reasoning Error')",
    )
    pipeline_stage: str = PydanticField(
        ...,
        description="The stage of the pipeline where the error occurred. If there are no relevant logs included in the input, then there is no way of knowing which stage the error was introduced in. The pipeline stage should thus be 'unknown'.",
        # Limit to known pipeline stages
        pattern="^(schema generation|extraction|merge|answer generation|unknown)$",
    )
    error_description: str = PydanticField(..., description="Detailed explanation of what went wrong and why")
    improvement_suggestion: str = PydanticField(
        ..., description="Concrete suggestions for how to fix or prevent this error in the future"
    )

    # class Config:
    #     extra = "forbid"  # Prevent additional properties


class RephrasedQuestion(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="Thought process for how to rephrase the question.",
    )
    question: str = PydanticField(
        ...,
        description="The rephrased question. The question should be different from the original question, but should be equivalent in meaning.",
    )


class TableProcessingNeeded(BaseModel):
    reasoning: str = PydanticField(
        ...,
        description="The reasoning for the decision to process the tables for this question. If the question can be directly answered from the tables, then we do not need to process the tables.",
    )
    processing_needed: bool = PydanticField(
        ...,
        description="Whether the tables should be processed for this question.",
    )


@dataclass
class ExtractedTable:
    name: str
    tables: list[Table]
    sql_query: Output
    dataframe: pd.DataFrame
    dataframe_table_name: str
    table_str: str = None
    actions: list[dict] = None
