import time


from sliders.document import Document
from sliders.llm_models import Field, Tables, Table

from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler


class GenerateSchema:
    def __init__(self, config: dict, model_config: dict):
        self.config = config
        self.model_config = model_config

    @staticmethod
    def create_generate_schema_chain(**kwargs):
        llm_client = get_llm_client(**kwargs)
        generate_schema_template = load_fewshot_prompt_template(
            template_file="sliders/generate_schema_qa.prompt",
            template_blocks=[],
        )
        generate_schema_chain = generate_schema_template | llm_client.with_structured_output(Tables)
        return generate_schema_chain

    async def generate(
        self, question: str, documents: list[Document], metadata: dict, task_guidelines: str | None
    ) -> Tables:
        schema_start_time = time.time()
        handler = LoggingHandler(
            prompt_file="sliders/generate_schema_qa.prompt",
            metadata={
                "question": question,
                "stage": "generate_schema",
                **(metadata or {}),
            },
        )
        try:
            schema = await self.create_generate_schema_chain(
                **self.model_config["generate_schema"],
            ).ainvoke(
                {
                    "document_description": documents[0].description,
                    "question": question,
                    "task_guidelines": task_guidelines,
                },
                config={"callbacks": [handler]},
            )

            # Update schema metadata
            metadata["timing"]["schema_generation"]["generation_time"] = time.time() - schema_start_time
            metadata["schema"]["generated_classes"] = len(schema.tables)
            metadata["schema"]["total_fields"] = sum(len(cls.fields) for cls in schema.tables)
            metadata["schema"]["generation_time"] = time.time() - schema_start_time

            # Store schema complexity metrics
            metadata["schema"]["average_fields_per_class"] = (
                metadata["schema"]["total_fields"] / metadata["schema"]["generated_classes"]
                if metadata["schema"]["generated_classes"] > 0
                else 0
            )

            # Store field types distribution
            field_types = {}
            for cls in schema.tables:
                for field in cls.fields:
                    field_type = field.data_type
                    field_types[field_type] = field_types.get(field_type, 0) + 1
            metadata["schema"]["field_types_distribution"] = field_types

            # Store the schema object for backward compatibility
            metadata["schema"]["schema_object"] = schema.model_dump()

            if self.config.get("add_extra_information_class", False):
                schema.tables.append(
                    Table(
                        name="AdditionalInformation",
                        fields=[
                            Field(
                                name="additional_information",
                                data_type="str",
                                description="Additional information that is useful for answering the question, but isn't covered by the other relationship schema.",
                                unit=None,
                                scale=None,
                                required=False,
                                normalization=None,
                            )
                        ],
                    )
                )

            return schema

        except Exception as e:
            metadata["errors"].append(
                {
                    "stage": "schema_generation",
                    "error": str(e),
                    "question": question,
                    "document_descriptions": [doc.description for doc in documents],
                }
            )
            metadata["timing"]["schema_generation"]["generation_time"] = time.time() - schema_start_time
            raise
