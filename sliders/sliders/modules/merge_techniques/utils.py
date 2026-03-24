from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sliders.llm_models import Tables, Field


def get_table_schema(table_name: str, schema: Tables) -> list[Field]:
    for table in schema.tables:
        if table.name == table_name:
            return table.fields
    return None


def sanitize_table_name(name: str) -> str:
    # Replace special characters and spaces with underscore
    import re

    # Remove all special characters except alphanumeric and underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")
    return sanitized


def format_table(table_data):
    return (
        str(tuple(table_data.columns.to_list()))
        + "\n"
        + "\n".join([str(row) for row in table_data.to_records(index=False)])
    )
