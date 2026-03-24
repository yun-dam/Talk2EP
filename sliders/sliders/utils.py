from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from pydantic import BaseModel

from sliders.log_utils import logger

if TYPE_CHECKING:
    from sliders.llm_models import Tables


def string_to_type(value_type: str) -> Type:
    # Add typing module types to the local namespace
    local_namespace = {
        "List": List,
        "list": list,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "Literal": Literal,
    }

    # Handle special case for Literal types
    original_value_type = value_type
    value_type = value_type.strip()

    if value_type.lower().startswith("literal["):
        # Extract the arguments part and reconstruct with proper capitalization
        args_part = value_type[value_type.find("[") :]
        value_type = f"Literal{args_part}"

    try:
        # Evaluate the string safely using eval
        return eval(value_type, {"__builtins__": None}, local_namespace)
    except Exception as e:
        logger.warning(f"Failed to parse type: {original_value_type} -> {value_type}, error: {e}")
        try:
            # If eval fails with the modified string, try with the original
            return eval(original_value_type, {"__builtins__": None}, local_namespace)
        except Exception as e2:
            raise ValueError(f"Invalid type string: {original_value_type}") from e2


def type_to_str(tp):
    """Convert a type annotation to string, handling containers and Pydantic models."""
    origin = get_origin(tp)
    args = get_args(tp)

    if origin is None:
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            # Recurse into nested BaseModel
            fields = ", ".join(f"{k}: {type_to_str(v.annotation)}" for k, v in tp.model_fields.items())
            return f"{tp.__name__}({fields})"
        return tp.__name__ if hasattr(tp, "__name__") else str(tp)

    elif origin in (list, List):
        return f"List[{type_to_str(args[0])}]"
    elif origin in (dict, Dict):
        return f"Dict[{type_to_str(args[0])}, {type_to_str(args[1])}]"
    elif origin is Union:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) < len(args):
            return f"Optional[{type_to_str(non_none[0])}]"
        return "Union[" + ", ".join(type_to_str(a) for a in args) + "]"
    else:
        return str(tp)


def pydantic_model_to_signature(model: type[BaseModel], func_name: str = "my_function") -> str:
    type_hints = get_type_hints(model)
    args = []

    for name, field in model.model_fields.items():
        annotation = type_hints.get(name, Any)
        type_str = type_to_str(annotation)

        if field.is_required:
            args.append(f"{name}: {type_str}")
        else:
            default_repr = repr(field.default)
            args.append(f"{name}: {type_str} = {default_repr}")

    return f"def {func_name}({', '.join(args)}):\n    pass"


def prepare_schema_repr(schema: "Tables") -> str:
    class_repr = ""
    for one_table in schema.tables:
        one_class_repr = f"## Table Name: {one_table.name} ({one_table.description})\n"
        for field in one_table.fields:
            one_class_repr += f"### Field Name: {field.name}\n"
            one_class_repr += f"Description: {field.description}\n"
            one_class_repr += f"Data Type: {field.data_type}\n"
            if field.data_type == "enum":
                one_class_repr += f"Enum Values: {field.enum_values}\n"
            one_class_repr += f"Unit: {field.unit}\n"
            one_class_repr += f"Scale: {field.scale}\n"
            # one_class_repr += f"Required: {field.required}\n"
            one_class_repr += f"Normalization: {field.normalization}\n"
        class_repr += one_class_repr + "\n"
    return class_repr
