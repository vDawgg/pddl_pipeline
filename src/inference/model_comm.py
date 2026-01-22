import inspect
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import (
    Any,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from src.utils.prompts import Prompts, get_prompt

logger = logging.getLogger(__name__)


def make_assistant_message(message: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": message,
            },
        ],
    }


def make_user_message(message: str) -> dict:
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": message,
            },
        ],
    }


def make_user_message_with_image(message: str, b64_images: list[str]) -> dict:
    user_message = make_user_message(message)
    user_message["content"].extend(
        [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64_image}",
                },
            }
            for b64_image in b64_images
        ]
    )
    return user_message


def get_complex_type(
    origin: Any, py_type: Any, param_name: str | None, defs: dict
) -> dict | None:
    if origin is None:
        return None
    args = get_args(py_type)
    if origin is UnionType or origin is Union:
        assert param_name is not None
        return {
            "anyOf": [python_type_to_json_schema(arg, defs) for arg in args],
            "title": param_name,
        }
    elif origin is list:
        if args:
            return {
                "type": "array",
                "items": python_type_to_json_schema(args[0], defs),
            }
        return {"type": "array"}
    elif origin is dict:
        return {"type": "object"}
    elif origin is tuple:
        return {
            "maxItems": len(args),
            "minItems": len(args),
            "prefixItems": [python_type_to_json_schema(arg, defs) for arg in args],
            "type": "array",
        }


def get_enum_type(py_type: Any, defs: dict) -> dict:
    enum_name = py_type.__name__
    if enum_name not in defs:
        defs[enum_name] = {
            "enum": [e.value for e in py_type],
            "title": enum_name,
            "type": "string",
        }
    return {"$ref": f"#/$defs/{enum_name}"}


def get_primitive_types(py_type: Any) -> dict:
    if py_type is str:
        return {"type": "string"}
    elif py_type is int:
        return {"type": "integer"}
    elif py_type is float:
        return {"type": "number"}
    elif py_type is bool:
        return {"type": "boolean"}
    elif py_type is list:
        return {"type": "array"}
    elif py_type is dict:
        return {"type": "object"}
    elif py_type is type(None):
        return {"type": "null"}
    return {"type": "string"}


def python_type_to_json_schema(
    py_type: Any, defs: dict, param_name: str | None = None
) -> dict:
    """
    Convert a Python type to a JSON Schema dict.
    Handles enums by adding them to $defs and returning a $ref.
    """
    origin = get_origin(py_type)
    if generic_type := get_complex_type(origin, py_type, param_name, defs):
        return generic_type
    if isinstance(py_type, type) and issubclass(py_type, Enum):
        return get_enum_type(py_type, defs)
    return get_primitive_types(py_type)


def make_tool(func: Callable) -> dict:
    func_name = func.__name__
    doc = inspect.getdoc(func)
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    properties = {}
    required = []
    defs = {}

    for param_name, param in signature.parameters.items():
        param_type = type_hints.get(param_name, str)
        schema = python_type_to_json_schema(param_type, defs, param_name)

        if param.default is inspect.Parameter.empty:
            properties[param_name] = schema
            required.append(param_name)
        else:
            properties[param_name] = schema

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    if defs:
        parameters["$defs"] = defs

    tool_json = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": doc,
            "parameters": parameters,
        },
    }
    return tool_json


@dataclass
class ReactResponse:
    thought: str
    tool_name: str
    tool_args: dict


def parse_react_message(response: str) -> ReactResponse | None:
    thought_header = r"\[\[\s*--\s*thought\s*--\s*\]\]"
    tool_name_header = r"\[\[\s*--\s*tool_name\s*--\s*\]\]"
    tool_args_header = r"\[\[\s*--\s*tool_args\s*--\s*\]\]"
    block_content = r"\s*(.*?)\s*(?=\n\s*\n|\Z)"
    json_object = r"\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
    thought_block = re.compile(thought_header + block_content, re.DOTALL)
    tool_name_block = re.compile(tool_name_header + block_content, re.DOTALL)
    tool_args_block = re.compile(tool_args_header + json_object, re.DOTALL)
    if (
        (thought := thought_block.search(response))
        and (tool_name := tool_name_block.search(response))
        and (tool_args := tool_args_block.search(response))
    ):
        try:
            return ReactResponse(
                thought=thought.group(1).strip(),
                tool_name=tool_name.group(1).strip(),
                tool_args=json.loads(tool_args.group(1).strip().replace("'", '"')),
            )
        except Exception:
            return None
    return None


def make_prompt_with_trajectory(
    input_prompt: str,
    parsed_responses: list[ReactResponse],
    results: list[str],
    max_past_steps: int,
    domain_file_path: Path | None = None,
    problem_file_path: Path | None = None,
) -> str:
    if len(parsed_responses) > 0:
        if len(parsed_responses) > max_past_steps:
            parsed_responses_working_copy = parsed_responses[-max_past_steps:]
            results_working_copy = results[-max_past_steps:]
        else:
            parsed_responses_working_copy = parsed_responses
            results_working_copy = results
        unformatted_trajectory = get_prompt(Prompts.TRAJECTORY)
        trajectory = ""
        assert len(results_working_copy) == len(parsed_responses_working_copy)
        for i in range(len(parsed_responses_working_copy)):
            parsed_response = parsed_responses_working_copy[i]
            unformatted_iteration = get_prompt(Prompts.ITERATION)
            trajectory += (
                unformatted_iteration.format(
                    iteration=i,
                    thought=parsed_response.thought,
                    tool_name=parsed_response.tool_name,
                    tool_args=parsed_response.tool_args,
                    tool_result=results_working_copy[i],
                )
                + "\n"
            )
        return unformatted_trajectory.format(
            domain_file_path=domain_file_path or "Not yet created",
            problem_file_path=problem_file_path or "Not yet created",
            input_prompt=input_prompt,
            trajectory=trajectory,
        )
    return input_prompt
