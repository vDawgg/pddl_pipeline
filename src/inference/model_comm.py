import inspect
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from types import UnionType
from typing import (
    Any,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import openai
from pydantic import BaseModel

from src.constants import project_root
from src.inference import Models, provider_hosts, provider_keys
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
        return ReactResponse(
            thought=thought.group(1).strip(),
            tool_name=tool_name.group(1).strip(),
            tool_args=json.loads(tool_args.group(1).strip().replace("'", '"')),
        )
    return None


# TODO: Think of a cleaner way to structure the results here.
def make_prompt_with_trajectory(
    input_prompt: str, parsed_responses: list[ReactResponse], results: list[str]
) -> str:
    if len(parsed_responses) > 0:
        unformatted_trajectory = get_prompt(Prompts.TRAJECTORY)
        trajectory = ""
        for i in range(len(parsed_responses)):
            parsed_response = parsed_responses[i]
            unformatted_iteration = get_prompt(Prompts.ITERATION)
            trajectory += (
                unformatted_iteration.format(
                    iteration=i,
                    thought=parsed_response.thought,
                    tool_name=parsed_response.tool_name,
                    tool_args=parsed_response.tool_args,
                    tool_result=results[i],
                )
                + "\n"
            )
        return unformatted_trajectory.format(
            input_prompt=input_prompt, trajectory=trajectory
        )
    return input_prompt


T = TypeVar("T", bound=BaseModel)


def make_request[T](
    input_prompt: str,
    model_name: Models,
    messages: list[Any] | None = None,
    format: type[T] | None = None,
    imgs: list[str] | None = None,
) -> tuple[T | str, list[Any]]:
    messages = messages or []

    logger.debug("# User Message")
    logger.debug(input_prompt)

    key_path = project_root / provider_keys[model_name]
    if not key_path.exists():
        raise FileNotFoundError(f"API key file not found: {key_path}")

    client = openai.OpenAI(
        base_url=provider_hosts[model_name],
        api_key=open(str(key_path)).readline().strip(),
    )

    if imgs:
        messages.append(make_user_message_with_image(input_prompt, imgs))
    else:
        messages.append(make_user_message(input_prompt))

    if format:
        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=format,
        )
        res = response.choices[0].message.parsed
        assert res
        return res, messages
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        res = response.choices[0].message.content
        logger.debug("# Assistant Message")
        logger.debug(res)
        assert res is not None
        res = res.removeprefix("```pddl")
        res = res.removesuffix("```")
        return res.strip(), messages


# TODO: This might have to be dumbed down with the logic transferred to the pipeline
#       Otherwise we might have problems with the structure here.
# TODO: Try with native openai tool-calling as well
#       -> This structure seems to be supported by most of the newer models.
def make_react_workflow(
    model_name: str,
    input_prompt: str,
    tools: list[Callable],
    max_iters=10,
) -> str:
    tools_json = [make_tool(t) for t in tools]
    tools_dict = {t.__name__: t for t in tools}

    unformatted_base_prompt = get_prompt(Prompts.REACT_BASE)
    base_prompt = unformatted_base_prompt.format(tools=tools_json)
    input_prompt = base_prompt + input_prompt

    key_path = project_root / provider_keys[model_name]
    if not key_path.exists():
        raise FileNotFoundError(f"API key file not found: {key_path}")
    client = openai.OpenAI(
        base_url=provider_hosts[model_name],
        api_key=open(str(key_path)).readline().strip(),
    )

    res = None
    parsed_responses = []
    tool_results = []
    for _ in range(max_iters):
        prompt_with_trajectory = make_prompt_with_trajectory(
            input_prompt, parsed_responses, tool_results
        )
        res = (
            client.chat.completions.create(
                model=model_name,
                messages=[
                    make_user_message(prompt_with_trajectory)  # type: ignore
                ],
            )
            .choices[0]
            .message.content
        )
        assert res is not None
        logger.debug("# Assistant Message")
        logger.debug(res)

        parsed_response = parse_react_message(res)
        if parsed_response is None:
            logger.error("Assistant answered with malformed response")
            # TODO: Think about what we actually want to do here
            continue

        parsed_responses.append(parsed_response)
        if parsed_response.tool_name == "finish":
            break
        tool_results.append(
            tools_dict[parsed_response.tool_name](**parsed_response.tool_args)
        )
        logger.debug("# Tool Call")
        logger.debug(f"## Tool: {parsed_response.tool_name}")
        logger.debug(f"## Tool args: {parsed_response.tool_args}")
        logger.debug(f"## Tool result: {tool_results[-1]}")
    assert res is not None
    return res.strip()
