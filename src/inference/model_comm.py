import os
from typing import Any, Type, TypeVar

import openai
from pydantic import BaseModel

from src.inference import Models


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


T = TypeVar("T", bound=BaseModel)


def make_request(
    input_prompt: str,
    model_name: Models,
    messages: list[Any] | None = None,
    format: Type[T] | None = None,
    imgs: list[str] | None = None,
) -> tuple[T | str, list[Any]]:
    messages = messages or []

    client = openai.OpenAI(
        base_url=os.environ.get("SERVER_ADDRESS", "http://localhost:8080/v1"),
        api_key="sk-no-key-required",
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
        messages.append(make_assistant_message(res.model_dump_json()))
        return res, messages
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        res = response.choices[0].message.content
        assert res
        messages.append(make_assistant_message(res))
        return res, messages
