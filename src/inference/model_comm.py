import os

from typing import Any, Type, TypeVar

import openai
from pydantic import BaseModel


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
    messages: list[Any] = [],
    format: Type[T] | None = None,
    imgs: list[str] | None = None,
) -> T | str | None:
    model_name = "Qwen/Qwen3-VL-8B-Instruct"
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
        return response.choices[0].message.parsed
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        return response.choices[0].message.content
