import logging
from typing import Any, Type, TypeVar

import openai
from pydantic import BaseModel

from src.inference import Models, provider_keys, provider_hosts
from src.constants import project_root

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


T = TypeVar("T", bound=BaseModel)


def make_request(
    input_prompt: str,
    model_name: Models,
    messages: list[Any] | None = None,
    format: Type[T] | None = None,
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
        assert res
        return res, messages
