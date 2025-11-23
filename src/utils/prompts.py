from enum import StrEnum

from src.constants import prompts_dir


class Prompts(StrEnum):
    BASELINE = "baseline.md"


def get_prompt(prompt: Prompts) -> str:
    with open(prompts_dir / prompt, "r") as f:
        return f.read()
