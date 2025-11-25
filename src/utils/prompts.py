from enum import StrEnum

from src.constants import prompts_dir


class Prompts(StrEnum):
    BASELINE_CONTEXT = "baseline_context.md"
    BASELINE_DOMAIN = "baseline_domain.md"
    BASELINE_PROBLEM = "baseline_problem.md"


def get_prompt(*prompts: Prompts) -> str:
    parts = []
    for p in prompts:
        with open(prompts_dir / p, "r") as f:
            parts.append(f.read())

    return "\n\n".join(parts)
