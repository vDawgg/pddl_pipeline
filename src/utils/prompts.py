from enum import StrEnum

from src.constants import prompts_dir


class Prompts(StrEnum):
    BASELINE_CONTEXT = "baseline_context.md"
    BASELINE_DOMAIN = "baseline_domain.md"
    BASELINE_PROBLEM = "baseline_problem.md"
    VAL_FEEDBACK_CONTEXT = "val_feedback_context.md"
    VAL_FEEDBACK_DOMAIN = "val_feedback_domain.md"
    VAL_FEEDBACK_PROBLEM = "val_feedback_problem.md"
    PLANNER_CONTEXT = "planner_context.md"
    PLANNER_TASK = "planner_task.md"
    PLANNER_TRANSLATE_CONTEXT = "planner_translate_context.md"
    PLANNER_TRANSLATE_TASK = "planner_translate_task.md"


def get_prompt(*prompts: Prompts) -> str:
    parts = []
    for p in prompts:
        with open(prompts_dir / p, "r") as f:
            parts.append(f.read())

    return "\n\n".join(parts)
