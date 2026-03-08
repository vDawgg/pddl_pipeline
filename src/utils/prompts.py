from enum import StrEnum

from src.constants import prompts_dir
from src.utils.domains import Domains


class Prompts(StrEnum):
    GENERATION_CONTEXT = "generation_context.md"
    GENERATION_CONTEXT_TOOLS = "generation_context_tools.md"
    GENERATION_CONTEXT_TOOLS_IMAGES = "generation_context_tools_images.md"
    GENERATION_CONTEXT_IMAGES = "generation_context_images.md"
    GENERATION_CONTEXT_PLAN_FEEDBACK = "generation_context_plan_feedback.md"
    BLOCKSWORLD_DOMAIN = "blocksworld_domain.md"
    BLOCKSWORLD_PROBLEM = "blocksworld_problem.md"
    RING_AND_PEG = "ring_and_peg.md"
    RING_AND_PEG_DOMAIN = "ring_and_peg_domain.md"
    RING_AND_PEG_PROBLEM = "ring_and_peg_problem.md"
    RING_AND_PEG_PLAN = "ring_and_peg_plan.md"
    VAL_FEEDBACK_CONTEXT = "val_feedback_context.md"
    VAL_FEEDBACK_CONTEXT_IMAGES = "val_feedback_context_images.md"
    VAL_FEEDBACK_DOMAIN = "val_feedback_domain.md"
    VAL_FEEDBACK_PROBLEM = "val_feedback_problem.md"
    PLANNER_CONTEXT = "planner_context.md"
    PLANNER_CONTEXT_IMAGES = "planner_context_images.md"
    PLANNER_TASK = "planner_task.md"
    PLANNER_TRANSLATE_CONTEXT = "planner_translate_context.md"
    PLANNER_TRANSLATE_CONTEXT_IMAGES = "planner_translate_context_images.md"
    PLANNER_TRANSLATE_TASK = "planner_translate_task.md"
    PLAN_FEEDBACK = "plan_feedback.md"


def get_prompt(*prompts: Prompts) -> str:
    parts = []
    for p in prompts:
        with open(prompts_dir / p) as f:
            parts.append(f.read())

    return "\n\n".join(parts)


domain_prompts = {
    Domains.BLOCKSWORLD: get_prompt(
        Prompts.GENERATION_CONTEXT, Prompts.BLOCKSWORLD_DOMAIN
    ),
    Domains.RING_AND_PEG: get_prompt(
        Prompts.GENERATION_CONTEXT, Prompts.RING_AND_PEG_DOMAIN
    ),
}

domain_prompts_tools = {
    Domains.BLOCKSWORLD: get_prompt(
        Prompts.GENERATION_CONTEXT_TOOLS, Prompts.BLOCKSWORLD_DOMAIN
    ),
    Domains.RING_AND_PEG: get_prompt(
        Prompts.GENERATION_CONTEXT_TOOLS, Prompts.RING_AND_PEG_DOMAIN
    ),
}

domain_prompts_images = {
    Domains.BLOCKSWORLD: get_prompt(
        Prompts.GENERATION_CONTEXT_IMAGES, Prompts.BLOCKSWORLD_DOMAIN
    ),
    Domains.RING_AND_PEG: get_prompt(
        Prompts.GENERATION_CONTEXT_IMAGES, Prompts.RING_AND_PEG_DOMAIN
    ),
}

problem_prompts = {
    Domains.BLOCKSWORLD: get_prompt(Prompts.BLOCKSWORLD_PROBLEM),
    Domains.RING_AND_PEG: get_prompt(Prompts.RING_AND_PEG_PROBLEM),
}


def add_line_numbers(lines: list[str]) -> list[str]:
    lines_with_line_numbers = []
    for i, line in enumerate(lines):
        lines_with_line_numbers.append(f"{i}:\t" + line)
    return lines_with_line_numbers


def make_file_view(file_content: str) -> str:
    lines = [line + "\n" for line in file_content.split("\n")]
    return "```pddl\n" + "".join(add_line_numbers(lines)) + "```"
