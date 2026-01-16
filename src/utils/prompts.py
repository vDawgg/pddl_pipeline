from enum import StrEnum

from src.constants import prompts_dir
from src.utils.domains import Domains


class Prompts(StrEnum):
    GENERATION_CONTEXT = "generation_context.md"
    GENERATION_CONTEXT_TOOLS = "generation_context_tools.md"
    BLOCKSWORLD_DOMAIN = "blocksworld_domain.md"
    BLOCKSWORLD_PROBLEM = "blocksworld_problem.md"
    RING_AND_PEG = "ring_and_peg.md"
    RING_AND_PEG_DOMAIN = "ring_and_peg_domain.md"
    RING_AND_PEG_PROBLEM = "ring_and_peg_problem.md"
    VAL_FEEDBACK_CONTEXT = "val_feedback_context.md"
    VAL_FEEDBACK_DOMAIN = "val_feedback_domain.md"
    VAL_FEEDBACK_PROBLEM = "val_feedback_problem.md"
    PLANNER_CONTEXT = "planner_context.md"
    PLANNER_TASK = "planner_task.md"
    PLANNER_TRANSLATE_CONTEXT = "planner_translate_context.md"
    PLANNER_TRANSLATE_TASK = "planner_translate_task.md"
    REACT_BASE = "react_base.md"
    TRAJECTORY = "trajectory.md"
    ITERATION = "iteration.md"


def get_prompt(*prompts: Prompts) -> str:
    parts = []
    for p in prompts:
        with open(prompts_dir / p) as f:
            parts.append(f.read())

    return "\n\n".join(parts)


domain_pompts = {
    Domains.BLOCKSWORLD: get_prompt(
        Prompts.GENERATION_CONTEXT, Prompts.BLOCKSWORLD_DOMAIN
    ),
    Domains.RING_AND_PEG: get_prompt(
        Prompts.GENERATION_CONTEXT, Prompts.RING_AND_PEG_DOMAIN
    ),
}

problem_prompts = {
    Domains.BLOCKSWORLD: get_prompt(Prompts.BLOCKSWORLD_PROBLEM),
    Domains.RING_AND_PEG: get_prompt(Prompts.RING_AND_PEG_PROBLEM),
}
