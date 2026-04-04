from src.base.schemas import Prompts
from src.constants import prompts_dir


def get_prompt(*prompts: Prompts) -> str:
    parts = []
    for p in prompts:
        with open(prompts_dir / p) as f:
            parts.append(f.read())

    return "\n\n".join(parts)


def get_domain_problem_prompt(domain: Prompts, problem: Prompts) -> str:
    return get_prompt(Prompts.DOMAIN_AND_PROBLEM).format(
        domain=get_prompt(domain),
        problem=get_prompt(problem),
    )


def get_domain_problem_and_context_prompt(
    context: Prompts, domain: Prompts, problem: Prompts
) -> str:
    return "\n\n".join(
        [
            get_prompt(context),
            get_domain_problem_prompt(domain, problem),
        ]
    )


def add_line_numbers(lines: list[str]) -> list[str]:
    lines_with_line_numbers = []
    for i, line in enumerate(lines):
        lines_with_line_numbers.append(f"{i}:\t" + line)
    return lines_with_line_numbers


def make_file_view(file_content: str) -> str:
    lines = [line + "\n" for line in file_content.split("\n")]
    return "```pddl\n" + "".join(add_line_numbers(lines)) + "```"
