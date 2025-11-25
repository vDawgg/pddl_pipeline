from src.inference.model_comm import make_request
from src.utils.prompts import Prompts, get_prompt


def baseline() -> tuple[str, str]:
    domain, messages = make_request(
        get_prompt(Prompts.BASELINE_CONTEXT, Prompts.BASELINE_DOMAIN)
    )
    problem, messages = make_request(
        get_prompt(Prompts.BASELINE_PROBLEM),
        messages=messages,
    )
    return domain, problem
