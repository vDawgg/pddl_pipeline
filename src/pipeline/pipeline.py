from src.inference.model_comm import make_request
from src.utils.prompts import Prompts, get_prompt


def simple_pipeline() -> str | None:
    return make_request("What is PDDL?")


def baseline() -> str | None:
    return make_request(
        get_prompt(Prompts.BASELINE)
    )
