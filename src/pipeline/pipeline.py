from src.inference.model_comm import make_request


def simple_pipeline() -> str | None:
    return make_request("What is PDDL?")
