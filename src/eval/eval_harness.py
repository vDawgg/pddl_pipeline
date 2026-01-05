import logging

from tqdm import tqdm

from src.base.pipeline import PipelineBase, PipelineError


logger = logging.getLogger(__name__)


# TODO: This function will need to persist results so nothing gets lost
#       → Does not need to be the full PDDL results
def run_eval(iterations: int, pipeline: PipelineBase) -> tuple[int, int, int, int]:
    num_model_errors = 0
    num_syntactically_incorrect_domains = 0
    num_syntactically_incorrect_problems = 0
    num_failed_plans = 0
    plan_failures = []
    num_successfull_plans = 0
    for _ in tqdm(range(iterations), "Running Evaluation"):
        try:
            result = pipeline.run()
            match result:
                case PipelineError.DOMAIN_FAILURE:
                    num_syntactically_incorrect_domains += 1
                case PipelineError.PROBLEM_FAILURE:
                    num_syntactically_incorrect_problems += 1
                case PipelineError.PLAN_FAILURE:
                    num_failed_plans += 1
                case _:
                    num_successfull_plans += 1
        except Exception:
            num_model_errors += 1
            continue
    logger.info(plan_failures)
    return (
        num_model_errors,
        num_syntactically_incorrect_domains,
        num_syntactically_incorrect_problems,
        num_failed_plans,
    )
