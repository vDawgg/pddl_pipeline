from tqdm import tqdm

from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.pipeline.pipeline_base import PipelineBase
from src.utils.io import write_temp_pddl_file


# TODO: This function will need to persist results so nothing gets lost
#       → Does not need to be the full PDDL results
def run_eval(iterations: int, pipeline: PipelineBase) -> tuple[int, int, int]:
    num_syntactically_incorrect_domains = 0
    num_syntactically_incorrect_problems = 0
    num_failed_plans = 0
    for _ in tqdm(range(iterations), "Running Evaluation"):
        domain, problem = pipeline.run()

        domain_file = write_temp_pddl_file(domain)
        domain_error_info = get_syntax_mistakes_domain(domain_file)
        if domain_error_info.num_errors > 0:
            num_syntactically_incorrect_domains += 1
            continue

        problem_file = write_temp_pddl_file(problem)
        problem_error_info = get_syntax_mistakes_problem(domain_file, problem_file)
        if problem_error_info.num_errors > 0:
            num_syntactically_incorrect_problems += 1
            continue

        # TODO: Add the same for the planning results here as well
    return (
        num_syntactically_incorrect_domains,
        num_syntactically_incorrect_problems,
        num_failed_plans,
    )
