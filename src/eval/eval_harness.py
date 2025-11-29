from tqdm import tqdm

from src.eval.fast_downward import generate_plan, FDErrorInfo, ExitCodes
from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.pipeline.pipeline_base import PipelineBase
from src.utils.io import write_temp_pddl_file


# TODO: This function will need to persist results so nothing gets lost
#       → Does not need to be the full PDDL results
def run_eval(iterations: int, pipeline: PipelineBase) -> tuple[int, int, int, int]:
    num_model_errors = 0
    num_syntactically_incorrect_domains = 0
    num_syntactically_incorrect_problems = 0
    num_failed_plans = 0
    plan_failures = []
    for _ in tqdm(range(iterations), "Running Evaluation"):
        try:
            domain, problem = pipeline.run()
        except Exception:
            num_model_errors += 1
            continue

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

        fd_output = generate_plan(domain_file, problem_file)
        if type(fd_output) is FDErrorInfo:
            num_failed_plans += 1
            # TODO: We need to handle the different cases properly here! VAL does not seem
            #       to give us enough output to assume that the planner will be able to translate
            #       our input
            plan_failures.append(fd_output.exit_code)
    print(plan_failures)
    return (
        num_model_errors,
        num_syntactically_incorrect_domains,
        num_syntactically_incorrect_problems,
        num_failed_plans,
    )
