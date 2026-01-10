import logging

from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.eval.fast_downward import generate_plan, FDErrorInfo
from src.inference.model_comm import make_request, make_assistant_message
from src.base.pipeline import Pipelines
from src.base.schema import PipelineError, PipelineResult
from src.pipeline.baseline import Baseline
from src.utils.io import write_temp_pddl_file
from src.utils.prompts import Prompts, get_prompt, domain_pompts, problem_prompts


logger = logging.getLogger(__name__)


class ValFeedbackPipeline(Baseline):
    def __init__(self, model, domain):
        super().__init__(model, domain)
        self.pipeline = Pipelines.VAL_FEEDBACK

    def fix_domain(self, domain: str, num_tries: int = 5) -> tuple[str, int]:
        iterations = 0
        for i in range(num_tries):
            domain_file = write_temp_pddl_file(domain)
            err_info = get_syntax_mistakes_domain(domain_file)
            if err_info.num_errors == 0:
                break
            else:
                logger.debug(f"Iterations domain syntax fixes: {i}")
                iterations = i
                prompt = get_prompt(
                    Prompts.VAL_FEEDBACK_CONTEXT, Prompts.VAL_FEEDBACK_DOMAIN
                ).format(
                    domain=domain,
                    errors=err_info.get_lines_with_errors(),
                )
                domain, _ = make_request(
                    prompt,
                    model_name=self.model,
                )
        return domain, iterations

    def fix_problem(
        self, domain_file: str, domain: str, problem: str, num_tries: int = 5
    ) -> tuple[str, int]:
        iterations = 0
        for i in range(num_tries):
            problem_file = write_temp_pddl_file(problem)
            err_info = get_syntax_mistakes_problem(domain_file, problem_file)
            if err_info.num_errors == 0:
                break
            else:
                logger.debug(f"Iterations problem syntax fixes: {i}")
                iterations = i
                prompt = get_prompt(
                    Prompts.VAL_FEEDBACK_CONTEXT, Prompts.VAL_FEEDBACK_DOMAIN
                ).format(
                    domain=domain,
                    problem=problem,
                    errors=err_info.get_lines_with_errors(),
                )
                domain, _ = make_request(
                    prompt,
                    model_name=self.model,
                )
        return problem, iterations

    def run(self) -> PipelineResult:
        iterations: dict[str, int] = {}

        domain, messages = make_request(
            domain_pompts[self.domain],
            model_name=self.model,
        )
        domain, domain_iters = self.fix_domain(domain)
        iterations["domain_fixes"] = domain_iters

        domain_file = write_temp_pddl_file(domain)
        if not self.is_domain_valid(domain_file):
            return PipelineResult(
                error=PipelineError.DOMAIN_FAILURE, iterations=iterations
            )

        problem, messages = make_request(
            problem_prompts[self.domain],
            model_name=self.model,
            messages=[*messages, make_assistant_message(domain)],
        )
        problem, problem_iters = self.fix_problem(domain_file, domain, problem)
        iterations["problem_fixes"] = problem_iters

        problem_file = write_temp_pddl_file(problem)
        if not self.is_problem_valid(domain_file, problem_file):
            return PipelineResult(
                error=PipelineError.DOMAIN_FAILURE, iterations=iterations
            )

        planner_output = generate_plan(
            domain_file, problem_file, self.model, self.pipeline, self.domain
        )
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return PipelineResult(
                error=PipelineError.PLAN_FAILURE, iterations=iterations
            )
        logger.debug("# Successfully generated a plan")
        return PipelineResult(iterations=iterations)
