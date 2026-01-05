import logging

from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.eval.fast_downward import generate_plan, FDErrorInfo, Plan
from src.inference.model_comm import make_request, make_assistant_message
from src.base.pipeline import PipelineError
from src.pipeline.baseline import Baseline
from src.utils.io import write_temp_pddl_file
from src.utils.prompts import Prompts, get_prompt


logger = logging.getLogger(__name__)


class ValFeedbackPipeline(Baseline):
    def fix_domain(self, domain: str, num_tries: int = 5) -> str:
        for i in range(num_tries):
            logger.debug(f"Iterations domain syntax fixes: {i}")
            domain_file = write_temp_pddl_file(domain)
            err_info = get_syntax_mistakes_domain(domain_file)
            if err_info.num_errors == 0:
                break
            else:
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
        return domain

    def fix_problem(
        self, domain_file: str, domain: str, problem: str, num_tries: int = 5
    ) -> str:
        for i in range(num_tries):
            logger.debug(f"Iterations problem syntax fixes: {i}")
            problem_file = write_temp_pddl_file(problem)
            err_info = get_syntax_mistakes_problem(domain_file, problem_file)
            if err_info.num_errors == 0:
                break
            else:
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
        return problem

    def run(self) -> PipelineError | Plan:
        domain, messages = make_request(
            get_prompt(Prompts.BASELINE_CONTEXT, Prompts.BASELINE_DOMAIN),
            model_name=self.model,
        )
        domain = self.fix_domain(domain)
        domain_file = write_temp_pddl_file(domain)
        if not self.is_domain_valid(domain_file, domain):
            return PipelineError.DOMAIN_FAILURE

        problem, messages = make_request(
            get_prompt(Prompts.BASELINE_PROBLEM),
            model_name=self.model,
            messages=[*messages, make_assistant_message(domain)],
        )
        problem = self.fix_problem(domain_file, domain, problem)
        problem_file = write_temp_pddl_file(problem)
        if not self.is_problem_valid(domain_file, problem_file, domain):
            return PipelineError.DOMAIN_FAILURE

        planner_output = generate_plan(domain_file, problem_file)
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return PipelineError.PLAN_FAILURE
        logger.debug("# Plan\n\n")
        logger.debug(planner_output)
        return planner_output
