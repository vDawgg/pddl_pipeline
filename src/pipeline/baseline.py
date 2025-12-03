import logging

from src.eval.fast_downward import generate_plan, FDErrorInfo, Plan
from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.inference.model_comm import make_request, make_assistant_message
from src.pipeline.pipeline_base import PipelineBase, PipelineError
from src.utils.io import write_temp_pddl_file
from src.utils.prompts import Prompts, get_prompt


logger = logging.getLogger(__name__)


class Baseline(PipelineBase):
    def is_domain_valid(self, domain_file: str, domain: str) -> bool:
        err_info = get_syntax_mistakes_domain(domain_file)
        if err_info.num_errors > 0:
            logger.debug("Failed to create a syntactically valid domain")
            return False
        logger.debug("Generated syntactically valid domain")
        return True

    def is_problem_valid(
        self, domain_file: str, problem_file: str, problem: str
    ) -> bool:
        err_info = get_syntax_mistakes_problem(domain_file, problem_file)
        if err_info.num_errors > 0:
            logger.debug("Failed to create a syntactically valid problem")
            return False
        logger.debug("Generated syntactically valid problem")
        return True

    def run(self) -> PipelineError | Plan:
        # TODO: We should add natural language descriptions of the actions we pass to the model
        # TODO: We should probably also test with the thinking variant of the model
        domain, messages = make_request(
            get_prompt(Prompts.BASELINE_CONTEXT, Prompts.BASELINE_DOMAIN),
            self.model,
        )
        domain_file = write_temp_pddl_file(domain)
        if not self.is_domain_valid(domain_file, domain):
            return PipelineError.DOMAIN_FAILURE

        problem, messages = make_request(
            get_prompt(Prompts.BASELINE_PROBLEM),
            self.model,
            messages=[*messages, make_assistant_message(domain)],
        )
        problem_file = write_temp_pddl_file(problem)
        if not self.is_problem_valid(domain_file, problem_file, problem):
            return PipelineError.PROBLEM_FAILURE

        planner_output = generate_plan(domain_file, problem_file)
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return PipelineError.PLAN_FAILURE
        logger.debug("# Plan\n\n")
        logger.debug(planner_output)

        return planner_output
