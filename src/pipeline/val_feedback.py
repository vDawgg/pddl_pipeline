import logging

from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.eval.fast_downward import generate_plan, FDErrorInfo, Plan
from src.inference.model_comm import make_request, make_assistant_message
from src.pipeline.pipeline_base import PipelineBase, PipelineError
from src.utils.io import write_temp_pddl_file
from src.utils.prompts import Prompts, get_prompt


logger = logging.getLogger(__name__)


class ValFeedbackPipeline(PipelineBase):
    def run(self) -> PipelineError | Plan:
        domain, messages = make_request(
            get_prompt(Prompts.BASELINE_CONTEXT, Prompts.BASELINE_DOMAIN),
            model_name=self.model,
        )
        # TODO: This should be configurable
        # TODO: We should probably also collect the amount of iterations we run through in total
        #       If informative could also be collected for each step
        for i in range(5):
            logger.debug(f"Iterations domain syntax fixes: {i}")
            domain_file = write_temp_pddl_file(domain)
            err_info = get_syntax_mistakes_domain(domain_file)
            if err_info.num_errors == 0:
                break
            else:
                domain, _ = make_request(
                    get_prompt(Prompts.VAL_FEEDBACK_DOMAIN) + str(err_info.errors),
                    model_name=self.model,
                    messages=[*messages, make_assistant_message(domain)],
                )
        domain_file = write_temp_pddl_file(domain)
        err_info = get_syntax_mistakes_domain(domain_file)
        if err_info.num_errors > 0:
            logger.debug("Failed to create a syntactically valid domain")
            return PipelineError.DOMAIN_FAILURE
        logger.debug("# Domain\n\n")
        logger.debug(domain)

        problem, messages = make_request(
            get_prompt(Prompts.BASELINE_PROBLEM),
            model_name=self.model,
            messages=[*messages, make_assistant_message(domain)],
        )
        for i in range(5):
            logger.debug(f"Iterations problem syntax fixes: {i}")
            problem_file = write_temp_pddl_file(problem)
            err_info = get_syntax_mistakes_problem(domain_file, problem_file)
            if err_info.num_errors == 0:
                break
            else:
                domain, messages = make_request(
                    get_prompt(Prompts.VAL_FEEDBACK_DOMAIN) + str(err_info.errors),
                    model_name=self.model,
                    messages=[*messages, make_assistant_message(problem)],
                )
        problem_file = write_temp_pddl_file(problem)
        err_info = get_syntax_mistakes_problem(domain_file, problem_file)
        if err_info.num_errors > 0:
            logger.debug("Failed to create a syntactically valid problem")
            return PipelineError.PROBLEM_FAILURE
        logger.debug("# Problem\n\n")
        logger.debug(problem)

        planner_output = generate_plan(domain_file, problem_file)
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return PipelineError.PLAN_FAILURE
        logger.debug("# Plan\n\n")
        logger.debug(planner_output)
        return planner_output
