import logging

from src.inference.model_comm import make_assistant_message, make_request
from src.eval.fast_downward import Plan, generate_plan, FDErrorInfo
from src.pipeline.pipeline_base import PipelineError
from src.pipeline.val_feedback import ValFeedbackPipeline
from src.utils.io import write_temp_pddl_file
from src.utils.prompts import Prompts, get_prompt


logger = logging.getLogger(__name__)


class ValAndPlannerFeedbackPipeline(ValFeedbackPipeline):
    # NOTE: This does not account for syntax mistakes the model might make after the basis was syntactically correct
    # TODO: We should probably add the planners output due to the above
    #       -> Also try and see whether we can get some form of reasoning unsolvable plans
    def fix_planning(
        self,
        domain_file: str,
        problem_file: str,
        domain: str,
        problem: str,
        num_tries: int = 5,
    ) -> Plan | FDErrorInfo:
        assert num_tries > 0
        planner_output = None
        for i in range(num_tries):
            logger.debug(f"Iterations planning fixes: {i}")
            planner_output = generate_plan(domain_file, problem_file)
            if isinstance(planner_output, Plan):
                break
            unformatted_prompt = get_prompt(
                Prompts.PLANNER_CONTEXT, Prompts.PLANNER_TASK
            )
            prompt = unformatted_prompt.format(
                file="domain", domain=domain, problem=problem
            )
            domain, _ = make_request(
                prompt,
                model_name=self.model,
            )
            prompt = unformatted_prompt.format(
                file="problem", domain=domain, problem=problem
            )
            problem, _ = make_request(
                prompt,
                model_name=self.model,
            )
            domain_file = write_temp_pddl_file(domain)
            problem_file = write_temp_pddl_file(problem)
        assert planner_output is not None
        return planner_output

    def run(self) -> PipelineError | Plan:
        domain, messages = make_request(
            get_prompt(Prompts.BASELINE_CONTEXT, Prompts.BASELINE_DOMAIN),
            model_name=self.model,
        )
        domain = self.fix_domain(domain, messages)
        domain_file = write_temp_pddl_file(domain)
        if not self.is_domain_valid(domain_file, domain):
            return PipelineError.DOMAIN_FAILURE

        problem, messages = make_request(
            get_prompt(Prompts.BASELINE_PROBLEM),
            model_name=self.model,
            messages=[*messages, make_assistant_message(domain)],
        )
        problem = self.fix_problem(domain_file, problem, messages)
        problem_file = write_temp_pddl_file(problem)
        if not self.is_problem_valid(domain_file, problem_file, problem):
            logger.debug("Problem failure")
            return PipelineError.PROBLEM_FAILURE

        planner_output = self.fix_planning(domain_file, problem_file, domain, problem)
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return PipelineError.PLAN_FAILURE
        logger.debug("# Plan")
        logger.debug(planner_output)
        return planner_output
