import logging

from src.inference.model_comm import make_assistant_message, make_request
from src.eval.fast_downward import Plan, generate_plan, FDErrorInfo, ExitCodes
from src.base.pipeline import PipelineError
from src.base.schema import PDDLFiles
from src.pipeline.val_feedback import ValFeedbackPipeline
from src.utils.io import write_temp_pddl_file
from src.utils.prompts import Prompts, get_prompt


logger = logging.getLogger(__name__)


# NOTE: The models do not properly incorporate feedback.
#       -> Larger models might fix this but will mean that we will not be able to run the pipeline on Jetson.
#       -> Another approach could be to introduce the checks as tools instead of using them for the feedback loops.
class ValAndPlannerFeedbackPipeline(ValFeedbackPipeline):
    def fix_plan_not_found(self, domain: str, problem: str) -> tuple[str, str]:
        unformatted_prompt = get_prompt(Prompts.PLANNER_CONTEXT, Prompts.PLANNER_TASK)
        prompt = unformatted_prompt.format(
            file=PDDLFiles.DOMAIN, domain=domain, problem=problem
        )
        domain, _ = make_request(
            prompt,
            model_name=self.model,
        )
        prompt = unformatted_prompt.format(
            file=PDDLFiles.PROBLEM, domain=domain, problem=problem
        )
        problem, _ = make_request(
            prompt,
            model_name=self.model,
        )
        domain_file = write_temp_pddl_file(domain)
        problem_file = write_temp_pddl_file(problem)
        return domain_file, problem_file

    def fix_parsing_error(
        self, domain: str, problem: str, planner_output: FDErrorInfo
    ) -> tuple[str, str]:
        unformatted_prompt = get_prompt(
            Prompts.PLANNER_TRANSLATE_CONTEXT, Prompts.PLANNER_TRANSLATE_TASK
        )
        if planner_output.file == PDDLFiles.DOMAIN:
            content = domain
        else:
            content = problem
        prompt = unformatted_prompt.format(
            file=planner_output.file,
            err_msg=planner_output.error_message,
            content=content,
        )
        output, _ = make_request(
            prompt,
            model_name=self.model,
        )
        if planner_output.file == PDDLFiles.DOMAIN:
            domain = output
        else:
            problem = output
        domain_file = write_temp_pddl_file(domain)
        problem_file = write_temp_pddl_file(problem)
        return domain_file, problem_file

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
            elif planner_output.exit_code in [
                ExitCodes.TRANSLATE_CRITICAL_ERROR,
                ExitCodes.TRANSLATE_INPUT_ERROR,
            ]:
                logger.debug("Parsing error")
                print(planner_output.exit_code)
                print(planner_output.error_message)
                assert planner_output.file is not None
                domain_file, problem_file = self.fix_parsing_error(
                    domain, problem, planner_output
                )
            else:
                logger.debug("Planning error")
                domain_file, problem_file = self.fix_plan_not_found(domain, problem)
        assert planner_output is not None
        return planner_output

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
