import logging

from src.inference.model_comm import make_assistant_message, make_request
from src.eval.fast_downward import generate_plan, FDErrorInfo, ExitCodes
from src.base.pipeline import Pipelines
from src.base.schema import PDDLFiles, PipelineResult, PipelineError
from src.pipeline.val_feedback import ValFeedbackPipeline
from src.utils.io import write_temp_pddl_file
from src.utils.prompts import Prompts, get_prompt, domain_pompts, problem_prompts


logger = logging.getLogger(__name__)


# NOTE: The models do not properly incorporate feedback.
#       -> Larger models might fix this but will mean that we will not be able to run the pipeline on Jetson.
#       -> Another approach could be to introduce the checks as tools instead of using them for the feedback loops.
class ValAndPlannerFeedbackPipeline(ValFeedbackPipeline):
    def __init__(self, model, domain):
        super().__init__(model, domain)
        self.pipeline = Pipelines.VAL_AND_PLANNER_FEEDBACK

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
    ) -> tuple[FDErrorInfo | None, int]:
        assert num_tries > 0
        planner_output = None
        iterations = 0
        for i in range(num_tries):
            logger.debug(f"Iterations planning fixes: {i}")
            planner_output = generate_plan(
                domain_file, problem_file, self.model, self.pipeline, self.domain
            )
            if planner_output is None:
                break
            elif planner_output.exit_code in [
                ExitCodes.TRANSLATE_CRITICAL_ERROR,
                ExitCodes.TRANSLATE_INPUT_ERROR,
            ]:
                logger.debug("Parsing error")
                assert planner_output.file is not None
                iterations = i
                domain_file, problem_file = self.fix_parsing_error(
                    domain, problem, planner_output
                )
            else:
                logger.debug("Planning error")
                iterations = i
                domain_file, problem_file = self.fix_plan_not_found(domain, problem)
        return planner_output, iterations

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
            logger.debug("Problem failure")
            return PipelineResult(
                error=PipelineError.PROBLEM_FAILURE, iterations=iterations
            )

        planner_output, planner_iters = self.fix_planning(
            domain_file, problem_file, domain, problem
        )
        iterations["planner_fixes"] = planner_iters

        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return PipelineResult(
                error=PipelineError.PLAN_FAILURE, iterations=iterations
            )
        logger.debug("# Successfully generated a plan")
        return PipelineResult(iterations=iterations)
