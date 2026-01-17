import logging
from pathlib import Path

from src.base.pipeline import Pipelines
from src.base.schema import PDDLFiles, PipelineError, PipelineResult
from src.eval.fast_downward import ExitCodes, FDErrorInfo, generate_plan
from src.inference import Models
from src.inference.model_comm import make_assistant_message, make_request
from src.pipeline.val_feedback import ValFeedbackPipeline
from src.utils.domains import Domains
from src.utils.io import read_pddl_file, write_pddl_file
from src.utils.prompts import Prompts, domain_pompts, get_prompt, problem_prompts

logger = logging.getLogger(__name__)


# NOTE: The models do not properly incorporate feedback.
#       -> Larger models might fix this but will mean that we will not be able to run the pipeline on Jetson.
#       -> Another approach could be to introduce the checks as tools instead of using them for the feedback loops.
class ValAndPlannerFeedbackPipeline(ValFeedbackPipeline):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        super().__init__(model, domain, pipeline or Pipelines.VAL_AND_PLANNER_FEEDBACK)

    def fix_plan_not_found(self, domain_file: Path, problem_file: Path):
        unformatted_prompt = get_prompt(Prompts.PLANNER_CONTEXT, Prompts.PLANNER_TASK)
        prompt = unformatted_prompt.format(
            file=PDDLFiles.DOMAIN,
            domain=read_pddl_file(domain_file),
            problem=read_pddl_file(problem_file),
        )
        domain, _ = make_request(
            prompt,
            model_name=self.model,
        )
        prompt = unformatted_prompt.format(
            file=PDDLFiles.PROBLEM,
            domain=read_pddl_file(domain_file),
            problem=read_pddl_file(problem_file),
        )
        problem, _ = make_request(
            prompt,
            model_name=self.model,
        )
        domain_file = write_pddl_file(domain, domain_file)
        problem_file = write_pddl_file(problem, problem_file)
        return

    def fix_parsing_error(
        self, domain_file: Path, problem_file: Path, planner_output: FDErrorInfo
    ):
        unformatted_prompt = get_prompt(
            Prompts.PLANNER_TRANSLATE_CONTEXT, Prompts.PLANNER_TRANSLATE_TASK
        )
        if planner_output.file == PDDLFiles.DOMAIN:
            content = read_pddl_file(domain_file)
        else:
            content = read_pddl_file(problem_file)
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
            write_pddl_file(output, file=domain_file)
        else:
            write_pddl_file(output, file=problem_file)
        return

    def fix_planning(
        self,
        domain_file: Path,
        problem_file: Path,
        num_tries: int = 5,
    ) -> tuple[FDErrorInfo | Path, int]:
        assert num_tries > 0
        iterations = 0
        planner_output = generate_plan(domain_file, problem_file, self.name)
        for i in range(num_tries - 1):
            if type(planner_output) is str:
                break
            elif type(planner_output) is FDErrorInfo and planner_output.exit_code in [
                ExitCodes.TRANSLATE_CRITICAL_ERROR,
                ExitCodes.TRANSLATE_INPUT_ERROR,
            ]:
                logger.debug("Parsing error")
                assert planner_output.file is not None
                iterations = i + 1
                self.fix_parsing_error(domain_file, problem_file, planner_output)
            else:
                logger.debug("Planning error")
                iterations = i + 1
                self.fix_plan_not_found(domain_file, problem_file)
            logger.debug(f"Iterations planning fixes: {iterations}")
            planner_output = generate_plan(domain_file, problem_file, self.name)
        return planner_output, iterations

    def _run_impl(self) -> PipelineResult:
        domain, messages = make_request(
            domain_pompts[self.domain],
            model_name=self.model,
        )
        domain_file = write_pddl_file(
            domain, name=self.name, pddl_file_type=PDDLFiles.DOMAIN
        )
        domain_iters = self.fix_domain(domain_file)
        if not self.is_domain_valid(domain_file):
            return PipelineResult(
                elapsed_time=self.elapsed_time,
                error=PipelineError.DOMAIN_FAILURE,
                domain_file=domain_file,
                num_domain_fixes=domain_iters,
            )

        problem, messages = make_request(
            problem_prompts[self.domain],
            model_name=self.model,
            messages=[*messages, make_assistant_message(domain)],
        )
        problem_file = write_pddl_file(
            problem, name=self.name, pddl_file_type=PDDLFiles.PROBLEM
        )
        problem_iters = self.fix_problem(domain_file, problem_file)
        if not self.is_problem_valid(domain_file, problem_file):
            logger.debug("Problem failure")
            return PipelineResult(
                elapsed_time=self.elapsed_time,
                error=PipelineError.PROBLEM_FAILURE,
                domain_file=domain_file,
                problem_file=problem_file,
                num_domain_fixes=domain_iters,
                num_problem_fixes=problem_iters,
            )

        # TODO: It might be cleaner to implement a is_plan_valid method here as well.
        planner_output, planner_iters = self.fix_planning(domain_file, problem_file)
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return PipelineResult(
                elapsed_time=self.elapsed_time,
                error=PipelineError.PLAN_FAILURE,
                domain_file=domain_file,
                problem_file=problem_file,
                num_domain_fixes=domain_iters,
                num_problem_fixes=problem_iters,
                num_planner_fixes=planner_iters,
            )
        logger.debug("# Successfully generated a plan")
        return PipelineResult(
            elapsed_time=self.elapsed_time,
            domain_file=domain_file,
            problem_file=problem_file,
            plan_file=planner_output,
            num_domain_fixes=domain_iters,
            num_problem_fixes=problem_iters,
            num_planner_fixes=planner_iters,
        )
