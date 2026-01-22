import logging
from pathlib import Path

from src.base.pipeline import Pipelines
from src.base.schema import PDDLFiles, PipelineError, PipelineResult
from src.eval.fast_downward import ExitCodes, FDErrorInfo
from src.inference import Models
from src.inference.model_comm import make_assistant_message
from src.pipeline.val_feedback import ValFeedbackPipeline
from src.utils.domains import Domains
from src.utils.prompts import Prompts, domain_pompts, get_prompt, problem_prompts

logger = logging.getLogger(__name__)


class ValAndPlannerFeedbackPipeline(ValFeedbackPipeline):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        super().__init__(model, domain, pipeline or Pipelines.VAL_AND_PLANNER_FEEDBACK)

    def fix_plan_not_found(self, domain_file: Path, problem_file: Path):
        unformatted_prompt = get_prompt(Prompts.PLANNER_CONTEXT, Prompts.PLANNER_TASK)
        prompt = unformatted_prompt.format(
            file=PDDLFiles.DOMAIN,
            domain=self._read_pddl_file(domain_file),
            problem=self._read_pddl_file(problem_file),
        )
        domain, _ = self.make_request(
            prompt,
        )
        prompt = unformatted_prompt.format(
            file=PDDLFiles.PROBLEM,
            domain=self._read_pddl_file(domain_file),
            problem=self._read_pddl_file(problem_file),
        )
        problem, _ = self.make_request(
            prompt,
        )
        domain_file = self._write_pddl_file(domain, domain_file)
        problem_file = self._write_pddl_file(problem, problem_file)
        return

    def fix_parsing_error(
        self, domain_file: Path, problem_file: Path, planner_output: FDErrorInfo
    ):
        unformatted_prompt = get_prompt(
            Prompts.PLANNER_TRANSLATE_CONTEXT, Prompts.PLANNER_TRANSLATE_TASK
        )
        if planner_output.file == PDDLFiles.DOMAIN:
            content = self._read_pddl_file(domain_file)
        else:
            content = self._read_pddl_file(problem_file)
        prompt = unformatted_prompt.format(
            file=planner_output.file,
            err_msg=planner_output.error_message,
            content=content,
        )
        output, _ = self.make_request(
            prompt,
        )
        if planner_output.file == PDDLFiles.DOMAIN:
            self._write_pddl_file(output, file=domain_file)
        else:
            self._write_pddl_file(output, file=problem_file)
        return

    def fix_planning(
        self,
        domain_file: Path,
        problem_file: Path,
        num_tries: int = 5,
    ) -> FDErrorInfo | Path:
        assert num_tries > 0
        iterations = 0
        planner_output = None
        for i in range(num_tries):
            planner_output = self._generate_plan(domain_file, problem_file)
            self.generate_plan_calls += 1
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
        assert planner_output is not None
        return planner_output

    def _run_impl(self) -> PipelineResult:
        domain, messages = self.make_request(
            domain_pompts[self.domain],
        )
        self.domain_file = self._write_pddl_file(
            domain, pddl_file_type=PDDLFiles.DOMAIN
        )
        self.fix_domain(self.domain_file)
        if not self.is_domain_valid(self.domain_file):
            return self.create_result(
                error=PipelineError.DOMAIN_FAILURE,
            )

        problem, messages = self.make_request(
            problem_prompts[self.domain],
            messages=[*messages, make_assistant_message(domain)],
        )
        self.problem_file = self._write_pddl_file(
            problem, pddl_file_type=PDDLFiles.PROBLEM
        )
        self.fix_problem(self.domain_file, self.problem_file)
        if not self.is_problem_valid(self.domain_file, self.problem_file):
            logger.debug("Problem failure")
            return self.create_result(
                error=PipelineError.PROBLEM_FAILURE,
            )

        planner_output = self.fix_planning(self.domain_file, self.problem_file)
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return self.create_result(
                error=planner_output.to_pipeline_error(),
            )
        logger.debug("# Successfully generated a plan")
        self.plan_file = planner_output
        return self.create_result()
