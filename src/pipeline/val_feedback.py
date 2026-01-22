import logging
from pathlib import Path

from src.base.pipeline import Pipelines
from src.base.schema import PDDLFiles, PipelineError, PipelineResult
from src.eval.fast_downward import FDErrorInfo
from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.inference import Models
from src.inference.model_comm import make_assistant_message
from src.pipeline.baseline import Baseline
from src.utils.domains import Domains
from src.utils.prompts import Prompts, domain_pompts, get_prompt, problem_prompts

logger = logging.getLogger(__name__)


class ValFeedbackPipeline(Baseline):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        super().__init__(model, domain, pipeline or Pipelines.VAL_FEEDBACK)

    def fix_domain(self, domain_file: Path, num_tries: int = 5):
        for i in range(num_tries):
            err_info = get_syntax_mistakes_domain(domain_file)
            self.domain_syntax_errors_calls += 1
            if err_info.num_errors == 0:
                break
            else:
                logger.debug(f"Iterations domain syntax fixes: {i}")
                prompt = get_prompt(
                    Prompts.VAL_FEEDBACK_CONTEXT, Prompts.VAL_FEEDBACK_DOMAIN
                ).format(
                    domain=self._read_pddl_file(domain_file),
                    errors=err_info.get_lines_with_errors(),
                )
                domain, _ = self.make_request(
                    prompt,
                )
                domain_file = self._write_pddl_file(domain, file=domain_file)

    def fix_problem(self, domain_file: Path, problem_file: Path, num_tries: int = 5):
        for i in range(num_tries):
            err_info = get_syntax_mistakes_problem(domain_file, problem_file)
            self.problem_syntax_mistakes_calls += 1
            if err_info.num_errors == 0:
                break
            else:
                logger.debug(f"Iterations problem syntax fixes: {i}")
                prompt = get_prompt(
                    Prompts.VAL_FEEDBACK_CONTEXT, Prompts.VAL_FEEDBACK_DOMAIN
                ).format(
                    domain=self._read_pddl_file(domain_file),
                    problem=self._read_pddl_file(problem_file),
                    errors=err_info.get_lines_with_errors(),
                )
                problem, _ = self.make_request(
                    prompt,
                )
                problem_file = self._write_pddl_file(problem, file=problem_file)

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
            return self.create_result(
                error=PipelineError.DOMAIN_FAILURE,
            )

        planner_output = self._generate_plan(self.domain_file, self.problem_file)
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return self.create_result(
                error=planner_output.to_pipeline_error(),
            )
        logger.debug("# Successfully generated a plan")
        self.plan_file = planner_output
        return self.create_result()
