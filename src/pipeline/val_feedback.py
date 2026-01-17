import logging
from pathlib import Path

from src.base.pipeline import Pipelines
from src.base.schema import PDDLFiles, PipelineError, PipelineResult
from src.eval.fast_downward import FDErrorInfo, generate_plan
from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.inference import Models
from src.inference.model_comm import make_assistant_message
from src.pipeline.baseline import Baseline
from src.utils.domains import Domains
from src.utils.io import read_pddl_file, write_pddl_file
from src.utils.prompts import Prompts, domain_pompts, get_prompt, problem_prompts

logger = logging.getLogger(__name__)


class ValFeedbackPipeline(Baseline):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        super().__init__(model, domain, pipeline or Pipelines.VAL_FEEDBACK)

    def fix_domain(self, domain_file: Path, num_tries: int = 5) -> int:
        iterations = 0
        for i in range(num_tries):
            err_info = get_syntax_mistakes_domain(domain_file)
            if err_info.num_errors == 0:
                break
            else:
                logger.debug(f"Iterations domain syntax fixes: {i}")
                iterations = i
                prompt = get_prompt(
                    Prompts.VAL_FEEDBACK_CONTEXT, Prompts.VAL_FEEDBACK_DOMAIN
                ).format(
                    domain=read_pddl_file(domain_file),
                    errors=err_info.get_lines_with_errors(),
                )
                domain, _ = self.make_request(
                    prompt,
                )
                domain_file = write_pddl_file(domain, file=domain_file)
        return iterations

    def fix_problem(
        self, domain_file: Path, problem_file: Path, num_tries: int = 5
    ) -> int:
        iterations = 0
        for i in range(num_tries):
            err_info = get_syntax_mistakes_problem(domain_file, problem_file)
            if err_info.num_errors == 0:
                break
            else:
                logger.debug(f"Iterations problem syntax fixes: {i}")
                iterations = i
                prompt = get_prompt(
                    Prompts.VAL_FEEDBACK_CONTEXT, Prompts.VAL_FEEDBACK_DOMAIN
                ).format(
                    domain=read_pddl_file(domain_file),
                    problem=read_pddl_file(problem_file),
                    errors=err_info.get_lines_with_errors(),
                )
                problem, _ = self.make_request(
                    prompt,
                )
                problem_file = write_pddl_file(problem, file=problem_file)
        return iterations

    def _run_impl(self) -> PipelineResult:
        domain, messages = self.make_request(
            domain_pompts[self.domain],
        )
        self.domain_file = write_pddl_file(
            domain, name=self.name, pddl_file_type=PDDLFiles.DOMAIN
        )
        domain_iters = self.fix_domain(self.domain_file)
        if not self.is_domain_valid(self.domain_file):
            return self.create_result(
                error=PipelineError.DOMAIN_FAILURE,
                num_domain_fixes=domain_iters,
            )

        problem, messages = self.make_request(
            problem_prompts[self.domain],
            messages=[*messages, make_assistant_message(domain)],
        )
        self.problem_file = write_pddl_file(
            problem, name=self.name, pddl_file_type=PDDLFiles.PROBLEM
        )
        problem_iters = self.fix_problem(self.domain_file, self.problem_file)
        if not self.is_problem_valid(self.domain_file, self.problem_file):
            return self.create_result(
                error=PipelineError.DOMAIN_FAILURE,
                num_domain_fixes=domain_iters,
                num_problem_fixes=problem_iters,
            )

        planner_output = generate_plan(self.domain_file, self.problem_file, self.name)
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return self.create_result(
                error=PipelineError.PLAN_FAILURE,
                num_domain_fixes=domain_iters,
                num_problem_fixes=problem_iters,
            )
        logger.debug("# Successfully generated a plan")
        self.plan_file = planner_output
        return self.create_result(
            num_domain_fixes=domain_iters,
            num_problem_fixes=problem_iters,
        )
