import logging
from pathlib import Path

from src.base.pipeline import PipelineBase, Pipelines
from src.base.schema import PDDLFiles, PipelineError, PipelineResult
from src.eval.fast_downward import FDErrorInfo, generate_plan
from src.eval.val import get_syntax_mistakes_domain, get_syntax_mistakes_problem
from src.inference import Models
from src.inference.model_comm import make_assistant_message
from src.utils.domains import Domains
from src.utils.io import write_pddl_file
from src.utils.prompts import domain_pompts, problem_prompts

logger = logging.getLogger(__name__)


class Baseline(PipelineBase):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        super().__init__(model, domain, pipeline or Pipelines.BASELINE)

    def is_domain_valid(self, domain_file: Path) -> bool:
        err_info = get_syntax_mistakes_domain(domain_file)
        if err_info.num_errors > 0:
            logger.debug("Failed to create a syntactically valid domain")
            return False
        logger.debug("Generated syntactically valid domain")
        return True

    def is_problem_valid(
        self,
        domain_file: Path,
        problem_file: Path,
    ) -> bool:
        err_info = get_syntax_mistakes_problem(domain_file, problem_file)
        if err_info.num_errors > 0:
            logger.debug("Failed to create a syntactically valid problem")
            return False
        logger.debug("Generated syntactically valid problem")
        return True

    def _run_impl(self) -> PipelineResult:
        domain, messages = self.make_request(
            domain_pompts[self.domain],
        )
        self.domain_file = write_pddl_file(
            domain, name=self.name, pddl_file_type=PDDLFiles.DOMAIN
        )
        if not self.is_domain_valid(self.domain_file):
            return self.create_result(error=PipelineError.DOMAIN_FAILURE)

        problem, messages = self.make_request(
            problem_prompts[self.domain],
            messages=[*messages, make_assistant_message(domain)],
        )
        self.problem_file = write_pddl_file(
            problem, name=self.name, pddl_file_type=PDDLFiles.PROBLEM
        )
        if not self.is_problem_valid(self.domain_file, self.problem_file):
            return self.create_result(error=PipelineError.PROBLEM_FAILURE)

        planner_output = generate_plan(self.domain_file, self.problem_file, self.name)
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return self.create_result(error=PipelineError.PLAN_FAILURE)
        logger.debug("# Successfully generated a plan")
        self.plan_file = planner_output
        return self.create_result()
