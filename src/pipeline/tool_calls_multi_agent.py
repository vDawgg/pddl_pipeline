import logging

from src.base.schema import PipelineError, PipelineResult
from src.eval.fast_downward import FDErrorInfo, generate_plan
from src.inference.model_comm import make_react_workflow
from src.pipeline.tool_calls import ToolCallPipeline
from src.utils.prompts import Prompts, get_prompt

logger = logging.getLogger(__name__)


class ToolCallPipelineMutltiAgent(ToolCallPipeline):
    def __init__(self, model, domain, pipeline=None):
        super().__init__(model, domain, pipeline)
        self.name = "tool_call_pipeline_multi_agent"

    def _run_impl(self) -> PipelineResult:
        make_react_workflow(
            model_name=self.model,
            input_prompt=get_prompt(
                Prompts.GENERATION_CONTEXT_TOOLS, Prompts.RING_AND_PEG_DOMAIN
            ),
            tools=[
                self.create_pddl_file,
                self.read_pddl_file,
                self.edit_lines,
                self.get_syntax_mistakes_domain,
            ],
        )
        if self.domain_file is None or not self.is_domain_valid(self.domain_file):
            return PipelineResult(
                elapsed_time=self.elapsed_time,
                error=PipelineError.DOMAIN_FAILURE,
                domain_file=self.domain_file,
                _number_of_fixes=self.get_total_tool_calls(),
            )
        assert self.domain_file is not None
        domain_info = (
            f"The previously generated domain can be found under: {self.domain_file}"
        )
        make_react_workflow(
            model_name=self.model,
            input_prompt=get_prompt(
                Prompts.GENERATION_CONTEXT_TOOLS, Prompts.RING_AND_PEG_PROBLEM
            )
            + domain_info,
            tools=[
                self.create_pddl_file,
                self.read_pddl_file,
                self.edit_lines,
                self.get_syntax_mistakes_domain,
                self.get_syntax_mistakes_problem,
            ],
        )
        if self.problem_file is None or not self.is_problem_valid(
            self.domain_file, self.problem_file
        ):
            PipelineResult(
                elapsed_time=self.elapsed_time,
                error=PipelineError.PROBLEM_FAILURE,
                domain_file=self.domain_file,
                problem_file=self.problem_file,
                _number_of_fixes=self.get_total_tool_calls(),
            )
        assert self.problem_file is not None
        plan = generate_plan(self.domain_file, self.problem_file, self.name)
        if isinstance(plan, FDErrorInfo):
            logger.debug(
                f"# Failed to generate solvable domain and problem: {plan.error_message}"
            )
            return PipelineResult(
                elapsed_time=self.elapsed_time,
                error=PipelineError.PLAN_FAILURE,
                domain_file=self.domain_file,
                problem_file=self.problem_file,
                _number_of_fixes=self.get_total_tool_calls(),
            )
        return PipelineResult(
            elapsed_time=self.elapsed_time,
            domain_file=self.domain_file,
            problem_file=self.problem_file,
            plan_file=plan,
            _number_of_fixes=self.get_total_tool_calls(),
        )
