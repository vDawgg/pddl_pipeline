import logging

from src.base.pipeline import Pipelines
from src.base.schema import PipelineError
from src.constants import images_dir
from src.eval.fast_downward import FDErrorInfo
from src.pipeline.tool_calls import ToolCallPipeline
from src.utils.prompts import Prompts, get_prompt

logger = logging.getLogger(__name__)


class ToolCallImagePipeline(ToolCallPipeline):
    def __init__(self, model, domain, pipeline=None):
        super().__init__(model, domain, pipeline or Pipelines.TOOL_CALL_IMAGE)

    def _run_impl(self):
        # TODO: This should be loaded depending on the currently used domain
        img_paths = [str(images_dir / "peg_and_ring_plan_start.png")]
        self.make_react_workflow(
            # TODO: Update the prompt to reference the images
            input_prompt=get_prompt(
                Prompts.GENERATION_CONTEXT_TOOLS_IMAGES, Prompts.RING_AND_PEG
            ),
            tools=[
                self.create_pddl_file,
                self.read_pddl_file,
                self.edit_lines,
                self.get_syntax_mistakes_domain,
                self.get_syntax_mistakes_problem,
                self.translate_pddl,
                self.generate_plan,
                self.finish,
            ],
            image_paths=img_paths,
            max_iters=20,
        )
        error = None
        if self.domain_file is None or not self.is_domain_valid(self.domain_file):
            error = PipelineError.DOMAIN_FAILURE
        elif self.problem_file is None or not self.is_problem_valid(
            self.domain_file, self.problem_file
        ):
            error = PipelineError.PROBLEM_FAILURE
        else:
            plan = self._generate_plan(self.domain_file, self.problem_file)
            if isinstance(plan, FDErrorInfo):
                logger.debug(
                    f"# Failed to generate solvable domain and problem: {plan.error_message}"
                )
                error = plan.to_pipeline_error()
            else:
                self.plan_file = plan
        return self.create_result(
            error=error,
        )
