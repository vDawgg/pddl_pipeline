import logging

from src.base.pipeline import Pipelines
from src.base.schema import PDDLFiles, PipelineError
from src.constants import images_dir
from src.eval.fast_downward import FDErrorInfo
from src.inference.model_comm import make_assistant_message
from src.pipeline.val_and_planner_feedback import ValAndPlannerFeedbackPipeline
from src.utils.prompts import domain_prompts_images, problem_prompts

logger = logging.getLogger()


class ValAndPLannerFeedbackImagePipeline(ValAndPlannerFeedbackPipeline):
    def __init__(self, model, domain, pipeline=None):
        super().__init__(
            model, domain, pipeline or Pipelines.VAL_AND_PLANNER_FEEDBACK_IMAGE
        )

    def _run_impl(self):
        # TODO: This should be loaded depending on the currently used domain
        img_paths = [str(images_dir / "peg_and_ring_plan_start.png")]
        domain, messages = self.make_request(
            domain_prompts_images[self.domain],
            img_paths=img_paths,
        )
        self.domain_file = self._write_pddl_file(
            domain, pddl_file_type=PDDLFiles.DOMAIN
        )
        self.fix_domain(self.domain_file, image_paths=img_paths)
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
        self.fix_problem(self.domain_file, self.problem_file, image_paths=img_paths)
        if not self.is_problem_valid(self.domain_file, self.problem_file):
            logger.debug("Problem failure")
            return self.create_result(
                error=PipelineError.PROBLEM_FAILURE,
            )

        planner_output = self.fix_planning(
            self.domain_file, self.problem_file, image_paths=img_paths
        )
        if isinstance(planner_output, FDErrorInfo):
            logger.debug("Failed to generate a plan")
            return self.create_result(
                error=planner_output.to_pipeline_error(),
            )
        logger.debug("# Successfully generated a plan")
        self.plan_file = planner_output
        return self.create_result()
