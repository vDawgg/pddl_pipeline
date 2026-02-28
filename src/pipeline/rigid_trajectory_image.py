import logging

import dspy

from src.base.pipeline import Pipelines
from src.base.schema import PDDLFiles, PipelineError
from src.constants import images_dir
from src.eval.fast_downward import FDErrorInfo
from src.eval.val import (
    is_domain_valid,
    is_problem_valid,
)
from src.inference import Models
from src.pipeline.rigid_trajectory import RigidTrajectoryPipeline
from src.utils.domains import Domains

logger = logging.getLogger(__name__)


class RigidTrajectoryImagePipeline(RigidTrajectoryPipeline):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        super().__init__(model, domain, pipeline or Pipelines.RIGID_TRAJECTORY_IMAGE)

    def forward(
        self, domain_description: str, problem_description: str
    ) -> dspy.Prediction:
        image = dspy.Image(str(images_dir / "peg_and_ring_plan_start.png"))

        self.vars.num_model_calls += 1
        domain_result = self.generate_domain_module_image(
            scene=image, domain_description=domain_description
        )
        domain = domain_result.domain_pddl
        self.print_and_clear_history()

        self.vars.domain_file = self._write_pddl_file(
            domain, pddl_file_type=PDDLFiles.DOMAIN
        )
        self.fix_domain(self.vars.domain_file, image=image)
        if not is_domain_valid(self.vars.domain_file):
            self.print_and_clear_history()
            logger.debug("Domain failure")
            return dspy.Prediction(out=PipelineError.DOMAIN_FAILURE)
        self.print_and_clear_history()

        self.vars.num_model_calls += 1
        problem_result = self.generate_problem_module_image(
            scene=image, domain_pddl=domain, problem_description=problem_description
        )
        problem = problem_result.problem_pddl
        self.print_and_clear_history()

        self.vars.problem_file = self._write_pddl_file(
            problem, pddl_file_type=PDDLFiles.PROBLEM
        )
        self.fix_problem(self.vars.domain_file, self.vars.problem_file, image=image)
        if not is_problem_valid(self.vars.domain_file, self.vars.problem_file):
            self.print_and_clear_history()
            logger.debug("Problem failure")
            return dspy.Prediction(out=PipelineError.PROBLEM_FAILURE)
        self.print_and_clear_history()

        planner_output = self.fix_planning(
            self.vars.domain_file, self.vars.problem_file, image=image
        )
        if isinstance(planner_output, FDErrorInfo):
            self.print_and_clear_history()
            logger.debug("Failed to generate a plan")
            return dspy.Prediction(out=planner_output.to_pipeline_error())
        logger.debug("# Successfully generated a plan")
        self.vars.plan_file = planner_output
        self.print_and_clear_history()
        return dspy.Prediction(out=None)
