import logging
from typing import Literal

import dspy

from src.base.pipeline import FDErrorInfo
from src.base.schema import PipelineError, Pipelines, Tools
from src.constants import images_dir
from src.eval.val import is_domain_valid, is_problem_valid
from src.inference import Models
from src.pipeline.tool_call import ToolCallPipeline
from src.utils.domains import Domains
from src.utils.prompts import Prompts, get_prompt

logger = logging.getLogger(__name__)


class GeneratePddlSignature(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT_TOOLS)

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
    task_description: str = dspy.InputField(
        desc="Natural language description of the planning task"
    )
    success: Literal["success", "failure"] = dspy.OutputField(
        desc="Whether the final PDDL domain and problem are syntactically valid and solvable."
    )


class PlanFeedbackSignature(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT_PLAN_FEEDBACK)

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
    task_and_plan: str = dspy.InputField(
        desc="The generated plan and a natural language description of the task the plan should be applied to."
    )
    feedback: str = dspy.OutputField(desc="Actionable feedback on the provided plan.")


class ToolCallImagePipeline(ToolCallPipeline):
    def __init__(
        self,
        model: Models,
        domain: Domains,
        ablate_tools: list[Tools] | None = None,
        pipeline: Pipelines | None = None,
    ):
        super().__init__(
            model, domain, ablate_tools, pipeline or Pipelines.TOOL_CALL_IMAGE
        )
        self.generate_pddl_module = dspy.ReAct(
            GeneratePddlSignature,
            tools=list(self.tools.values()),
            max_iters=30,
        )

    def get_plan_feedback(self, plan_file: str) -> str:
        """
        Get detailed feedback on the feasibility of a generated plan given the current task.

        :param domain_file: the path to the domain file
        :type domain_file: str
        :param problem_file: the path to the problem file
        :type problem_file: str
        :param plan_file: the path to the plan_file
        :type problem_file: str
        :return: Feedback on the physical/logical feasibility of the generated plan.
        :rtype: str
        """
        base_plan_feedback_prompt = get_prompt(Prompts.PLAN_FEEDBACK)
        with open(plan_file) as f:
            plan = f.read()
        plan_feedback_prompt = base_plan_feedback_prompt.format(
            # TODO: We need to update this to accept/gather other domain/problem instructions than ring_and_peg in the future
            task=get_prompt(Prompts.RING_AND_PEG_IMAGE),
            plan=plan,
        )
        return self.plan_feedback_module(
            scene=dspy.Image(str(images_dir / "peg_and_ring_plan_start.png")),
            task_and_plan=plan_feedback_prompt,
        ).feedback

    def forward(
        self, task_description: str, scene: dspy.Image | None = None
    ) -> dspy.Prediction:
        error = None
        self.generate_pddl_module(
            scene=scene,
            task_description=task_description,
        )
        if self.vars.domain_file is None or not is_domain_valid(self.vars.domain_file):
            error = PipelineError.DOMAIN_FAILURE
        elif self.vars.problem_file is None or not is_problem_valid(
            self.vars.domain_file, self.vars.problem_file
        ):
            error = PipelineError.PROBLEM_FAILURE
        else:
            plan = self._generate_plan(self.vars.domain_file, self.vars.problem_file)
            if isinstance(plan, FDErrorInfo):
                logger.debug(
                    f"# Failed to generate solvable domain and problem: {plan.error_message}"
                )
                error = plan.to_pipeline_error()
            else:
                self.vars.plan_file = plan
                return dspy.Prediction(out=error, plan_file=plan)
        return dspy.Prediction(out=error, plan_file=None)

    def _run_impl(self):
        prediction = self(
            dspy.Image(str(images_dir / "peg_and_ring_plan_start.png")),
            get_prompt(Prompts.RING_AND_PEG_IMAGE),
        )
        token_usage = prediction.get_lm_usage()
        assert token_usage is not None
        token_usage = token_usage[f"openai/{self._model_config.api_model_name}"]
        self.vars.input_tokens = token_usage["prompt_tokens"]
        self.vars.output_tokens = token_usage["completion_tokens"]
        self.log_and_clear_history()
        return self.create_result(error=prediction.out)
