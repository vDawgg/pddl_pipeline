import logging
from typing import Literal

import dspy

from src.base.mappings import ACTION_SCHEMA_PROMPTS, DOMAIN_PROMPTS, IMAGES, OBJECT_NAME_PROMPTS, PROBLEM_PROMPTS
from src.base.pipeline import FDErrorInfo
from src.base.schemas import (
    Domains,
    PDDLFiles,
    PipelineError,
    Pipelines,
    Problems,
    Tools,
)
from src.constants import images_dir
from src.eval.val import is_domain_valid, is_problem_valid
from src.inference import Models
from src.pipeline.tool_call_curated import DSPyToolCallPipelineCurated
from src.utils.prompts import Prompts, get_domain_problem_prompt, get_prompt

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


class ActionMappingSignature(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT_ACTION_MAPPING)

    scene: dspy.Image = dspy.InputField(
        desc="An image of the initial scene of the planning task"
    )
    plan_and_actions: str = dspy.InputField(
        desc="The generated plan, PDDL domain and schema of the robots actions."
    )
    mapped_plan: str = dspy.OutputField(
        desc="The plan with the actions mapped to the robots action schema."
    )


class ToolCallImagePipeline(DSPyToolCallPipelineCurated):
    def __init__(
        self,
        model: Models,
        domain: Domains,
        problems: Problems,
        ablate_tools: list[Tools] | None = None,
        pipeline: Pipelines | None = None,
        optimized_program: str | None = None,
    ):
        super().__init__(
            model, domain, problems, ablate_tools, pipeline or Pipelines.TOOL_CALL_IMAGE
        )
        self.generate_pddl_module = dspy.ReAct(
            GeneratePddlSignature,
            tools=list(self.tools.values()),
            max_iters=50,
        )
        self.plan_feedback_module = dspy.Predict(PlanFeedbackSignature)
        self.action_mapping_module = dspy.Predict(ActionMappingSignature)

    def map_plan(self, plan_file: str) -> str:
        base_action_mapping_prompt = get_prompt(Prompts.ACTION_MAPPING)
        with open(plan_file) as f:
            action_mapping_prompt = base_action_mapping_prompt.format(
                domain=self.read_pddl_file(PDDLFiles.DOMAIN),
                plan=f.read(),
                action_schema=self.vars.action_schema,
                object_names=self.vars.object_names,
            )
            plan_mapping_out = self.action_mapping_module(
                scene=dspy.Image(str(images_dir / IMAGES[self.problem])),
                plan_and_actions=action_mapping_prompt,
            )
        with open(plan_file, "w") as f:
            f.write(plan_mapping_out.mapped_plan)
        return plan_mapping_out.mapped_plan

    def get_plan_feedback(self) -> str:
        """
        Get detailed feedback on the feasibility of a generated plan given the current task.

        :return: Feedback on the physical/logical feasibility of the generated plan.
        :rtype: str
        """
        self.vars.get_plan_feedback_calls += 1
        base_plan_feedback_prompt = get_prompt(Prompts.PLAN_FEEDBACK)
        if self.vars.plan_file is None:
            return "Could not generate feedback, as no plan has been generated to provide feedback on."
        mapped_plan = self.map_plan(self.vars.plan_file)
        plan_feedback_prompt = base_plan_feedback_prompt.format(
            task=get_domain_problem_prompt(
                DOMAIN_PROMPTS[self.domain],
                PROBLEM_PROMPTS[self.problem],
            ),
            plan=mapped_plan,
            action_schema=self.vars.action_schema,
        )
        return self.plan_feedback_module(
            scene=dspy.Image(str(images_dir / IMAGES[self.problem])),
            task_and_plan=plan_feedback_prompt,
        ).feedback

    def forward(
        self,
        task_description: str,
        action_schema: str,
        object_names: str,
        scene: dspy.Image | None = None,
    ) -> dspy.Prediction:
        error = None
        self.vars.task_description = task_description
        self.vars.action_schema = action_schema
        self.vars.object_names = object_names
        self.generate_pddl_module(
            scene=scene,
            task_description=task_description,
        )
        if self.vars.plan_file is not None:
            # Log history before gathering new instance below
            self.log_and_clear_history()
            self.map_plan(self.vars.plan_file)
        elif self.vars.domain_file is None or not is_domain_valid(
            self.vars.domain_file
        ):
            error = PipelineError.DOMAIN_FAILURE
        elif self.vars.problem_file is None or not is_problem_valid(
            self.vars.domain_file, self.vars.problem_file
        ):
            error = PipelineError.PROBLEM_FAILURE
        else:
            # Generated syntactically correct domain and problem, but no plan.
            plan = self._generate_plan(self.vars.domain_file, self.vars.problem_file)
            if isinstance(plan, FDErrorInfo):
                logger.debug(
                    f"# Failed to generate solvable domain and problem: {plan.error_message}"
                )
                error = plan.to_pipeline_error()
            else:
                self.vars.plan_file = plan
                self.map_plan(self.vars.plan_file)
        return dspy.Prediction(out=error, plan_file=None)

    def _run_impl(self):
        prediction = self(
            task_description=get_domain_problem_prompt(
                DOMAIN_PROMPTS[self.domain],
                PROBLEM_PROMPTS[self.problem],
            ),
            action_schema=get_prompt(ACTION_SCHEMA_PROMPTS[self.domain]),
            object_names=get_prompt(OBJECT_NAME_PROMPTS[self.domain]),
            scene=dspy.Image(str(images_dir / IMAGES[self.problem])),
        )
        token_usage = prediction.get_lm_usage()
        assert token_usage is not None
        token_usage = token_usage[f"openai/{self._model_config.api_model_name}"]
        self.vars.input_tokens = token_usage["prompt_tokens"]
        self.vars.output_tokens = token_usage["completion_tokens"]
        self.log_and_clear_history()
        return self.create_result(error=prediction.out)
