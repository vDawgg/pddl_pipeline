import logging
from typing import Literal

import dspy

from src.base.pipeline import CombinedMeta, FDErrorInfo, Pipelines
from src.base.schema import PipelineError
from src.constants import project_root
from src.inference import Models
from src.pipeline.tool_calls import ToolCallPipeline
from src.utils.domains import Domains
from src.utils.prompts import Prompts, get_prompt

logger = logging.getLogger(__name__)


class GeneratePddlSignature(dspy.Signature):
    __doc__ = get_prompt(Prompts.GENERATION_CONTEXT_TOOLS)

    task_description: str = dspy.InputField(
        desc="Natural language description of the planning task"
    )
    success: Literal["success", "failure"] = dspy.OutputField(
        desc="Whether the final PDDL domain and problem are syntactically valid and solvable."
    )


class DSPyToolCallPipeline(dspy.Module, ToolCallPipeline, metaclass=CombinedMeta):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        dspy.Module.__init__(self)
        ToolCallPipeline.__init__(
            self, model, domain, pipeline or Pipelines.DSPY_TOOL_CALL
        )

        key_path = project_root / self._model_config.key_file
        if not key_path.exists():
            raise FileNotFoundError(f"API key file not found: {key_path}")
        api_key = key_path.read_text().strip().split("\n")[0]
        self.lm = dspy.LM(
            model="openai/" + self._model_config.api_model_name,
            api_key=api_key,
            api_base=self._model_config.base_url,
            cache=False,  # Disable DSPy's built-in response caching
        )
        dspy.configure(lm=self.lm)

        self.generate_pddl_module = dspy.ReAct(
            GeneratePddlSignature,
            tools=[
                self.create_pddl_file,
                self.read_pddl_file,
                self.edit_lines,
                self.get_syntax_mistakes_domain,
                self.get_syntax_mistakes_problem,
                self.translate_pddl,
                self.generate_plan,
            ],
            max_iters=20,
        )

    def forward(self) -> dspy.Prediction:
        error = None
        self.generate_pddl_module(
            task_description=get_prompt(Prompts.RING_AND_PEG),
        )
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
        return dspy.Prediction(out=error)

    def compile_module(self):
        # TODO: Implement one of the dspy optimization functions for this module here.
        #       The function can take old PDDL problems and descriptions as input.
        #       -> Either take them from old competitions or look into one of the existing
        #          LLM PDDL generation benchmarking papers.
        ...

    def _run_impl(self):
        prediction = self()
        # Log interaction history in one shot and clear everything after
        # to always get interactions from last run only
        logger.debug(self.lm.inspect_history())
        self.num_model_calls = len(self.lm.history)
        self.lm.history.clear()
        return self.create_result(error=prediction.out)
