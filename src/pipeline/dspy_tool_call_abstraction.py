from pathlib import Path

import dspy

from src.base.pipeline import Pipelines
from src.constants import project_root
from src.eval.fast_downward import FDErrorInfo, UnsolvabilityFeedback
from src.inference import Models
from src.pipeline.dspy_tool_calls import DSPyToolCallPipeline, GeneratePddlSignature
from src.pipeline.tool_calls import ToolCallPipeline
from src.utils.domains import Domains


class DSPyToolCallPipelineAbstraction(DSPyToolCallPipeline):
    """
    Version of dspy tool call pipeline with unsolvabiliity feedback.
    The feedback includes information on which predicates could get removed from the domain which would lead to
    it being solvable.
    The basis for this unsolvability feedback is originally from:
    Sreedharan, Sarath, et al. "Why Can't You Do That HAL? Explaining Unsolvability of Planning Tasks." IJCAI. 2019.
    """

    def generate_plan(self, domain_file: str, problem_file: str) -> str:
        """
        Try to generate a plan given a domain and problem file using the fast downward planning system.
        The function either answers witht the path to the generated plan if successfull or with error information if planning was unsuccessfull.
        The main types of errors that happen either stem from the plan / domain containing syntax mistakes or the domain / problem being ill-defined leading to the planner not being able to find a plan.

        :param domain_file: the path to the domain file
        :type domain_file: str
        :param problem_file: the path to the problem file
        :type problem_file: str
        :return: The path to the problem file or an error description
        :rtype: str
        """
        self.generate_plan_calls += 1
        plan_output = self._generate_plan(
            Path(domain_file), Path(problem_file), UnsolvabilityFeedback.ABSTRACTION
        )
        if isinstance(plan_output, FDErrorInfo):
            return "Fast Downward was unable to generate a plan" + plan_output.to_str()
        return f"Fast Downward successfully generated a plan under {plan_output}"

    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        dspy.Module.__init__(self)
        ToolCallPipeline.__init__(
            self, model, domain, pipeline or Pipelines.DSPY_TOOL_CALL_ABSTRACTION
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
