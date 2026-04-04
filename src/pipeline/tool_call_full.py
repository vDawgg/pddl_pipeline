from pathlib import Path

import dspy

from src.base.schemas import Domains, Pipelines, Problems, Tools
from src.constants import project_root
from src.eval.fast_downward import FDErrorInfo
from src.inference import Models
from src.pipeline.tool_call import GeneratePddlSignature, ToolCallPipeline


class DSPyToolCallPipelineFull(ToolCallPipeline):
    """Version of dspy tool call pipeline with full unsolvabiliity feedback"""

    def generate_plan(self) -> str:
        """
        Try to generate a plan given a domain and problem file using the fast downward planning system.
        The function answers with the generated plan or error information if the generation fails.
        The main types of errors that happen either stem from the plan / domain containing syntax mistakes or the domain / problem being ill-defined leading to the planner not being able to find a plan.

        :return: The outcome of the planning process
        :rtype: str
        """
        self.vars.generate_plan_calls += 1
        if self.vars.domain_file is None:
            return "Domain file has not yet been created"
        elif self.vars.problem_file is None:
            return "Problem file has not yet been created"
        plan_output = self._generate_plan(
            Path(self.vars.domain_file), Path(self.vars.problem_file)
        )
        if isinstance(plan_output, FDErrorInfo):
            return "Fast Downward was unable to generate a plan" + plan_output.to_str()
        return f"Fast Downward successfully generated a plan under {plan_output}"

    def __init__(
        self,
        model: Models,
        domain: Domains,
        problem: Problems,
        ablate_tools: list[Tools] | None = None,
        pipeline: Pipelines | None = None,
        optimized_program: str | None = None,
    ):
        dspy.Module.__init__(self)
        ToolCallPipeline.__init__(
            self, model, domain, problem, pipeline=pipeline or Pipelines.TOOL_CALL_FULL
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
