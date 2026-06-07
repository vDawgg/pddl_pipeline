import dspy

from src.base.schemas import Domains, Pipelines, Problems, Tools
from src.eval.fast_downward import FDErrorInfo, UnsolvabilityFeedback
from src.inference import Models
from src.pipeline.tool_call import GeneratePddlSignature, ToolCallPipeline


class ToolCallPipelineAbstraction(ToolCallPipeline):
    """
    Version of dspy tool call pipeline with unsolvabiliity feedback.
    The feedback includes information on which predicates could get removed from the domain which would lead to
    it being solvable.
    The basis for this unsolvability feedback is originally from:
    Sreedharan, Sarath, et al. "Why Can't You Do That HAL? Explaining Unsolvability of Planning Tasks." IJCAI. 2019.
    """

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
            self.vars.domain_file,
            self.vars.problem_file,
            UnsolvabilityFeedback.ABSTRACTION,
        )
        if isinstance(plan_output, FDErrorInfo):
            return (
                "Fast Downward was unable to generate a plan\n" + plan_output.to_str()
            )
        self.vars.plan_file = plan_output
        with open(plan_output) as f:
            return f"Fast Downward successfully generated a plan.\n\nGenerated plan:\n{f.read()}"

    def __init__(
        self,
        model: Models,
        domain: Domains,
        problem: Problems,
        ablate_tools: list[Tools] | None = None,
        pipeline: Pipelines | None = None,
        optimized_program: str | None = None,
    ):
        super().__init__(
            model,
            domain,
            problem,
            ablate_tools,
            pipeline or Pipelines.TOOL_CALL_ABSTRACTION,
            optimized_program,
        )

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
                self.get_plan_feedback,
            ],
            max_iters=50,
        )
