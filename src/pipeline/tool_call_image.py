from typing import Literal

import dspy

from src.base.pipeline import Pipelines
from src.constants import images_dir
from src.inference import Models
from src.pipeline.tool_call import ToolCallPipeline
from src.utils.domains import Domains
from src.utils.prompts import Prompts, get_prompt


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


class ToolCallImagePipeline(ToolCallPipeline):
    def __init__(
        self, model: Models, domain: Domains, pipeline: Pipelines | None = None
    ):
        super().__init__(model, domain, pipeline or Pipelines.TOOL_CALL_IMAGE)
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

    def _run_impl(self):
        prediction = self(
            dspy.Image(str(images_dir / "peg_and_ring_plan_start.png")),
            get_prompt(Prompts.RING_AND_PEG),
        )
        self.print_and_clear_history()
        return self.create_result(error=prediction.out)
